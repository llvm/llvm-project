//===- CycleConvergenceExtend.cpp - Extend cycle body for convergence -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// NOTE: It is not clear if the effects of this transform can survive other
// control flow transforms such as jump-threading. Whether or not every such
// transform can preserve this CFG, and even if it can, whether that transform
// should preserve this CFG has not been determined yet.
//
// For now, this transform is meant to be used as late as possible, when
// preparing the CFG for code generation on targets that support convergence
// control tokens, such as AMDGPU. It is possible that the transform may
// eventually be merged into the structurizer or similar passes.
//
// But notably, this transform serves as a good WYSIWYM demonstration of
// convergence control tokens.
// ===----------------------------------------------------------------------===//
//
// This file implements a pass to extend cycles: if a token T defined in a cycle
// L is used at U outside of L, then the entire cycle nest is modified so that
// every path P from L to U is included in the body of L, including any sibling
// cycles whose header lies on P.
//
// Input CFG:
//
//         +-------------------+
//         | A: token %a = ... | <+
//         +-------------------+  |
//           |                    |
//           v                    |
//    +--> +-------------------+  |
//    |    | B: token %b = ... |  |
//    +--- +-------------------+  |
//           |                    |
//           v                    |
//         +-------------------+  |
//         |         C         | -+
//         +-------------------+
//           |
//           v
//         +-------------------+
//         |  D: use token %b  |
//         |     use token %a  |
//         +-------------------+
//
// Both cycles in the above nest need to be extended to contain the respective
// uses %d1 and %d2. To make this work, the block D needs to be split into two
// blocks "D1;D2" so that D1 is absorbed by the inner cycle while D2 is absorbed
// by the outer cycle.
//
// Transformed CFG:
//
//            +-------------------+
//            | A: token %a = ... | <-----+
//            +-------------------+       |
//              |                         |
//              v                         |
//            +-------------------+       |
//    +-----> | B: token %b = ... | -+    |
//    |       +-------------------+  |    |
//    |         |                    |    |
//    |         v                    |    |
//    |       +-------------------+  |    |
//    |    +- |         C         |  |    |
//    |    |  +-------------------+  |    |
//    |    |    |                    |    |
//    |    |    v                    |    |
//    |    |  +-------------------+  |    |
//    |    |  | D1: use token %b  |  |    |
//    |    |  +-------------------+  |    |
//    |    |    |                    |    |
//    |    |    v                    |    |
//    |    |  +-------------------+  |    |
//    +----+- |       Flow1       | <+    |
//         |  +-------------------+       |
//         |    |                         |
//         |    v                         |
//         |  +-------------------+       |
//         |  | D2: use token %a  |       |
//         |  +-------------------+       |
//         |    |                         |
//         |    v                         |
//         |  +-------------------+       |
//         +> |       Flow2       | ------+
//            +-------------------+
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ExtendCycleConvergence.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "extend-cycle-convergence"

using namespace llvm;

using BBSetVector = SetVector<BasicBlock *>;
using ExtensionMap = DenseMap<Cycle *, SmallVector<CallBase *>>;
// A single BB very rarely defines more than one token.
using TokenDefsMap = DenseMap<BasicBlock *, SmallVector<CallBase *, 1>>;
using TokenDefUsesMap = DenseMap<CallBase *, SmallVector<CallBase *>>;

static void updateTokenDefs(TokenDefsMap &TokenDefs, BasicBlock &BB) {
  TokenDefsMap::mapped_type Defs;
  for (Instruction &I : BB) {
    if (isa<ConvergenceControlInst>(I))
      Defs.push_back(cast<CallBase>(&I));
  }
  if (Defs.empty()) {
    TokenDefs.erase(&BB);
    return;
  }
  TokenDefs.insert_or_assign(&BB, std::move(Defs));
}

static bool splitForExtension(CycleInfo &CI, Cycle *DefCycle, BasicBlock *BB,
                              CallBase *TokenUse, TokenDefsMap &TokenDefs,
                              DomTreeUpdater &DTU) {
  if (DefCycle->contains(BB))
    return false;
  BasicBlock *NewBB = BB->splitBasicBlockBefore(TokenUse->getNextNode(),
                                                BB->getName() + ".ext");
  DTU.getDomTree().splitBlock(NewBB);
  if (Cycle *BBCycle = CI.getCycle(BB))
    CI.addBlockToCycle(NewBB, BBCycle);
  updateTokenDefs(TokenDefs, *BB);
  updateTokenDefs(TokenDefs, *NewBB);
  return true;
}

static void locateExtensions(CycleInfo &CI, Cycle *DefCycle, BasicBlock *BB,
                             TokenDefsMap &TokenDefs,
                             TokenDefUsesMap &TokenDefUses, DomTreeUpdater &DTU,
                             SmallVectorImpl<CallBase *> &ExtPoints) {
  if (auto Iter = TokenDefs.find(BB); Iter != TokenDefs.end()) {
    for (CallBase *Def : Iter->second) {
      for (CallBase *TokenUse : TokenDefUses[Def]) {
        BasicBlock *BB = TokenUse->getParent();
        if (splitForExtension(CI, DefCycle, BB, TokenUse, TokenDefs, DTU)) {
          ExtPoints.push_back(TokenUse);
        }
      }
    }
  }
}

static void initialize(ExtensionMap &ExtBorder, TokenDefsMap &TokenDefs,
                       TokenDefUsesMap &TokenDefUses, Function &F,
                       CycleInfo &CI, DomTreeUpdater &DTU) {
  for (BasicBlock *BB : depth_first(&F)) {
    updateTokenDefs(TokenDefs, *BB);
    for (Instruction &I : *BB) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (auto *TokenDef =
                cast_or_null<CallBase>(CB->getConvergenceControlToken())) {
          TokenDefUses[TokenDef].push_back(CB);
        }
      }
    }
  }

  for (BasicBlock *BB : depth_first(&F)) {
    if (Cycle *DefCycle = CI.getCycle(BB)) {
      SmallVector<CallBase *> ExtPoints;
      locateExtensions(CI, DefCycle, BB, TokenDefs, TokenDefUses, DTU,
                       ExtPoints);
      if (!ExtPoints.empty()) {
        auto Success = ExtBorder.try_emplace(DefCycle, std::move(ExtPoints));
        (void)Success;
        assert(Success.second);
      }
    }
  }
}

static bool hasSuccInsideCycle(BasicBlock *BB, Cycle *C) {
  for (BasicBlock *Succ : successors(BB)) {
    if (C->contains(Succ))
      return true;
  }
  return false;
}

PreservedAnalyses ExtendCycleConvergencePass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "===== Extend cycle convergence for function "
                    << F.getName() << "\n");

  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  CycleInfo &CI = AM.getResult<CycleAnalysis>(F);
  ExtensionMap ExtBorder;
  TokenDefsMap TokenDefs;
  TokenDefUsesMap TokenDefUses;

  initialize(ExtBorder, TokenDefs, TokenDefUses, F, CI, DTU);
  if (ExtBorder.empty())
    return PreservedAnalyses::all();

  for (auto Iter : ExtBorder) {
    Cycle *DefCycle = Iter.first;
    auto &ExtList = Iter.second;
    SmallVector<BasicBlock *> TransferredBlocks;

    LLVM_DEBUG(dbgs() << "Extend cycle with header "
                      << DefCycle->getHeader()->getName());
    assert(!ExtList.empty());
    for (auto I = ExtList.begin(); I != ExtList.end(); ++I) {
      CallBase *ExtPoint = *I;
      if (DefCycle->contains(ExtPoint->getParent()))
        continue;
      LLVM_DEBUG(dbgs() << "\n  up to " << ExtPoint->getParent()->getName()
                        << "\n  for token used:  " << *ExtPoint << "\n");
      CI.extendCycle(DefCycle, ExtPoint->getParent(), &TransferredBlocks);
      for (BasicBlock *BB : TransferredBlocks) {
        locateExtensions(CI, DefCycle, BB, TokenDefs, TokenDefUses, DTU,
                         ExtList);
      }
    };

    LLVM_DEBUG(dbgs() << "After extension:\n" << CI.print(DefCycle) << "\n");

    // Now that we have absorbed the convergence extensions into the cycle, we
    // need to introduce dummy backedges so that the cycle remains strongly
    // connected.
    BBSetVector Incoming, Outgoing;
    SmallVector<BasicBlock *> GuardBlocks;

    for (CallBase *ExtPoint : ExtList) {
      BasicBlock *BB = ExtPoint->getParent();
      if (!hasSuccInsideCycle(BB, DefCycle))
        Incoming.insert(BB);
    }
    for (BasicBlock *BB : Incoming) {
      for (BasicBlock *Succ : successors(BB)) {
        if (!DefCycle->contains(Succ))
          Outgoing.insert(Succ);
      }
    }

    // Redirect the backedges as well, just to add non-trivial edges to the ones
    // being redirected.
    for (BasicBlock *Pred : predecessors(DefCycle->getHeader())) {
      if (DefCycle->contains(Pred))
        Incoming.insert(Pred);
    }
    // We don't touch the exiting edges of the latches simply because
    // redirecting them is not a post-condition of this transform. Separately,
    // the header must be the last Outgoing block so that the entire chain of
    // guard blocks is included in the cycle.
    Outgoing.insert(DefCycle->getHeader());

    CreateControlFlowHub(&DTU, GuardBlocks, Incoming, Outgoing, "Extend");
    for (BasicBlock *BB : GuardBlocks)
      CI.addBlockToCycle(BB, DefCycle);
    DTU.flush();
  }

#if !defined(NDEBUG)
#if defined(EXPENSIVE_CHECKS)
  assert(DT.verify(DominatorTree::VerificationLevel::Full));
#else
  assert(DT.verify(DominatorTree::VerificationLevel::Fast));
#endif // EXPENSIVE_CHECKS
  CI.validateTree();
#endif // NDEBUG

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<CycleAnalysis>();
  return PA;
}
