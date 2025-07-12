//===- IsolatePath.cpp - Code to isolate paths with UB --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass isolates code paths with undefined behavior from paths without
// undefined behavior, and then add a trap instruction on that path. This
// prevents code generation where, after the UB instruction's eliminated, the
// code can wander off the end of a function.
//
// For example, a nullptr dereference:
//
//   foo:
//     %phi.val = phi ptr [ %arrayidx.i, %pred1 ], [ null, %pred2 ]
//     %load.val = load i32, ptr %phi.val, align 4
//
// is converted into:
//
//   foo:
//     %load.val = load i32, ptr %ptr.val, align 4
//
//   foo.ub.path:
//     %load.val.ub = load volatile i32, ptr null, align 4
//     tail call void @llvm.trap()
//     unreachable
//
// Note: we allow the NULL dereference to actually occur so that code that
// wishes to catch the signal can do so.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/IsolatePath.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "isolate-path"

STATISTIC(NumIsolatedBlocks, "Number of isolated blocks");

static cl::opt<bool> ConvertUBToTrapUnreachable(
    "isolate-ub-path-to-trap-unreachable", cl::init(false), cl::Hidden,
    cl::desc("Isolate the UB path into one with a 'trap-unreachable' pair."));

/// Look through GEPs to see if the nullptr is accessed.
static bool HasUBAccess(BasicBlock *Parent, GetElementPtrInst *GEP) {
  for (Value *V : GEP->materialized_users()) {
    if (auto *G = dyn_cast<GetElementPtrInst>(V)) {
      if (G->getParent() != Parent)
        return false;
      return HasUBAccess(Parent, G);
    } else if (auto *LI = dyn_cast<LoadInst>(V)) {
      if (LI->getParent() != Parent)
        return false;
      if (GEP == LI->getPointerOperand())
        return true;
    } else if (auto *SI = dyn_cast<StoreInst>(V)) {
      if (SI->getParent() != Parent)
        return false;
      if (GEP == SI->getPointerOperand())
        return true;
    }
  }

  return false;
}

static std::pair<PHINode *, Instruction *> GetFirstUBInst(BasicBlock *BB) {
  // Find PHIs that have 'nullptr' inputs.
  SmallPtrSet<PHINode *, 4> NullptrPhis;
  for (PHINode &PN : BB->phis()) {
    if (!PN.getType()->isPointerTy())
      continue;

    for (Value *V : PN.incoming_values())
      if (isa<ConstantPointerNull>(V)) {
        NullptrPhis.insert(&PN);
        break;
      }
  }
  if (NullptrPhis.empty())
    return {};

  // Grab instructions that may be UB.
  SmallDenseMap<PHINode *, SmallPtrSet<Instruction *, 4>> MaybeUBInsts;
  for (PHINode *PN : NullptrPhis) {
    for (Value *V : PN->materialized_users()) {
      if (auto *LI = dyn_cast<LoadInst>(V)) {
        if (LI->getParent() == BB && PN == LI->getPointerOperand())
          MaybeUBInsts[PN].insert(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(V)) {
        if (SI->getParent() == BB && PN == SI->getPointerOperand())
          MaybeUBInsts[PN].insert(SI);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
        if (GEP->getParent() == BB && HasUBAccess(BB, GEP))
          MaybeUBInsts[PN].insert(GEP);
      }
    }
  }
  if (MaybeUBInsts.empty())
    return {};

  // Get the first UB instruction.
  PHINode *FirstUBPhiNode = nullptr;
  Instruction *FirstUBInst = nullptr;
  for (auto Element : MaybeUBInsts) {
    PHINode *PN = Element.getFirst();
    SmallPtrSetImpl<Instruction *> &Insts = Element.getSecond();

    for (Instruction &I : *BB) {
      if (&I == FirstUBInst)
        break;

      if (Insts.contains(&I)) {
        FirstUBPhiNode = PN;
        FirstUBInst = &I;
        break;
      }
    }
  }

  return std::make_pair(FirstUBPhiNode, FirstUBInst);
}

/// Convert any accesses of a nullptr within the BB into a trap.
bool IsolatePathPass::ProcessPointerUndefinedBehavior(BasicBlock *BB,
                                                      DomTreeUpdater *DTU) {
  if (!BB->canSplitPredecessors())
    return false;

  // Get the first UB instruction and associated PHI node.
  auto [FirstUBPhiNode, FirstUBInst] = GetFirstUBInst(BB);
  if (!FirstUBInst)
    return false;

  // Now that we have the first UB instruction and the PHI node associated with
  // it, determine how to split the predecessors.
  SmallPtrSet<BasicBlock *, 4> UBPhiPreds;
  SmallPtrSet<BasicBlock *, 4> NonUBPhiPreds;
  unsigned Index = 0;
  for (Value *V : FirstUBPhiNode->incoming_values())
    if (isa<ConstantPointerNull>(V))
      UBPhiPreds.insert(FirstUBPhiNode->getIncomingBlock(Index++));
    else
      NonUBPhiPreds.insert(FirstUBPhiNode->getIncomingBlock(Index++));

  if (NonUBPhiPreds.empty())
    // All paths have undefined behavior. Other passes will deal with this
    // code.
    return false;

  SmallVector<DominatorTree::UpdateType, 8> Updates;
  BasicBlock *UBBlock = nullptr;

  // Clone the block, isolating the UB instructions on their own path.
  ValueToValueMapTy VMap;
  UBBlock = CloneBasicBlock(BB, VMap, ".ub.path", BB->getParent());
  VMap[BB] = UBBlock;
  ++NumIsolatedBlocks;

  // Replace the UB predecessors' terminators' targets with the new block.
  llvm::for_each(UBPhiPreds, [&](BasicBlock *Pred) {
    Pred->getTerminator()->replaceSuccessorWith(BB, UBBlock);
  });

  // Remove predecessors of isolated paths from the original PHI nodes.
  for (PHINode &PN : BB->phis())
    PN.removeIncomingValueIf([&](unsigned I) {
      return UBPhiPreds.contains(PN.getIncomingBlock(I));
    });

  // Remove predecessors of valid paths from the isolated path PHI nodes.
  for (PHINode &PN : UBBlock->phis())
    PN.removeIncomingValueIf([&](unsigned I) {
      return NonUBPhiPreds.contains(PN.getIncomingBlock(I));
    });

  // Rewrite the instructions in the cloned block to refer to the instructions
  // in the cloned block.
  for (auto &I : *UBBlock) {
    RemapDbgRecordRange(BB->getModule(), I.getDbgRecordRange(), VMap,
                        RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    RemapInstruction(&I, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  }

  // Update the dominator tree.
  for (auto *Pred : UBPhiPreds) {
    Updates.push_back({DominatorTree::Insert, Pred, UBBlock});
    Updates.push_back({DominatorTree::Delete, Pred, BB});
  }

  if (ConvertUBToTrapUnreachable) {
    // Isolate the UB path into a trap-unreachable instruction.
    unsigned UBIndex = 0;

    // Get the index into the block of the first UB instruction.
    for (auto Iter = BB->begin(); Iter != BB->end(); ++Iter, ++UBIndex)
      if (&*Iter == FirstUBInst) {
        if (isa<LoadInst>(FirstUBInst))
          ++UBIndex;
        break;
      }

    // Remove the instructions following the nullptr dereference.
    for (unsigned Index = UBBlock->size(); Index > UBIndex; --Index)
      UBBlock->rbegin()->eraseFromParent();

    // Allow the NULL dereference to actually occur so that code that wishes to
    // catch the signal can do so.
    if (const auto *LI = dyn_cast<LoadInst>(&*UBBlock->rbegin()))
      const_cast<LoadInst *>(LI)->setVolatile(true);

    // Add a 'trap()' call.
    IRBuilder<> Builder(UBBlock);
    Function *TrapDecl =
        Intrinsic::getOrInsertDeclaration(BB->getModule(), Intrinsic::trap);
    Builder.CreateCall(TrapDecl);
  } else {
    // Remove all instructions.
    for (unsigned Index = UBBlock->size(); Index > 0; --Index)
      UBBlock->rbegin()->eraseFromParent();
  }

  // End in an 'unreachable' instruction.
  IRBuilder<> Builder(UBBlock);
  Builder.CreateUnreachable();

  if (!Updates.empty())
    DTU->applyUpdates(Updates);

  SplitUBBlocks.insert(UBBlock);
  return true;
}

PreservedAnalyses IsolatePathPass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  bool Changed = false;

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(&DT, &PDT, DomTreeUpdater::UpdateStrategy::Eager);

  // Use a worklist of blocks because we'll be adding new blocks to the
  // function and potentially processing the same block multiple times.
  std::vector<BasicBlock *> Blocks;
  Blocks.reserve(F.size());
  llvm::transform(F, std::back_inserter(Blocks),
                  [](BasicBlock &BB) { return &BB; });

  while (!Blocks.empty()) {
    BasicBlock *BB = Blocks.back();
    Blocks.pop_back();
    if (SplitUBBlocks.contains(BB))
      continue;

    // No PHI nodes.
    if (BB->phis().empty())
      continue;

    // Ignore landing and EH pads for now.
    // FIXME: Should we support them?
    if (BB->isLandingPad() || BB->isEHPad())
      continue;

    // Support some of the more common predecessor terminators.
    // FIXME: Add support for 'SwitchInst'.
    if (llvm::any_of(predecessors(BB), [&](BasicBlock *Pred) {
          Instruction *TI = Pred->getTerminator();
          return !isa<BranchInst>(TI) && !isa<ReturnInst>(TI) &&
                 !isa<SwitchInst>(TI);
        }))
      continue;

    if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator()))
      // If a BB has an edge to itself, then duplication of BB could result in
      // reallocation of the BB's PHI nodes.
      if (llvm::any_of(BI->successors(),
                       [&](BasicBlock *B) { return B == BB; }))
        continue;

    if (ProcessPointerUndefinedBehavior(BB, &DTU)) {
      // Reprocess the block to handle further UB instructions.
      Blocks.push_back(BB);
      Changed = true;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // FIXME: Should we update LoopInfo and LCCSA like in SplitBlockPredecessors?
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  return PA;
}
