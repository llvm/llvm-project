//===- SSAUpdaterBulk.cpp - Unstructured SSA Update Tool ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSAUpdaterBulk class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SSAUpdaterBulk.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"

using namespace llvm;

#define DEBUG_TYPE "ssaupdaterbulk"

/// Helper function for finding a block which should have a value for the given
/// user. For PHI-nodes this block is the corresponding predecessor, for other
/// instructions it's their parent block.
static BasicBlock *getUserBB(Use *U) {
  auto *User = cast<Instruction>(U->getUser());

  if (auto *UserPN = dyn_cast<PHINode>(User))
    return UserPN->getIncomingBlock(*U);
  else
    return User->getParent();
}

/// Add a new variable to the SSA rewriter. This needs to be called before
/// AddAvailableValue or AddUse calls.
unsigned SSAUpdaterBulk::AddVariable(StringRef Name, Type *Ty) {
  unsigned Var = Rewrites.size();
  LLVM_DEBUG(dbgs() << "SSAUpdater: Var=" << Var << ": initialized with Ty = "
                    << *Ty << ", Name = " << Name << "\n");
  RewriteInfo RI(Name, Ty);
  Rewrites.push_back(RI);
  return Var;
}

/// Indicate that a rewritten value is available in the specified block with the
/// specified value.
void SSAUpdaterBulk::AddAvailableValue(unsigned Var, BasicBlock *BB, Value *V) {
  assert(Var < Rewrites.size() && "Variable not found!");
  LLVM_DEBUG(dbgs() << "SSAUpdater: Var=" << Var
                    << ": added new available value " << *V << " in "
                    << BB->getName() << "\n");
  Rewrites[Var].Defines.emplace_back(BB, V);
}

/// Record a use of the symbolic value. This use will be updated with a
/// rewritten value when RewriteAllUses is called.
void SSAUpdaterBulk::AddUse(unsigned Var, Use *U) {
  assert(Var < Rewrites.size() && "Variable not found!");
  LLVM_DEBUG(dbgs() << "SSAUpdater: Var=" << Var << ": added a use" << *U->get()
                    << " in " << getUserBB(U)->getName() << "\n");
  Rewrites[Var].Uses.push_back(U);
}

/// Given sets of UsingBlocks and DefBlocks, compute the set of LiveInBlocks.
/// This is basically a subgraph limited by DefBlocks and UsingBlocks.
static void
ComputeLiveInBlocks(const SmallPtrSetImpl<BasicBlock *> &UsingBlocks,
                    const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
                    SmallPtrSetImpl<BasicBlock *> &LiveInBlocks,
                    PredIteratorCache &PredCache) {
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(UsingBlocks.begin(),
                                                    UsingBlocks.end());

  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (BasicBlock *P : PredCache.get(BB)) {
      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

struct BBValueInfo {
  Value *LiveInValue = nullptr;
  Value *LiveOutValue = nullptr;
};

/// Perform all the necessary updates, including new PHI-nodes insertion and the
/// requested uses update.
void SSAUpdaterBulk::RewriteAllUses(DominatorTree *DT,
                                    SmallVectorImpl<PHINode *> *InsertedPHIs) {
  DenseMap<BasicBlock *, BBValueInfo> BBInfos;
  for (RewriteInfo &R : Rewrites) {
    BBInfos.clear();

    // Compute locations for new phi-nodes.
    // For that we need to initialize DefBlocks from definitions in R.Defines,
    // UsingBlocks from uses in R.Uses, then compute LiveInBlocks, and then use
    // this set for computing iterated dominance frontier (IDF).
    // The IDF blocks are the blocks where we need to insert new phi-nodes.
    ForwardIDFCalculator IDF(*DT);
    LLVM_DEBUG(dbgs() << "SSAUpdater: rewriting " << R.Uses.size()
                      << " use(s)\n");

    SmallPtrSet<BasicBlock *, 2> DefBlocks(llvm::from_range,
                                           llvm::make_first_range(R.Defines));
    IDF.setDefiningBlocks(DefBlocks);

    SmallPtrSet<BasicBlock *, 2> UsingBlocks;
    for (Use *U : R.Uses)
      UsingBlocks.insert(getUserBB(U));

    SmallVector<BasicBlock *, 32> IDFBlocks;
    SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
    ComputeLiveInBlocks(UsingBlocks, DefBlocks, LiveInBlocks, PredCache);
    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.calculate(IDFBlocks);

    // Reserve sufficient buckets to prevent map growth. [1]
    BBInfos.reserve(LiveInBlocks.size() + DefBlocks.size());

    for (auto [BB, V] : R.Defines)
      BBInfos[BB].LiveOutValue = V;

    // We've computed IDF, now insert new phi-nodes there.
    for (BasicBlock *FrontierBB : IDFBlocks) {
      IRBuilder<> B(FrontierBB, FrontierBB->begin());
      PHINode *PN = B.CreatePHI(R.Ty, 0, R.Name);
      BBInfos[FrontierBB].LiveInValue = PN;
      if (InsertedPHIs)
        InsertedPHIs->push_back(PN);
    }

    // IsLiveOut indicates whether we are computing live-out values (true) or
    // live-in values (false).
    auto ComputeValue = [&](BasicBlock *BB, bool IsLiveOut) -> Value * {
      BBValueInfo *BBInfo = &BBInfos[BB];

      if (IsLiveOut && BBInfo->LiveOutValue)
        return BBInfo->LiveOutValue;

      if (BBInfo->LiveInValue)
        return BBInfo->LiveInValue;

      SmallVector<BBValueInfo *, 4> Stack = {BBInfo};
      Value *V = nullptr;

      while (DT->isReachableFromEntry(BB) && !PredCache.get(BB).empty() &&
             (BB = DT->getNode(BB)->getIDom()->getBlock())) {
        BBInfo = &BBInfos[BB];

        if (BBInfo->LiveOutValue) {
          V = BBInfo->LiveOutValue;
          break;
        }

        if (BBInfo->LiveInValue) {
          V = BBInfo->LiveInValue;
          break;
        }

        Stack.emplace_back(BBInfo);
      }

      if (!V)
        V = UndefValue::get(R.Ty);

      for (BBValueInfo *BBInfo : Stack)
        // Loop above can insert new entries into the BBInfos map: assume the
        // map shouldn't grow due to [1] and BBInfo references are valid.
        BBInfo->LiveInValue = V;

      return V;
    };

    // Fill in arguments of the inserted PHIs.
    for (BasicBlock *BB : IDFBlocks) {
      auto *PHI = cast<PHINode>(&BB->front());
      for (BasicBlock *Pred : PredCache.get(BB))
        PHI->addIncoming(ComputeValue(Pred, /*IsLiveOut=*/true), Pred);
    }

    // Rewrite actual uses with the inserted definitions.
    SmallPtrSet<Use *, 4> ProcessedUses;
    for (Use *U : R.Uses) {
      if (!ProcessedUses.insert(U).second)
        continue;

      auto *User = cast<Instruction>(U->getUser());
      BasicBlock *BB = getUserBB(U);
      Value *V = ComputeValue(BB, /*IsLiveOut=*/BB != User->getParent());
      Value *OldVal = U->get();
      assert(OldVal && "Invalid use!");
      // Notify that users of the existing value that it is being replaced.
      if (OldVal != V && OldVal->hasValueHandle())
        ValueHandleBase::ValueIsRAUWd(OldVal, V);
      LLVM_DEBUG(dbgs() << "SSAUpdater: replacing " << *OldVal << " with " << *V
                        << "\n");
      U->set(V);
    }
  }
}

// Perform a single pass of simplification over the worklist of PHIs.
// This should be called after RewriteAllUses() because simplifying PHIs
// immediately after creation would require updating all references to those
// PHIs in the BBValueInfo structures, which would necessitate additional
// reference tracking overhead.
static void simplifyPass(MutableArrayRef<PHINode *> Worklist,
                         const DataLayout &DL) {
  for (PHINode *&PHI : Worklist) {
    if (Value *Simplified = simplifyInstruction(PHI, DL)) {
      PHI->replaceAllUsesWith(Simplified);
      PHI->eraseFromParent();
      PHI = nullptr; // Mark as removed.
    }
  }
}

#ifndef NDEBUG // Should this be under EXPENSIVE_CHECKS?
// New PHI nodes should not reference one another but they may reference
// themselves or existing PHI nodes, and existing PHI nodes may reference new
// PHI nodes.
static bool
PHIAreRefEachOther(const iterator_range<BasicBlock::phi_iterator> NewPHIs) {
  SmallPtrSet<PHINode *, 8> NewPHISet;
  for (PHINode &PN : NewPHIs)
    NewPHISet.insert(&PN);
  for (PHINode &PHI : NewPHIs) {
    for (Value *V : PHI.incoming_values()) {
      PHINode *IncPHI = dyn_cast<PHINode>(V);
      if (IncPHI && IncPHI != &PHI && NewPHISet.contains(IncPHI))
        return true;
    }
  }
  return false;
}
#endif

static bool replaceIfIdentical(PHINode &PHI, PHINode &ReplPHI) {
  if (!PHI.isIdenticalToWhenDefined(&ReplPHI))
    return false;
  PHI.replaceAllUsesWith(&ReplPHI);
  PHI.eraseFromParent();
  return true;
}

bool EliminateNewDuplicatePHINodes(BasicBlock *BB,
                                   BasicBlock::phi_iterator FirstExistingPN) {
  auto NewPHIs = make_range(BB->phis().begin(), FirstExistingPN);
  assert(!PHIAreRefEachOther(NewPHIs));

  // Deduplicate new PHIs first to reduce the number of comparisons on the
  // following new -> existing pass.
  bool Changed = false;
  for (auto I = BB->phis().begin(); I != FirstExistingPN; ++I) {
    for (auto J = std::next(I); J != FirstExistingPN;) {
      Changed |= replaceIfIdentical(*J++, *I);
    }
  }

  // Iterate over existing PHIs and replace identical new PHIs.
  for (PHINode &ExistingPHI : make_range(FirstExistingPN, BB->phis().end())) {
    auto I = BB->phis().begin();
    assert(I != FirstExistingPN); // Should be at least one new PHI.
    do {
      Changed |= replaceIfIdentical(*I++, ExistingPHI);
    } while (I != FirstExistingPN);
    if (BB->phis().begin() == FirstExistingPN)
      return Changed;
  }
  return Changed;
}

static void deduplicatePass(ArrayRef<PHINode *> Worklist) {
  SmallDenseMap<BasicBlock *, unsigned> BBs;
  for (PHINode *PHI : Worklist) {
    if (PHI)
      ++BBs[PHI->getParent()];
  }

  for (auto [BB, NumNewPHIs] : BBs) {
    auto FirstExistingPN = std::next(BB->phis().begin(), NumNewPHIs);
    EliminateNewDuplicatePHINodes(BB, FirstExistingPN);
  }
}

void SSAUpdaterBulk::RewriteAndOptimizeAllUses(DominatorTree &DT) {
  SmallVector<PHINode *, 4> PHIs;
  RewriteAllUses(&DT, &PHIs);
  if (PHIs.empty())
    return;

  simplifyPass(PHIs, PHIs.front()->getParent()->getDataLayout());
  deduplicatePass(PHIs);
}
