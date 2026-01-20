//===- CodeMoverUtils.cpp - CodeMover Utilities ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform movements on basic blocks, and instructions
// contained within a function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;

#define DEBUG_TYPE "codemover-utils"

STATISTIC(HasDependences,
          "Cannot move across instructions that has memory dependences");
STATISTIC(MayThrowException, "Cannot move across instructions that may throw");
STATISTIC(NotMovedPHINode, "Movement of PHINodes are not supported");
STATISTIC(NotMovedTerminator, "Movement of Terminator are not supported");

static bool domTreeLevelBefore(DominatorTree *DT, const Instruction *InstA,
                               const Instruction *InstB) {
  // Use ordered basic block in case the 2 instructions are in the same
  // block.
  if (InstA->getParent() == InstB->getParent())
    return InstA->comesBefore(InstB);

  DomTreeNode *DA = DT->getNode(InstA->getParent());
  DomTreeNode *DB = DT->getNode(InstB->getParent());
  return DA->getLevel() < DB->getLevel();
}

static bool reportInvalidCandidate(const Instruction &I,
                                   llvm::Statistic &Stat) {
  ++Stat;
  LLVM_DEBUG(dbgs() << "Unable to move instruction: " << I << ". "
                    << Stat.getDesc());
  return false;
}

/// Collect all instructions in between \p StartInst and \p EndInst, and store
/// them in \p InBetweenInsts.
static void
collectInstructionsInBetween(Instruction &StartInst, const Instruction &EndInst,
                             SmallPtrSetImpl<Instruction *> &InBetweenInsts) {
  assert(InBetweenInsts.empty() && "Expecting InBetweenInsts to be empty");

  /// Get the next instructions of \p I, and push them to \p WorkList.
  auto getNextInsts = [](Instruction &I,
                         SmallPtrSetImpl<Instruction *> &WorkList) {
    if (Instruction *NextInst = I.getNextNode())
      WorkList.insert(NextInst);
    else {
      assert(I.isTerminator() && "Expecting a terminator instruction");
      for (BasicBlock *Succ : successors(&I))
        WorkList.insert(&Succ->front());
    }
  };

  SmallPtrSet<Instruction *, 10> WorkList;
  getNextInsts(StartInst, WorkList);
  while (!WorkList.empty()) {
    Instruction *CurInst = *WorkList.begin();
    WorkList.erase(CurInst);

    if (CurInst == &EndInst)
      continue;

    if (!InBetweenInsts.insert(CurInst).second)
      continue;

    getNextInsts(*CurInst, WorkList);
  }
}

bool llvm::isSafeToMoveBefore(Instruction &I, Instruction &InsertPoint,
                              DominatorTree &DT, const PostDominatorTree *PDT,
                              DependenceInfo *DI, bool CheckForEntireBlock) {
  // Skip tests when we don't have PDT or DI
  if (!PDT || !DI)
    return false;

  // Cannot move itself before itself.
  if (&I == &InsertPoint)
    return false;

  // Not moved.
  if (I.getNextNode() == &InsertPoint)
    return true;

  if (isa<PHINode>(I) || isa<PHINode>(InsertPoint))
    return reportInvalidCandidate(I, NotMovedPHINode);

  if (I.isTerminator())
    return reportInvalidCandidate(I, NotMovedTerminator);

  if (isReachedBefore(&I, &InsertPoint, &DT, PDT))
    for (const Use &U : I.uses())
      if (auto *UserInst = dyn_cast<Instruction>(U.getUser())) {
        // If InsertPoint is in a BB that comes after I, then we cannot move if
        // I is used in the terminator of the current BB.
        if (I.getParent() == InsertPoint.getParent() &&
            UserInst == I.getParent()->getTerminator())
          return false;
        if (UserInst != &InsertPoint && !DT.dominates(&InsertPoint, U)) {
          // If UserInst is an instruction that appears later in the same BB as
          // I, then it is okay to move since I will still be available when
          // UserInst is executed.
          if (CheckForEntireBlock && I.getParent() == UserInst->getParent() &&
              DT.dominates(&I, UserInst))
            continue;
          return false;
        }
      }
  if (isReachedBefore(&InsertPoint, &I, &DT, PDT))
    for (const Value *Op : I.operands())
      if (auto *OpInst = dyn_cast<Instruction>(Op)) {
        if (&InsertPoint == OpInst)
          return false;
        // If OpInst is an instruction that appears earlier in the same BB as
        // I, then it is okay to move since OpInst will still be available.
        if (CheckForEntireBlock && I.getParent() == OpInst->getParent() &&
            DT.dominates(OpInst, &I))
          continue;
        if (!DT.dominates(OpInst, &InsertPoint))
          return false;
      }

  DT.updateDFSNumbers();
  const bool MoveForward = domTreeLevelBefore(&DT, &I, &InsertPoint);
  Instruction &StartInst = (MoveForward ? I : InsertPoint);
  Instruction &EndInst = (MoveForward ? InsertPoint : I);
  SmallPtrSet<Instruction *, 10> InstsToCheck;
  collectInstructionsInBetween(StartInst, EndInst, InstsToCheck);
  if (!MoveForward)
    InstsToCheck.insert(&InsertPoint);

  // Check if there exists instructions which may throw, may synchonize, or may
  // never return, from I to InsertPoint.
  if (!isSafeToSpeculativelyExecute(&I))
    if (llvm::any_of(InstsToCheck, [](Instruction *I) {
          if (I->mayThrow())
            return true;

          const CallBase *CB = dyn_cast<CallBase>(I);
          if (!CB)
            return false;
          if (!CB->hasFnAttr(Attribute::WillReturn))
            return true;
          if (!CB->hasFnAttr(Attribute::NoSync))
            return true;

          return false;
        })) {
      return reportInvalidCandidate(I, MayThrowException);
    }

  // Check if I has any output/flow/anti dependences with instructions from \p
  // StartInst to \p EndInst.
  if (llvm::any_of(InstsToCheck, [&DI, &I](Instruction *CurInst) {
        auto DepResult = DI->depends(&I, CurInst);
        if (DepResult && (DepResult->isOutput() || DepResult->isFlow() ||
                          DepResult->isAnti()))
          return true;
        return false;
      }))
    return reportInvalidCandidate(I, HasDependences);

  return true;
}

bool llvm::isSafeToMoveBefore(BasicBlock &BB, Instruction &InsertPoint,
                              DominatorTree &DT, const PostDominatorTree *PDT,
                              DependenceInfo *DI) {
  return llvm::all_of(BB, [&](Instruction &I) {
    if (BB.getTerminator() == &I)
      return true;

    return isSafeToMoveBefore(I, InsertPoint, DT, PDT, DI,
                              /*CheckForEntireBlock=*/true);
  });
}

void llvm::moveInstructionsToTheBeginning(BasicBlock &FromBB, BasicBlock &ToBB,
                                          DominatorTree &DT,
                                          const PostDominatorTree &PDT,
                                          DependenceInfo &DI) {
  for (Instruction &I :
       llvm::make_early_inc_range(llvm::drop_begin(llvm::reverse(FromBB)))) {
    BasicBlock::iterator MovePos = ToBB.getFirstNonPHIOrDbg();

    if (isSafeToMoveBefore(I, *MovePos, DT, &PDT, &DI))
      I.moveBeforePreserving(MovePos);
  }
}

void llvm::moveInstructionsToTheEnd(BasicBlock &FromBB, BasicBlock &ToBB,
                                    DominatorTree &DT,
                                    const PostDominatorTree &PDT,
                                    DependenceInfo &DI) {
  Instruction *MovePos = ToBB.getTerminator();
  while (FromBB.size() > 1) {
    Instruction &I = FromBB.front();
    if (isSafeToMoveBefore(I, *MovePos, DT, &PDT, &DI))
      I.moveBeforePreserving(MovePos->getIterator());
  }
}

bool llvm::nonStrictlyPostDominate(const BasicBlock *ThisBlock,
                                   const BasicBlock *OtherBlock,
                                   const DominatorTree *DT,
                                   const PostDominatorTree *PDT) {
  const BasicBlock *CommonDominator =
      DT->findNearestCommonDominator(ThisBlock, OtherBlock);
  if (CommonDominator == nullptr)
    return false;

  /// Recursively check the predecessors of \p ThisBlock up to
  /// their common dominator, and see if any of them post-dominates
  /// \p OtherBlock.
  SmallVector<const BasicBlock *, 8> WorkList;
  SmallPtrSet<const BasicBlock *, 8> Visited;
  WorkList.push_back(ThisBlock);
  while (!WorkList.empty()) {
    const BasicBlock *CurBlock = WorkList.pop_back_val();
    Visited.insert(CurBlock);
    if (PDT->dominates(CurBlock, OtherBlock))
      return true;

    for (const auto *Pred : predecessors(CurBlock)) {
      if (Pred == CommonDominator || Visited.count(Pred))
        continue;
      WorkList.push_back(Pred);
    }
  }
  return false;
}

bool llvm::isReachedBefore(const Instruction *I0, const Instruction *I1,
                           const DominatorTree *DT,
                           const PostDominatorTree *PDT) {
  const BasicBlock *BB0 = I0->getParent();
  const BasicBlock *BB1 = I1->getParent();
  if (BB0 == BB1)
    return DT->dominates(I0, I1);

  return nonStrictlyPostDominate(BB1, BB0, DT, PDT);
}
