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
STATISTIC(NotControlFlowEquivalent,
          "Instructions are not control flow equivalent");
STATISTIC(NotMovedPHINode, "Movement of PHINodes are not supported");
STATISTIC(NotMovedTerminator, "Movement of Terminator are not supported");

bool llvm::isControlFlowEquivalent(const Instruction &I0, const Instruction &I1,
                                   const DominatorTree &DT,
                                   const PostDominatorTree &PDT) {
  return isControlFlowEquivalent(*I0.getParent(), *I1.getParent(), DT, PDT);
}

bool llvm::isControlFlowEquivalent(const BasicBlock &BB0, const BasicBlock &BB1,
                                   const DominatorTree &DT,
                                   const PostDominatorTree &PDT) {
  if (&BB0 == &BB1)
    return true;

  return ((DT.dominates(&BB0, &BB1) && PDT.dominates(&BB1, &BB0)) ||
          (PDT.dominates(&BB0, &BB1) && DT.dominates(&BB1, &BB0)));
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
                              const DominatorTree &DT,
                              const PostDominatorTree &PDT,
                              DependenceInfo &DI) {
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

  // TODO remove this limitation.
  if (!isControlFlowEquivalent(I, InsertPoint, DT, PDT))
    return reportInvalidCandidate(I, NotControlFlowEquivalent);

  // As I and InsertPoint are control flow equivalent, if I dominates
  // InsertPoint, then I comes before InsertPoint.
  const bool MoveForward = DT.dominates(&I, &InsertPoint);
  if (MoveForward) {
    // When I is being moved forward, we need to make sure the InsertPoint
    // dominates every users. Or else, a user may be using an undefined I.
    for (const Use &U : I.uses())
      if (auto *UserInst = dyn_cast<Instruction>(U.getUser()))
        if (UserInst != &InsertPoint && !DT.dominates(&InsertPoint, U))
          return false;
  } else {
    // When I is being moved backward, we need to make sure all its opernads
    // dominates the InsertPoint. Or else, an operand may be undefined for I.
    for (const Value *Op : I.operands())
      if (auto *OpInst = dyn_cast<Instruction>(Op))
        if (&InsertPoint == OpInst || !DT.dominates(OpInst, &InsertPoint))
          return false;
  }

  Instruction &StartInst = (MoveForward ? I : InsertPoint);
  Instruction &EndInst = (MoveForward ? InsertPoint : I);
  SmallPtrSet<Instruction *, 10> InstsToCheck;
  collectInstructionsInBetween(StartInst, EndInst, InstsToCheck);
  if (!MoveForward)
    InstsToCheck.insert(&InsertPoint);

  // Check if there exists instructions which may throw, may synchonize, or may
  // never return, from I to InsertPoint.
  if (!isSafeToSpeculativelyExecute(&I))
    if (std::any_of(InstsToCheck.begin(), InstsToCheck.end(),
                    [](Instruction *I) {
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
  if (std::any_of(InstsToCheck.begin(), InstsToCheck.end(),
                  [&DI, &I](Instruction *CurInst) {
                    auto DepResult = DI.depends(&I, CurInst, true);
                    if (DepResult &&
                        (DepResult->isOutput() || DepResult->isFlow() ||
                         DepResult->isAnti()))
                      return true;
                    return false;
                  }))
    return reportInvalidCandidate(I, HasDependences);

  return true;
}

void llvm::moveInstsBottomUp(BasicBlock &FromBB, BasicBlock &ToBB,
                             const DominatorTree &DT,
                             const PostDominatorTree &PDT, DependenceInfo &DI) {
  for (auto It = ++FromBB.rbegin(); It != FromBB.rend();) {
    Instruction *MovePos = ToBB.getFirstNonPHIOrDbg();
    Instruction &I = *It;
    // Increment the iterator before modifying FromBB.
    ++It;

    if (isSafeToMoveBefore(I, *MovePos, DT, PDT, DI))
      I.moveBefore(MovePos);
  }
}
