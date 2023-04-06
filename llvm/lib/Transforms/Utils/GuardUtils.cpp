//===-- GuardUtils.cpp - Utils for work with guards -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utils that are used to perform transformations related to guards and their
// conditions.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/GuardUtils.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<uint32_t> PredicatePassBranchWeight(
    "guards-predicate-pass-branch-weight", cl::Hidden, cl::init(1 << 20),
    cl::desc("The probability of a guard failing is assumed to be the "
             "reciprocal of this value (default = 1 << 20)"));

void llvm::makeGuardControlFlowExplicit(Function *DeoptIntrinsic,
                                        CallInst *Guard, bool UseWC) {
  OperandBundleDef DeoptOB(*Guard->getOperandBundle(LLVMContext::OB_deopt));
  SmallVector<Value *, 4> Args(drop_begin(Guard->args()));

  auto *CheckBB = Guard->getParent();
  auto *DeoptBlockTerm =
      SplitBlockAndInsertIfThen(Guard->getArgOperand(0), Guard, true);

  auto *CheckBI = cast<BranchInst>(CheckBB->getTerminator());

  // SplitBlockAndInsertIfThen inserts control flow that branches to
  // DeoptBlockTerm if the condition is true.  We want the opposite.
  CheckBI->swapSuccessors();

  CheckBI->getSuccessor(0)->setName("guarded");
  CheckBI->getSuccessor(1)->setName("deopt");

  if (auto *MD = Guard->getMetadata(LLVMContext::MD_make_implicit))
    CheckBI->setMetadata(LLVMContext::MD_make_implicit, MD);

  MDBuilder MDB(Guard->getContext());
  CheckBI->setMetadata(LLVMContext::MD_prof,
                       MDB.createBranchWeights(PredicatePassBranchWeight, 1));

  IRBuilder<> B(DeoptBlockTerm);
  auto *DeoptCall = B.CreateCall(DeoptIntrinsic, Args, {DeoptOB}, "");

  if (DeoptIntrinsic->getReturnType()->isVoidTy()) {
    B.CreateRetVoid();
  } else {
    DeoptCall->setName("deoptcall");
    B.CreateRet(DeoptCall);
  }

  DeoptCall->setCallingConv(Guard->getCallingConv());
  DeoptBlockTerm->eraseFromParent();

  if (UseWC) {
    // We want the guard to be expressed as explicit control flow, but still be
    // widenable. For that, we add Widenable Condition intrinsic call to the
    // guard's condition.
    IRBuilder<> B(CheckBI);
    auto *WC = B.CreateIntrinsic(Intrinsic::experimental_widenable_condition,
                                 {}, {}, nullptr, "widenable_cond");
    CheckBI->setCondition(B.CreateAnd(CheckBI->getCondition(), WC,
                                      "exiplicit_guard_cond"));
    assert(isWidenableBranch(CheckBI) && "Branch must be widenable.");
  }
}

#ifndef NDEBUG
static bool isLoopInvariantWidenableCond(Value *WCCond, const Loop &L) {
  bool IsLoopInv = L.isLoopInvariant(WCCond);
  if (IsLoopInv || !isa<Instruction>(WCCond))
    return IsLoopInv;

  return L.hasLoopInvariantOperands(cast<Instruction>(WCCond));
}
#endif

void llvm::widenWidenableBranch(BranchInst *WidenableBR, Value *NewCond,
                                const LoopInfo &LI) {
  assert(isWidenableBranch(WidenableBR) && "precondition");

  // The tempting trivially option is to produce something like this:
  // br (and oldcond, newcond) where oldcond is assumed to contain a widenable
  // condition, but that doesn't match the pattern parseWidenableBranch expects
  // so we have to be more sophisticated.

  Use *C, *WC;
  BasicBlock *IfTrueBB, *IfFalseBB;
  parseWidenableBranch(WidenableBR, C, WC, IfTrueBB, IfFalseBB);
#ifndef NDEBUG
  bool IsLoopInvariantWidenableBR = false;
  auto *L = LI.getLoopFor(WidenableBR->getParent());
  if (L)
    IsLoopInvariantWidenableBR =
        isLoopInvariantWidenableCond(WidenableBR->getCondition(), *L);
#endif
  if (!C) {
    // br (wc()), ... form
    IRBuilder<> B(WidenableBR);
    WidenableBR->setCondition(B.CreateAnd(NewCond, WC->get()));
  } else {
    // br (wc & C), ... form
    IRBuilder<> B(WidenableBR);
    C->set(B.CreateAnd(NewCond, C->get()));
    Instruction *WCAnd = cast<Instruction>(WidenableBR->getCondition());
    // Condition is only guaranteed to dominate branch
    WCAnd->moveBefore(WidenableBR);    
  }
#ifndef NDEBUG
  if (IsLoopInvariantWidenableBR) {
    auto *L = LI.getLoopFor(WidenableBR->getParent());
    // NB! This can also be loop-varying if we missed hoisting the IR out, which
    // was infact loop-invariant. This would imply a pass-ordering issue and the
    // assert should be dealt with as such.
    assert(isLoopInvariantWidenableCond(WidenableBR->getCondition(), *L) &&
           "Loop invariant widenable condition made loop variant!");
  }
#endif
  assert(isWidenableBranch(WidenableBR) && "preserve widenabiliy");
}

void llvm::setWidenableBranchCond(BranchInst *WidenableBR, Value *NewCond,
                                  const LoopInfo &LI) {
  assert(isWidenableBranch(WidenableBR) && "precondition");

  Use *C, *WC;
  BasicBlock *IfTrueBB, *IfFalseBB;
  parseWidenableBranch(WidenableBR, C, WC, IfTrueBB, IfFalseBB);
#ifndef NDEBUG
  bool IsLoopInvariantWidenableBR = false;
  auto *L = LI.getLoopFor(WidenableBR->getParent());
  if (L)
    IsLoopInvariantWidenableBR =
        isLoopInvariantWidenableCond(WidenableBR->getCondition(), *L);
#endif
  if (!C) {
    // br (wc()), ... form
    IRBuilder<> B(WidenableBR);
    WidenableBR->setCondition(B.CreateAnd(NewCond, WC->get()));
  } else {
    // br (wc & C), ... form
    Instruction *WCAnd = cast<Instruction>(WidenableBR->getCondition());
    // Condition is only guaranteed to dominate branch
    WCAnd->moveBefore(WidenableBR);
    C->set(NewCond);
  }
#ifndef NDEBUG
  if (IsLoopInvariantWidenableBR) {
    auto *L = LI.getLoopFor(WidenableBR->getParent());
    // NB! This can also be loop-varying if we missed hoisting the IR out, which
    // was infact loop-invariant. This would imply a pass-ordering issue and the
    // assert should be dealt with as such.
    assert(isLoopInvariantWidenableCond(WidenableBR->getCondition(), *L) &&
           "Loop invariant widenable condition made loop variant!");
  }
#endif
  assert(isWidenableBranch(WidenableBR) && "preserve widenabiliy");
}
