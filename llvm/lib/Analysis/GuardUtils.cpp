//===-- GuardUtils.cpp - Utils for work with guards -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utils that are used to perform analyzes related to guards and their
// conditions.
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/GuardUtils.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;
using namespace llvm::PatternMatch;

bool llvm::isGuard(const User *U) {
  return match(U, m_Intrinsic<Intrinsic::experimental_guard>());
}

bool llvm::isWidenableBranch(const User *U) {
  Value *Condition, *WidenableCondition;
  BasicBlock *GuardedBB, *DeoptBB;
  return parseWidenableBranch(U, Condition, WidenableCondition, GuardedBB,
                              DeoptBB);
}

bool llvm::isGuardAsWidenableBranch(const User *U) {
  Value *Condition, *WidenableCondition;
  BasicBlock *GuardedBB, *DeoptBB;
  if (!parseWidenableBranch(U, Condition, WidenableCondition, GuardedBB,
                            DeoptBB))
    return false;
  for (auto &Insn : *DeoptBB) {
    if (match(&Insn, m_Intrinsic<Intrinsic::experimental_deoptimize>()))
      return true;
    if (Insn.mayHaveSideEffects())
      return false;
  }
  return false;
}

bool llvm::parseWidenableBranch(const User *U, Value *&Condition,
                                Value *&WidenableCondition,
                                BasicBlock *&IfTrueBB, BasicBlock *&IfFalseBB) {
  if (match(U, m_Br(m_Intrinsic<Intrinsic::experimental_widenable_condition>(),
                    IfTrueBB, IfFalseBB)) &&
      cast<BranchInst>(U)->getCondition()->hasOneUse()) {
    WidenableCondition = cast<BranchInst>(U)->getCondition();
    Condition = ConstantInt::getTrue(IfTrueBB->getContext());
    return true;
  }

  // Check for two cases:
  // 1) br (i1 (and A, WC())), label %IfTrue, label %IfFalse
  // 2) br (i1 (and WC(), B)), label %IfTrue, label %IfFalse
  // We do not check for more generalized and trees as we should canonicalize
  // to the form above in instcombine. (TODO)
  if (!match(U, m_Br(m_And(m_Value(Condition), m_Value(WidenableCondition)),
                     IfTrueBB, IfFalseBB)))
    return false;
  if (!match(WidenableCondition,
             m_Intrinsic<Intrinsic::experimental_widenable_condition>())) {
    if (!match(Condition,
               m_Intrinsic<Intrinsic::experimental_widenable_condition>()))
      return false;
    std::swap(Condition, WidenableCondition);
  }
    
  // For the branch to be (easily) widenable, it must not correlate with other
  // branches.  Thus, the widenable condition must have a single use.
  return (WidenableCondition->hasOneUse() &&
          cast<BranchInst>(U)->getCondition()->hasOneUse());
}
