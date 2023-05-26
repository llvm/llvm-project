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
  SmallPtrSet<const BasicBlock *, 2> Visited;
  Visited.insert(DeoptBB);
  do {
    for (auto &Insn : *DeoptBB) {
      if (match(&Insn, m_Intrinsic<Intrinsic::experimental_deoptimize>()))
        return true;
      if (Insn.mayHaveSideEffects())
        return false;
    }
    DeoptBB = DeoptBB->getUniqueSuccessor();
    if (!DeoptBB)
      return false;
  } while (Visited.insert(DeoptBB).second);
  return false;
}

bool llvm::parseWidenableBranch(const User *U, Value *&Condition,
                                Value *&WidenableCondition,
                                BasicBlock *&IfTrueBB, BasicBlock *&IfFalseBB) {

  Use *C, *WC;
  if (parseWidenableBranch(const_cast<User*>(U), C, WC, IfTrueBB, IfFalseBB)) {
    if (C)
      Condition = C->get();
    else
      Condition = ConstantInt::getTrue(IfTrueBB->getContext());
    WidenableCondition = WC->get();
    return true;
  }
  return false;
}

bool llvm::parseWidenableBranch(User *U, Use *&C,Use *&WC,
                                BasicBlock *&IfTrueBB, BasicBlock *&IfFalseBB) {

  auto *BI = dyn_cast<BranchInst>(U);
  if (!BI || !BI->isConditional())
    return false;
  auto *Cond = BI->getCondition();
  if (!Cond->hasOneUse())
    return false;

  IfTrueBB = BI->getSuccessor(0);
  IfFalseBB = BI->getSuccessor(1);

  if (match(Cond, m_Intrinsic<Intrinsic::experimental_widenable_condition>())) {
    WC = &BI->getOperandUse(0);
    C = nullptr;
    return true;
  }

  // Check for two cases:
  // 1) br (i1 (and A, WC())), label %IfTrue, label %IfFalse
  // 2) br (i1 (and WC(), B)), label %IfTrue, label %IfFalse
  // We do not check for more generalized and trees as we should canonicalize
  // to the form above in instcombine. (TODO)
  Value *A, *B;
  if (!match(Cond, m_And(m_Value(A), m_Value(B))))
    return false;
  auto *And = dyn_cast<Instruction>(Cond);
  if (!And)
    // Could be a constexpr
    return false;

  if (match(A, m_Intrinsic<Intrinsic::experimental_widenable_condition>()) &&
      A->hasOneUse()) {
    WC = &And->getOperandUse(0);
    C = &And->getOperandUse(1);
    return true;
  }

  if (match(B, m_Intrinsic<Intrinsic::experimental_widenable_condition>()) &&
      B->hasOneUse()) {
    WC = &And->getOperandUse(1);
    C = &And->getOperandUse(0);
    return true;
  }
  return false;
}
