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

bool llvm::isGuard(const User *U) {
  using namespace llvm::PatternMatch;
  return match(U, m_Intrinsic<Intrinsic::experimental_guard>());
}

bool llvm::isGuardAsWidenableBranch(const User *U) {
  Value *Condition, *WidenableCondition;
  BasicBlock *GuardedBB, *DeoptBB;
  if (!parseWidenableBranch(U, Condition, WidenableCondition, GuardedBB,
                            DeoptBB))
    return false;
  using namespace llvm::PatternMatch;
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
  using namespace llvm::PatternMatch;
  return match(U, m_Br(m_And(m_Value(Condition), m_Value(WidenableCondition)),
                       IfTrueBB, IfFalseBB));
}
