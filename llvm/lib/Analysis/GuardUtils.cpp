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

bool llvm::isWidenableCondition(const Value *V) {
  return match(V, m_Intrinsic<Intrinsic::experimental_widenable_condition>());
}

bool llvm::isWidenableBranch(const User *U) {
  return extractWidenableCondition(U) != nullptr;
}

bool llvm::isGuardAsWidenableBranch(const User *U) {
  if (!isWidenableBranch(U))
    return false;
  BasicBlock *DeoptBB = cast<BranchInst>(U)->getSuccessor(1);
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

template <typename CallbackType>
static void parseCondition(Value *Condition,
                           CallbackType RecordCheckOrWidenableCond) {
  SmallVector<Value *, 4> Worklist(1, Condition);
  SmallPtrSet<Value *, 4> Visited;
  Visited.insert(Condition);
  do {
    Value *Check = Worklist.pop_back_val();
    Value *LHS, *RHS;
    if (match(Check, m_And(m_Value(LHS), m_Value(RHS)))) {
      if (Visited.insert(LHS).second)
        Worklist.push_back(LHS);
      if (Visited.insert(RHS).second)
        Worklist.push_back(RHS);
      continue;
    }
    if (!RecordCheckOrWidenableCond(Check))
      break;
  } while (!Worklist.empty());
}

void llvm::parseWidenableGuard(const User *U,
                               llvm::SmallVectorImpl<Value *> &Checks) {
  assert((isGuard(U) || isWidenableBranch(U)) && "Should be");
  Value *Condition = isGuard(U) ? cast<IntrinsicInst>(U)->getArgOperand(0)
                                : cast<BranchInst>(U)->getCondition();

  parseCondition(Condition, [&](Value *Check) {
    if (!isWidenableCondition(Check))
      Checks.push_back(Check);
    return true;
  });
}

Value *llvm::extractWidenableCondition(const User *U) {
  auto *BI = dyn_cast<BranchInst>(U);
  if (!BI || !BI->isConditional())
    return nullptr;

  auto Condition = BI->getCondition();
  if (!Condition->hasOneUse())
    return nullptr;

  Value *WidenableCondition = nullptr;
  parseCondition(Condition, [&](Value *Check) {
    // We require widenable_condition has only one use, otherwise we don't
    // consider appropriate branch as widenable.
    if (isWidenableCondition(Check) && Check->hasOneUse()) {
      WidenableCondition = Check;
      return false;
    }
    return true;
  });
  return WidenableCondition;
}
