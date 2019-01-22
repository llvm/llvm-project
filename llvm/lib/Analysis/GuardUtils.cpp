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
  using namespace llvm::PatternMatch;
  const BranchInst *BI = dyn_cast<BranchInst>(U);

  // We are looking for the following pattern:
  //   br i1 %cond & widenable_condition(), label %guarded, label %deopt
  // deopt:
  //   <non-side-effecting instructions>
  //   deoptimize()
  if (!BI || !BI->isConditional())
    return false;

  if (!match(BI->getCondition(),
             m_And(m_Value(),
                   m_Intrinsic<Intrinsic::experimental_widenable_condition>())))
    return false;

  const BasicBlock *DeoptBlock = BI->getSuccessor(1);
  for (auto &Insn : *DeoptBlock) {
    if (match(&Insn, m_Intrinsic<Intrinsic::experimental_deoptimize>()))
      return true;
    if (Insn.mayHaveSideEffects())
      return false;
  }
  return false;
}
