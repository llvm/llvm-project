//===- SLPReductionUtils.cpp - SLP reduction match helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPReductionUtils.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

static bool matchRdxBop(Instruction *I, Value *&V0, Value *&V1) {
  if (match(I, m_BinOp(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMaxNum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMinNum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMaximum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMinimum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smin>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umin>(m_Value(V0), m_Value(V1))))
    return true;
  return false;
}

Instruction *getNonPhiOperand(Instruction *I, PHINode *Phi) {
  Value *Op0 = nullptr;
  Value *Op1 = nullptr;
  if (!matchRdxBop(I, Op0, Op1))
    return nullptr;
  return dyn_cast<Instruction>(Op0 == Phi ? Op1 : Op0);
}

bool isReductionCandidate(Instruction *I) {
  bool IsSelect = match(I, m_Select(m_Value(), m_Value(), m_Value()));
  Value *B0 = nullptr, *B1 = nullptr;
  bool IsBinop = matchRdxBop(I, B0, B1);
  return IsBinop || IsSelect;
}

} // namespace llvm::slpvectorizer
