//===- FloatingPointPredicateUtils.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FloatingPointPredicateUtils.h"
#include "llvm/IR/PatternMatch.h"
#include <optional>

namespace llvm {

using namespace PatternMatch;

template <>
DenormalMode FloatingPointPredicateUtils::queryDenormalMode(const Function &F,
                                                            Value *Val) {
  Type *Ty = Val->getType()->getScalarType();
  return F.getDenormalMode(Ty->getFltSemantics());
}

template <>
bool FloatingPointPredicateUtils::lookThroughFAbs(const Function &F, Value *LHS,
                                                  Value *&Src) {
  return match(LHS, m_FAbs(m_Value(Src)));
}

template <>
std::optional<APFloat>
FloatingPointPredicateUtils::matchConstantFloat(const Function &F, Value *Val) {
  const APFloat *ConstVal;

  if (!match(Val, m_APFloatAllowPoison(ConstVal)))
    return std::nullopt;

  return *ConstVal;
}

} // namespace llvm
