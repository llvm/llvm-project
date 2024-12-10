//===- ScalarEvolutionPatternMatch.h - Match on SCEVs -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on SCEVs, based on LLVM's IR pattern matchers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTIONPATTERNMATCH_H
#define LLVM_ANALYSIS_SCALAREVOLUTIONPATTERNMATCH_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"

namespace llvm {
namespace SCEVPatternMatch {

template <typename Val, typename Pattern>
bool match(const SCEV *S, const Pattern &P) {
  return P.match(S);
}

/// Match a specified integer value. \p BitWidth optionally specifies the
/// bitwidth the matched constant must have. If it is 0, the matched constant
/// can have any bitwidth.
template <unsigned BitWidth = 0> struct specific_intval {
  APInt Val;

  specific_intval(APInt V) : Val(std::move(V)) {}

  bool match(const SCEV *S) const {
    const auto *C = dyn_cast<SCEVConstant>(S);
    if (!C)
      return false;

    if (BitWidth != 0 && C->getAPInt().getBitWidth() != BitWidth)
      return false;
    return APInt::isSameValue(C->getAPInt(), Val);
  }
};

inline specific_intval<0> m_scev_Zero() {
  return specific_intval<0>(APInt(64, 0));
}
inline specific_intval<0> m_scev_One() {
  return specific_intval<0>(APInt(64, 1));
}
inline specific_intval<0> m_scev_MinusOne() {
  return specific_intval<0>(APInt(64, -1));
}

} // namespace SCEVPatternMatch
} // namespace llvm

#endif
