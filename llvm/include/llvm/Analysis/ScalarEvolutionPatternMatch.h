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
#include "llvm/IR/PatternMatch.h"

namespace llvm {
namespace SCEVPatternMatch {

template <typename Val, typename Pattern>
bool match(const SCEV *S, const Pattern &P) {
  return P.match(S);
}

struct specific_intval64 : public PatternMatch::specific_intval64<false> {
  specific_intval64(uint64_t V) : PatternMatch::specific_intval64<false>(V) {}

  bool match(const SCEV *S) {
    auto *Cast = dyn_cast<SCEVConstant>(S);
    return Cast &&
           PatternMatch::specific_intval64<false>::match(Cast->getValue());
  }
};

inline specific_intval64 m_scev_Zero() { return specific_intval64(0); }
inline specific_intval64 m_scev_One() { return specific_intval64(1); }
inline specific_intval64 m_scev_MinusOne() { return specific_intval64(-1); }

} // namespace SCEVPatternMatch
} // namespace llvm

#endif
