//===----------------------------------------------------------------------===//
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

template <typename Predicate> struct cst_pred_ty : public Predicate {
  bool match(const SCEV *S) {
    assert((isa<SCEVCouldNotCompute>(S) || !S->getType()->isVectorTy()) &&
           "no vector types expected from SCEVs");
    auto *C = dyn_cast<SCEVConstant>(S);
    return C && this->isValue(C->getAPInt());
  }
};

struct is_zero {
  bool isValue(const APInt &C) { return C.isZero(); }
};
/// Match an integer 0.
inline cst_pred_ty<is_zero> m_scev_Zero() { return cst_pred_ty<is_zero>(); }

struct is_one {
  bool isValue(const APInt &C) { return C.isOne(); }
};
/// Match an integer 1.
inline cst_pred_ty<is_one> m_scev_One() { return cst_pred_ty<is_one>(); }

struct is_all_ones {
  bool isValue(const APInt &C) { return C.isAllOnes(); }
};
/// Match an integer with all bits set.
inline cst_pred_ty<is_all_ones> m_scev_AllOnes() {
  return cst_pred_ty<is_all_ones>();
}

} // namespace SCEVPatternMatch
} // namespace llvm

#endif
