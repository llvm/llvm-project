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

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) const { return isa<Class>(V); }
};

template <typename Class> struct bind_ty {
  Class *&VR;

  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) const {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a SCEV, capturing it if we match.
inline bind_ty<const SCEV> m_SCEV(const SCEV *&V) { return V; }
inline bind_ty<const SCEVConstant> m_SCEVConstant(const SCEVConstant *&V) {
  return V;
}
inline bind_ty<const SCEVUnknown> m_SCEVUnknown(const SCEVUnknown *&V) {
  return V;
}

/// Match a specified const SCEV *.
struct specificscev_ty {
  const SCEV *Expr;

  specificscev_ty(const SCEV *Expr) : Expr(Expr) {}

  template <typename ITy> bool match(ITy *S) { return S == Expr; }
};

/// Match if we have a specific specified SCEV.
inline specificscev_ty m_Specific(const SCEV *S) { return S; }

/// Match a unary SCEV.
template <typename SCEVTy, typename Op0_t> struct SCEVUnaryExpr_match {
  Op0_t Op0;

  SCEVUnaryExpr_match(Op0_t Op0) : Op0(Op0) {}

  bool match(const SCEV *S) {
    auto *E = dyn_cast<SCEVTy>(S);
    return E && E->getNumOperands() == 1 && Op0.match(E->getOperand(0));
  }
};

template <typename SCEVTy, typename Op0_t>
inline SCEVUnaryExpr_match<SCEVTy, Op0_t> m_scev_Unary(const Op0_t &Op0) {
  return SCEVUnaryExpr_match<SCEVTy, Op0_t>(Op0);
}

template <typename Op0_t>
inline SCEVUnaryExpr_match<SCEVSignExtendExpr, Op0_t>
m_scev_SExt(const Op0_t &Op0) {
  return m_scev_Unary<SCEVSignExtendExpr>(Op0);
}

template <typename Op0_t>
inline SCEVUnaryExpr_match<SCEVZeroExtendExpr, Op0_t>
m_scev_ZExt(const Op0_t &Op0) {
  return m_scev_Unary<SCEVZeroExtendExpr>(Op0);
}

/// Match a binary SCEV.
template <typename SCEVTy, typename Op0_t, typename Op1_t>
struct SCEVBinaryExpr_match {
  Op0_t Op0;
  Op1_t Op1;

  SCEVBinaryExpr_match(Op0_t Op0, Op1_t Op1) : Op0(Op0), Op1(Op1) {}

  bool match(const SCEV *S) {
    auto *E = dyn_cast<SCEVTy>(S);
    return E && E->getNumOperands() == 2 && Op0.match(E->getOperand(0)) &&
           Op1.match(E->getOperand(1));
  }
};

template <typename SCEVTy, typename Op0_t, typename Op1_t>
inline SCEVBinaryExpr_match<SCEVTy, Op0_t, Op1_t>
m_scev_Binary(const Op0_t &Op0, const Op1_t &Op1) {
  return SCEVBinaryExpr_match<SCEVTy, Op0_t, Op1_t>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline SCEVBinaryExpr_match<SCEVAddExpr, Op0_t, Op1_t>
m_scev_Add(const Op0_t &Op0, const Op1_t &Op1) {
  return m_scev_Binary<SCEVAddExpr>(Op0, Op1);
}

} // namespace SCEVPatternMatch
} // namespace llvm

#endif
