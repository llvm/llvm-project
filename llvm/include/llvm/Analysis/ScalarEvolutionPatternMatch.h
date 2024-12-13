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

namespace detail {

template <typename TupleTy, typename Fn, std::size_t... Is>
bool CheckTupleElements(const TupleTy &Ops, Fn P, std::index_sequence<Is...>) {
  return (P(std::get<Is>(Ops), Is) && ...);
}

/// Helper to check if predicate \p P holds on all tuple elements in \p Ops
template <typename TupleTy, typename Fn>
bool all_of_tuple_elements(const TupleTy &Ops, Fn P) {
  return CheckTupleElements(
      Ops, P, std::make_index_sequence<std::tuple_size<TupleTy>::value>{});
}

} // namespace detail

template <typename Ops_t, typename SCEVTy> struct SCEV_match {
  Ops_t Ops;

  SCEV_match() : Ops() {
    static_assert(std::tuple_size<Ops_t>::value == 0 &&
                  "constructor can only be used with zero operands");
  }
  SCEV_match(Ops_t Ops) : Ops(Ops) {}
  template <typename A_t, typename B_t> SCEV_match(A_t A, B_t B) : Ops({A, B}) {
    static_assert(std::tuple_size<Ops_t>::value == 2 &&
                  "constructor can only be used for binary matcher");
  }

  bool match(const SCEV *S) const {
    auto *Cast = dyn_cast<SCEVTy>(S);
    if (!Cast || Cast->getNumOperands() != std::tuple_size<Ops_t>::value)
      return false;
    return detail::all_of_tuple_elements(Ops, [Cast](auto Op, unsigned Idx) {
      return Op.match(Cast->getOperand(Idx));
    });
  }
};

template <typename Op0_t, typename Op1_t, typename SCEVTy>
using BinarySCEV_match = SCEV_match<std::tuple<Op0_t, Op1_t>, SCEVTy>;

template <typename Op0_t, typename Op1_t, typename SCEVTy>
inline BinarySCEV_match<Op0_t, Op1_t, SCEVTy> m_scev_Binary(const Op0_t &Op0,
                                                            const Op1_t &Op1) {
  return BinarySCEV_match<Op0_t, Op1_t, SCEVTy>(Op0, Op1);
}

template <typename Op0_t, typename Op1_t>
inline BinarySCEV_match<Op0_t, Op1_t, SCEVAddExpr>
m_scev_Add(const Op0_t &Op0, const Op1_t &Op1) {
  return BinarySCEV_match<Op0_t, Op1_t, SCEVAddExpr>(Op0, Op1);
}

template <typename Op0_t, typename SCEVTy>
using UnarySCEV_match = SCEV_match<std::tuple<Op0_t>, SCEVTy>;

template <typename Op0_t, typename Op1_t, typename SCEVTy>
inline UnarySCEV_match<Op0_t, SCEVTy> m_scev_Unary(const Op0_t &Op0) {
  return UnarySCEV_match<Op0_t, SCEVTy>(Op0);
}

template <typename Op0_t>
inline UnarySCEV_match<Op0_t, SCEVSignExtendExpr>
m_scev_SExt(const Op0_t &Op0) {
  return UnarySCEV_match<Op0_t, SCEVSignExtendExpr>(Op0);
}

template <typename Op0_t>
inline UnarySCEV_match<Op0_t, SCEVZeroExtendExpr>
m_scev_ZExt(const Op0_t &Op0) {
  return UnarySCEV_match<Op0_t, SCEVZeroExtendExpr>(Op0);
}

} // namespace SCEVPatternMatch
} // namespace llvm

#endif
