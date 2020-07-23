//===- PatternMatch.h - Match on the LLVM IR --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the LLVM IR. The power of these routines is
// that it allows you to write concise patterns that are expressive and easy to
// understand. The other major advantage of this is that it allows you to
// trivially capture/bind elements in the pattern to variables. For example,
// you can do something like this:
//
//  Value *Exp = ...
//  Value *X, *Y;  ConstantInt *C1, *C2;      // (X & C1) | (Y & C2)
//  if (match(Exp, m_Or(m_And(m_Value(X), m_ConstantInt(C1)),
//                      m_And(m_Value(Y), m_ConstantInt(C2))))) {
//    ... Pattern is matched and variables are bound ...
//  }
//
// This is primarily useful to things like the instruction combiner, but can
// also be useful for static analysis tools or code generators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PATTERNMATCH_H
#define LLVM_IR_PATTERNMATCH_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

namespace llvm {
namespace PatternMatch {

template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern &>(P).match(V);
}

template <typename Pattern> bool match(ArrayRef<int> Mask, const Pattern &P) {
  return const_cast<Pattern &>(P).match(Mask);
}

template <typename SubPattern_t> struct OneUse_match {
  SubPattern_t SubPattern;

  OneUse_match(const SubPattern_t &SP) : SubPattern(SP) {}

  template <typename OpTy> bool match(OpTy *V) {
    return V->hasOneUse() && SubPattern.match(V);
  }
};

template <typename T> inline OneUse_match<T> m_OneUse(const T &SubPattern) {
  return SubPattern;
}

template <typename Class> struct class_match {
  template <typename ITy> bool match(ITy *V) { return isa<Class>(V); }
};

/// Match an arbitrary value and ignore it.
inline class_match<Value> m_Value() { return class_match<Value>(); }

/// Match an arbitrary unary operation and ignore it.
inline class_match<UnaryOperator> m_UnOp() {
  return class_match<UnaryOperator>();
}

/// Match an arbitrary binary operation and ignore it.
inline class_match<BinaryOperator> m_BinOp() {
  return class_match<BinaryOperator>();
}

/// Matches any compare instruction and ignore it.
inline class_match<CmpInst> m_Cmp() { return class_match<CmpInst>(); }

/// Match an arbitrary ConstantInt and ignore it.
inline class_match<ConstantInt> m_ConstantInt() {
  return class_match<ConstantInt>();
}

/// Match an arbitrary undef constant.
inline class_match<UndefValue> m_Undef() { return class_match<UndefValue>(); }

/// Match an arbitrary Constant and ignore it.
inline class_match<Constant> m_Constant() { return class_match<Constant>(); }

/// Match an arbitrary basic block value and ignore it.
inline class_match<BasicBlock> m_BasicBlock() {
  return class_match<BasicBlock>();
}

/// Inverting matcher
template <typename Ty> struct match_unless {
  Ty M;

  match_unless(const Ty &Matcher) : M(Matcher) {}

  template <typename ITy> bool match(ITy *V) { return !M.match(V); }
};

/// Match if the inner matcher does *NOT* match.
template <typename Ty> inline match_unless<Ty> m_Unless(const Ty &M) {
  return match_unless<Ty>(M);
}

/// Matching combinators
template <typename LTy, typename RTy> struct match_combine_or {
  LTy L;
  RTy R;

  match_combine_or(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      return true;
    if (R.match(V))
      return true;
    return false;
  }
};

template <typename LTy, typename RTy> struct match_combine_and {
  LTy L;
  RTy R;

  match_combine_and(const LTy &Left, const RTy &Right) : L(Left), R(Right) {}

  template <typename ITy> bool match(ITy *V) {
    if (L.match(V))
      if (R.match(V))
        return true;
    return false;
  }
};

/// Combine two pattern matchers matching L || R
template <typename LTy, typename RTy>
inline match_combine_or<LTy, RTy> m_CombineOr(const LTy &L, const RTy &R) {
  return match_combine_or<LTy, RTy>(L, R);
}

/// Combine two pattern matchers matching L && R
template <typename LTy, typename RTy>
inline match_combine_and<LTy, RTy> m_CombineAnd(const LTy &L, const RTy &R) {
  return match_combine_and<LTy, RTy>(L, R);
}

struct apint_match {
  const APInt *&Res;
  bool AllowUndef;

  apint_match(const APInt *&Res, bool AllowUndef)
    : Res(Res), AllowUndef(AllowUndef) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      Res = &CI->getValue();
      return true;
    }
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CI = dyn_cast_or_null<ConstantInt>(
                C->getSplatValue(AllowUndef))) {
          Res = &CI->getValue();
          return true;
        }
    return false;
  }
};
// Either constexpr if or renaming ConstantFP::getValueAPF to
// ConstantFP::getValue is needed to do it via single template
// function for both apint/apfloat.
struct apfloat_match {
  const APFloat *&Res;
  bool AllowUndef;

  apfloat_match(const APFloat *&Res, bool AllowUndef)
      : Res(Res), AllowUndef(AllowUndef) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CI = dyn_cast<ConstantFP>(V)) {
      Res = &CI->getValueAPF();
      return true;
    }
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CI = dyn_cast_or_null<ConstantFP>(
                C->getSplatValue(AllowUndef))) {
          Res = &CI->getValueAPF();
          return true;
        }
    return false;
  }
};

/// Match a ConstantInt or splatted ConstantVector, binding the
/// specified pointer to the contained APInt.
inline apint_match m_APInt(const APInt *&Res) {
  // Forbid undefs by default to maintain previous behavior.
  return apint_match(Res, /* AllowUndef */ false);
}

/// Match APInt while allowing undefs in splat vector constants.
inline apint_match m_APIntAllowUndef(const APInt *&Res) {
  return apint_match(Res, /* AllowUndef */ true);
}

/// Match APInt while forbidding undefs in splat vector constants.
inline apint_match m_APIntForbidUndef(const APInt *&Res) {
  return apint_match(Res, /* AllowUndef */ false);
}

/// Match a ConstantFP or splatted ConstantVector, binding the
/// specified pointer to the contained APFloat.
inline apfloat_match m_APFloat(const APFloat *&Res) {
  // Forbid undefs by default to maintain previous behavior.
  return apfloat_match(Res, /* AllowUndef */ false);
}

/// Match APFloat while allowing undefs in splat vector constants.
inline apfloat_match m_APFloatAllowUndef(const APFloat *&Res) {
  return apfloat_match(Res, /* AllowUndef */ true);
}

/// Match APFloat while forbidding undefs in splat vector constants.
inline apfloat_match m_APFloatForbidUndef(const APFloat *&Res) {
  return apfloat_match(Res, /* AllowUndef */ false);
}

template <int64_t Val> struct constantint_match {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *CI = dyn_cast<ConstantInt>(V)) {
      const APInt &CIV = CI->getValue();
      if (Val >= 0)
        return CIV == static_cast<uint64_t>(Val);
      // If Val is negative, and CI is shorter than it, truncate to the right
      // number of bits.  If it is larger, then we have to sign extend.  Just
      // compare their negated values.
      return -CIV == -Val;
    }
    return false;
  }
};

/// Match a ConstantInt with a specific value.
template <int64_t Val> inline constantint_match<Val> m_ConstantInt() {
  return constantint_match<Val>();
}

/// This helper class is used to match constant scalars, vector splats,
/// and fixed width vectors that satisfy a specified predicate.
/// For fixed width vector constants, undefined elements are ignored.
template <typename Predicate, typename ConstantVal>
struct cstval_pred_ty : public Predicate {
  template <typename ITy> bool match(ITy *V) {
    if (const auto *CV = dyn_cast<ConstantVal>(V))
      return this->isValue(CV->getValue());
    if (const auto *VTy = dyn_cast<VectorType>(V->getType())) {
      if (const auto *C = dyn_cast<Constant>(V)) {
        if (const auto *CV = dyn_cast_or_null<ConstantVal>(C->getSplatValue()))
          return this->isValue(CV->getValue());

        // Number of elements of a scalable vector unknown at compile time
        auto *FVTy = dyn_cast<FixedVectorType>(VTy);
        if (!FVTy)
          return false;

        // Non-splat vector constant: check each element for a match.
        unsigned NumElts = FVTy->getNumElements();
        assert(NumElts != 0 && "Constant vector with no elements?");
        bool HasNonUndefElements = false;
        for (unsigned i = 0; i != NumElts; ++i) {
          Constant *Elt = C->getAggregateElement(i);
          if (!Elt)
            return false;
          if (isa<UndefValue>(Elt))
            continue;
          auto *CV = dyn_cast<ConstantVal>(Elt);
          if (!CV || !this->isValue(CV->getValue()))
            return false;
          HasNonUndefElements = true;
        }
        return HasNonUndefElements;
      }
    }
    return false;
  }
};

/// specialization of cstval_pred_ty for ConstantInt
template <typename Predicate>
using cst_pred_ty = cstval_pred_ty<Predicate, ConstantInt>;

/// specialization of cstval_pred_ty for ConstantFP
template <typename Predicate>
using cstfp_pred_ty = cstval_pred_ty<Predicate, ConstantFP>;

/// This helper class is used to match scalar and vector constants that
/// satisfy a specified predicate, and bind them to an APInt.
template <typename Predicate> struct api_pred_ty : public Predicate {
  const APInt *&Res;

  api_pred_ty(const APInt *&R) : Res(R) {}

  template <typename ITy> bool match(ITy *V) {
    if (const auto *CI = dyn_cast<ConstantInt>(V))
      if (this->isValue(CI->getValue())) {
        Res = &CI->getValue();
        return true;
      }
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue()))
          if (this->isValue(CI->getValue())) {
            Res = &CI->getValue();
            return true;
          }

    return false;
  }
};

///////////////////////////////////////////////////////////////////////////////
//
// Encapsulate constant value queries for use in templated predicate matchers.
// This allows checking if constants match using compound predicates and works
// with vector constants, possibly with relaxed constraints. For example, ignore
// undef values.
//
///////////////////////////////////////////////////////////////////////////////

struct is_any_apint {
  bool isValue(const APInt &C) { return true; }
};
/// Match an integer or vector with any integral constant.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_any_apint> m_AnyIntegralConstant() {
  return cst_pred_ty<is_any_apint>();
}

struct is_all_ones {
  bool isValue(const APInt &C) { return C.isAllOnesValue(); }
};
/// Match an integer or vector with all bits set.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_all_ones> m_AllOnes() {
  return cst_pred_ty<is_all_ones>();
}

struct is_maxsignedvalue {
  bool isValue(const APInt &C) { return C.isMaxSignedValue(); }
};
/// Match an integer or vector with values having all bits except for the high
/// bit set (0x7f...).
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_maxsignedvalue> m_MaxSignedValue() {
  return cst_pred_ty<is_maxsignedvalue>();
}
inline api_pred_ty<is_maxsignedvalue> m_MaxSignedValue(const APInt *&V) {
  return V;
}

struct is_negative {
  bool isValue(const APInt &C) { return C.isNegative(); }
};
/// Match an integer or vector of negative values.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_negative> m_Negative() {
  return cst_pred_ty<is_negative>();
}
inline api_pred_ty<is_negative> m_Negative(const APInt *&V) {
  return V;
}

struct is_nonnegative {
  bool isValue(const APInt &C) { return C.isNonNegative(); }
};
/// Match an integer or vector of non-negative values.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_nonnegative> m_NonNegative() {
  return cst_pred_ty<is_nonnegative>();
}
inline api_pred_ty<is_nonnegative> m_NonNegative(const APInt *&V) {
  return V;
}

struct is_strictlypositive {
  bool isValue(const APInt &C) { return C.isStrictlyPositive(); }
};
/// Match an integer or vector of strictly positive values.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_strictlypositive> m_StrictlyPositive() {
  return cst_pred_ty<is_strictlypositive>();
}
inline api_pred_ty<is_strictlypositive> m_StrictlyPositive(const APInt *&V) {
  return V;
}

struct is_nonpositive {
  bool isValue(const APInt &C) { return C.isNonPositive(); }
};
/// Match an integer or vector of non-positive values.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_nonpositive> m_NonPositive() {
  return cst_pred_ty<is_nonpositive>();
}
inline api_pred_ty<is_nonpositive> m_NonPositive(const APInt *&V) { return V; }

struct is_one {
  bool isValue(const APInt &C) { return C.isOneValue(); }
};
/// Match an integer 1 or a vector with all elements equal to 1.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_one> m_One() {
  return cst_pred_ty<is_one>();
}

struct is_zero_int {
  bool isValue(const APInt &C) { return C.isNullValue(); }
};
/// Match an integer 0 or a vector with all elements equal to 0.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_zero_int> m_ZeroInt() {
  return cst_pred_ty<is_zero_int>();
}

struct is_zero {
  template <typename ITy> bool match(ITy *V) {
    auto *C = dyn_cast<Constant>(V);
    // FIXME: this should be able to do something for scalable vectors
    return C && (C->isNullValue() || cst_pred_ty<is_zero_int>().match(C));
  }
};
/// Match any null constant or a vector with all elements equal to 0.
/// For vectors, this includes constants with undefined elements.
inline is_zero m_Zero() {
  return is_zero();
}

struct is_power2 {
  bool isValue(const APInt &C) { return C.isPowerOf2(); }
};
/// Match an integer or vector power-of-2.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_power2> m_Power2() {
  return cst_pred_ty<is_power2>();
}
inline api_pred_ty<is_power2> m_Power2(const APInt *&V) {
  return V;
}

struct is_negated_power2 {
  bool isValue(const APInt &C) { return (-C).isPowerOf2(); }
};
/// Match a integer or vector negated power-of-2.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_negated_power2> m_NegatedPower2() {
  return cst_pred_ty<is_negated_power2>();
}
inline api_pred_ty<is_negated_power2> m_NegatedPower2(const APInt *&V) {
  return V;
}

struct is_power2_or_zero {
  bool isValue(const APInt &C) { return !C || C.isPowerOf2(); }
};
/// Match an integer or vector of 0 or power-of-2 values.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_power2_or_zero> m_Power2OrZero() {
  return cst_pred_ty<is_power2_or_zero>();
}
inline api_pred_ty<is_power2_or_zero> m_Power2OrZero(const APInt *&V) {
  return V;
}

struct is_sign_mask {
  bool isValue(const APInt &C) { return C.isSignMask(); }
};
/// Match an integer or vector with only the sign bit(s) set.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_sign_mask> m_SignMask() {
  return cst_pred_ty<is_sign_mask>();
}

struct is_lowbit_mask {
  bool isValue(const APInt &C) { return C.isMask(); }
};
/// Match an integer or vector with only the low bit(s) set.
/// For vectors, this includes constants with undefined elements.
inline cst_pred_ty<is_lowbit_mask> m_LowBitMask() {
  return cst_pred_ty<is_lowbit_mask>();
}

struct icmp_pred_with_threshold {
  ICmpInst::Predicate Pred;
  const APInt *Thr;
  bool isValue(const APInt &C) {
    switch (Pred) {
    case ICmpInst::Predicate::ICMP_EQ:
      return C.eq(*Thr);
    case ICmpInst::Predicate::ICMP_NE:
      return C.ne(*Thr);
    case ICmpInst::Predicate::ICMP_UGT:
      return C.ugt(*Thr);
    case ICmpInst::Predicate::ICMP_UGE:
      return C.uge(*Thr);
    case ICmpInst::Predicate::ICMP_ULT:
      return C.ult(*Thr);
    case ICmpInst::Predicate::ICMP_ULE:
      return C.ule(*Thr);
    case ICmpInst::Predicate::ICMP_SGT:
      return C.sgt(*Thr);
    case ICmpInst::Predicate::ICMP_SGE:
      return C.sge(*Thr);
    case ICmpInst::Predicate::ICMP_SLT:
      return C.slt(*Thr);
    case ICmpInst::Predicate::ICMP_SLE:
      return C.sle(*Thr);
    default:
      llvm_unreachable("Unhandled ICmp predicate");
    }
  }
};
/// Match an integer or vector with every element comparing 'pred' (eg/ne/...)
/// to Threshold. For vectors, this includes constants with undefined elements.
inline cst_pred_ty<icmp_pred_with_threshold>
m_SpecificInt_ICMP(ICmpInst::Predicate Predicate, const APInt &Threshold) {
  cst_pred_ty<icmp_pred_with_threshold> P;
  P.Pred = Predicate;
  P.Thr = &Threshold;
  return P;
}

struct is_nan {
  bool isValue(const APFloat &C) { return C.isNaN(); }
};
/// Match an arbitrary NaN constant. This includes quiet and signalling nans.
/// For vectors, this includes constants with undefined elements.
inline cstfp_pred_ty<is_nan> m_NaN() {
  return cstfp_pred_ty<is_nan>();
}

struct is_inf {
  bool isValue(const APFloat &C) { return C.isInfinity(); }
};
/// Match a positive or negative infinity FP constant.
/// For vectors, this includes constants with undefined elements.
inline cstfp_pred_ty<is_inf> m_Inf() {
  return cstfp_pred_ty<is_inf>();
}

struct is_any_zero_fp {
  bool isValue(const APFloat &C) { return C.isZero(); }
};
/// Match a floating-point negative zero or positive zero.
/// For vectors, this includes constants with undefined elements.
inline cstfp_pred_ty<is_any_zero_fp> m_AnyZeroFP() {
  return cstfp_pred_ty<is_any_zero_fp>();
}

struct is_pos_zero_fp {
  bool isValue(const APFloat &C) { return C.isPosZero(); }
};
/// Match a floating-point positive zero.
/// For vectors, this includes constants with undefined elements.
inline cstfp_pred_ty<is_pos_zero_fp> m_PosZeroFP() {
  return cstfp_pred_ty<is_pos_zero_fp>();
}

struct is_neg_zero_fp {
  bool isValue(const APFloat &C) { return C.isNegZero(); }
};
/// Match a floating-point negative zero.
/// For vectors, this includes constants with undefined elements.
inline cstfp_pred_ty<is_neg_zero_fp> m_NegZeroFP() {
  return cstfp_pred_ty<is_neg_zero_fp>();
}

///////////////////////////////////////////////////////////////////////////////

template <typename Class> struct bind_ty {
  Class *&VR;

  bind_ty(Class *&V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (auto *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

/// Match a value, capturing it if we match.
inline bind_ty<Value> m_Value(Value *&V) { return V; }
inline bind_ty<const Value> m_Value(const Value *&V) { return V; }

/// Match an instruction, capturing it if we match.
inline bind_ty<Instruction> m_Instruction(Instruction *&I) { return I; }
/// Match a unary operator, capturing it if we match.
inline bind_ty<UnaryOperator> m_UnOp(UnaryOperator *&I) { return I; }
/// Match a binary operator, capturing it if we match.
inline bind_ty<BinaryOperator> m_BinOp(BinaryOperator *&I) { return I; }
/// Match a with overflow intrinsic, capturing it if we match.
inline bind_ty<WithOverflowInst> m_WithOverflowInst(WithOverflowInst *&I) { return I; }

/// Match a ConstantInt, capturing the value if we match.
inline bind_ty<ConstantInt> m_ConstantInt(ConstantInt *&CI) { return CI; }

/// Match a Constant, capturing the value if we match.
inline bind_ty<Constant> m_Constant(Constant *&C) { return C; }

/// Match a ConstantFP, capturing the value if we match.
inline bind_ty<ConstantFP> m_ConstantFP(ConstantFP *&C) { return C; }

/// Match a basic block value, capturing it if we match.
inline bind_ty<BasicBlock> m_BasicBlock(BasicBlock *&V) { return V; }
inline bind_ty<const BasicBlock> m_BasicBlock(const BasicBlock *&V) {
  return V;
}

/// Match a specified Value*.
struct specificval_ty {
  const Value *Val;

  specificval_ty(const Value *V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) { return V == Val; }
};

/// Match if we have a specific specified value.
inline specificval_ty m_Specific(const Value *V) { return V; }

/// Stores a reference to the Value *, not the Value * itself,
/// thus can be used in commutative matchers.
template <typename Class> struct deferredval_ty {
  Class *const &Val;

  deferredval_ty(Class *const &V) : Val(V) {}

  template <typename ITy> bool match(ITy *const V) { return V == Val; }
};

/// A commutative-friendly version of m_Specific().
inline deferredval_ty<Value> m_Deferred(Value *const &V) { return V; }
inline deferredval_ty<const Value> m_Deferred(const Value *const &V) {
  return V;
}

/// Match a specified floating point value or vector of all elements of
/// that value.
struct specific_fpval {
  double Val;

  specific_fpval(double V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (const auto *CFP = dyn_cast<ConstantFP>(V))
      return CFP->isExactlyValue(Val);
    if (V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        if (auto *CFP = dyn_cast_or_null<ConstantFP>(C->getSplatValue()))
          return CFP->isExactlyValue(Val);
    return false;
  }
};

/// Match a specific floating point value or vector with all elements
/// equal to the value.
inline specific_fpval m_SpecificFP(double V) { return specific_fpval(V); }

/// Match a float 1.0 or vector with all elements equal to 1.0.
inline specific_fpval m_FPOne() { return m_SpecificFP(1.0); }

struct bind_const_intval_ty {
  uint64_t &VR;

  bind_const_intval_ty(uint64_t &V) : VR(V) {}

  template <typename ITy> bool match(ITy *V) {
    if (const auto *CV = dyn_cast<ConstantInt>(V))
      if (CV->getValue().ule(UINT64_MAX)) {
        VR = CV->getZExtValue();
        return true;
      }
    return false;
  }
};

/// Match a specified integer value or vector of all elements of that
/// value.
struct specific_intval {
  APInt Val;

  specific_intval(APInt V) : Val(std::move(V)) {}

  template <typename ITy> bool match(ITy *V) {
    const auto *CI = dyn_cast<ConstantInt>(V);
    if (!CI && V->getType()->isVectorTy())
      if (const auto *C = dyn_cast<Constant>(V))
        CI = dyn_cast_or_null<ConstantInt>(C->getSplatValue());

    return CI && APInt::isSameValue(CI->getValue(), Val);
  }
};

/// Match a specific integer value or vector with all elements equal to
/// the value.
inline specific_intval m_SpecificInt(APInt V) {
  return specific_intval(std::move(V));
}

inline specific_intval m_SpecificInt(uint64_t V) {
  return m_SpecificInt(APInt(64, V));
}

/// Match a ConstantInt and bind to its value.  This does not match
/// ConstantInts wider than 64-bits.
inline bind_const_intval_ty m_ConstantInt(uint64_t &V) { return V; }

/// Match a specified basic block value.
struct specific_bbval {
  BasicBlock *Val;

  specific_bbval(BasicBlock *Val) : Val(Val) {}

  template <typename ITy> bool match(ITy *V) {
    const auto *BB = dyn_cast<BasicBlock>(V);
    return BB && BB == Val;
  }
};

/// Match a specific basic block value.
inline specific_bbval m_SpecificBB(BasicBlock *BB) {
  return specific_bbval(BB);
}

/// A commutative-friendly version of m_Specific().
inline deferredval_ty<BasicBlock> m_Deferred(BasicBlock *const &BB) {
  return BB;
}
inline deferredval_ty<const BasicBlock>
m_Deferred(const BasicBlock *const &BB) {
  return BB;
}

//===----------------------------------------------------------------------===//
// Matcher for any binary operator.
//
template <typename LHS_t, typename RHS_t, bool Commutable = false>
struct AnyBinaryOp_match {
  LHS_t L;
  RHS_t R;

  // The evaluation order is always stable, regardless of Commutability.
  // The LHS is always matched first.
  AnyBinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<BinaryOperator>(V))
      return (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) ||
             (Commutable && L.match(I->getOperand(1)) &&
              R.match(I->getOperand(0)));
    return false;
  }
};

template <typename LHS, typename RHS>
inline AnyBinaryOp_match<LHS, RHS> m_BinOp(const LHS &L, const RHS &R) {
  return AnyBinaryOp_match<LHS, RHS>(L, R);
}

//===----------------------------------------------------------------------===//
// Matcher for any unary operator.
// TODO fuse unary, binary matcher into n-ary matcher
//
template <typename OP_t> struct AnyUnaryOp_match {
  OP_t X;

  AnyUnaryOp_match(const OP_t &X) : X(X) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<UnaryOperator>(V))
      return X.match(I->getOperand(0));
    return false;
  }
};

template <typename OP_t> inline AnyUnaryOp_match<OP_t> m_UnOp(const OP_t &X) {
  return AnyUnaryOp_match<OP_t>(X);
}

//===----------------------------------------------------------------------===//
// Matchers for specific binary operators.
//

template <typename LHS_t, typename RHS_t, unsigned Opcode,
          bool Commutable = false>
struct BinaryOp_match {
  LHS_t L;
  RHS_t R;

  // The evaluation order is always stable, regardless of Commutability.
  // The LHS is always matched first.
  BinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<BinaryOperator>(V);
      return (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) ||
             (Commutable && L.match(I->getOperand(1)) &&
              R.match(I->getOperand(0)));
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode &&
             ((L.match(CE->getOperand(0)) && R.match(CE->getOperand(1))) ||
              (Commutable && L.match(CE->getOperand(1)) &&
               R.match(CE->getOperand(0))));
    return false;
  }
};

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add> m_Add(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FAdd> m_FAdd(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FAdd>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Sub> m_Sub(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Sub>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FSub> m_FSub(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FSub>(L, R);
}

template <typename Op_t> struct FNeg_match {
  Op_t X;

  FNeg_match(const Op_t &Op) : X(Op) {}
  template <typename OpTy> bool match(OpTy *V) {
    auto *FPMO = dyn_cast<FPMathOperator>(V);
    if (!FPMO) return false;

    if (FPMO->getOpcode() == Instruction::FNeg)
      return X.match(FPMO->getOperand(0));

    if (FPMO->getOpcode() == Instruction::FSub) {
      if (FPMO->hasNoSignedZeros()) {
        // With 'nsz', any zero goes.
        if (!cstfp_pred_ty<is_any_zero_fp>().match(FPMO->getOperand(0)))
          return false;
      } else {
        // Without 'nsz', we need fsub -0.0, X exactly.
        if (!cstfp_pred_ty<is_neg_zero_fp>().match(FPMO->getOperand(0)))
          return false;
      }

      return X.match(FPMO->getOperand(1));
    }

    return false;
  }
};

/// Match 'fneg X' as 'fsub -0.0, X'.
template <typename OpTy>
inline FNeg_match<OpTy>
m_FNeg(const OpTy &X) {
  return FNeg_match<OpTy>(X);
}

/// Match 'fneg X' as 'fsub +-0.0, X'.
template <typename RHS>
inline BinaryOp_match<cstfp_pred_ty<is_any_zero_fp>, RHS, Instruction::FSub>
m_FNegNSZ(const RHS &X) {
  return m_FSub(m_AnyZeroFP(), X);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul> m_Mul(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FMul> m_FMul(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FMul>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::UDiv> m_UDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::UDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SDiv> m_SDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FDiv> m_FDiv(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FDiv>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::URem> m_URem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::URem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::SRem> m_SRem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::SRem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FRem> m_FRem(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FRem>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And> m_And(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Or> m_Or(const LHS &L,
                                                      const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor> m_Xor(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shl> m_Shl(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shl>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::LShr> m_LShr(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::LShr>(L, R);
}

template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::AShr> m_AShr(const LHS &L,
                                                          const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::AShr>(L, R);
}

template <typename LHS_t, typename RHS_t, unsigned Opcode,
          unsigned WrapFlags = 0>
struct OverflowingBinaryOp_match {
  LHS_t L;
  RHS_t R;

  OverflowingBinaryOp_match(const LHS_t &LHS, const RHS_t &RHS)
      : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *Op = dyn_cast<OverflowingBinaryOperator>(V)) {
      if (Op->getOpcode() != Opcode)
        return false;
      if (WrapFlags & OverflowingBinaryOperator::NoUnsignedWrap &&
          !Op->hasNoUnsignedWrap())
        return false;
      if (WrapFlags & OverflowingBinaryOperator::NoSignedWrap &&
          !Op->hasNoSignedWrap())
        return false;
      return L.match(Op->getOperand(0)) && R.match(Op->getOperand(1));
    }
    return false;
  }
};

template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWAdd(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWSub(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWMul(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                 OverflowingBinaryOperator::NoSignedWrap>
m_NSWShl(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                   OverflowingBinaryOperator::NoSignedWrap>(
      L, R);
}

template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWAdd(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Add,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWSub(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Sub,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWMul(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Mul,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}
template <typename LHS, typename RHS>
inline OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                 OverflowingBinaryOperator::NoUnsignedWrap>
m_NUWShl(const LHS &L, const RHS &R) {
  return OverflowingBinaryOp_match<LHS, RHS, Instruction::Shl,
                                   OverflowingBinaryOperator::NoUnsignedWrap>(
      L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches a group of binary opcodes.
//
template <typename LHS_t, typename RHS_t, typename Predicate>
struct BinOpPred_match : Predicate {
  LHS_t L;
  RHS_t R;

  BinOpPred_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      return this->isOpType(I->getOpcode()) && L.match(I->getOperand(0)) &&
             R.match(I->getOperand(1));
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      return this->isOpType(CE->getOpcode()) && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

struct is_shift_op {
  bool isOpType(unsigned Opcode) { return Instruction::isShift(Opcode); }
};

struct is_right_shift_op {
  bool isOpType(unsigned Opcode) {
    return Opcode == Instruction::LShr || Opcode == Instruction::AShr;
  }
};

struct is_logical_shift_op {
  bool isOpType(unsigned Opcode) {
    return Opcode == Instruction::LShr || Opcode == Instruction::Shl;
  }
};

struct is_bitwiselogic_op {
  bool isOpType(unsigned Opcode) {
    return Instruction::isBitwiseLogicOp(Opcode);
  }
};

struct is_idiv_op {
  bool isOpType(unsigned Opcode) {
    return Opcode == Instruction::SDiv || Opcode == Instruction::UDiv;
  }
};

struct is_irem_op {
  bool isOpType(unsigned Opcode) {
    return Opcode == Instruction::SRem || Opcode == Instruction::URem;
  }
};

/// Matches shift operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_shift_op> m_Shift(const LHS &L,
                                                      const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_shift_op>(L, R);
}

/// Matches logical shift operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_right_shift_op> m_Shr(const LHS &L,
                                                          const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_right_shift_op>(L, R);
}

/// Matches logical shift operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_logical_shift_op>
m_LogicalShift(const LHS &L, const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_logical_shift_op>(L, R);
}

/// Matches bitwise logic operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_bitwiselogic_op>
m_BitwiseLogic(const LHS &L, const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_bitwiselogic_op>(L, R);
}

/// Matches integer division operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_idiv_op> m_IDiv(const LHS &L,
                                                    const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_idiv_op>(L, R);
}

/// Matches integer remainder operations.
template <typename LHS, typename RHS>
inline BinOpPred_match<LHS, RHS, is_irem_op> m_IRem(const LHS &L,
                                                    const RHS &R) {
  return BinOpPred_match<LHS, RHS, is_irem_op>(L, R);
}

//===----------------------------------------------------------------------===//
// Class that matches exact binary ops.
//
template <typename SubPattern_t> struct Exact_match {
  SubPattern_t SubPattern;

  Exact_match(const SubPattern_t &SP) : SubPattern(SP) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *PEO = dyn_cast<PossiblyExactOperator>(V))
      return PEO->isExact() && SubPattern.match(V);
    return false;
  }
};

template <typename T> inline Exact_match<T> m_Exact(const T &SubPattern) {
  return SubPattern;
}

//===----------------------------------------------------------------------===//
// Matchers for CmpInst classes
//

template <typename LHS_t, typename RHS_t, typename Class, typename PredicateTy,
          bool Commutable = false>
struct CmpClass_match {
  PredicateTy &Predicate;
  LHS_t L;
  RHS_t R;

  // The evaluation order is always stable, regardless of Commutability.
  // The LHS is always matched first.
  CmpClass_match(PredicateTy &Pred, const LHS_t &LHS, const RHS_t &RHS)
      : Predicate(Pred), L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<Class>(V)) {
      if (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) {
        Predicate = I->getPredicate();
        return true;
      } else if (Commutable && L.match(I->getOperand(1)) &&
           R.match(I->getOperand(0))) {
        Predicate = I->getSwappedPredicate();
        return true;
      }
    }
    return false;
  }
};

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, CmpInst, CmpInst::Predicate>
m_Cmp(CmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, CmpInst, CmpInst::Predicate>(Pred, L, R);
}

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate>
m_ICmp(ICmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate>(Pred, L, R);
}

template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, FCmpInst, FCmpInst::Predicate>
m_FCmp(FCmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, FCmpInst, FCmpInst::Predicate>(Pred, L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for instructions with a given opcode and number of operands.
//

/// Matches instructions with Opcode and three operands.
template <typename T0, unsigned Opcode> struct OneOps_match {
  T0 Op1;

  OneOps_match(const T0 &Op1) : Op1(Op1) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<Instruction>(V);
      return Op1.match(I->getOperand(0));
    }
    return false;
  }
};

/// Matches instructions with Opcode and three operands.
template <typename T0, typename T1, unsigned Opcode> struct TwoOps_match {
  T0 Op1;
  T1 Op2;

  TwoOps_match(const T0 &Op1, const T1 &Op2) : Op1(Op1), Op2(Op2) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<Instruction>(V);
      return Op1.match(I->getOperand(0)) && Op2.match(I->getOperand(1));
    }
    return false;
  }
};

/// Matches instructions with Opcode and three operands.
template <typename T0, typename T1, typename T2, unsigned Opcode>
struct ThreeOps_match {
  T0 Op1;
  T1 Op2;
  T2 Op3;

  ThreeOps_match(const T0 &Op1, const T1 &Op2, const T2 &Op3)
      : Op1(Op1), Op2(Op2), Op3(Op3) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (V->getValueID() == Value::InstructionVal + Opcode) {
      auto *I = cast<Instruction>(V);
      return Op1.match(I->getOperand(0)) && Op2.match(I->getOperand(1)) &&
             Op3.match(I->getOperand(2));
    }
    return false;
  }
};

/// Matches SelectInst.
template <typename Cond, typename LHS, typename RHS>
inline ThreeOps_match<Cond, LHS, RHS, Instruction::Select>
m_Select(const Cond &C, const LHS &L, const RHS &R) {
  return ThreeOps_match<Cond, LHS, RHS, Instruction::Select>(C, L, R);
}

/// This matches a select of two constants, e.g.:
/// m_SelectCst<-1, 0>(m_Value(V))
template <int64_t L, int64_t R, typename Cond>
inline ThreeOps_match<Cond, constantint_match<L>, constantint_match<R>,
                      Instruction::Select>
m_SelectCst(const Cond &C) {
  return m_Select(C, m_ConstantInt<L>(), m_ConstantInt<R>());
}

/// Matches FreezeInst.
template <typename OpTy>
inline OneOps_match<OpTy, Instruction::Freeze> m_Freeze(const OpTy &Op) {
  return OneOps_match<OpTy, Instruction::Freeze>(Op);
}

/// Matches InsertElementInst.
template <typename Val_t, typename Elt_t, typename Idx_t>
inline ThreeOps_match<Val_t, Elt_t, Idx_t, Instruction::InsertElement>
m_InsertElt(const Val_t &Val, const Elt_t &Elt, const Idx_t &Idx) {
  return ThreeOps_match<Val_t, Elt_t, Idx_t, Instruction::InsertElement>(
      Val, Elt, Idx);
}

/// Matches ExtractElementInst.
template <typename Val_t, typename Idx_t>
inline TwoOps_match<Val_t, Idx_t, Instruction::ExtractElement>
m_ExtractElt(const Val_t &Val, const Idx_t &Idx) {
  return TwoOps_match<Val_t, Idx_t, Instruction::ExtractElement>(Val, Idx);
}

/// Matches shuffle.
template <typename T0, typename T1, typename T2> struct Shuffle_match {
  T0 Op1;
  T1 Op2;
  T2 Mask;

  Shuffle_match(const T0 &Op1, const T1 &Op2, const T2 &Mask)
      : Op1(Op1), Op2(Op2), Mask(Mask) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<ShuffleVectorInst>(V)) {
      return Op1.match(I->getOperand(0)) && Op2.match(I->getOperand(1)) &&
             Mask.match(I->getShuffleMask());
    }
    return false;
  }
};

struct m_Mask {
  ArrayRef<int> &MaskRef;
  m_Mask(ArrayRef<int> &MaskRef) : MaskRef(MaskRef) {}
  bool match(ArrayRef<int> Mask) {
    MaskRef = Mask;
    return true;
  }
};

struct m_ZeroMask {
  bool match(ArrayRef<int> Mask) {
    return all_of(Mask, [](int Elem) { return Elem == 0 || Elem == -1; });
  }
};

struct m_SpecificMask {
  ArrayRef<int> &MaskRef;
  m_SpecificMask(ArrayRef<int> &MaskRef) : MaskRef(MaskRef) {}
  bool match(ArrayRef<int> Mask) { return MaskRef == Mask; }
};

struct m_SplatOrUndefMask {
  int &SplatIndex;
  m_SplatOrUndefMask(int &SplatIndex) : SplatIndex(SplatIndex) {}
  bool match(ArrayRef<int> Mask) {
    auto First = find_if(Mask, [](int Elem) { return Elem != -1; });
    if (First == Mask.end())
      return false;
    SplatIndex = *First;
    return all_of(Mask,
                  [First](int Elem) { return Elem == *First || Elem == -1; });
  }
};

/// Matches ShuffleVectorInst independently of mask value.
template <typename V1_t, typename V2_t>
inline TwoOps_match<V1_t, V2_t, Instruction::ShuffleVector>
m_Shuffle(const V1_t &v1, const V2_t &v2) {
  return TwoOps_match<V1_t, V2_t, Instruction::ShuffleVector>(v1, v2);
}

template <typename V1_t, typename V2_t, typename Mask_t>
inline Shuffle_match<V1_t, V2_t, Mask_t>
m_Shuffle(const V1_t &v1, const V2_t &v2, const Mask_t &mask) {
  return Shuffle_match<V1_t, V2_t, Mask_t>(v1, v2, mask);
}

/// Matches LoadInst.
template <typename OpTy>
inline OneOps_match<OpTy, Instruction::Load> m_Load(const OpTy &Op) {
  return OneOps_match<OpTy, Instruction::Load>(Op);
}

/// Matches StoreInst.
template <typename ValueOpTy, typename PointerOpTy>
inline TwoOps_match<ValueOpTy, PointerOpTy, Instruction::Store>
m_Store(const ValueOpTy &ValueOp, const PointerOpTy &PointerOp) {
  return TwoOps_match<ValueOpTy, PointerOpTy, Instruction::Store>(ValueOp,
                                                                  PointerOp);
}

//===----------------------------------------------------------------------===//
// Matchers for CastInst classes
//

template <typename Op_t, unsigned Opcode> struct CastClass_match {
  Op_t Op;

  CastClass_match(const Op_t &OpMatch) : Op(OpMatch) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *O = dyn_cast<Operator>(V))
      return O->getOpcode() == Opcode && Op.match(O->getOperand(0));
    return false;
  }
};

/// Matches BitCast.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::BitCast> m_BitCast(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::BitCast>(Op);
}

/// Matches PtrToInt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::PtrToInt> m_PtrToInt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::PtrToInt>(Op);
}

/// Matches Trunc.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::Trunc> m_Trunc(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::Trunc>(Op);
}

template <typename OpTy>
inline match_combine_or<CastClass_match<OpTy, Instruction::Trunc>, OpTy>
m_TruncOrSelf(const OpTy &Op) {
  return m_CombineOr(m_Trunc(Op), Op);
}

/// Matches SExt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::SExt> m_SExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::SExt>(Op);
}

/// Matches ZExt.
template <typename OpTy>
inline CastClass_match<OpTy, Instruction::ZExt> m_ZExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::ZExt>(Op);
}

template <typename OpTy>
inline match_combine_or<CastClass_match<OpTy, Instruction::ZExt>, OpTy>
m_ZExtOrSelf(const OpTy &Op) {
  return m_CombineOr(m_ZExt(Op), Op);
}

template <typename OpTy>
inline match_combine_or<CastClass_match<OpTy, Instruction::SExt>, OpTy>
m_SExtOrSelf(const OpTy &Op) {
  return m_CombineOr(m_SExt(Op), Op);
}

template <typename OpTy>
inline match_combine_or<CastClass_match<OpTy, Instruction::ZExt>,
                        CastClass_match<OpTy, Instruction::SExt>>
m_ZExtOrSExt(const OpTy &Op) {
  return m_CombineOr(m_ZExt(Op), m_SExt(Op));
}

template <typename OpTy>
inline match_combine_or<
    match_combine_or<CastClass_match<OpTy, Instruction::ZExt>,
                     CastClass_match<OpTy, Instruction::SExt>>,
    OpTy>
m_ZExtOrSExtOrSelf(const OpTy &Op) {
  return m_CombineOr(m_ZExtOrSExt(Op), Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::UIToFP> m_UIToFP(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::UIToFP>(Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::SIToFP> m_SIToFP(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::SIToFP>(Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::FPToUI> m_FPToUI(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::FPToUI>(Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::FPToSI> m_FPToSI(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::FPToSI>(Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::FPTrunc> m_FPTrunc(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::FPTrunc>(Op);
}

template <typename OpTy>
inline CastClass_match<OpTy, Instruction::FPExt> m_FPExt(const OpTy &Op) {
  return CastClass_match<OpTy, Instruction::FPExt>(Op);
}

//===----------------------------------------------------------------------===//
// Matchers for control flow.
//

struct br_match {
  BasicBlock *&Succ;

  br_match(BasicBlock *&Succ) : Succ(Succ) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *BI = dyn_cast<BranchInst>(V))
      if (BI->isUnconditional()) {
        Succ = BI->getSuccessor(0);
        return true;
      }
    return false;
  }
};

inline br_match m_UnconditionalBr(BasicBlock *&Succ) { return br_match(Succ); }

template <typename Cond_t, typename TrueBlock_t, typename FalseBlock_t>
struct brc_match {
  Cond_t Cond;
  TrueBlock_t T;
  FalseBlock_t F;

  brc_match(const Cond_t &C, const TrueBlock_t &t, const FalseBlock_t &f)
      : Cond(C), T(t), F(f) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *BI = dyn_cast<BranchInst>(V))
      if (BI->isConditional() && Cond.match(BI->getCondition()))
        return T.match(BI->getSuccessor(0)) && F.match(BI->getSuccessor(1));
    return false;
  }
};

template <typename Cond_t>
inline brc_match<Cond_t, bind_ty<BasicBlock>, bind_ty<BasicBlock>>
m_Br(const Cond_t &C, BasicBlock *&T, BasicBlock *&F) {
  return brc_match<Cond_t, bind_ty<BasicBlock>, bind_ty<BasicBlock>>(
      C, m_BasicBlock(T), m_BasicBlock(F));
}

template <typename Cond_t, typename TrueBlock_t, typename FalseBlock_t>
inline brc_match<Cond_t, TrueBlock_t, FalseBlock_t>
m_Br(const Cond_t &C, const TrueBlock_t &T, const FalseBlock_t &F) {
  return brc_match<Cond_t, TrueBlock_t, FalseBlock_t>(C, T, F);
}

//===----------------------------------------------------------------------===//
// Matchers for max/min idioms, eg: "select (sgt x, y), x, y" -> smax(x,y).
//

template <typename CmpInst_t, typename LHS_t, typename RHS_t, typename Pred_t,
          bool Commutable = false>
struct MaxMin_match {
  LHS_t L;
  RHS_t R;

  // The evaluation order is always stable, regardless of Commutability.
  // The LHS is always matched first.
  MaxMin_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template <typename OpTy> bool match(OpTy *V) {
    // Look for "(x pred y) ? x : y" or "(x pred y) ? y : x".
    auto *SI = dyn_cast<SelectInst>(V);
    if (!SI)
      return false;
    auto *Cmp = dyn_cast<CmpInst_t>(SI->getCondition());
    if (!Cmp)
      return false;
    // At this point we have a select conditioned on a comparison.  Check that
    // it is the values returned by the select that are being compared.
    Value *TrueVal = SI->getTrueValue();
    Value *FalseVal = SI->getFalseValue();
    Value *LHS = Cmp->getOperand(0);
    Value *RHS = Cmp->getOperand(1);
    if ((TrueVal != LHS || FalseVal != RHS) &&
        (TrueVal != RHS || FalseVal != LHS))
      return false;
    typename CmpInst_t::Predicate Pred =
        LHS == TrueVal ? Cmp->getPredicate() : Cmp->getInversePredicate();
    // Does "(x pred y) ? x : y" represent the desired max/min operation?
    if (!Pred_t::match(Pred))
      return false;
    // It does!  Bind the operands.
    return (L.match(LHS) && R.match(RHS)) ||
           (Commutable && L.match(RHS) && R.match(LHS));
  }
};

/// Helper class for identifying signed max predicates.
struct smax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SGT || Pred == CmpInst::ICMP_SGE;
  }
};

/// Helper class for identifying signed min predicates.
struct smin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_SLT || Pred == CmpInst::ICMP_SLE;
  }
};

/// Helper class for identifying unsigned max predicates.
struct umax_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_UGT || Pred == CmpInst::ICMP_UGE;
  }
};

/// Helper class for identifying unsigned min predicates.
struct umin_pred_ty {
  static bool match(ICmpInst::Predicate Pred) {
    return Pred == CmpInst::ICMP_ULT || Pred == CmpInst::ICMP_ULE;
  }
};

/// Helper class for identifying ordered max predicates.
struct ofmax_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_OGT || Pred == CmpInst::FCMP_OGE;
  }
};

/// Helper class for identifying ordered min predicates.
struct ofmin_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_OLT || Pred == CmpInst::FCMP_OLE;
  }
};

/// Helper class for identifying unordered max predicates.
struct ufmax_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_UGT || Pred == CmpInst::FCMP_UGE;
  }
};

/// Helper class for identifying unordered min predicates.
struct ufmin_pred_ty {
  static bool match(FCmpInst::Predicate Pred) {
    return Pred == CmpInst::FCMP_ULT || Pred == CmpInst::FCMP_ULE;
  }
};

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty> m_SMax(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty> m_SMin(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty> m_UMax(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty>(L, R);
}

template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty> m_UMin(const LHS &L,
                                                             const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty>(L, R);
}

/// Match an 'ordered' floating point maximum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'maximum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ogt/ge, L, R), L, R) semantics matched by this predicate.
///
///                         max(L, R)  iff L and R are not NaN
///  m_OrdFMax(L, R) =      R          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ofmax_pred_ty> m_OrdFMax(const LHS &L,
                                                                 const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ofmax_pred_ty>(L, R);
}

/// Match an 'ordered' floating point minimum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'minimum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(olt/le, L, R), L, R) semantics matched by this predicate.
///
///                         min(L, R)  iff L and R are not NaN
///  m_OrdFMin(L, R) =      R          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ofmin_pred_ty> m_OrdFMin(const LHS &L,
                                                                 const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ofmin_pred_ty>(L, R);
}

/// Match an 'unordered' floating point maximum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'maximum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ugt/ge, L, R), L, R) semantics matched by this predicate.
///
///                         max(L, R)  iff L and R are not NaN
///  m_UnordFMax(L, R) =    L          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ufmax_pred_ty>
m_UnordFMax(const LHS &L, const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ufmax_pred_ty>(L, R);
}

/// Match an 'unordered' floating point minimum function.
/// Floating point has one special value 'NaN'. Therefore, there is no total
/// order. However, if we can ignore the 'NaN' value (for example, because of a
/// 'no-nans-float-math' flag) a combination of a fcmp and select has 'minimum'
/// semantics. In the presence of 'NaN' we have to preserve the original
/// select(fcmp(ult/le, L, R), L, R) semantics matched by this predicate.
///
///                          min(L, R)  iff L and R are not NaN
///  m_UnordFMin(L, R) =     L          iff L or R are NaN
template <typename LHS, typename RHS>
inline MaxMin_match<FCmpInst, LHS, RHS, ufmin_pred_ty>
m_UnordFMin(const LHS &L, const RHS &R) {
  return MaxMin_match<FCmpInst, LHS, RHS, ufmin_pred_ty>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for overflow check patterns: e.g. (a + b) u< a, (a ^ -1) <u b
// Note that S might be matched to other instructions than AddInst.
//

template <typename LHS_t, typename RHS_t, typename Sum_t>
struct UAddWithOverflow_match {
  LHS_t L;
  RHS_t R;
  Sum_t S;

  UAddWithOverflow_match(const LHS_t &L, const RHS_t &R, const Sum_t &S)
      : L(L), R(R), S(S) {}

  template <typename OpTy> bool match(OpTy *V) {
    Value *ICmpLHS, *ICmpRHS;
    ICmpInst::Predicate Pred;
    if (!m_ICmp(Pred, m_Value(ICmpLHS), m_Value(ICmpRHS)).match(V))
      return false;

    Value *AddLHS, *AddRHS;
    auto AddExpr = m_Add(m_Value(AddLHS), m_Value(AddRHS));

    // (a + b) u< a, (a + b) u< b
    if (Pred == ICmpInst::ICMP_ULT)
      if (AddExpr.match(ICmpLHS) && (ICmpRHS == AddLHS || ICmpRHS == AddRHS))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpLHS);

    // a >u (a + b), b >u (a + b)
    if (Pred == ICmpInst::ICMP_UGT)
      if (AddExpr.match(ICmpRHS) && (ICmpLHS == AddLHS || ICmpLHS == AddRHS))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpRHS);

    Value *Op1;
    auto XorExpr = m_OneUse(m_Xor(m_Value(Op1), m_AllOnes()));
    // (a ^ -1) <u b
    if (Pred == ICmpInst::ICMP_ULT) {
      if (XorExpr.match(ICmpLHS))
        return L.match(Op1) && R.match(ICmpRHS) && S.match(ICmpLHS);
    }
    //  b > u (a ^ -1)
    if (Pred == ICmpInst::ICMP_UGT) {
      if (XorExpr.match(ICmpRHS))
        return L.match(Op1) && R.match(ICmpLHS) && S.match(ICmpRHS);
    }

    // Match special-case for increment-by-1.
    if (Pred == ICmpInst::ICMP_EQ) {
      // (a + 1) == 0
      // (1 + a) == 0
      if (AddExpr.match(ICmpLHS) && m_ZeroInt().match(ICmpRHS) &&
          (m_One().match(AddLHS) || m_One().match(AddRHS)))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpLHS);
      // 0 == (a + 1)
      // 0 == (1 + a)
      if (m_ZeroInt().match(ICmpLHS) && AddExpr.match(ICmpRHS) &&
          (m_One().match(AddLHS) || m_One().match(AddRHS)))
        return L.match(AddLHS) && R.match(AddRHS) && S.match(ICmpRHS);
    }

    return false;
  }
};

/// Match an icmp instruction checking for unsigned overflow on addition.
///
/// S is matched to the addition whose result is being checked for overflow, and
/// L and R are matched to the LHS and RHS of S.
template <typename LHS_t, typename RHS_t, typename Sum_t>
UAddWithOverflow_match<LHS_t, RHS_t, Sum_t>
m_UAddWithOverflow(const LHS_t &L, const RHS_t &R, const Sum_t &S) {
  return UAddWithOverflow_match<LHS_t, RHS_t, Sum_t>(L, R, S);
}

template <typename Opnd_t> struct Argument_match {
  unsigned OpI;
  Opnd_t Val;

  Argument_match(unsigned OpIdx, const Opnd_t &V) : OpI(OpIdx), Val(V) {}

  template <typename OpTy> bool match(OpTy *V) {
    // FIXME: Should likely be switched to use `CallBase`.
    if (const auto *CI = dyn_cast<CallInst>(V))
      return Val.match(CI->getArgOperand(OpI));
    return false;
  }
};

/// Match an argument.
template <unsigned OpI, typename Opnd_t>
inline Argument_match<Opnd_t> m_Argument(const Opnd_t &Op) {
  return Argument_match<Opnd_t>(OpI, Op);
}

/// Intrinsic matchers.
struct IntrinsicID_match {
  unsigned ID;

  IntrinsicID_match(Intrinsic::ID IntrID) : ID(IntrID) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (const auto *CI = dyn_cast<CallInst>(V))
      if (const auto *F = CI->getCalledFunction())
        return F->getIntrinsicID() == ID;
    return false;
  }
};

/// Intrinsic matches are combinations of ID matchers, and argument
/// matchers. Higher arity matcher are defined recursively in terms of and-ing
/// them with lower arity matchers. Here's some convenient typedefs for up to
/// several arguments, and more can be added as needed
template <typename T0 = void, typename T1 = void, typename T2 = void,
          typename T3 = void, typename T4 = void, typename T5 = void,
          typename T6 = void, typename T7 = void, typename T8 = void,
          typename T9 = void, typename T10 = void>
struct m_Intrinsic_Ty;
template <typename T0> struct m_Intrinsic_Ty<T0> {
  using Ty = match_combine_and<IntrinsicID_match, Argument_match<T0>>;
};
template <typename T0, typename T1> struct m_Intrinsic_Ty<T0, T1> {
  using Ty =
      match_combine_and<typename m_Intrinsic_Ty<T0>::Ty, Argument_match<T1>>;
};
template <typename T0, typename T1, typename T2>
struct m_Intrinsic_Ty<T0, T1, T2> {
  using Ty =
      match_combine_and<typename m_Intrinsic_Ty<T0, T1>::Ty,
                        Argument_match<T2>>;
};
template <typename T0, typename T1, typename T2, typename T3>
struct m_Intrinsic_Ty<T0, T1, T2, T3> {
  using Ty =
      match_combine_and<typename m_Intrinsic_Ty<T0, T1, T2>::Ty,
                        Argument_match<T3>>;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct m_Intrinsic_Ty<T0, T1, T2, T3, T4> {
  using Ty = match_combine_and<typename m_Intrinsic_Ty<T0, T1, T2, T3>::Ty,
                               Argument_match<T4>>;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
struct m_Intrinsic_Ty<T0, T1, T2, T3, T4, T5> {
  using Ty = match_combine_and<typename m_Intrinsic_Ty<T0, T1, T2, T3, T4>::Ty,
                               Argument_match<T5>>;
};

/// Match intrinsic calls like this:
/// m_Intrinsic<Intrinsic::fabs>(m_Value(X))
template <Intrinsic::ID IntrID> inline IntrinsicID_match m_Intrinsic() {
  return IntrinsicID_match(IntrID);
}

template <Intrinsic::ID IntrID, typename T0>
inline typename m_Intrinsic_Ty<T0>::Ty m_Intrinsic(const T0 &Op0) {
  return m_CombineAnd(m_Intrinsic<IntrID>(), m_Argument<0>(Op0));
}

template <Intrinsic::ID IntrID, typename T0, typename T1>
inline typename m_Intrinsic_Ty<T0, T1>::Ty m_Intrinsic(const T0 &Op0,
                                                       const T1 &Op1) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0), m_Argument<1>(Op1));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2>
inline typename m_Intrinsic_Ty<T0, T1, T2>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1), m_Argument<2>(Op2));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2,
          typename T3>
inline typename m_Intrinsic_Ty<T0, T1, T2, T3>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2, const T3 &Op3) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1, Op2), m_Argument<3>(Op3));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2,
          typename T3, typename T4>
inline typename m_Intrinsic_Ty<T0, T1, T2, T3, T4>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2, const T3 &Op3,
            const T4 &Op4) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1, Op2, Op3),
                      m_Argument<4>(Op4));
}

template <Intrinsic::ID IntrID, typename T0, typename T1, typename T2,
          typename T3, typename T4, typename T5>
inline typename m_Intrinsic_Ty<T0, T1, T2, T3, T4, T5>::Ty
m_Intrinsic(const T0 &Op0, const T1 &Op1, const T2 &Op2, const T3 &Op3,
            const T4 &Op4, const T5 &Op5) {
  return m_CombineAnd(m_Intrinsic<IntrID>(Op0, Op1, Op2, Op3, Op4),
                      m_Argument<5>(Op5));
}

// Helper intrinsic matching specializations.
template <typename Opnd0>
inline typename m_Intrinsic_Ty<Opnd0>::Ty m_BitReverse(const Opnd0 &Op0) {
  return m_Intrinsic<Intrinsic::bitreverse>(Op0);
}

template <typename Opnd0>
inline typename m_Intrinsic_Ty<Opnd0>::Ty m_BSwap(const Opnd0 &Op0) {
  return m_Intrinsic<Intrinsic::bswap>(Op0);
}

template <typename Opnd0>
inline typename m_Intrinsic_Ty<Opnd0>::Ty m_FAbs(const Opnd0 &Op0) {
  return m_Intrinsic<Intrinsic::fabs>(Op0);
}

template <typename Opnd0>
inline typename m_Intrinsic_Ty<Opnd0>::Ty m_FCanonicalize(const Opnd0 &Op0) {
  return m_Intrinsic<Intrinsic::canonicalize>(Op0);
}

template <typename Opnd0, typename Opnd1>
inline typename m_Intrinsic_Ty<Opnd0, Opnd1>::Ty m_FMin(const Opnd0 &Op0,
                                                        const Opnd1 &Op1) {
  return m_Intrinsic<Intrinsic::minnum>(Op0, Op1);
}

template <typename Opnd0, typename Opnd1>
inline typename m_Intrinsic_Ty<Opnd0, Opnd1>::Ty m_FMax(const Opnd0 &Op0,
                                                        const Opnd1 &Op1) {
  return m_Intrinsic<Intrinsic::maxnum>(Op0, Op1);
}

//===----------------------------------------------------------------------===//
// Matchers for two-operands operators with the operators in either order
//

/// Matches a BinaryOperator with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline AnyBinaryOp_match<LHS, RHS, true> m_c_BinOp(const LHS &L, const RHS &R) {
  return AnyBinaryOp_match<LHS, RHS, true>(L, R);
}

/// Matches an ICmp with a predicate over LHS and RHS in either order.
/// Swaps the predicate if operands are commuted.
template <typename LHS, typename RHS>
inline CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate, true>
m_c_ICmp(ICmpInst::Predicate &Pred, const LHS &L, const RHS &R) {
  return CmpClass_match<LHS, RHS, ICmpInst, ICmpInst::Predicate, true>(Pred, L,
                                                                       R);
}

/// Matches a Add with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add, true> m_c_Add(const LHS &L,
                                                                const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add, true>(L, R);
}

/// Matches a Mul with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul, true> m_c_Mul(const LHS &L,
                                                                const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul, true>(L, R);
}

/// Matches an And with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And, true> m_c_And(const LHS &L,
                                                                const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And, true>(L, R);
}

/// Matches an Or with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Or, true> m_c_Or(const LHS &L,
                                                              const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or, true>(L, R);
}

/// Matches an Xor with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor, true> m_c_Xor(const LHS &L,
                                                                const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor, true>(L, R);
}

/// Matches a 'Neg' as 'sub 0, V'.
template <typename ValTy>
inline BinaryOp_match<cst_pred_ty<is_zero_int>, ValTy, Instruction::Sub>
m_Neg(const ValTy &V) {
  return m_Sub(m_ZeroInt(), V);
}

/// Matches a 'Not' as 'xor V, -1' or 'xor -1, V'.
template <typename ValTy>
inline BinaryOp_match<ValTy, cst_pred_ty<is_all_ones>, Instruction::Xor, true>
m_Not(const ValTy &V) {
  return m_c_Xor(V, m_AllOnes());
}

/// Matches an SMin with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty, true>
m_c_SMin(const LHS &L, const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smin_pred_ty, true>(L, R);
}
/// Matches an SMax with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty, true>
m_c_SMax(const LHS &L, const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, smax_pred_ty, true>(L, R);
}
/// Matches a UMin with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty, true>
m_c_UMin(const LHS &L, const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umin_pred_ty, true>(L, R);
}
/// Matches a UMax with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty, true>
m_c_UMax(const LHS &L, const RHS &R) {
  return MaxMin_match<ICmpInst, LHS, RHS, umax_pred_ty, true>(L, R);
}

/// Matches FAdd with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FAdd, true>
m_c_FAdd(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FAdd, true>(L, R);
}

/// Matches FMul with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::FMul, true>
m_c_FMul(const LHS &L, const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::FMul, true>(L, R);
}

template <typename Opnd_t> struct Signum_match {
  Opnd_t Val;
  Signum_match(const Opnd_t &V) : Val(V) {}

  template <typename OpTy> bool match(OpTy *V) {
    unsigned TypeSize = V->getType()->getScalarSizeInBits();
    if (TypeSize == 0)
      return false;

    unsigned ShiftWidth = TypeSize - 1;
    Value *OpL = nullptr, *OpR = nullptr;

    // This is the representation of signum we match:
    //
    //  signum(x) == (x >> 63) | (-x >>u 63)
    //
    // An i1 value is its own signum, so it's correct to match
    //
    //  signum(x) == (x >> 0)  | (-x >>u 0)
    //
    // for i1 values.

    auto LHS = m_AShr(m_Value(OpL), m_SpecificInt(ShiftWidth));
    auto RHS = m_LShr(m_Neg(m_Value(OpR)), m_SpecificInt(ShiftWidth));
    auto Signum = m_Or(LHS, RHS);

    return Signum.match(V) && OpL == OpR && Val.match(OpL);
  }
};

/// Matches a signum pattern.
///
/// signum(x) =
///      x >  0  ->  1
///      x == 0  ->  0
///      x <  0  -> -1
template <typename Val_t> inline Signum_match<Val_t> m_Signum(const Val_t &V) {
  return Signum_match<Val_t>(V);
}

template <int Ind, typename Opnd_t> struct ExtractValue_match {
  Opnd_t Val;
  ExtractValue_match(const Opnd_t &V) : Val(V) {}

  template <typename OpTy> bool match(OpTy *V) {
    if (auto *I = dyn_cast<ExtractValueInst>(V))
      return I->getNumIndices() == 1 && I->getIndices()[0] == Ind &&
             Val.match(I->getAggregateOperand());
    return false;
  }
};

/// Match a single index ExtractValue instruction.
/// For example m_ExtractValue<1>(...)
template <int Ind, typename Val_t>
inline ExtractValue_match<Ind, Val_t> m_ExtractValue(const Val_t &V) {
  return ExtractValue_match<Ind, Val_t>(V);
}

/// Matches patterns for `vscale`. This can either be a call to `llvm.vscale` or
/// the constant expression
///  `ptrtoint(gep <vscale x 1 x i8>, <vscale x 1 x i8>* null, i32 1>`
/// under the right conditions determined by DataLayout.
struct VScaleVal_match {
private:
  template <typename Base, typename Offset>
  inline BinaryOp_match<Base, Offset, Instruction::GetElementPtr>
  m_OffsetGep(const Base &B, const Offset &O) {
    return BinaryOp_match<Base, Offset, Instruction::GetElementPtr>(B, O);
  }

public:
  const DataLayout &DL;
  VScaleVal_match(const DataLayout &DL) : DL(DL) {}

  template <typename ITy> bool match(ITy *V) {
    if (m_Intrinsic<Intrinsic::vscale>().match(V))
      return true;

    if (m_PtrToInt(m_OffsetGep(m_Zero(), m_SpecificInt(1))).match(V)) {
      Type *PtrTy = cast<Operator>(V)->getOperand(0)->getType();
      auto *DerefTy = PtrTy->getPointerElementType();
      if (isa<ScalableVectorType>(DerefTy) &&
          DL.getTypeAllocSizeInBits(DerefTy).getKnownMinSize() == 8)
        return true;
    }

    return false;
  }
};

inline VScaleVal_match m_VScale(const DataLayout &DL) {
  return VScaleVal_match(DL);
}

} // end namespace PatternMatch
} // end namespace llvm

#endif // LLVM_IR_PATTERNMATCH_H
