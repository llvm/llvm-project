//===- llvm/Support/GenericFloatingPointPredicateUtils.h -----*- C++-*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities for dealing with flags related to floating point properties and
/// mode controls.
///
//===----------------------------------------------------------------------===/

#ifndef LLVM_ADT_GENERICFLOATINGPOINTPREDICATEUTILS_H
#define LLVM_ADT_GENERICFLOATINGPOINTPREDICATEUTILS_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Instructions.h"
#include <optional>

namespace llvm {

template <typename ContextT> class GenericFloatingPointPredicateUtils {
  using ValueRefT = typename ContextT::ValueRefT;
  using FunctionT = typename ContextT::FunctionT;

  constexpr static ValueRefT Invalid = {};

private:
  static DenormalMode queryDenormalMode(const FunctionT &F, ValueRefT Val);

  static bool lookThroughFAbs(const FunctionT &F, ValueRefT LHS,
                              ValueRefT &Src);

  static std::optional<APFloat> matchConstantFloat(const FunctionT &F,
                                                   ValueRefT Val);

  /// Return the return value for fcmpImpliesClass for a compare that produces
  /// an exact class test.
  static std::tuple<ValueRefT, FPClassTest, FPClassTest>
  exactClass(ValueRefT V, FPClassTest M) {
    return {V, M, ~M};
  }

public:
  /// Returns a pair of values, which if passed to llvm.is.fpclass, returns the
  /// same result as an fcmp with the given operands.
  static std::pair<ValueRefT, FPClassTest>
  fcmpToClassTest(FCmpInst::Predicate Pred, const FunctionT &F, ValueRefT LHS,
                  ValueRefT RHS, bool LookThroughSrc) {
    std::optional<APFloat> ConstRHS = matchConstantFloat(F, RHS);
    if (!ConstRHS)
      return {Invalid, fcAllFlags};

    return fcmpToClassTest(Pred, F, LHS, *ConstRHS, LookThroughSrc);
  }

  static std::pair<ValueRefT, FPClassTest>
  fcmpToClassTest(FCmpInst::Predicate Pred, const FunctionT &F, ValueRefT LHS,
                  const APFloat &ConstRHS, bool LookThroughSrc) {

    auto [Src, ClassIfTrue, ClassIfFalse] =
        fcmpImpliesClass(Pred, F, LHS, ConstRHS, LookThroughSrc);

    if (Src && ClassIfTrue == ~ClassIfFalse)
      return {Src, ClassIfTrue};

    return {Invalid, fcAllFlags};
  }

  /// Compute the possible floating-point classes that \p LHS could be based on
  /// fcmp \Pred \p LHS, \p RHS.
  ///
  /// \returns { TestedValue, ClassesIfTrue, ClassesIfFalse }
  ///
  /// If the compare returns an exact class test, ClassesIfTrue ==
  /// ~ClassesIfFalse
  ///
  /// This is a less exact version of fcmpToClassTest (e.g. fcmpToClassTest will
  /// only succeed for a test of x > 0 implies positive, but not x > 1).
  ///
  /// If \p LookThroughSrc is true, consider the input value when computing the
  /// mask. This may look through sign bit operations.
  ///
  /// If \p LookThroughSrc is false, ignore the source value (i.e. the first
  /// pair element will always be LHS.
  ///
  static std::tuple<ValueRefT, FPClassTest, FPClassTest>
  fcmpImpliesClass(CmpInst::Predicate Pred, const FunctionT &F, ValueRefT LHS,
                   FPClassTest RHSClass, bool LookThroughSrc) {
    assert(RHSClass != fcNone);
    ValueRefT Src = LHS;

    if (Pred == FCmpInst::FCMP_TRUE)
      return exactClass(Src, fcAllFlags);

    if (Pred == FCmpInst::FCMP_FALSE)
      return exactClass(Src, fcNone);

    const FPClassTest OrigClass = RHSClass;

    const bool IsNegativeRHS = (RHSClass & fcNegative) == RHSClass;
    const bool IsPositiveRHS = (RHSClass & fcPositive) == RHSClass;
    const bool IsNaN = (RHSClass & ~fcNan) == fcNone;

    if (IsNaN) {
      // fcmp o__ x, nan -> false
      // fcmp u__ x, nan -> true
      return exactClass(Src, CmpInst::isOrdered(Pred) ? fcNone : fcAllFlags);
    }

    // fcmp ord x, zero|normal|subnormal|inf -> ~fcNan
    if (Pred == FCmpInst::FCMP_ORD)
      return exactClass(Src, ~fcNan);

    // fcmp uno x, zero|normal|subnormal|inf -> fcNan
    if (Pred == FCmpInst::FCMP_UNO)
      return exactClass(Src, fcNan);

    const bool IsFabs = LookThroughSrc && lookThroughFAbs(F, LHS, Src);
    if (IsFabs)
      RHSClass = llvm::inverse_fabs(RHSClass);

    const bool IsZero = (OrigClass & fcZero) == OrigClass;
    if (IsZero) {
      assert(Pred != FCmpInst::FCMP_ORD && Pred != FCmpInst::FCMP_UNO);
      // Compares with fcNone are only exactly equal to fcZero if input
      // denormals are not flushed.
      // TODO: Handle DAZ by expanding masks to cover subnormal cases.
      DenormalMode Mode = queryDenormalMode(F, LHS);
      if (Mode.Input != DenormalMode::IEEE)
        return {Invalid, fcAllFlags, fcAllFlags};

      switch (Pred) {
      case FCmpInst::FCMP_OEQ: // Match x == 0.0
        return exactClass(Src, fcZero);
      case FCmpInst::FCMP_UEQ: // Match isnan(x) || (x == 0.0)
        return exactClass(Src, fcZero | fcNan);
      case FCmpInst::FCMP_UNE: // Match (x != 0.0)
        return exactClass(Src, ~fcZero);
      case FCmpInst::FCMP_ONE: // Match !isnan(x) && x != 0.0
        return exactClass(Src, ~fcNan & ~fcZero);
      case FCmpInst::FCMP_ORD:
        // Canonical form of ord/uno is with a zero. We could also handle
        // non-canonical other non-NaN constants or LHS == RHS.
        return exactClass(Src, ~fcNan);
      case FCmpInst::FCMP_UNO:
        return exactClass(Src, fcNan);
      case FCmpInst::FCMP_OGT: // x > 0
        return exactClass(Src, fcPosSubnormal | fcPosNormal | fcPosInf);
      case FCmpInst::FCMP_UGT: // isnan(x) || x > 0
        return exactClass(Src, fcPosSubnormal | fcPosNormal | fcPosInf | fcNan);
      case FCmpInst::FCMP_OGE: // x >= 0
        return exactClass(Src, fcPositive | fcNegZero);
      case FCmpInst::FCMP_UGE: // isnan(x) || x >= 0
        return exactClass(Src, fcPositive | fcNegZero | fcNan);
      case FCmpInst::FCMP_OLT: // x < 0
        return exactClass(Src, fcNegSubnormal | fcNegNormal | fcNegInf);
      case FCmpInst::FCMP_ULT: // isnan(x) || x < 0
        return exactClass(Src, fcNegSubnormal | fcNegNormal | fcNegInf | fcNan);
      case FCmpInst::FCMP_OLE: // x <= 0
        return exactClass(Src, fcNegative | fcPosZero);
      case FCmpInst::FCMP_ULE: // isnan(x) || x <= 0
        return exactClass(Src, fcNegative | fcPosZero | fcNan);
      default:
        llvm_unreachable("all compare types are handled");
      }

      return {Invalid, fcAllFlags, fcAllFlags};
    }

    const bool IsDenormalRHS = (OrigClass & fcSubnormal) == OrigClass;

    const bool IsInf = (OrigClass & fcInf) == OrigClass;
    if (IsInf) {
      FPClassTest Mask = fcAllFlags;

      switch (Pred) {
      case FCmpInst::FCMP_OEQ:
      case FCmpInst::FCMP_UNE: {
        // Match __builtin_isinf patterns
        //
        //   fcmp oeq x, +inf -> is_fpclass x, fcPosInf
        //   fcmp oeq fabs(x), +inf -> is_fpclass x, fcInf
        //   fcmp oeq x, -inf -> is_fpclass x, fcNegInf
        //   fcmp oeq fabs(x), -inf -> is_fpclass x, 0 -> false
        //
        //   fcmp une x, +inf -> is_fpclass x, ~fcPosInf
        //   fcmp une fabs(x), +inf -> is_fpclass x, ~fcInf
        //   fcmp une x, -inf -> is_fpclass x, ~fcNegInf
        //   fcmp une fabs(x), -inf -> is_fpclass x, fcAllFlags -> true
        if (IsNegativeRHS) {
          Mask = fcNegInf;
          if (IsFabs)
            Mask = fcNone;
        } else {
          Mask = fcPosInf;
          if (IsFabs)
            Mask |= fcNegInf;
        }
        break;
      }
      case FCmpInst::FCMP_ONE:
      case FCmpInst::FCMP_UEQ: {
        // Match __builtin_isinf patterns
        //   fcmp one x, -inf -> is_fpclass x, fcNegInf
        //   fcmp one fabs(x), -inf -> is_fpclass x, ~fcNegInf & ~fcNan
        //   fcmp one x, +inf -> is_fpclass x, ~fcNegInf & ~fcNan
        //   fcmp one fabs(x), +inf -> is_fpclass x, ~fcInf & fcNan
        //
        //   fcmp ueq x, +inf -> is_fpclass x, fcPosInf|fcNan
        //   fcmp ueq (fabs x), +inf -> is_fpclass x, fcInf|fcNan
        //   fcmp ueq x, -inf -> is_fpclass x, fcNegInf|fcNan
        //   fcmp ueq fabs(x), -inf -> is_fpclass x, fcNan
        if (IsNegativeRHS) {
          Mask = ~fcNegInf & ~fcNan;
          if (IsFabs)
            Mask = ~fcNan;
        } else {
          Mask = ~fcPosInf & ~fcNan;
          if (IsFabs)
            Mask &= ~fcNegInf;
        }

        break;
      }
      case FCmpInst::FCMP_OLT:
      case FCmpInst::FCMP_UGE: {
        if (IsNegativeRHS) {
          // No value is ordered and less than negative infinity.
          // All values are unordered with or at least negative infinity.
          // fcmp olt x, -inf -> false
          // fcmp uge x, -inf -> true
          Mask = fcNone;
          break;
        }

        // fcmp olt fabs(x), +inf -> fcFinite
        // fcmp uge fabs(x), +inf -> ~fcFinite
        // fcmp olt x, +inf -> fcFinite|fcNegInf
        // fcmp uge x, +inf -> ~(fcFinite|fcNegInf)
        Mask = fcFinite;
        if (!IsFabs)
          Mask |= fcNegInf;
        break;
      }
      case FCmpInst::FCMP_OGE:
      case FCmpInst::FCMP_ULT: {
        if (IsNegativeRHS) {
          // fcmp oge x, -inf -> ~fcNan
          // fcmp oge fabs(x), -inf -> ~fcNan
          // fcmp ult x, -inf -> fcNan
          // fcmp ult fabs(x), -inf -> fcNan
          Mask = ~fcNan;
          break;
        }

        // fcmp oge fabs(x), +inf -> fcInf
        // fcmp oge x, +inf -> fcPosInf
        // fcmp ult fabs(x), +inf -> ~fcInf
        // fcmp ult x, +inf -> ~fcPosInf
        Mask = fcPosInf;
        if (IsFabs)
          Mask |= fcNegInf;
        break;
      }
      case FCmpInst::FCMP_OGT:
      case FCmpInst::FCMP_ULE: {
        if (IsNegativeRHS) {
          // fcmp ogt x, -inf -> fcmp one x, -inf
          // fcmp ogt fabs(x), -inf -> fcmp ord x, x
          // fcmp ule x, -inf -> fcmp ueq x, -inf
          // fcmp ule fabs(x), -inf -> fcmp uno x, x
          Mask = IsFabs ? ~fcNan : ~(fcNegInf | fcNan);
          break;
        }

        // No value is ordered and greater than infinity.
        Mask = fcNone;
        break;
      }
      case FCmpInst::FCMP_OLE:
      case FCmpInst::FCMP_UGT: {
        if (IsNegativeRHS) {
          Mask = IsFabs ? fcNone : fcNegInf;
          break;
        }

        // fcmp ole x, +inf -> fcmp ord x, x
        // fcmp ole fabs(x), +inf -> fcmp ord x, x
        // fcmp ole x, -inf -> fcmp oeq x, -inf
        // fcmp ole fabs(x), -inf -> false
        Mask = ~fcNan;
        break;
      }
      default:
        llvm_unreachable("all compare types are handled");
      }

      // Invert the comparison for the unordered cases.
      if (FCmpInst::isUnordered(Pred))
        Mask = ~Mask;

      return exactClass(Src, Mask);
    }

    if (Pred == FCmpInst::FCMP_OEQ)
      return {Src, RHSClass, fcAllFlags};

    if (Pred == FCmpInst::FCMP_UEQ) {
      FPClassTest Class = RHSClass | fcNan;
      return {Src, Class, ~fcNan};
    }

    if (Pred == FCmpInst::FCMP_ONE)
      return {Src, ~fcNan, RHSClass | fcNan};

    if (Pred == FCmpInst::FCMP_UNE)
      return {Src, fcAllFlags, RHSClass};

    assert((RHSClass == fcNone || RHSClass == fcPosNormal ||
            RHSClass == fcNegNormal || RHSClass == fcNormal ||
            RHSClass == fcPosSubnormal || RHSClass == fcNegSubnormal ||
            RHSClass == fcSubnormal) &&
           "should have been recognized as an exact class test");

    if (IsNegativeRHS) {
      // TODO: Handle fneg(fabs)
      if (IsFabs) {
        // fabs(x) o> -k -> fcmp ord x, x
        // fabs(x) u> -k -> true
        // fabs(x) o< -k -> false
        // fabs(x) u< -k -> fcmp uno x, x
        switch (Pred) {
        case FCmpInst::FCMP_OGT:
        case FCmpInst::FCMP_OGE:
          return {Src, ~fcNan, fcNan};
        case FCmpInst::FCMP_UGT:
        case FCmpInst::FCMP_UGE:
          return {Src, fcAllFlags, fcNone};
        case FCmpInst::FCMP_OLT:
        case FCmpInst::FCMP_OLE:
          return {Src, fcNone, fcAllFlags};
        case FCmpInst::FCMP_ULT:
        case FCmpInst::FCMP_ULE:
          return {Src, fcNan, ~fcNan};
        default:
          break;
        }

        return {Invalid, fcAllFlags, fcAllFlags};
      }

      FPClassTest ClassesLE = fcNegInf | fcNegNormal;
      FPClassTest ClassesGE = fcPositive | fcNegZero | fcNegSubnormal;

      if (IsDenormalRHS)
        ClassesLE |= fcNegSubnormal;
      else
        ClassesGE |= fcNegNormal;

      switch (Pred) {
      case FCmpInst::FCMP_OGT:
      case FCmpInst::FCMP_OGE:
        return {Src, ClassesGE, ~ClassesGE | RHSClass};
      case FCmpInst::FCMP_UGT:
      case FCmpInst::FCMP_UGE:
        return {Src, ClassesGE | fcNan, ~(ClassesGE | fcNan) | RHSClass};
      case FCmpInst::FCMP_OLT:
      case FCmpInst::FCMP_OLE:
        return {Src, ClassesLE, ~ClassesLE | RHSClass};
      case FCmpInst::FCMP_ULT:
      case FCmpInst::FCMP_ULE:
        return {Src, ClassesLE | fcNan, ~(ClassesLE | fcNan) | RHSClass};
      default:
        break;
      }
    } else if (IsPositiveRHS) {
      FPClassTest ClassesGE = fcPosNormal | fcPosInf;
      FPClassTest ClassesLE = fcNegative | fcPosZero | fcPosSubnormal;
      if (IsDenormalRHS)
        ClassesGE |= fcPosSubnormal;
      else
        ClassesLE |= fcPosNormal;

      if (IsFabs) {
        ClassesGE = llvm::inverse_fabs(ClassesGE);
        ClassesLE = llvm::inverse_fabs(ClassesLE);
      }

      switch (Pred) {
      case FCmpInst::FCMP_OGT:
      case FCmpInst::FCMP_OGE:
        return {Src, ClassesGE, ~ClassesGE | RHSClass};
      case FCmpInst::FCMP_UGT:
      case FCmpInst::FCMP_UGE:
        return {Src, ClassesGE | fcNan, ~(ClassesGE | fcNan) | RHSClass};
      case FCmpInst::FCMP_OLT:
      case FCmpInst::FCMP_OLE:
        return {Src, ClassesLE, ~ClassesLE | RHSClass};
      case FCmpInst::FCMP_ULT:
      case FCmpInst::FCMP_ULE:
        return {Src, ClassesLE | fcNan, ~(ClassesLE | fcNan) | RHSClass};
      default:
        break;
      }
    }

    return {Invalid, fcAllFlags, fcAllFlags};
  }

  static std::tuple<ValueRefT, FPClassTest, FPClassTest>
  fcmpImpliesClass(CmpInst::Predicate Pred, const FunctionT &F, ValueRefT LHS,
                   const APFloat &ConstRHS, bool LookThroughSrc) {
    // We can refine checks against smallest normal / largest denormal to an
    // exact class test.
    if (!ConstRHS.isNegative() && ConstRHS.isSmallestNormalized()) {
      ValueRefT Src = LHS;
      const bool IsFabs = LookThroughSrc && lookThroughFAbs(F, LHS, Src);

      FPClassTest Mask;
      // Match pattern that's used in __builtin_isnormal.
      switch (Pred) {
      case FCmpInst::FCMP_OLT:
      case FCmpInst::FCMP_UGE: {
        // fcmp olt x, smallest_normal ->
        // fcNegInf|fcNegNormal|fcSubnormal|fcZero fcmp olt fabs(x),
        // smallest_normal -> fcSubnormal|fcZero fcmp uge x, smallest_normal ->
        // fcNan|fcPosNormal|fcPosInf fcmp uge fabs(x), smallest_normal ->
        // ~(fcSubnormal|fcZero)
        Mask = fcZero | fcSubnormal;
        if (!IsFabs)
          Mask |= fcNegNormal | fcNegInf;

        break;
      }
      case FCmpInst::FCMP_OGE:
      case FCmpInst::FCMP_ULT: {
        // fcmp oge x, smallest_normal -> fcPosNormal | fcPosInf
        // fcmp oge fabs(x), smallest_normal -> fcInf | fcNormal
        // fcmp ult x, smallest_normal -> ~(fcPosNormal | fcPosInf)
        // fcmp ult fabs(x), smallest_normal -> ~(fcInf | fcNormal)
        Mask = fcPosInf | fcPosNormal;
        if (IsFabs)
          Mask |= fcNegInf | fcNegNormal;
        break;
      }
      default:
        return fcmpImpliesClass(Pred, F, LHS, ConstRHS.classify(),
                                LookThroughSrc);
      }

      // Invert the comparison for the unordered cases.
      if (FCmpInst::isUnordered(Pred))
        Mask = ~Mask;

      return exactClass(Src, Mask);
    }

    return fcmpImpliesClass(Pred, F, LHS, ConstRHS.classify(), LookThroughSrc);
  }

  static std::tuple<ValueRefT, FPClassTest, FPClassTest>
  fcmpImpliesClass(CmpInst::Predicate Pred, const FunctionT &F, ValueRefT LHS,
                   ValueRefT RHS, bool LookThroughSrc) {
    std::optional<APFloat> ConstRHS = matchConstantFloat(F, RHS);
    if (!ConstRHS)
      return {Invalid, fcAllFlags, fcAllFlags};

    // TODO: Just call computeKnownFPClass for RHS to handle non-constants.
    return fcmpImpliesClass(Pred, F, LHS, *ConstRHS, LookThroughSrc);
  }
};

} // namespace llvm

#endif // LLVM_ADT_GENERICFLOATINGPOINTPREDICATEUTILS_H
