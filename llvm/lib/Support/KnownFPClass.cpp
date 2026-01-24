//===- llvm/Support/KnownFPClass.h - Stores known fplcass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a class for representing known fpclasses used by
// computeKnownFPClass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownFPClass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

KnownFPClass::KnownFPClass(const APFloat &C)
    : KnownFPClasses(C.classify()), SignBit(C.isNegative()) {}

/// Return true if it's possible to assume IEEE treatment of input denormals in
/// \p F for \p Val.
static bool inputDenormalIsIEEE(DenormalMode Mode) {
  return Mode.Input == DenormalMode::IEEE;
}

static bool inputDenormalIsIEEEOrPosZero(DenormalMode Mode) {
  return Mode.Input == DenormalMode::IEEE ||
         Mode.Input == DenormalMode::PositiveZero;
}

bool KnownFPClass::isKnownNeverLogicalZero(DenormalMode Mode) const {
  return isKnownNeverZero() &&
         (isKnownNeverSubnormal() || inputDenormalIsIEEE(Mode));
}

bool KnownFPClass::isKnownNeverLogicalNegZero(DenormalMode Mode) const {
  return isKnownNeverNegZero() &&
         (isKnownNeverNegSubnormal() || inputDenormalIsIEEEOrPosZero(Mode));
}

bool KnownFPClass::isKnownNeverLogicalPosZero(DenormalMode Mode) const {
  if (!isKnownNeverPosZero())
    return false;

  // If we know there are no denormals, nothing can be flushed to zero.
  if (isKnownNeverSubnormal())
    return true;

  switch (Mode.Input) {
  case DenormalMode::IEEE:
    return true;
  case DenormalMode::PreserveSign:
    // Negative subnormal won't flush to +0
    return isKnownNeverPosSubnormal();
  case DenormalMode::PositiveZero:
  default:
    // Both positive and negative subnormal could flush to +0
    return false;
  }

  llvm_unreachable("covered switch over denormal mode");
}

void KnownFPClass::propagateDenormal(const KnownFPClass &Src,
                                     DenormalMode Mode) {
  KnownFPClasses = Src.KnownFPClasses;
  // If we aren't assuming the source can't be a zero, we don't have to check if
  // a denormal input could be flushed.
  if (!Src.isKnownNeverPosZero() && !Src.isKnownNeverNegZero())
    return;

  // If we know the input can't be a denormal, it can't be flushed to 0.
  if (Src.isKnownNeverSubnormal())
    return;

  if (!Src.isKnownNeverPosSubnormal() && Mode != DenormalMode::getIEEE())
    KnownFPClasses |= fcPosZero;

  if (!Src.isKnownNeverNegSubnormal() && Mode != DenormalMode::getIEEE()) {
    if (Mode != DenormalMode::getPositiveZero())
      KnownFPClasses |= fcNegZero;

    if (Mode.Input == DenormalMode::PositiveZero ||
        Mode.Output == DenormalMode::PositiveZero ||
        Mode.Input == DenormalMode::Dynamic ||
        Mode.Output == DenormalMode::Dynamic)
      KnownFPClasses |= fcPosZero;
  }
}

KnownFPClass KnownFPClass::minMaxLike(const KnownFPClass &LHS_,
                                      const KnownFPClass &RHS_, MinMaxKind Kind,
                                      DenormalMode Mode) {
  KnownFPClass KnownLHS = LHS_;
  KnownFPClass KnownRHS = RHS_;

  bool NeverNaN = KnownLHS.isKnownNeverNaN() || KnownRHS.isKnownNeverNaN();
  KnownFPClass Known = KnownLHS | KnownRHS;

  // If either operand is not NaN, the result is not NaN.
  if (NeverNaN &&
      (Kind == MinMaxKind::minnum || Kind == MinMaxKind::maxnum ||
       Kind == MinMaxKind::minimumnum || Kind == MinMaxKind::maximumnum))
    Known.knownNot(fcNan);

  if (Kind == MinMaxKind::maxnum || Kind == MinMaxKind::maximumnum) {
    // If at least one operand is known to be positive, the result must be
    // positive.
    if ((KnownLHS.cannotBeOrderedLessThanZero() &&
         KnownLHS.isKnownNeverNaN()) ||
        (KnownRHS.cannotBeOrderedLessThanZero() && KnownRHS.isKnownNeverNaN()))
      Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
  } else if (Kind == MinMaxKind::maximum) {
    // If at least one operand is known to be positive, the result must be
    // positive.
    if (KnownLHS.cannotBeOrderedLessThanZero() ||
        KnownRHS.cannotBeOrderedLessThanZero())
      Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);
  } else if (Kind == MinMaxKind::minnum || Kind == MinMaxKind::minimumnum) {
    // If at least one operand is known to be negative, the result must be
    // negative.
    if ((KnownLHS.cannotBeOrderedGreaterThanZero() &&
         KnownLHS.isKnownNeverNaN()) ||
        (KnownRHS.cannotBeOrderedGreaterThanZero() &&
         KnownRHS.isKnownNeverNaN()))
      Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);
  } else if (Kind == MinMaxKind::minimum) {
    // If at least one operand is known to be negative, the result must be
    // negative.
    if (KnownLHS.cannotBeOrderedGreaterThanZero() ||
        KnownRHS.cannotBeOrderedGreaterThanZero())
      Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);
  } else
    llvm_unreachable("unhandled intrinsic");

  // Fixup zero handling if denormals could be returned as a zero.
  //
  // As there's no spec for denormal flushing, be conservative with the
  // treatment of denormals that could be flushed to zero. For older
  // subtargets on AMDGPU the min/max instructions would not flush the
  // output and return the original value.
  //
  if ((Known.KnownFPClasses & fcZero) != fcNone &&
      !Known.isKnownNeverSubnormal()) {
    if (Mode != DenormalMode::getIEEE())
      Known.KnownFPClasses |= fcZero;
  }

  if (Known.isKnownNeverNaN()) {
    if (KnownLHS.SignBit && KnownRHS.SignBit &&
        *KnownLHS.SignBit == *KnownRHS.SignBit) {
      if (*KnownLHS.SignBit)
        Known.signBitMustBeOne();
      else
        Known.signBitMustBeZero();
    } else if ((Kind == MinMaxKind::maximum || Kind == MinMaxKind::minimum ||
                Kind == MinMaxKind::maximumnum ||
                Kind == MinMaxKind::minimumnum) ||
               // FIXME: Should be using logical zero versions
               ((KnownLHS.isKnownNeverNegZero() ||
                 KnownRHS.isKnownNeverPosZero()) &&
                (KnownLHS.isKnownNeverPosZero() ||
                 KnownRHS.isKnownNeverNegZero()))) {
      // Don't take sign bit from NaN operands.
      if (!KnownLHS.isKnownNeverNaN())
        KnownLHS.SignBit = std::nullopt;
      if (!KnownRHS.isKnownNeverNaN())
        KnownRHS.SignBit = std::nullopt;
      if ((Kind == MinMaxKind::maximum || Kind == MinMaxKind::maximumnum ||
           Kind == MinMaxKind::maxnum) &&
          (KnownLHS.SignBit == false || KnownRHS.SignBit == false))
        Known.signBitMustBeZero();
      else if ((Kind == MinMaxKind::minimum || Kind == MinMaxKind::minimumnum ||
                Kind == MinMaxKind::minnum) &&
               (KnownLHS.SignBit == true || KnownRHS.SignBit == true))
        Known.signBitMustBeOne();
    }
  }

  return Known;
}

KnownFPClass KnownFPClass::canonicalize(const KnownFPClass &KnownSrc,
                                        DenormalMode DenormMode) {
  KnownFPClass Known;

  // This is essentially a stronger form of
  // propagateCanonicalizingSrc. Other "canonicalizing" operations don't
  // actually have an IR canonicalization guarantee.

  // Canonicalize may flush denormals to zero, so we have to consider the
  // denormal mode to preserve known-not-0 knowledge.
  Known.KnownFPClasses = KnownSrc.KnownFPClasses | fcZero | fcQNan;

  // Stronger version of propagateNaN
  // Canonicalize is guaranteed to quiet signaling nans.
  if (KnownSrc.isKnownNeverNaN())
    Known.knownNot(fcNan);
  else
    Known.knownNot(fcSNan);

  // FIXME: Missing check of IEEE like types.

  // If the parent function flushes denormals, the canonical output cannot be a
  // denormal.
  if (DenormMode == DenormalMode::getIEEE()) {
    if (KnownSrc.isKnownNever(fcPosZero))
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNever(fcNegZero))
      Known.knownNot(fcNegZero);
    return Known;
  }

  if (DenormMode.inputsAreZero() || DenormMode.outputsAreZero())
    Known.knownNot(fcSubnormal);

  if (DenormMode == DenormalMode::getPreserveSign()) {
    if (KnownSrc.isKnownNever(fcPosZero | fcPosSubnormal))
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNever(fcNegZero | fcNegSubnormal))
      Known.knownNot(fcNegZero);
    return Known;
  }

  if (DenormMode.Input == DenormalMode::PositiveZero ||
      (DenormMode.Output == DenormalMode::PositiveZero &&
       DenormMode.Input == DenormalMode::IEEE))
    Known.knownNot(fcNegZero);

  return Known;
}

// Handle known sign bit and nan cases for fadd.
static KnownFPClass fadd_impl(const KnownFPClass &KnownLHS,
                              const KnownFPClass &KnownRHS, DenormalMode Mode) {
  KnownFPClass Known;

  // Adding positive and negative infinity produces NaN.
  // TODO: Check sign of infinities.
  if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
      (KnownLHS.isKnownNeverInfinity() || KnownRHS.isKnownNeverInfinity()))
    Known.knownNot(fcNan);

  if (KnownLHS.cannotBeOrderedLessThanZero() &&
      KnownRHS.cannotBeOrderedLessThanZero()) {
    Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

    // This can't underflow if one of the operands is known normal.
    if (KnownLHS.isKnownNever(fcZero | fcPosSubnormal) ||
        KnownRHS.isKnownNever(fcZero | fcPosSubnormal))
      Known.knownNot(fcZero);
  }

  if (KnownLHS.cannotBeOrderedGreaterThanZero() &&
      KnownRHS.cannotBeOrderedGreaterThanZero()) {
    Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

    // This can't underflow if one of the operands is known normal.
    if (KnownLHS.isKnownNever(fcZero | fcNegSubnormal) ||
        KnownRHS.isKnownNever(fcZero | fcNegSubnormal))
      Known.knownNot(fcZero);
  }

  return Known;
}

KnownFPClass KnownFPClass::fadd(const KnownFPClass &KnownLHS,
                                const KnownFPClass &KnownRHS,
                                DenormalMode Mode) {
  KnownFPClass Known = fadd_impl(KnownLHS, KnownRHS, Mode);

  // (fadd x, 0.0) is guaranteed to return +0.0, not -0.0.
  if ((KnownLHS.isKnownNeverLogicalNegZero(Mode) ||
       KnownRHS.isKnownNeverLogicalNegZero(Mode)) &&
      // Make sure output negative denormal can't flush to -0
      (Mode.Output == DenormalMode::IEEE ||
       Mode.Output == DenormalMode::PositiveZero))
    Known.knownNot(fcNegZero);

  return Known;
}

KnownFPClass KnownFPClass::fadd_self(const KnownFPClass &KnownSrc,
                                     DenormalMode Mode) {
  KnownFPClass Known = fadd(KnownSrc, KnownSrc, Mode);

  // Doubling 0 will give the same 0.
  if (KnownSrc.isKnownNeverLogicalPosZero(Mode) &&
      (Mode.Output == DenormalMode::IEEE ||
       (Mode.Output == DenormalMode::PreserveSign &&
        KnownSrc.isKnownNeverPosSubnormal()) ||
       (Mode.Output == DenormalMode::PositiveZero &&
        KnownSrc.isKnownNeverSubnormal())))
    Known.knownNot(fcPosZero);

  return Known;
}

KnownFPClass KnownFPClass::fsub(const KnownFPClass &KnownLHS,
                                const KnownFPClass &KnownRHS,
                                DenormalMode Mode) {
  return fadd(KnownLHS, fneg(KnownRHS), Mode);
}

KnownFPClass KnownFPClass::fmul(const KnownFPClass &KnownLHS,
                                const KnownFPClass &KnownRHS,
                                DenormalMode Mode) {
  KnownFPClass Known;

  // xor sign bit.
  if ((KnownLHS.isKnownNever(fcNegative) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcPositive) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcNegative);

  if ((KnownLHS.isKnownNever(fcPositive) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcNegative) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcPositive);

  // inf * anything => inf or nan
  if (KnownLHS.isKnownAlways(fcInf | fcNan) ||
      KnownRHS.isKnownAlways(fcInf | fcNan))
    Known.knownNot(fcNormal | fcSubnormal | fcZero);

  // 0 * anything => 0 or nan
  if (KnownRHS.isKnownAlways(fcZero | fcNan) ||
      KnownLHS.isKnownAlways(fcZero | fcNan))
    Known.knownNot(fcNormal | fcSubnormal | fcInf);

  // +/-0 * +/-inf = nan
  if ((KnownLHS.isKnownAlways(fcZero | fcNan) &&
       KnownRHS.isKnownAlways(fcInf | fcNan)) ||
      (KnownLHS.isKnownAlways(fcInf | fcNan) &&
       KnownRHS.isKnownAlways(fcZero | fcNan)))
    Known.knownNot(~fcNan);

  if (!KnownLHS.isKnownNeverNaN() || !KnownRHS.isKnownNeverNaN())
    return Known;

  if (KnownLHS.SignBit && KnownRHS.SignBit) {
    if (*KnownLHS.SignBit == *KnownRHS.SignBit)
      Known.signBitMustBeZero();
    else
      Known.signBitMustBeOne();
  }

  // If 0 * +/-inf produces NaN.
  if ((KnownRHS.isKnownNeverInfinity() ||
       KnownLHS.isKnownNeverLogicalZero(Mode)) &&
      (KnownLHS.isKnownNeverInfinity() ||
       KnownRHS.isKnownNeverLogicalZero(Mode)))
    Known.knownNot(fcNan);

  return Known;
}

KnownFPClass KnownFPClass::fdiv(const KnownFPClass &KnownLHS,
                                const KnownFPClass &KnownRHS,
                                DenormalMode Mode) {
  KnownFPClass Known;

  // Only 0/0, Inf/Inf produce NaN.
  if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
      (KnownLHS.isKnownNeverInfinity() || KnownRHS.isKnownNeverInfinity()) &&
      (KnownLHS.isKnownNeverLogicalZero(Mode) ||
       KnownRHS.isKnownNeverLogicalZero(Mode))) {
    Known.knownNot(fcNan);
  }

  // xor sign bit.
  // X / -0.0 is -Inf (or NaN).
  // +X / +X is +X
  if ((KnownLHS.isKnownNever(fcNegative) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcPositive) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcNegative);

  if ((KnownLHS.isKnownNever(fcPositive) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcNegative) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcPositive);

  // 0 / x => 0 or nan
  if (KnownLHS.isKnownAlways(fcZero))
    Known.knownNot(fcSubnormal | fcNormal | fcInf);

  // x / 0 => nan or inf
  if (KnownRHS.isKnownAlways(fcZero))
    Known.knownNot(fcFinite);

  return Known;
}

KnownFPClass KnownFPClass::fdiv_self(const KnownFPClass &KnownSrc,
                                     DenormalMode Mode) {
  // X / X is always exactly 1.0 or a NaN.
  KnownFPClass Known(fcNan | fcPosNormal);

  if (KnownSrc.isKnownNeverInfOrNaN() && KnownSrc.isKnownNeverLogicalZero(Mode))
    Known.knownNot(fcNan);
  else if (KnownSrc.isKnownNever(fcSNan))
    Known.knownNot(fcSNan);

  return Known;
}
KnownFPClass KnownFPClass::frem_self(const KnownFPClass &KnownSrc,
                                     DenormalMode Mode) {
  // X % X is always exactly [+-]0.0 or a NaN.
  KnownFPClass Known(fcNan | fcZero);

  if (KnownSrc.isKnownNeverInfOrNaN() && KnownSrc.isKnownNeverLogicalZero(Mode))
    Known.knownNot(fcNan);
  else if (KnownSrc.isKnownNever(fcSNan))
    Known.knownNot(fcSNan);

  return Known;
}

KnownFPClass KnownFPClass::fma(const KnownFPClass &KnownLHS,
                               const KnownFPClass &KnownRHS,
                               const KnownFPClass &KnownAddend,
                               DenormalMode Mode) {
  KnownFPClass Mul = fmul(KnownLHS, KnownRHS, Mode);

  // FMA differs from the base fmul + fadd handling only in the treatment of -0
  // results.
  //
  // If the multiply is a -0 due to rounding, the final -0 + 0 will be -0,
  // unlike for a separate fadd.
  return fadd_impl(Mul, KnownAddend, Mode);
}

KnownFPClass KnownFPClass::fma_square(const KnownFPClass &KnownSquared,
                                      const KnownFPClass &KnownAddend,
                                      DenormalMode Mode) {
  KnownFPClass Squared = square(KnownSquared, Mode);
  KnownFPClass Known = fadd_impl(Squared, KnownAddend, Mode);

  // Since we know the squared input must be positive, the add of opposite sign
  // infinities nan hazard only applies for negative inf.
  //
  // TODO: Alternatively to proving addend is not -inf, we could know Squared is
  // not pinf. Other than the degenerate always-subnormal input case, we can't
  // prove that without a known range.
  if (KnownAddend.isKnownNever(fcNegInf | fcNan) && Squared.isKnownNever(fcNan))
    Known.knownNot(fcNan);

  return Known;
}

KnownFPClass KnownFPClass::exp(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;
  Known.knownNot(fcNegative);

  Known.propagateNaN(KnownSrc);

  if (KnownSrc.cannotBeOrderedLessThanZero()) {
    // If the source is positive this cannot underflow.
    Known.knownNot(fcPosZero);

    // Cannot introduce denormal values.
    Known.knownNot(fcPosSubnormal);
  }

  // If the source is negative, this cannot overflow to infinity.
  if (KnownSrc.cannotBeOrderedGreaterThanZero())
    Known.knownNot(fcPosInf);

  return Known;
}

void KnownFPClass::propagateCanonicalizingSrc(const KnownFPClass &Src,
                                              DenormalMode Mode) {
  propagateDenormal(Src, Mode);
  propagateNaN(Src, /*PreserveSign=*/true);
}

KnownFPClass KnownFPClass::log(const KnownFPClass &KnownSrc,
                               DenormalMode Mode) {
  KnownFPClass Known;
  Known.knownNot(fcNegZero);

  if (KnownSrc.isKnownNeverPosInfinity())
    Known.knownNot(fcPosInf);

  if (KnownSrc.isKnownNeverNaN() && KnownSrc.cannotBeOrderedLessThanZero())
    Known.knownNot(fcNan);

  if (KnownSrc.isKnownNeverLogicalZero(Mode))
    Known.knownNot(fcNegInf);

  return Known;
}

KnownFPClass KnownFPClass::sqrt(const KnownFPClass &KnownSrc,
                                DenormalMode Mode) {
  KnownFPClass Known;
  Known.knownNot(fcPosSubnormal);

  if (KnownSrc.isKnownNeverPosInfinity())
    Known.knownNot(fcPosInf);
  if (KnownSrc.isKnownNever(fcSNan))
    Known.knownNot(fcSNan);

  // Any negative value besides -0 returns a nan.
  if (KnownSrc.isKnownNeverNaN() && KnownSrc.cannotBeOrderedLessThanZero())
    Known.knownNot(fcNan);

  // The only negative value that can be returned is -0 for -0 inputs.
  Known.knownNot(fcNegInf | fcNegSubnormal | fcNegNormal);

  // If the input denormal mode could be PreserveSign, a negative
  // subnormal input could produce a negative zero output.
  if (KnownSrc.isKnownNeverLogicalNegZero(Mode))
    Known.knownNot(fcNegZero);

  return Known;
}

KnownFPClass KnownFPClass::fpext(const KnownFPClass &KnownSrc,
                                 const fltSemantics &DstTy,
                                 const fltSemantics &SrcTy) {
  // Infinity, nan and zero propagate from source.
  KnownFPClass Known = KnownSrc;

  // All subnormal inputs should be in the normal range in the result type.
  if (APFloat::isRepresentableAsNormalIn(SrcTy, DstTy)) {
    if (Known.KnownFPClasses & fcPosSubnormal)
      Known.KnownFPClasses |= fcPosNormal;
    if (Known.KnownFPClasses & fcNegSubnormal)
      Known.KnownFPClasses |= fcNegNormal;
    Known.knownNot(fcSubnormal);
  }

  // Sign bit of a nan isn't guaranteed.
  if (!Known.isKnownNeverNaN())
    Known.SignBit = std::nullopt;

  return Known;
}

KnownFPClass KnownFPClass::fptrunc(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // Sign should be preserved
  // TODO: Handle cannot be ordered greater than zero
  if (KnownSrc.cannotBeOrderedLessThanZero())
    Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

  Known.propagateNaN(KnownSrc, true);

  // Infinity needs a range check.
  return Known;
}

KnownFPClass KnownFPClass::roundToIntegral(const KnownFPClass &KnownSrc,
                                           bool IsTrunc,
                                           bool IsMultiUnitFPType) {
  KnownFPClass Known;

  // Integer results cannot be subnormal.
  Known.knownNot(fcSubnormal);

  Known.propagateNaN(KnownSrc, true);

  // Pass through infinities, except PPC_FP128 is a special case for
  // intrinsics other than trunc.
  if (IsTrunc || !IsMultiUnitFPType) {
    if (KnownSrc.isKnownNeverPosInfinity())
      Known.knownNot(fcPosInf);
    if (KnownSrc.isKnownNeverNegInfinity())
      Known.knownNot(fcNegInf);
  }

  // Negative round ups to 0 produce -0
  if (KnownSrc.isKnownNever(fcPosFinite))
    Known.knownNot(fcPosFinite);
  if (KnownSrc.isKnownNever(fcNegFinite))
    Known.knownNot(fcNegFinite);

  return Known;
}

KnownFPClass KnownFPClass::frexp_mant(const KnownFPClass &KnownSrc,
                                      DenormalMode Mode) {
  KnownFPClass Known;
  Known.knownNot(fcSubnormal);

  if (KnownSrc.isKnownNever(fcNegative))
    Known.knownNot(fcNegative);
  else {
    if (KnownSrc.isKnownNeverLogicalNegZero(Mode))
      Known.knownNot(fcNegZero);
    if (KnownSrc.isKnownNever(fcNegInf))
      Known.knownNot(fcNegInf);
  }

  if (KnownSrc.isKnownNever(fcPositive))
    Known.knownNot(fcPositive);
  else {
    if (KnownSrc.isKnownNeverLogicalPosZero(Mode))
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNever(fcPosInf))
      Known.knownNot(fcPosInf);
  }

  Known.propagateNaN(KnownSrc);
  return Known;
}
