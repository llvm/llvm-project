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
#include "llvm/Support/KnownBits.h"
#include <climits>

using namespace llvm;

KnownFPClass::KnownFPClass(const APFloat &C)
    : KnownFPClass(C.classify(), C.isNegative()) {}

KnownFPClass KnownFPClass::applyInputDenormalMode(const KnownFPClass &KnownSrc,
                                                  DenormalMode Mode) {
  KnownFPClass Known = KnownSrc;
  switch (Mode.Input) {
  case DenormalMode::IEEE:
    return Known;
  case DenormalMode::PreserveSign:
    if (KnownSrc.getKnownFPClasses() & fcPosSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    if (KnownSrc.getKnownFPClasses() & fcNegSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcNegZero);
    return Known;
  case DenormalMode::PositiveZero:
    if (KnownSrc.getKnownFPClasses() & fcSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    return Known;
  default:
    if (KnownSrc.getKnownFPClasses() & fcSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    if (KnownSrc.getKnownFPClasses() & fcNegSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcNegZero);
    return Known;
  }

  llvm_unreachable("covered switch over denormal mode");
}

KnownFPClass KnownFPClass::applyOutputDenormalMode(const KnownFPClass &KnownSrc,
                                                   DenormalMode Mode) {
  KnownFPClass Known = KnownSrc;
  switch (Mode.Output) {
  case DenormalMode::IEEE:
    return Known;
  case DenormalMode::PreserveSign:
    if (KnownSrc.getKnownFPClasses() & fcPosSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    if (KnownSrc.getKnownFPClasses() & fcNegSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcNegZero);
    return Known;
  case DenormalMode::PositiveZero:
    if (KnownSrc.getKnownFPClasses() & fcSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    return Known;
  default:
    if (KnownSrc.getKnownFPClasses() & fcSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcPosZero);
    if (KnownSrc.getKnownFPClasses() & fcNegSubnormal)
      Known.setKnownFPClasses(Known.getKnownFPClasses() | fcNegZero);
    return Known;
  }

  llvm_unreachable("covered switch over denormal mode");
}

bool KnownFPClass::isKnownNeverLogicalZero(DenormalMode Mode) const {
  return applyInputDenormalMode(*this, Mode).isKnownNeverZero();
}

bool KnownFPClass::isKnownNeverLogicalNegZero(DenormalMode Mode) const {
  return applyInputDenormalMode(*this, Mode).isKnownNeverNegZero();
}

bool KnownFPClass::isKnownNeverLogicalPosZero(DenormalMode Mode) const {
  return applyInputDenormalMode(*this, Mode).isKnownNeverPosZero();
}

KnownFPClass KnownFPClass::minMaxLike(const KnownFPClass &KnownLHS_,
                                      const KnownFPClass &KnownRHS_,
                                      MinMaxKind Kind, DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);

  KnownFPClass Known;

  Known.propagateNonNaN(KnownLHS, KnownRHS);

  // The *num functions return the non-NaN operand if only one operand is NaN.
  // FIXME: minnum/maxnum may return a NaN if either operand is a signaling NaN.
  if ((KnownLHS.isKnownNeverNaN() || KnownRHS.isKnownNeverNaN()) &&
      (Kind == MinMaxKind::minnum || Kind == MinMaxKind::maxnum ||
       Kind == MinMaxKind::minimumnum || Kind == MinMaxKind::maximumnum))
    Known.knownNot(fcNan);

  switch (Kind) {
  case MinMaxKind::maxnum: {
    if (KnownLHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyLess(KnownLHS.getKnownFPClasses(),
                                         /*OrderedZeroSign=*/false));
    if (KnownRHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyLess(KnownRHS.getKnownFPClasses(),
                                         /*OrderedZeroSign=*/false));
    break;
  }
  case MinMaxKind::minnum: {
    if (KnownLHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                            /*OrderedZeroSign=*/false));
    if (KnownRHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyGreater(KnownRHS.getKnownFPClasses(),
                                            /*OrderedZeroSign=*/false));
    break;
  }
  case MinMaxKind::maximumnum: {
    if (KnownLHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyLess(KnownLHS.getKnownFPClasses(),
                                         /*OrderedZeroSign=*/true));
    if (KnownRHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyLess(KnownRHS.getKnownFPClasses(),
                                         /*OrderedZeroSign=*/true));
    break;
  }
  case MinMaxKind::minimumnum: {
    if (KnownLHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                            /*OrderedZeroSign=*/true));
    if (KnownRHS.isKnownNeverNaN())
      Known.knownNot(orderedStrictlyGreater(KnownRHS.getKnownFPClasses(),
                                            /*OrderedZeroSign=*/true));
    break;
  }
  case MinMaxKind::maximum: {
    Known.knownNot(orderedStrictlyLess(KnownLHS.getKnownFPClasses(),
                                       /*OrderedZeroSign=*/true) |
                   orderedStrictlyLess(KnownRHS.getKnownFPClasses(),
                                       /*OrderedZeroSign=*/true));
    break;
  }
  case MinMaxKind::minimum: {
    Known.knownNot(orderedStrictlyGreater(KnownLHS.getKnownFPClasses(),
                                          /*OrderedZeroSign=*/true) |
                   orderedStrictlyGreater(KnownRHS.getKnownFPClasses(),
                                          /*OrderedZeroSign=*/true));
    break;
  }
  }

  // Remove impossible non-NaN classes.
  FPClassTest PossibleClasses =
      KnownLHS.getKnownFPClasses() | KnownRHS.getKnownFPClasses() | fcNan;
  Known.knownNot(~PossibleClasses);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::canonicalize(const KnownFPClass &KnownSrc,
                                        DenormalMode DenormMode) {
  KnownFPClass Known;

  // Canonicalize may flush denormals to zero, so we have to consider the
  // denormal mode to preserve known-not-0 knowledge.
  Known.setKnownFPClasses(KnownSrc.getKnownFPClasses() | fcZero | fcQNan);

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
       DenormMode.Input == DenormalMode::IEEE)) {
    // -0.0 is not a subnormal and should not be flushed.
    if (KnownSrc.isKnownNever(fcNegZero))
      Known.knownNot(fcNegZero);

    if (KnownSrc.isKnownNever(fcPosZero | fcSubnormal))
      Known.knownNot(fcPosZero);
  }

  return Known;
}

KnownFPClass KnownFPClass::bitcast(const fltSemantics &FltSemantics,
                                   const KnownBits &Bits) {
  assert(FltSemantics.sizeInBits == Bits.getBitWidth() &&
         "Bitcast operand has incorrect bit width");
  KnownFPClass Known;

  // Conflicting known bits do not describe a concrete value. Return unknown.
  if (Bits.hasConflict())
    return Known;

  // Return unknown for types we have not validated.
  auto IsSupported = [](const fltSemantics &Semantics) {
    switch (APFloat::SemanticsToEnum(Semantics)) {
    case APFloatBase::S_IEEEhalf:
    case APFloatBase::S_BFloat:
    case APFloatBase::S_IEEEsingle:
    case APFloatBase::S_IEEEdouble:
    case APFloatBase::S_IEEEquad:
    case APFloatBase::S_x87DoubleExtended:
      return true;
    default:
      return false;
    }
  };
  if (!IsSupported(FltSemantics))
    return Known;

  // Transfer information from the sign bit.
  if (Bits.isNonNegative())
    Known.signBitMustBeZero();
  else if (Bits.isNegative())
    Known.signBitMustBeOne();

  switch (APFloatBase::SemanticsToEnum(FltSemantics)) {
  case APFloatBase::S_IEEEhalf:
  case APFloatBase::S_BFloat:
  case APFloatBase::S_IEEEsingle:
  case APFloatBase::S_IEEEdouble:
  case APFloatBase::S_IEEEquad: {
    const unsigned MantissaBits = FltSemantics.precision - 1;
    const APInt ExponentMask = APInt::getBitsSet(
        FltSemantics.sizeInBits, MantissaBits, FltSemantics.sizeInBits - 1);
    const APInt MantissaMask =
        APInt::getLowBitsSet(FltSemantics.sizeInBits, MantissaBits);

    const bool ExponentKnownAllZeros =
        (Bits.Zero & ExponentMask) == ExponentMask;
    const bool ExponentKnownAllOnes = (Bits.One & ExponentMask) == ExponentMask;
    const bool ExponentKnownNotAllZeros = !(Bits.One & ExponentMask).isZero();
    const bool ExponentKnownNotAllOnes = !(Bits.Zero & ExponentMask).isZero();

    const bool MantissaKnownAllZeros =
        (Bits.Zero & MantissaMask) == MantissaMask;
    const bool MantissaKnownNotAllZeros = !(Bits.One & MantissaMask).isZero();

    // Zero and subnormal require an exponent with all zero bits.
    if (ExponentKnownNotAllZeros)
      Known.knownNot(fcZero | fcSubnormal);

    // Infinity and NaN require an exponent with all one bits.
    if (ExponentKnownNotAllOnes)
      Known.knownNot(fcInf | fcNan);

    // Normal values have an exponent that is not all zeros or all ones.
    if (ExponentKnownAllZeros || ExponentKnownAllOnes)
      Known.knownNot(fcNormal);

    // Zero and infinity require a mantissa with all zero bits.
    if (MantissaKnownNotAllZeros)
      Known.knownNot(fcZero | fcInf);

    // Subnormal and NaN require a non-zero mantissa.
    if (MantissaKnownAllZeros)
      Known.knownNot(fcSubnormal | fcNan);

    const bool QuietBitKnownSet = Bits.One[MantissaBits - 1];
    const bool QuietBitKnownClear = Bits.Zero[MantissaBits - 1];

    if (QuietBitKnownSet)
      Known.knownNot(fcSNan);
    else if (QuietBitKnownClear)
      Known.knownNot(fcQNan);

    break;
  }
  case APFloatBase::S_x87DoubleExtended: {
    // Layout of the x87 floating point type:
    // [0:62] fraction
    // [63] explicit integer bit
    // [64:78] exponent
    // [79] sign bit
    // APFloat treats the x87 pseudo encodings as documented in
    // fltSemantics::hasExplicitIntegerBit.
    constexpr unsigned BitWidth = 80;
    constexpr unsigned FractionBits = 63;
    constexpr unsigned IntegerBitIndex = FractionBits;
    const APInt FractionMask = APInt::getLowBitsSet(BitWidth, FractionBits);
    const APInt ExponentMask =
        APInt::getBitsSet(BitWidth, FractionBits + 1, BitWidth - 1);

    const bool ExponentKnownAllZeros =
        (Bits.Zero & ExponentMask) == ExponentMask;
    const bool ExponentKnownAllOnes = (Bits.One & ExponentMask) == ExponentMask;
    const bool ExponentKnownNotAllZeros = !(Bits.One & ExponentMask).isZero();
    const bool ExponentKnownNotAllOnes = !(Bits.Zero & ExponentMask).isZero();
    const bool FractionKnownAllZeros =
        (Bits.Zero & FractionMask) == FractionMask;
    const bool FractionKnownNotAllZeros = !(Bits.One & FractionMask).isZero();
    const bool IntegerBitKnownSet = Bits.One[IntegerBitIndex];
    const bool IntegerBitKnownClear = Bits.Zero[IntegerBitIndex];

    // zero and subnormal require an exponent with all zero bits and a cleared
    // integer bit.
    if (ExponentKnownNotAllZeros || IntegerBitKnownSet)
      Known.knownNot(fcZero | fcSubnormal);

    // Zero requires a zero fraction.
    if (FractionKnownNotAllZeros)
      Known.knownNot(fcZero);

    // Infinity requires an exponent with all one bits, a set integer bit, and
    // a zero fraction.
    if (ExponentKnownNotAllOnes || IntegerBitKnownClear ||
        FractionKnownNotAllZeros)
      Known.knownNot(fcInf);

    // A normal x87 value has its integer bit set. A zero exponent with that
    // bit set is the x87 pseudodenormal encoding, which APFloat treats as
    // normal.
    if (IntegerBitKnownClear || ExponentKnownAllOnes)
      Known.knownNot(fcNormal);

    // A subnormal has a nonzero fraction; NaNs are possible with a zero
    // fraction through pseudoinfinity and unnormal encodings.
    if (FractionKnownAllZeros)
      Known.knownNot(fcSubnormal);

    // We can rule out NaN if the value has all exponent bits set to zero, or
    // if the value is a canonical normal or infinity.
    if (ExponentKnownAllZeros ||
        (IntegerBitKnownSet &&
         (ExponentKnownNotAllOnes || FractionKnownAllZeros)))
      Known.knownNot(fcNan);

    const bool QuietBitKnownSet = Bits.One[FractionBits - 1];
    const bool QuietBitKnownClear = Bits.Zero[FractionBits - 1];

    if (QuietBitKnownSet)
      Known.knownNot(fcSNan);
    else if (QuietBitKnownClear)
      Known.knownNot(fcQNan);

    break;
  }
  default:
    llvm_unreachable("unhandled supported type");
  }

  return Known;
}

KnownBits KnownFPClass::toKnownBits(const fltSemantics &FltSemantics) const {
  KnownBits Known(FltSemantics.sizeInBits);
  const FPClassTest FPClasses = getKnownFPClasses();

  // Return unknown if poison.
  if (FPClasses == fcNone)
    return Known;

  // Return unknown for types we have not validated.
  auto IsSupported = [](const fltSemantics &Semantics) {
    switch (APFloat::SemanticsToEnum(Semantics)) {
    case APFloatBase::S_IEEEhalf:
    case APFloatBase::S_BFloat:
    case APFloatBase::S_IEEEsingle:
    case APFloatBase::S_IEEEdouble:
    case APFloatBase::S_IEEEquad:
    case APFloatBase::S_x87DoubleExtended:
      return true;
    default:
      return false;
    }
  };
  if (!IsSupported(FltSemantics))
    return Known;

  switch (APFloat::SemanticsToEnum(FltSemantics)) {
  case APFloatBase::S_IEEEhalf:
  case APFloatBase::S_BFloat:
  case APFloatBase::S_IEEEsingle:
  case APFloatBase::S_IEEEdouble:
  case APFloatBase::S_IEEEquad: {
    // For ieee types, we cannot deduce anything if the source could be normal.
    if (FPClasses & fcNormal)
      break;

    Known.setAllConflict();

    const unsigned BitWidth = FltSemantics.sizeInBits;
    const unsigned MantissaBits = FltSemantics.precision - 1;
    const unsigned ExponentBits = BitWidth - MantissaBits - 1;

    APInt MantissaMask = APInt::getLowBitsSet(BitWidth, MantissaBits);
    APInt ExponentMask =
        APInt::getBitsSet(BitWidth, MantissaBits, MantissaBits + ExponentBits);

    const unsigned QuietBitIndex = MantissaBits - 1;
    APInt PayloadMask = MantissaMask;
    PayloadMask.clearBit(QuietBitIndex);

    if (FPClasses & fcNan) {
      // Exponent bits cannot be zeros.
      Known.Zero &= ~ExponentMask;
      // No individual payload bit is known.
      Known.Zero &= ~PayloadMask;
      Known.One &= ~PayloadMask;

      if (FPClasses & fcQNan)
        Known.Zero.clearBit(QuietBitIndex);
      if (FPClasses & fcSNan)
        Known.One.clearBit(QuietBitIndex);
    }
    if (FPClasses & fcInf) {
      // Exponent bits cannot be zeros.
      Known.Zero &= ~ExponentMask;
      // Mantissa bits cannot be ones.
      Known.One &= ~MantissaMask;
    }
    if (FPClasses & fcSubnormal) {
      // Exponent bits cannot be ones.
      Known.One &= ~ExponentMask;
      // Unknown mantissa.
      Known.One &= ~MantissaMask;
      Known.Zero &= ~MantissaMask;
    }
    if (FPClasses & fcZero) {
      // Exponent bits cannot be ones.
      Known.One &= ~ExponentMask;
      // Mantissa cannot be ones.
      Known.One &= ~MantissaMask;
    }

    break;
  }
  case APFloatBase::S_x87DoubleExtended: {
    if (!isKnownNever(fcNormal | fcSubnormal | fcNan))
      break;

    Known.setAllConflict();

    if (FPClasses & fcInf)
      Known = Known.intersectWith(KnownBits::makeConstant(
          APFloat::getInf(FltSemantics).bitcastToAPInt()));

    if (FPClasses & fcZero)
      Known = Known.intersectWith(
          KnownBits::makeConstant(APInt::getZero(FltSemantics.sizeInBits)));

    break;
  }
  default:
    llvm_unreachable("unhandled supported semantics");
  }

  Known.Zero.clearSignBit();
  Known.One.clearSignBit();
  if (std::optional<bool> Sign = getSignBit()) {
    if (*Sign)
      Known.makeNegative();
    else
      Known.makeNonNegative();
  }

  return Known;
}

static KnownFPClass scaleByExpRange(const KnownFPClass &KnownSrc, int MinExp,
                                    int MaxExp, bool Negate,
                                    const fltSemantics &Flt) {
  // Return unknown if the range contradicts itself.
  if (MinExp > MaxExp)
    return KnownFPClass();

  // We are assuming that fcNormal has a magnitude strictly larger than
  // fcSubnormal, which does not hold for PPCDoubleDouble.
  if (&Flt == &APFloat::PPCDoubleDouble())
    return KnownFPClass();

  // Note: This assumes round-to-nearest with ties-to-even.
  KnownFPClass Known;

  Known.propagateNonNaN(KnownSrc);

  const int Precision = APFloat::semanticsPrecision(Flt);

  const int MinNormalExp = APFloat::semanticsMinExponent(Flt);
  const int MinSubnormalExp = MinNormalExp - Precision + 1;
  const int UnderflowExp = MinSubnormalExp - 1;
  const int OverflowExp = APFloat::semanticsMaxExponent(Flt) + 1;

  const bool SubnormalMayBeZero = MinExp < 0;
  const bool SubnormalMayBeSubnormal =
      MinExp < MinNormalExp - MinSubnormalExp &&
      MaxExp > UnderflowExp - MinNormalExp;
  const bool SubnormalMayBeNormal =
      MinExp < OverflowExp - MinSubnormalExp && MaxExp > 0;
  const bool SubnormalMayBeInfinity = MaxExp > OverflowExp - MinNormalExp;

  const bool NormalMayBeZero = MinExp < MinSubnormalExp - MinNormalExp;
  const bool NormalMayBeSubnormal =
      MinExp < 0 && MaxExp > UnderflowExp - OverflowExp;
  const bool NormalMayBeNormal = MaxExp > MinNormalExp - OverflowExp - 1 &&
                                 MinExp < OverflowExp - MinNormalExp;
  const bool NormalMayBeInfinity = MaxExp > 0;

  if (KnownSrc.isKnownNever(fcPosZero) &&
      (KnownSrc.isKnownNever(fcPosSubnormal) || !SubnormalMayBeZero) &&
      (KnownSrc.isKnownNever(fcPosNormal) || !NormalMayBeZero))
    Known.knownNot(fcPosZero);
  if (KnownSrc.isKnownNever(fcNegZero) &&
      (KnownSrc.isKnownNever(fcNegSubnormal) || !SubnormalMayBeZero) &&
      (KnownSrc.isKnownNever(fcNegNormal) || !NormalMayBeZero))
    Known.knownNot(fcNegZero);

  if ((KnownSrc.isKnownNever(fcPosSubnormal) || !SubnormalMayBeSubnormal) &&
      (KnownSrc.isKnownNever(fcPosNormal) || !NormalMayBeSubnormal))
    Known.knownNot(fcPosSubnormal);
  if ((KnownSrc.isKnownNever(fcNegSubnormal) || !SubnormalMayBeSubnormal) &&
      (KnownSrc.isKnownNever(fcNegNormal) || !NormalMayBeSubnormal))
    Known.knownNot(fcNegSubnormal);

  if ((KnownSrc.isKnownNever(fcPosSubnormal) || !SubnormalMayBeNormal) &&
      (KnownSrc.isKnownNever(fcPosNormal) || !NormalMayBeNormal))
    Known.knownNot(fcPosNormal);
  if ((KnownSrc.isKnownNever(fcNegSubnormal) || !SubnormalMayBeNormal) &&
      (KnownSrc.isKnownNever(fcNegNormal) || !NormalMayBeNormal))
    Known.knownNot(fcNegNormal);

  if (KnownSrc.isKnownNever(fcPosInf) &&
      (KnownSrc.isKnownNever(fcPosSubnormal) || !SubnormalMayBeInfinity) &&
      (KnownSrc.isKnownNever(fcPosNormal) || !NormalMayBeInfinity))
    Known.knownNot(fcPosInf);
  if (KnownSrc.isKnownNever(fcNegInf) &&
      (KnownSrc.isKnownNever(fcNegSubnormal) || !SubnormalMayBeInfinity) &&
      (KnownSrc.isKnownNever(fcNegNormal) || !NormalMayBeInfinity))
    Known.knownNot(fcNegInf);

  if (Negate)
    Known.fneg();

  return Known;
}

// Handle known sign bit and nan cases for fadd, fsub, and fma.
static KnownFPClass fadd_impl(const KnownFPClass &KnownLHS,
                              const KnownFPClass &KnownRHS, bool IsFMA) {
  KnownFPClass Known;

  // Adding positive and negative infinity produces NaN, but only if both
  // opposite-sign infinity combinations are possible.
  if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
      (KnownLHS.isKnownNever(fcPosInf) || KnownRHS.isKnownNever(fcNegInf)) &&
      (KnownLHS.isKnownNever(fcNegInf) || KnownRHS.isKnownNever(fcPosInf)))
    Known.knownNot(fcNan);

  if (KnownLHS.cannotBeOrderedLessThanZero() &&
      KnownRHS.cannotBeOrderedLessThanZero()) {
    Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

    // This can't underflow if one of the operands is known normal.
    if (KnownLHS.isKnownNever(fcZero | fcPosSubnormal) ||
        KnownRHS.isKnownNever(fcZero | fcPosSubnormal))
      Known.knownNot(fcZero | fcPosSubnormal);
  }

  if (KnownLHS.cannotBeOrderedGreaterThanZero() &&
      KnownRHS.cannotBeOrderedGreaterThanZero()) {
    Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

    // This can't underflow if one of the operands is known normal.
    if (KnownLHS.isKnownNever(fcZero | fcNegSubnormal) ||
        KnownRHS.isKnownNever(fcZero | fcNegSubnormal))
      Known.knownNot(fcZero | fcNegSubnormal);
  }

  if (!IsFMA) {
    // (fadd x, +0.0) is guaranteed to return +0.0, not -0.0.
    if (KnownLHS.isKnownNever(fcNegZero) || KnownRHS.isKnownNever(fcNegZero))
      Known.knownNot(fcNegZero);

    Known.propagateNonSNaN(KnownLHS, KnownRHS);
  }

  return Known;
}

KnownFPClass KnownFPClass::fadd(const KnownFPClass &KnownLHS_,
                                const KnownFPClass &KnownRHS_,
                                DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);

  KnownFPClass Known = fadd_impl(KnownLHS, KnownRHS, /*IsFMA=*/false);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fadd_self(const KnownFPClass &KnownSrc_,
                                     DenormalMode Mode) {
  KnownFPClass KnownSrc = applyInputDenormalMode(KnownSrc_, Mode);

  KnownFPClass Known = fadd_impl(KnownSrc, KnownSrc, /*IsFMA=*/false);

  if (KnownSrc.isKnownNever(fcPosSubnormal))
    Known.knownNot(fcPosSubnormal);
  if (KnownSrc.isKnownNever(fcNegSubnormal))
    Known.knownNot(fcNegSubnormal);

  // Doubling 0 will give the same 0.
  if (KnownSrc.isKnownNever(fcPosZero))
    Known.knownNot(fcPosZero);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fsub(const KnownFPClass &KnownLHS_,
                                const KnownFPClass &KnownRHS_,
                                DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);

  // fsub(x, y) == fadd(x, -y) as long as we negate after subnormals are
  // flushed.
  KnownRHS.fneg();

  KnownFPClass Known = fadd_impl(KnownLHS, KnownRHS, /*IsFMA=*/false);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fmul(const KnownFPClass &KnownLHS_,
                                const KnownFPClass &KnownRHS_,
                                DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);

  KnownFPClass Known;

  Known.propagateNonSNaN(KnownLHS, KnownRHS);

  // +X * +Y or -X * -Y => +Q
  // +X * -Y or -X * +Y => -Q
  if ((KnownLHS.isKnownNever(fcNegative) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcPositive) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcNegative);
  if ((KnownLHS.isKnownNever(fcPositive) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcNegative) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcPositive);

  // Inf * Y => Inf or NaN
  if (KnownLHS.isKnownAlways(fcInf | fcNan) ||
      KnownRHS.isKnownAlways(fcInf | fcNan))
    Known.knownNot(fcNormal | fcSubnormal | fcZero);

  // 0 * Y => 0 or NaN
  if (KnownRHS.isKnownAlways(fcZero | fcNan) ||
      KnownLHS.isKnownAlways(fcZero | fcNan))
    Known.knownNot(fcNormal | fcSubnormal | fcInf);

  // 0 * +/-inf => NaN
  if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
      (KnownRHS.isKnownNeverInfinity() || KnownLHS.isKnownNeverZero()) &&
      (KnownLHS.isKnownNeverInfinity() || KnownRHS.isKnownNeverZero()))
    Known.knownNot(fcNan);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fmul(const KnownFPClass &KnownLHS_,
                                const APFloat &CRHS, DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);

  const fltSemantics &Flt = CRHS.getSemantics();

  KnownFPClass Known;
  // TODO: Handle subnormal CRHS (which may flush to zero).
  if (CRHS.isNormal() && &Flt != &APFloat::PPCDoubleDouble()) {
    const int Exp = ilogb(CRHS);
    const bool IsPow2 = CRHS.getExactLog2Abs() != INT_MIN;

    // |C| in [2^Exp, 2^Exp] exactly when C is power of two
    // |C| in [2^Exp, 2^(Exp+1)) otherwise
    Known = scaleByExpRange(KnownLHS, Exp, IsPow2 ? Exp : Exp + 1,
                            CRHS.isNegative(), Flt);
  } else {
    KnownFPClass KnownRHS = applyInputDenormalMode(KnownFPClass(CRHS), Mode);
    Known = fmul(KnownLHS, KnownRHS, DenormalMode::getIEEE());

    const int MinKnownExponent = ilogb(CRHS);

    // Multiply of values <= 1 cannot introduce overflow.
    if (KnownLHS.isKnownNever(fcInf)) {
      if (MinKnownExponent < 0)
        Known.knownNot(fcInf);
      else if (MinKnownExponent == 0 &&
               CRHS.compareAbsoluteValue(APFloat::getOne(Flt)) ==
                   APFloat::cmpEqual)
        Known.knownNot(fcInf);
    }
  }

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fdiv(const KnownFPClass &KnownLHS_,
                                const KnownFPClass &KnownRHS_,
                                DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);

  KnownFPClass Known;

  Known.propagateNonSNaN(KnownLHS, KnownRHS);

  // Only 0/0, Inf/Inf produce NaN.
  if (KnownLHS.isKnownNeverNaN() && KnownRHS.isKnownNeverNaN() &&
      (KnownLHS.isKnownNeverInfinity() || KnownRHS.isKnownNeverInfinity()) &&
      (KnownLHS.isKnownNeverZero() || KnownRHS.isKnownNeverZero())) {
    Known.knownNot(fcNan);
  }

  //  X / -0.0 => -Inf (or NaN)
  // +X / +Y or -X / -Y => +Q
  // +X / -Y or -X / +Y => -Q
  if ((KnownLHS.isKnownNever(fcNegative) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcPositive) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcNegative);
  if ((KnownLHS.isKnownNever(fcPositive) &&
       KnownRHS.isKnownNever(fcNegative)) ||
      (KnownLHS.isKnownNever(fcNegative) && KnownRHS.isKnownNever(fcPositive)))
    Known.knownNot(fcPositive);

  // Normal and subnormal results require two non-zero finite operands.
  if ((KnownLHS.isKnownNever(fcNegNormal | fcNegSubnormal) &&
       KnownRHS.isKnownNever(fcNegNormal | fcNegSubnormal)) ||
      (KnownLHS.isKnownNever(fcPosNormal | fcPosSubnormal) &&
       KnownRHS.isKnownNever(fcPosNormal | fcPosSubnormal)))
    Known.knownNot(fcNegNormal | fcNegSubnormal);
  if ((KnownLHS.isKnownNever(fcNegNormal | fcNegSubnormal) &&
       KnownRHS.isKnownNever(fcPosNormal | fcPosSubnormal)) ||
      (KnownLHS.isKnownNever(fcPosNormal | fcPosSubnormal) &&
       KnownRHS.isKnownNever(fcNegNormal | fcNegSubnormal)))
    Known.knownNot(fcPosNormal | fcPosSubnormal);

  // 0 / X => 0 or NaN
  if (KnownLHS.isKnownAlways(fcZero))
    Known.knownNot(fcSubnormal | fcNormal | fcInf);

  // X / 0 => NaN or Inf
  if (KnownRHS.isKnownAlways(fcZero))
    Known.knownNot(fcFinite);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fdiv(const KnownFPClass &KnownLHS_,
                                const APFloat &CRHS, DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);

  const fltSemantics &Flt = CRHS.getSemantics();

  KnownFPClass Known;
  // TODO: Handle subnormal CRHS (which may flush to zero).
  if (CRHS.isNormal() && &Flt != &APFloat::PPCDoubleDouble()) {
    const int Exp = ilogb(CRHS);
    const bool IsPow2 = CRHS.getExactLog2Abs() != INT_MIN;

    // |1 / C| in [2^-Exp, 2^-Exp] exactly when C is power of two
    // |1 / C| in (2^(-Exp-1), 2^-Exp] otherwise
    Known = scaleByExpRange(KnownLHS, IsPow2 ? -Exp : -Exp - 1, -Exp,
                            CRHS.isNegative(), Flt);
  } else {
    KnownFPClass KnownRHS = applyInputDenormalMode(KnownFPClass(CRHS), Mode);
    Known = fdiv(KnownLHS, KnownRHS, DenormalMode::getIEEE());
  }

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fdiv_self(const KnownFPClass &KnownSrc,
                                     DenormalMode Mode) {
  // X / X is always exactly 1.0 or a NaN.
  KnownFPClass Known(fcNan | fcPosNormal);

  Known.propagateNonSNaN(KnownSrc);

  if (KnownSrc.isKnownNeverInfOrNaN() && KnownSrc.isKnownNeverLogicalZero(Mode))
    Known.knownNot(fcNan);

  return Known;
}

KnownFPClass KnownFPClass::frem(const KnownFPClass &KnownLHS,
                                const KnownFPClass &KnownRHS,
                                DenormalMode Mode) {
  KnownFPClass Known;

  // We are assuming that frem(x, Inf) is already handled elsewhere.

  Known.knownNot(fcInf);

  Known.propagateNonSNaN(KnownLHS, KnownRHS);

  // frem(Inf, x) and frem(x, 0) produce NaN.
  if (KnownLHS.isKnownNeverInfOrNaN() && KnownRHS.isKnownNeverNaN() &&
      KnownRHS.isKnownNeverLogicalZero(Mode)) {
    Known.knownNot(fcNan);
  }

  // The sign for frem is the same as the first operand.
  if (KnownLHS.isKnownNever(fcPosNormal | fcPosSubnormal))
    Known.knownNot(fcPosNormal | fcPosSubnormal);
  if (KnownLHS.isKnownNever(fcNegNormal | fcNegSubnormal))
    Known.knownNot(fcNegNormal | fcNegSubnormal);

  // A negative zero result requires a negative finite first operand.
  if (KnownLHS.isKnownNever(fcNegFinite))
    Known.knownNot(fcNegZero);

  // A positive zero result can additionally come from a negative finite
  // result being flushed to positive zero.
  if (KnownLHS.isKnownNever(fcPosFinite) &&
      KnownLHS.isKnownNeverLogicalPosZero(Mode) &&
      (!Mode.outputsMayBePositiveZero() ||
       KnownLHS.isKnownNever(fcNegNormal | fcNegSubnormal)))
    Known.knownNot(fcPosZero);

  return Known;
}

KnownFPClass KnownFPClass::frem_self(const KnownFPClass &KnownSrc,
                                     DenormalMode Mode) {
  // X % X is always exactly [+-]0.0 or a NaN.
  KnownFPClass Known(fcNan | fcZero);

  Known.propagateNonSNaN(KnownSrc);

  if (KnownSrc.isKnownNeverInfOrNaN() && KnownSrc.isKnownNeverLogicalZero(Mode))
    Known.knownNot(fcNan);

  // The sign of a zero result is the sign of the finite non-zero input.
  // We do not have to worry about subnormal values flushing to zero, since
  // frem(+-0.0, +-0.0) = NaN.
  if (KnownSrc.isKnownNever(fcPosNormal | fcPosSubnormal))
    Known.knownNot(fcPosZero);
  if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal))
    Known.knownNot(fcNegZero);

  return Known;
}

KnownFPClass KnownFPClass::fma(const KnownFPClass &KnownLHS_,
                               const KnownFPClass &KnownRHS_,
                               const KnownFPClass &KnownAddend_,
                               DenormalMode Mode) {
  KnownFPClass KnownLHS = applyInputDenormalMode(KnownLHS_, Mode);
  KnownFPClass KnownRHS = applyInputDenormalMode(KnownRHS_, Mode);
  KnownFPClass KnownAddend = applyInputDenormalMode(KnownAddend_, Mode);

  KnownFPClass Mul = fmul(KnownLHS, KnownRHS, DenormalMode::getIEEE());

  // FMA differs from the base fmul + fadd handling only in the treatment of -0
  // results.
  //
  // If the multiply is a -0 due to rounding, the final -0 + 0 will be -0,
  // unlike for a separate fadd.
  KnownFPClass Known = fadd_impl(Mul, KnownAddend, /*IsFMA=*/true);

  // propagateNonSNaN for 3 arguments.
  if (KnownLHS.isKnownNever(fcSNan) && KnownRHS.isKnownNever(fcSNan) &&
      KnownAddend.isKnownNever(fcSNan))
    Known.knownNot(fcSNan);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fmuladd(const KnownFPClass &KnownLHS,
                                   const KnownFPClass &KnownRHS,
                                   const KnownFPClass &KnownAddend,
                                   DenormalMode Mode) {
  // fmuladd can lower to either fma(x, y, z) or fadd(fmul(x, y), z). Compute
  // the intersection of both possibilities.
  KnownFPClass FMA = fma(KnownLHS, KnownRHS, KnownAddend, Mode);
  KnownFPClass MulAdd = fadd(fmul(KnownLHS, KnownRHS, Mode), KnownAddend, Mode);
  return FMA.intersectWith(MulAdd);
}

KnownFPClass KnownFPClass::fma_square(const KnownFPClass &KnownSquared_,
                                      const KnownFPClass &KnownAddend_,
                                      DenormalMode Mode) {
  KnownFPClass KnownSquared = applyInputDenormalMode(KnownSquared_, Mode);
  KnownFPClass KnownAddend = applyInputDenormalMode(KnownAddend_, Mode);

  KnownFPClass Squared = square(KnownSquared, DenormalMode::getIEEE());
  KnownFPClass Known = fadd_impl(Squared, KnownAddend, /*IsFMA=*/true);

  // Since we know the squared input must be positive, the add of opposite sign
  // infinities nan hazard only applies for negative inf.
  //
  // TODO: Alternatively to proving addend is not -inf, we could know Squared is
  // not pinf. Other than the degenerate always-subnormal input case, we can't
  // prove that without a known range.
  if (KnownAddend.isKnownNever(fcNegInf | fcNan) && Squared.isKnownNever(fcNan))
    Known.knownNot(fcNan);

  Known.propagateNonSNaN(KnownSquared, KnownAddend);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::fmuladd_square(const KnownFPClass &KnownSquared,
                                          const KnownFPClass &KnownAddend,
                                          DenormalMode Mode) {
  KnownFPClass FMA = fma_square(KnownSquared, KnownAddend, Mode);
  KnownFPClass MulAdd = fadd(square(KnownSquared, Mode), KnownAddend, Mode);
  return FMA.intersectWith(MulAdd);
}

KnownFPClass KnownFPClass::exp(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;
  Known.knownNot(fcNegative);

  Known.propagateNonNaN(KnownSrc);

  // The following deductions assume that exp10(-1.0) = +0.1 and
  // exp10(+1.0) = +10.0 are both finite normal values.

  // Zero can only be produced if x < -1.0.
  if (KnownSrc.isKnownNever(fcNegNormal | fcNegInf))
    Known.knownNot(fcPosZero);

  // Subnormals can only be produced if x < -1.0 and x is finite.
  if (KnownSrc.isKnownNever(fcNegNormal))
    Known.knownNot(fcPosSubnormal);

  // Infinity can only be produced if x > +1.0.
  if (KnownSrc.isKnownNever(fcPosNormal | fcPosInf))
    Known.knownNot(fcPosInf);

  return Known;
}

KnownFPClass KnownFPClass::log(const KnownFPClass &KnownSrc, DenormalMode Mode,
                               bool IsMultiUnitFPType) {
  KnownFPClass Known;

  // PPCDoubleDouble can represent the exact value of 1.0 + subnormal, and
  // log(1.0 + subnormal) = subnormal.
  if (!IsMultiUnitFPType)
    Known.knownNot(fcNegZero | fcSubnormal);

  Known.propagateNonSNaN(KnownSrc);

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

  Known.propagateNonSNaN(KnownSrc);

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

KnownFPClass KnownFPClass::sin(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  Known.propagateNonSNaN(KnownSrc);

  // Return NaN on infinite inputs.
  Known.knownNot(fcInf);
  if (KnownSrc.isKnownNeverNaN() && KnownSrc.isKnownNeverInfinity())
    Known.knownNot(fcNan);

  return Known;
}

KnownFPClass KnownFPClass::cos(const KnownFPClass &KnownSrc) {
  return sin(KnownSrc);
}

KnownFPClass KnownFPClass::tan(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // tan never returns Inf (tan(+-Inf) = NaN; tan(finite) = finite).
  Known.knownNot(fcInf);

  Known.propagateNonSNaN(KnownSrc);

  // NaN propagates. tan(+-Inf) is NaN.
  if (KnownSrc.isKnownNeverNaN() && KnownSrc.isKnownNeverInfinity())
    Known.knownNot(fcNan);

  return Known;
}

KnownFPClass KnownFPClass::sinh(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // sinh is sign-preserving: sinh(x) < 0 iff x < 0.
  if (KnownSrc.isKnownNever(fcNegative))
    Known.knownNot(fcNegative);

  // A nonzero positive result requires a nonzero positive input.
  // Zero remains possible: sinh(+/-0.0) = +/-0.0, and flushing a subnormal
  // may produce +0.0.
  if (KnownSrc.isKnownNever(fcPosSubnormal | fcPosNormal | fcPosInf))
    Known.knownNot(fcPosSubnormal | fcPosNormal | fcPosInf);

  Known.propagateNonNaN(KnownSrc);

  return Known;
}

KnownFPClass KnownFPClass::cosh(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // cosh(x) >= 1 for all real x; cosh(+-Inf) = +Inf. Never negative,
  // zero, or subnormal.
  Known.knownNot(fcNegative | fcZero | fcSubnormal);

  Known.propagateNonNaN(KnownSrc);

  return Known;
}

KnownFPClass KnownFPClass::tanh(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // tanh is bounded to (-1, 1), never Inf.
  Known.knownNot(fcInf);

  // tanh is sign-preserving: tanh(x) < 0 iff x < 0.
  if (KnownSrc.isKnownNever(fcNegSubnormal | fcNegNormal | fcNegInf)) {
    Known.knownNot(fcNegSubnormal | fcNegNormal);

    // -0.0 is preserved: tanh(-0.0) = -0.0.
    if (KnownSrc.isKnownNeverNegZero())
      Known.knownNot(fcNegZero);
  }

  // Likewise, tanh(x) > 0 iff x > 0. Zero is not ruled out: tanh(+-0.0) =
  // +-0.0, and a flushed subnormal may become +0.0.
  if (KnownSrc.isKnownNever(fcPosSubnormal | fcPosNormal | fcPosInf))
    Known.knownNot(fcPosSubnormal | fcPosNormal);

  Known.propagateNonNaN(KnownSrc);

  return Known;
}

KnownFPClass KnownFPClass::asin(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // asin is bounded to [-pi/2, pi/2], never Inf.
  Known.knownNot(fcInf);

  Known.propagateNonSNaN(KnownSrc);

  // asin is sign-preserving for finite arguments.
  if (KnownSrc.isKnownNever(fcNegFinite))
    Known.knownNot(fcNegFinite);

  if (KnownSrc.isKnownNever(fcPosSubnormal | fcPosNormal))
    Known.knownNot(fcPosSubnormal | fcPosNormal);

  // NaN propagates. asin(x) is also NaN for |x| > 1, so we cannot rule
  // out NaN without knowing the source is in [-1, 1].
  return Known;
}

KnownFPClass KnownFPClass::acos(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // acos(x) is bounded to [0, pi] for -1 <= x <= 1, and is never negative,
  // infinite, or subnormal. The smallest non-zero value occurs when x is
  // close to 1.0, where acos(x) can be approximated by sqrt(2 * (1 - x)).
  // Since sqrt cannot produce a subnormal result, we can conclude that
  // acos(x) will also never produce a subnormal result.
  Known.knownNot(fcNegative | fcInf | fcSubnormal);

  // acos(x) == +0.0 iff x == +1.0
  if (KnownSrc.isKnownNever(fcPosNormal))
    Known.knownNot(fcZero);

  Known.propagateNonSNaN(KnownSrc);

  // NaN propagates. acos(x) is also NaN for |x| > 1, so we cannot rule
  // out NaN without knowing the source is in [-1, 1].
  return Known;
}

KnownFPClass KnownFPClass::atan(const KnownFPClass &KnownSrc) {
  KnownFPClass Known;

  // atan is bounded to (-pi/2, pi/2), never Inf. atan(+-Inf) = +-pi/2 (finite).
  Known.knownNot(fcInf);

  // Ruling out Subnormal/Normal from the result requires the
  // input to never be finite-nonzero and never inf, since
  // atan(+-inf) = +-pi/2
  if (KnownSrc.isKnownNever(fcNegSubnormal | fcNegNormal | fcNegInf))
    Known.knownNot(fcNegSubnormal | fcNegNormal);

  if (KnownSrc.isKnownNever(fcPosSubnormal | fcPosNormal | fcPosInf))
    Known.knownNot(fcPosSubnormal | fcPosNormal);

  // atan(-0.0) = -0.0. Negative finite inputs may also produce -0.0
  // if the result rounds to a negative subnormal that is flushed to zero.
  if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal | fcNegZero))
    Known.knownNot(fcNegZero);

  Known.propagateNonNaN(KnownSrc);

  return Known;
}

KnownFPClass KnownFPClass::atan2(const KnownFPClass &KnownY_,
                                 const KnownFPClass &KnownX_,
                                 DenormalMode Mode) {
  KnownFPClass KnownY = applyInputDenormalMode(KnownY_, Mode);
  KnownFPClass KnownX = applyInputDenormalMode(KnownX_, Mode);
  KnownFPClass Known;

  // Even though these deductions are correct, we are ignoring the following
  // potentially erroneous cases:
  //   * atan2(y, inf) is not subnormal
  //   * atan2(inf, x) is not zero or subnormal

  // atan2 result is in (-pi, pi], never Inf.
  Known.knownNot(fcInf);

  Known.propagateNonNaN(KnownY, KnownX);

  // If x <= -0.0, then |atan2(y, x)| >= pi/2
  if (KnownX.isKnownNever(fcPositive))
    Known.knownNot(fcZero | fcSubnormal);

  // If y >= +0.0, then atan2(y, x) >= +0.0
  if (KnownY.isKnownNever(fcNegative))
    Known.knownNot(fcNegative);

  // If y <= -0.0, then atan2(y, x) <= -0.0
  if (KnownY.isKnownNever(fcPositive))
    Known.knownNot(fcPositive);

  return applyOutputDenormalMode(Known, Mode);
}

static KnownFPClass fpconvert(const KnownFPClass &KnownSrc_,
                              const fltSemantics &DstSem,
                              const fltSemantics &SrcSem, DenormalMode Mode) {
  KnownFPClass Known;
  APFloatBase::Semantics DstType = APFloatBase::SemanticsToEnum(DstSem);
  APFloatBase::Semantics SrcType = APFloatBase::SemanticsToEnum(SrcSem);
  auto IsSupported = [](APFloatBase::Semantics SemType) -> bool {
    switch (SemType) {
    case APFloatBase::S_IEEEhalf:
    case APFloatBase::S_BFloat:
    case APFloatBase::S_IEEEsingle:
    case APFloatBase::S_IEEEdouble:
    case APFloatBase::S_IEEEquad:
    case APFloatBase::S_x87DoubleExtended:
      return true;
    default:
      return false;
    }
  };

  // Return unknown for types we have not validated.
  // TODO: Add support for lossless casts to PPCDoubleDouble.
  if (!IsSupported(SrcType) || !IsSupported(DstType))
    return Known;

  KnownFPClass KnownSrc = KnownFPClass::applyInputDenormalMode(KnownSrc_, Mode);

  // TODO: Until we pass the rounding mode into fpconvert, the rounding mode is
  // assumed to be dynamic. This includes potential future rounding modes such
  // as roundAwayZero (not the same as roundNearestTiesAway) and roundToOdd
  // (jamming).

  Known.propagateNonNaN(KnownSrc);

  const bool NormalSrcIsFiniteDst = DstSem.maxExponent > SrcSem.maxExponent ||
                                    (DstSem.maxExponent == SrcSem.maxExponent &&
                                     DstSem.precision >= SrcSem.precision);

  // True if the dst can represent all non-zero finite values of src.
  const bool NonZeroFiniteSrcIsNonZeroInDst =
      DstSem.minExponent <= SrcSem.minExponent &&
      DstSem.precision >= SrcSem.precision;

  // For example: f32 = bf16, f128 = f80
  const bool NormalIsNormalAndSubnormalIsSubnormal =
      DstSem.minExponent == SrcSem.minExponent &&
      DstSem.maxExponent >= SrcSem.maxExponent &&
      DstSem.precision >= SrcSem.precision;

  // Rule out infinity.
  // We are assuming that subnormal values have a magnitude less than 1.0.
  if (KnownSrc.isKnownNever(fcPosInf) &&
      (KnownSrc.isKnownNever(fcPosNormal) || NormalSrcIsFiniteDst))
    Known.knownNot(fcPosInf);
  if (KnownSrc.isKnownNever(fcNegInf) &&
      (KnownSrc.isKnownNever(fcNegNormal) || NormalSrcIsFiniteDst))
    Known.knownNot(fcNegInf);

  // Rule out non-zero finite.
  if (NormalIsNormalAndSubnormalIsSubnormal) {
    // Normal never becomes subnormal, and vice-versa.
    if (KnownSrc.isKnownNever(fcPosNormal))
      Known.knownNot(fcPosNormal);
    if (KnownSrc.isKnownNever(fcNegNormal))
      Known.knownNot(fcNegNormal);
    if (KnownSrc.isKnownNever(fcPosSubnormal))
      Known.knownNot(fcPosSubnormal);
    if (KnownSrc.isKnownNever(fcNegSubnormal))
      Known.knownNot(fcNegSubnormal);
  } else {
    if (KnownSrc.isKnownNever(fcPosNormal | fcPosSubnormal))
      Known.knownNot(fcPosNormal | fcPosSubnormal);
    if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal))
      Known.knownNot(fcNegNormal | fcNegSubnormal);
  }

  // Rule out subnormal.
  if (APFloat::isRepresentableAsNormalIn(SrcSem, DstSem))
    Known.knownNot(fcSubnormal);

  // Rule out positive zero.
  if (KnownSrc.isKnownNever(fcPosZero) &&
      (KnownSrc.isKnownNever(fcPosNormal | fcPosSubnormal) ||
       NonZeroFiniteSrcIsNonZeroInDst))
    Known.knownNot(fcPosZero);

  // Rule out negative zero.
  if (KnownSrc.isKnownNever(fcNegZero) &&
      (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal) ||
       NonZeroFiniteSrcIsNonZeroInDst))
    Known.knownNot(fcNegZero);

  return KnownFPClass::applyOutputDenormalMode(Known, Mode);
}

// Determine what the input and output denormal modes are for fpconvert from
// the denormal modes for the dst and src types.
//
// TODO: We are currently conservative, and assume DenormalMode::getDynamic() if
// the dst or src denormal mode is not IEEE.
static DenormalMode getConvertDenormalMode(DenormalMode DstMode,
                                           DenormalMode SrcMode) {
  if (DstMode == DenormalMode::getIEEE() && SrcMode == DenormalMode::getIEEE())
    return DenormalMode::getIEEE();
  return DenormalMode::getDynamic();
}

KnownFPClass KnownFPClass::fpext(const KnownFPClass &KnownSrc,
                                 const fltSemantics &DstTy,
                                 const fltSemantics &SrcTy,
                                 DenormalMode DstMode, DenormalMode SrcMode) {
  // Note that fpext may round for some bizarre type combinations.
  // Such as ieee_binary256 = fpext(PPCDoubleDouble) if we add support for
  // ieee_binary256 in the future. PPCDoubleDouble can represent values such as
  // DBL_MAX + DBL_TRUE_MIN which would normally require 2098 bits of precision
  // to represent.
  return fpconvert(KnownSrc, DstTy, SrcTy,
                   getConvertDenormalMode(DstMode, SrcMode));
}

KnownFPClass KnownFPClass::fptrunc(const KnownFPClass &KnownSrc,
                                   const fltSemantics &DstTy,
                                   const fltSemantics &SrcTy,
                                   DenormalMode DstMode, DenormalMode SrcMode) {
  // TODO: pass rounding mode from fptrunc_round
  return fpconvert(KnownSrc, DstTy, SrcTy,
                   getConvertDenormalMode(DstMode, SrcMode));
}

KnownFPClass KnownFPClass::roundToIntegral(const KnownFPClass &KnownSrc,
                                           bool MayBeMultiUnitFPType,
                                           DenormalMode Mode) {
  KnownFPClass Known;

  // Integer results cannot be subnormal.
  Known.knownNot(fcSubnormal);

  Known.propagateNonNaN(KnownSrc);

  // Pass through infinities, except PPC_FP128 is a special case for intrinsics.
  if (!MayBeMultiUnitFPType) {
    if (KnownSrc.isKnownNeverPosInfinity())
      Known.knownNot(fcPosInf);
    if (KnownSrc.isKnownNeverNegInfinity())
      Known.knownNot(fcNegInf);
  }

  if (KnownSrc.isKnownNever(fcPosNormal | fcPosSubnormal))
    Known.knownNot(fcPosNormal);

  if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal))
    Known.knownNot(fcNegNormal);

  // Negative round ups towards zero produce negative zero.
  if (KnownSrc.isKnownNever(fcNegFinite))
    Known.knownNot(fcNegZero);

  // Negative subnormals may flush to positive zero.
  if (KnownSrc.isKnownNever(fcPosFinite) &&
      (KnownSrc.isKnownNever(fcNegSubnormal) ||
       !Mode.inputsMayBePositiveZero()))
    Known.knownNot(fcPosZero);

  return Known;
}

KnownFPClass KnownFPClass::trunc(const KnownFPClass &KnownSrc,
                                 DenormalMode Mode) {
  KnownFPClass Known = roundToIntegral(KnownSrc,
                                       /*MayBeMultiUnitFPType=*/true, Mode);

  if (KnownSrc.isKnownNeverPosInfinity())
    Known.knownNot(fcPosInf);
  if (KnownSrc.isKnownNeverNegInfinity())
    Known.knownNot(fcNegInf);

  return Known;
}

KnownFPClass KnownFPClass::floor(const KnownFPClass &KnownSrc,
                                 bool IsMultiUnitFPType, DenormalMode Mode) {
  KnownFPClass Known = roundToIntegral(KnownSrc, IsMultiUnitFPType, Mode);

  // Applicable for multi-unit floating point types.
  if (KnownSrc.isKnownNeverPosInfinity())
    Known.knownNot(fcPosInf);

  return Known;
}

KnownFPClass KnownFPClass::ceil(const KnownFPClass &KnownSrc,
                                bool IsMultiUnitFPType, DenormalMode Mode) {
  KnownFPClass Known = roundToIntegral(KnownSrc, IsMultiUnitFPType, Mode);

  // Applicable for multi-unit floating point types.
  if (KnownSrc.isKnownNeverNegInfinity())
    Known.knownNot(fcNegInf);

  return Known;
}

KnownFPClass KnownFPClass::frexp_mant(const KnownFPClass &KnownSrc,
                                      DenormalMode Mode) {
  KnownFPClass Known;

  Known.knownNot(fcSubnormal);

  Known.propagateNonNaN(KnownSrc);

  if (KnownSrc.isKnownNeverPosInfinity())
    Known.knownNot(fcPosInf);

  if (KnownSrc.isKnownNeverNegInfinity())
    Known.knownNot(fcNegInf);

  if (KnownSrc.isKnownNeverLogicalPosZero(Mode))
    Known.knownNot(fcPosZero);

  if (KnownSrc.isKnownNeverLogicalNegZero(Mode))
    Known.knownNot(fcNegZero);

  // TODO: These deductions can be improved if subnormal inputs are guaranteed
  // to be flushed to zero.
  if (KnownSrc.isKnownNever(fcPosNormal | fcPosSubnormal))
    Known.knownNot(fcPosNormal);

  if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal))
    Known.knownNot(fcNegNormal);

  return Known;
}

// The pre-existing ldexp code path, kept frozen for PPCDoubleDouble because
// scaleByExpRange does not work for PPCDoubleDouble.
static KnownFPClass ldexpPPCDoubleDouble(const KnownFPClass &KnownSrc_,
                                         const APInt &ConstantRangeExpMin,
                                         const APInt &ConstantRangeExpMax,
                                         DenormalMode Mode) {
  // Return unknown if the range contradicts itself.
  if (ConstantRangeExpMin.sgt(ConstantRangeExpMax))
    return KnownFPClass();

  KnownFPClass Known;

  KnownFPClass KnownSrc = KnownFPClass::applyInputDenormalMode(KnownSrc_, Mode);

  if (ConstantRangeExpMin.isZero() && ConstantRangeExpMax.isZero()) {
    // ldexp(x, 0) -> x
    // Here we make sure to properly propagate sNaN and denormals.
    Known.setKnownFPClasses(KnownSrc.getKnownFPClasses() | fcNan);
    Known.propagateNonNaN(KnownSrc);
    return KnownFPClass::applyOutputDenormalMode(Known, Mode);
  }

  Known.propagateNonNaN(KnownSrc);

  // Sign is preserved, but underflows may produce zeroes.
  if (KnownSrc.isKnownNever(fcNegative))
    Known.knownNot(fcNegative);
  else if (KnownSrc.cannotBeOrderedLessThanZero())
    Known.knownNot(KnownFPClass::OrderedLessThanZeroMask);

  if (KnownSrc.isKnownNever(fcPositive))
    Known.knownNot(fcPositive);
  else if (KnownSrc.cannotBeOrderedGreaterThanZero())
    Known.knownNot(KnownFPClass::OrderedGreaterThanZeroMask);

  if (ConstantRangeExpMax.isNonPositive()) {
    // If we know the power is <= 0, can't introduce inf
    if (KnownSrc.isKnownNeverPosInfinity())
      Known.knownNot(fcPosInf);
    if (KnownSrc.isKnownNeverNegInfinity())
      Known.knownNot(fcNegInf);
  } else if (ConstantRangeExpMin.isNonNegative()) {
    // If we know the power is >= 0, can't introduce subnormal or zero
    if (KnownSrc.isKnownNeverPosSubnormal())
      Known.knownNot(fcPosSubnormal);
    if (KnownSrc.isKnownNeverNegSubnormal())
      Known.knownNot(fcNegSubnormal);
    if (KnownSrc.isKnownNeverPosZero())
      Known.knownNot(fcPosZero);
    if (KnownSrc.isKnownNeverNegZero())
      Known.knownNot(fcNegZero);
  }

  return KnownFPClass::applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::ldexp(const KnownFPClass &KnownSrc_,
                                 const APInt &ConstantRangeExpMin,
                                 const APInt &ConstantRangeExpMax,
                                 const fltSemantics &Flt, DenormalMode Mode) {
  // Use frozen code path for PPCDoubleDouble.
  if (&Flt == &APFloat::PPCDoubleDouble())
    return ldexpPPCDoubleDouble(KnownSrc_, ConstantRangeExpMin,
                                ConstantRangeExpMax, Mode);

  KnownFPClass KnownSrc = applyInputDenormalMode(KnownSrc_, Mode);

  KnownFPClass Known;

  // The exponent operand can be any integer width. Truncating it could map a
  // huge exponent onto a small one and invert the tests below.
  auto SaturatedIntExp = [](const APInt &Exp) {
    if (Exp.isSignedIntN(sizeof(int) * CHAR_BIT))
      return static_cast<int>(Exp.getSExtValue());
    return Exp.isNegative() ? INT_MIN : INT_MAX;
  };

  Known = scaleByExpRange(KnownSrc, SaturatedIntExp(ConstantRangeExpMin),
                          SaturatedIntExp(ConstantRangeExpMax),
                          /*Negate=*/false, Flt);

  return applyOutputDenormalMode(Known, Mode);
}

KnownFPClass KnownFPClass::ldexp(const KnownFPClass &KnownSrc,
                                 const KnownBits &ExpBits,
                                 const fltSemantics &Flt, DenormalMode Mode) {
  return ldexp(KnownSrc, ExpBits.getSignedMinValue(),
               ExpBits.getSignedMaxValue(), Flt, Mode);
}

KnownFPClass KnownFPClass::pow(const KnownFPClass &KnownLHS,
                               const KnownFPClass &KnownRHS) {
  KnownFPClass Known;

  Known.propagateNonSNaN(KnownLHS, KnownRHS);

  // pow may return NaN if one of the arguments is NaN. NaN may be produced from
  // a non-zero-finite-negative base and a non-integer exponent.
  if (KnownLHS.isKnownNever(fcNan | fcNegNormal | fcNegSubnormal) &&
      KnownRHS.isKnownNeverNaN())
    Known.knownNot(fcNan);

  // We could rule out negative and subnormal results when exponent is known to
  // never be a normal value, but having either argument being known to never be
  // normal is unlikely and not worth considering.

  // Only a negative base raised to an odd power returns a negative value.
  if (KnownLHS.isKnownNever(fcNegative)) {
    Known.knownNot(fcNegative);
  } else if (KnownLHS.isKnownNever(fcNegNormal | fcNegSubnormal)) {
    Known.knownNot(fcNegNormal | fcNegSubnormal);
    // See if we can also rule out -0.0 or -inf.
    // Here at least one of -0.0 or -inf is a possible base.

    // pow(-0.0, odd-positive) = -0.0
    // pow(-inf, odd-negative) = -0.0
    if ((KnownLHS.isKnownNever(fcNegZero) ||
         KnownRHS.isKnownNever(fcPosNormal)) &&
        (KnownLHS.isKnownNever(fcNegInf) || KnownRHS.isKnownNever(fcNegNormal)))
      Known.knownNot(fcNegZero);

    // pow(-0.0, odd-negative) = -inf
    // pow(-inf, odd-positive) = -inf
    if ((KnownLHS.isKnownNever(fcNegZero) ||
         KnownRHS.isKnownNever(fcNegNormal)) &&
        (KnownLHS.isKnownNever(fcNegInf) || KnownRHS.isKnownNever(fcPosNormal)))
      Known.knownNot(fcNegInf);
  }

  return Known;
}

KnownFPClass KnownFPClass::powi(const KnownFPClass &KnownSrc,
                                const KnownBits &ExponentKnownBits) {
  KnownFPClass Known;
  Known.propagateNonNaN(KnownSrc);

  if (ExponentKnownBits.isZero()) {
    // powi(QNaN, 0) returns 1.0, and powi(SNaN, 0) may non-deterministically
    // return 1.0 or a NaN.
    if (KnownSrc.isKnownNever(fcSNan)) {
      Known.knownNot(~fcPosNormal);
      return Known;
    }

    Known.knownNot(~(fcPosNormal | fcNan));
    return Known;
  }

  // Given that exp is an integer, here are the
  // ways that powi can return a negative value:
  //
  //   powi(x, exp)    --> negative if exp is odd and x is negative.
  //   powi(-0, exp)   --> -inf if exp is negative odd.
  //   powi(-0, exp)   --> -0 if exp is positive odd.
  //   powi(-inf, exp) --> -0 if exp is negative odd.
  //   powi(-inf, exp) --> -inf if exp is positive odd.
  if (KnownSrc.isKnownNever(fcNegative) || ExponentKnownBits.isEven()) {
    Known.knownNot(fcNegative);
  } else if (KnownSrc.isKnownNever(fcNegNormal | fcNegSubnormal)) {
    Known.knownNot(fcNegNormal | fcNegSubnormal);
    // See if we can also rule out -0.0 or -inf.
    // Here at least one of -0.0 or -inf is a possible base.

    // We already know that ExponentKnownBits.isEven() is false here.
    const bool IsKnownNeverOddPositive = ExponentKnownBits.isNegative();
    const bool IsKnownNeverOddNegative = ExponentKnownBits.isNonNegative();

    // powi(-0.0, odd-positive) = -0.0
    // powi(-inf, odd-negative) = -0.0
    if ((KnownSrc.isKnownNever(fcNegZero) || IsKnownNeverOddPositive) &&
        (KnownSrc.isKnownNever(fcNegInf) || IsKnownNeverOddNegative))
      Known.knownNot(fcNegZero);

    // powi(-0.0, odd-negative) = -inf
    // powi(-inf, odd-positive) = -inf
    if ((KnownSrc.isKnownNever(fcNegZero) || IsKnownNeverOddNegative) &&
        (KnownSrc.isKnownNever(fcNegInf) || IsKnownNeverOddPositive))
      Known.knownNot(fcNegInf);
  }

  // powi(x, exp) --> inf
  // when:
  //   * powi(inf, exp), exp > 0
  //   * powi(+/-0, exp), exp < 0
  //   * powi(finite, exp), |exp| > 1
  //   * powi(subnormal, -1)
  // TODO:
  //   1. This simple all or nothing approach. We can do better
  //      and cover sign/parity and exp > 1 vs exp < -1 separately.
  //   2. powi(0/nan, exp), exp > 0 can be refinable
  //      to fcNan | fcZero | fcPosNormal.
  {
    APInt MinExp = ExponentKnownBits.getSignedMinValue();
    APInt MaxExp = ExponentKnownBits.getSignedMaxValue();

    // powi(inf, exp), exp > 0
    bool MayInfSrc =
        !KnownSrc.isKnownNever(fcInf) && MaxExp.isStrictlyPositive();

    // powi(+/-0, exp), exp < 0
    bool MayDivByZero = !KnownSrc.isKnownNever(fcZero) && MinExp.isNegative();

    // powi(finite, exp), |exp| > 1
    bool MayFinite = !KnownSrc.isKnownNever(fcNormal | fcSubnormal);
    bool MayAbsExpGT1 = MinExp.slt(-1) || MaxExp.sgt(1);
    bool MayFiniteOverflow = MayFinite && MayAbsExpGT1;

    // powi(subnormal, -1)
    bool MayBeNegOne = ExponentKnownBits.Zero.isZero();
    bool MaySubnormInv = !KnownSrc.isKnownNever(fcSubnormal) && MayBeNegOne;

    if (!MayInfSrc && !MayDivByZero && !MayFiniteOverflow && !MaySubnormInv)
      Known.knownNot(fcInf);
  }

  return Known;
}
