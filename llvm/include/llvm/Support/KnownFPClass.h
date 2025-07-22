//===- llvm/Support/KnownFPClass.h - Stores known fpclass -------*- C++ -*-===//
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

#ifndef LLVM_SUPPORT_KNOWNFPCLASS_H
#define LLVM_SUPPORT_KNOWNFPCLASS_H

#include "llvm/ADT/FloatingPointMode.h"
#include <optional>

namespace llvm {

struct KnownFPClass {
  /// Floating-point classes the value could be one of.
  FPClassTest KnownFPClasses = fcAllFlags;

  /// std::nullopt if the sign bit is unknown, true if the sign bit is
  /// definitely set or false if the sign bit is definitely unset.
  std::optional<bool> SignBit;

  bool operator==(KnownFPClass Other) const {
    return KnownFPClasses == Other.KnownFPClasses && SignBit == Other.SignBit;
  }

  /// Return true if it's known this can never be one of the mask entries.
  bool isKnownNever(FPClassTest Mask) const {
    return (KnownFPClasses & Mask) == fcNone;
  }

  bool isKnownAlways(FPClassTest Mask) const { return isKnownNever(~Mask); }

  bool isUnknown() const { return KnownFPClasses == fcAllFlags && !SignBit; }

  /// Return true if it's known this can never be a nan.
  bool isKnownNeverNaN() const { return isKnownNever(fcNan); }

  /// Return true if it's known this must always be a nan.
  bool isKnownAlwaysNaN() const { return isKnownAlways(fcNan); }

  /// Return true if it's known this can never be an infinity.
  bool isKnownNeverInfinity() const { return isKnownNever(fcInf); }

  /// Return true if it's known this can never be +infinity.
  bool isKnownNeverPosInfinity() const { return isKnownNever(fcPosInf); }

  /// Return true if it's known this can never be -infinity.
  bool isKnownNeverNegInfinity() const { return isKnownNever(fcNegInf); }

  /// Return true if it's known this can never be a subnormal
  bool isKnownNeverSubnormal() const { return isKnownNever(fcSubnormal); }

  /// Return true if it's known this can never be a positive subnormal
  bool isKnownNeverPosSubnormal() const { return isKnownNever(fcPosSubnormal); }

  /// Return true if it's known this can never be a negative subnormal
  bool isKnownNeverNegSubnormal() const { return isKnownNever(fcNegSubnormal); }

  /// Return true if it's known this can never be a zero. This means a literal
  /// [+-]0, and does not include denormal inputs implicitly treated as [+-]0.
  bool isKnownNeverZero() const { return isKnownNever(fcZero); }

  /// Return true if it's known this can never be a literal positive zero.
  bool isKnownNeverPosZero() const { return isKnownNever(fcPosZero); }

  /// Return true if it's known this can never be a negative zero. This means a
  /// literal -0 and does not include denormal inputs implicitly treated as -0.
  bool isKnownNeverNegZero() const { return isKnownNever(fcNegZero); }

  /// Return true if it's know this can never be interpreted as a zero. This
  /// extends isKnownNeverZero to cover the case where the assumed
  /// floating-point mode for the function interprets denormals as zero.
  bool isKnownNeverLogicalZero(DenormalMode Mode) const;

  /// Return true if it's know this can never be interpreted as a negative zero.
  bool isKnownNeverLogicalNegZero(DenormalMode Mode) const;

  /// Return true if it's know this can never be interpreted as a positive zero.
  bool isKnownNeverLogicalPosZero(DenormalMode Mode) const;

  static constexpr FPClassTest OrderedLessThanZeroMask =
      fcNegSubnormal | fcNegNormal | fcNegInf;
  static constexpr FPClassTest OrderedGreaterThanZeroMask =
      fcPosSubnormal | fcPosNormal | fcPosInf;

  /// Return true if we can prove that the analyzed floating-point value is
  /// either NaN or never less than -0.0.
  ///
  ///      NaN --> true
  ///       +0 --> true
  ///       -0 --> true
  ///   x > +0 --> true
  ///   x < -0 --> false
  bool cannotBeOrderedLessThanZero() const {
    return isKnownNever(OrderedLessThanZeroMask);
  }

  /// Return true if we can prove that the analyzed floating-point value is
  /// either NaN or never greater than -0.0.
  ///      NaN --> true
  ///       +0 --> true
  ///       -0 --> true
  ///   x > +0 --> false
  ///   x < -0 --> true
  bool cannotBeOrderedGreaterThanZero() const {
    return isKnownNever(OrderedGreaterThanZeroMask);
  }

  KnownFPClass &operator|=(const KnownFPClass &RHS) {
    KnownFPClasses = KnownFPClasses | RHS.KnownFPClasses;

    if (SignBit != RHS.SignBit)
      SignBit = std::nullopt;
    return *this;
  }

  void knownNot(FPClassTest RuleOut) {
    KnownFPClasses = KnownFPClasses & ~RuleOut;
    if (isKnownNever(fcNan) && !SignBit) {
      if (isKnownNever(fcNegative))
        SignBit = false;
      else if (isKnownNever(fcPositive))
        SignBit = true;
    }
  }

  void fneg() {
    KnownFPClasses = llvm::fneg(KnownFPClasses);
    if (SignBit)
      SignBit = !*SignBit;
  }

  void fabs() {
    if (KnownFPClasses & fcNegZero)
      KnownFPClasses |= fcPosZero;

    if (KnownFPClasses & fcNegInf)
      KnownFPClasses |= fcPosInf;

    if (KnownFPClasses & fcNegSubnormal)
      KnownFPClasses |= fcPosSubnormal;

    if (KnownFPClasses & fcNegNormal)
      KnownFPClasses |= fcPosNormal;

    signBitMustBeZero();
  }

  /// Return true if the sign bit must be 0, ignoring the sign of nans.
  bool signBitIsZeroOrNaN() const { return isKnownNever(fcNegative); }

  /// Assume the sign bit is zero.
  void signBitMustBeZero() {
    KnownFPClasses &= (fcPositive | fcNan);
    SignBit = false;
  }

  /// Assume the sign bit is one.
  void signBitMustBeOne() {
    KnownFPClasses &= (fcNegative | fcNan);
    SignBit = true;
  }

  void copysign(const KnownFPClass &Sign) {
    // Don't know anything about the sign of the source. Expand the possible set
    // to its opposite sign pair.
    if (KnownFPClasses & fcZero)
      KnownFPClasses |= fcZero;
    if (KnownFPClasses & fcSubnormal)
      KnownFPClasses |= fcSubnormal;
    if (KnownFPClasses & fcNormal)
      KnownFPClasses |= fcNormal;
    if (KnownFPClasses & fcInf)
      KnownFPClasses |= fcInf;

    // Sign bit is exactly preserved even for nans.
    SignBit = Sign.SignBit;

    // Clear sign bits based on the input sign mask.
    if (Sign.isKnownNever(fcPositive | fcNan) || (SignBit && *SignBit))
      KnownFPClasses &= (fcNegative | fcNan);
    if (Sign.isKnownNever(fcNegative | fcNan) || (SignBit && !*SignBit))
      KnownFPClasses &= (fcPositive | fcNan);
  }

  // Propagate knowledge that a non-NaN source implies the result can also not
  // be a NaN. For unconstrained operations, signaling nans are not guaranteed
  // to be quieted but cannot be introduced.
  void propagateNaN(const KnownFPClass &Src, bool PreserveSign = false) {
    if (Src.isKnownNever(fcNan)) {
      knownNot(fcNan);
      if (PreserveSign)
        SignBit = Src.SignBit;
    } else if (Src.isKnownNever(fcSNan))
      knownNot(fcSNan);
  }

  /// Propagate knowledge from a source value that could be a denormal or
  /// zero. We have to be conservative since output flushing is not guaranteed,
  /// so known-never-zero may not hold.
  ///
  /// This assumes a copy-like operation and will replace any currently known
  /// information.
  void propagateDenormal(const KnownFPClass &Src, DenormalMode Mode);

  /// Report known classes if \p Src is evaluated through a potentially
  /// canonicalizing operation. We can assume signaling nans will not be
  /// introduced, but cannot assume a denormal will be flushed under FTZ/DAZ.
  ///
  /// This assumes a copy-like operation and will replace any currently known
  /// information.
  void propagateCanonicalizingSrc(const KnownFPClass &Src, DenormalMode Mode);

  void resetAll() { *this = KnownFPClass(); }
};

inline KnownFPClass operator|(KnownFPClass LHS, const KnownFPClass &RHS) {
  LHS |= RHS;
  return LHS;
}

inline KnownFPClass operator|(const KnownFPClass &LHS, KnownFPClass &&RHS) {
  RHS |= LHS;
  return std::move(RHS);
}

} // namespace llvm

#endif
