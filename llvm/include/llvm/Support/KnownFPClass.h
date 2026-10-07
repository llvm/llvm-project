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

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
class APFloat;
class APInt;
struct fltSemantics;
struct KnownBits;

enum class FPClassMask : unsigned {
  kfcNone = 0,

  kfcNegQNan = 0x0001,
  kfcNegSNan = 0x0002,
  kfcNegInf = 0x0004,
  kfcNegNormal = 0x0008,
  kfcNegSubnormal = 0x0010,
  kfcNegZero = 0x0020,
  kfcPosZero = 0x0040,
  kfcPosSubnormal = 0x0080,
  kfcPosNormal = 0x0100,
  kfcPosInf = 0x0200,
  kfcPosSNan = 0x0400,
  kfcPosQNan = 0x0800,

  kfcSNan = kfcPosSNan | kfcNegSNan,
  kfcQNan = kfcPosQNan | kfcNegQNan,
  kfcPosNan = kfcPosSNan | kfcPosQNan,
  kfcNegNan = kfcNegSNan | kfcNegQNan,
  kfcNan = kfcSNan | kfcQNan,
  kfcInf = kfcPosInf | kfcNegInf,
  kfcNormal = kfcPosNormal | kfcNegNormal,
  kfcSubnormal = kfcPosSubnormal | kfcNegSubnormal,
  kfcZero = kfcPosZero | kfcNegZero,
  kfcPosFinite = kfcPosNormal | kfcPosSubnormal | kfcPosZero,
  kfcNegFinite = kfcNegNormal | kfcNegSubnormal | kfcNegZero,
  kfcFinite = kfcPosFinite | kfcNegFinite,
  kfcPositive = kfcPosFinite | kfcPosInf,
  kfcNegative = kfcNegFinite | kfcNegInf,
  kfcPosSignBit = kfcPositive | kfcPosNan,
  kfcNegSignBit = kfcNegative | kfcNegNan,

  kfcAllFlags = kfcNan | kfcInf | kfcFinite,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ kfcPosQNan)
};

class KnownFPClass {
  FPClassMask KnownFPMask = FPClassMask::kfcAllFlags;

  static constexpr FPClassMask toFPClassMask(FPClassTest Classes) {
    FPClassMask Mask = FPClassMask::kfcNone;

    if (Classes & fcQNan)
      Mask |= FPClassMask::kfcQNan;
    if (Classes & fcSNan)
      Mask |= FPClassMask::kfcSNan;
    if (Classes & fcNegInf)
      Mask |= FPClassMask::kfcNegInf;
    if (Classes & fcNegNormal)
      Mask |= FPClassMask::kfcNegNormal;
    if (Classes & fcNegSubnormal)
      Mask |= FPClassMask::kfcNegSubnormal;
    if (Classes & fcNegZero)
      Mask |= FPClassMask::kfcNegZero;
    if (Classes & fcPosZero)
      Mask |= FPClassMask::kfcPosZero;
    if (Classes & fcPosSubnormal)
      Mask |= FPClassMask::kfcPosSubnormal;
    if (Classes & fcPosNormal)
      Mask |= FPClassMask::kfcPosNormal;
    if (Classes & fcPosInf)
      Mask |= FPClassMask::kfcPosInf;

    return Mask;
  }

  static constexpr FPClassMask toFPClassMask(FPClassTest Classes,
                                             bool SignBit) {
    FPClassMask Mask = toFPClassMask(Classes);

    // This is the only way to generate a NaN with a specific sign from
    // FPClassTest. SignBit must agree with the input classes.
    if (!SignBit) {
      Mask &= ~FPClassMask::kfcNegNan;
      // If the SignBit is false, then we should not have any negative classes.
      if ((Mask & FPClassMask::kfcNegSignBit) == FPClassMask::kfcNone)
        return Mask;
    } else {
      Mask &= ~FPClassMask::kfcPosNan;
      // If the SignBit is true, then we should not have any positive classes.
      if ((Mask & FPClassMask::kfcPosSignBit) == FPClassMask::kfcNone)
        return Mask;
    }

    // SignBit is unknown or inconsistent with the input classes. Expand the
    // possible set to its opposite sign pair.
    if (Classes & fcQNan)
      Mask |= FPClassMask::kfcQNan;
    if (Classes & fcSNan)
      Mask |= FPClassMask::kfcSNan;
    if (Classes & fcInf)
      Mask |= FPClassMask::kfcInf;
    if (Classes & fcNormal)
      Mask |= FPClassMask::kfcNormal;
    if (Classes & fcSubnormal)
      Mask |= FPClassMask::kfcSubnormal;
    if (Classes & fcZero)
      Mask |= FPClassMask::kfcZero;

    return Mask;
  }

  static constexpr FPClassTest toFPClassTest(FPClassMask Mask) {
    FPClassTest Classes = fcNone;

    // Sign of qNaN and sNaN are lost in the conversion.
    if ((Mask & FPClassMask::kfcQNan) != FPClassMask::kfcNone)
      Classes |= fcQNan;
    if ((Mask & FPClassMask::kfcSNan) != FPClassMask::kfcNone)
      Classes |= fcSNan;

    if ((Mask & FPClassMask::kfcNegInf) != FPClassMask::kfcNone)
      Classes |= fcNegInf;
    if ((Mask & FPClassMask::kfcNegNormal) != FPClassMask::kfcNone)
      Classes |= fcNegNormal;
    if ((Mask & FPClassMask::kfcNegSubnormal) != FPClassMask::kfcNone)
      Classes |= fcNegSubnormal;
    if ((Mask & FPClassMask::kfcNegZero) != FPClassMask::kfcNone)
      Classes |= fcNegZero;
    if ((Mask & FPClassMask::kfcPosZero) != FPClassMask::kfcNone)
      Classes |= fcPosZero;
    if ((Mask & FPClassMask::kfcPosSubnormal) != FPClassMask::kfcNone)
      Classes |= fcPosSubnormal;
    if ((Mask & FPClassMask::kfcPosNormal) != FPClassMask::kfcNone)
      Classes |= fcPosNormal;
    if ((Mask & FPClassMask::kfcPosInf) != FPClassMask::kfcNone)
      Classes |= fcPosInf;

    return Classes;
  }

public:
  /// Floating-point classes the value could be one of.
  FPClassTest getKnownFPClasses() const { return toFPClassTest(KnownFPMask); }

  void setKnownFPClasses(FPClassTest Classes) {
    KnownFPMask = toFPClassMask(Classes);
  }

  /// std::nullopt if the sign bit is unknown, true if the sign bit is
  /// definitely set or false if the sign bit is definitely unset.
  /// By convention, returns false for kfcNone/poison.
  std::optional<bool> getSignBit() const {
    if (KnownFPMask == FPClassMask::kfcNone)
      return false;

    if ((KnownFPMask & FPClassMask::kfcPosSignBit) == KnownFPMask)
      return false;
    if ((KnownFPMask & FPClassMask::kfcNegSignBit) == KnownFPMask)
      return true;

    return std::nullopt;
  }

  void setSignBit(std::optional<bool> Sign) {
    if (Sign && !*Sign) {
      KnownFPMask &= FPClassMask::kfcPosSignBit;
      return;
    }
    if (Sign && *Sign) {
      KnownFPMask &= FPClassMask::kfcNegSignBit;
      return;
    }
    // Set sign to unknown.
    if ((KnownFPMask & FPClassMask::kfcQNan) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcQNan;
    if ((KnownFPMask & FPClassMask::kfcSNan) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcSNan;
    if ((KnownFPMask & FPClassMask::kfcInf) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcInf;
    if ((KnownFPMask & FPClassMask::kfcNormal) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcNormal;
    if ((KnownFPMask & FPClassMask::kfcSubnormal) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcSubnormal;
    if ((KnownFPMask & FPClassMask::kfcZero) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcZero;
  }

  KnownFPClass(FPClassMask Known = FPClassMask::kfcAllFlags)
      : KnownFPMask(Known) {}
  KnownFPClass(FPClassTest Known) : KnownFPMask(toFPClassMask(Known)) {}
  KnownFPClass(FPClassTest Known, bool Sign)
      : KnownFPMask(toFPClassMask(Known, Sign)) {}
  LLVM_ABI KnownFPClass(const APFloat &C);

  bool operator==(KnownFPClass Other) const {
    return KnownFPMask == Other.KnownFPMask;
  }

  /// Return true if it's known this can never be one of the mask entries.
  bool isKnownNever(FPClassMask Mask) const {
    return (KnownFPMask & Mask) == FPClassMask::kfcNone;
  }

  /// Return true if it's known this can never be one of the mask entries.
  bool isKnownNever(FPClassTest Mask) const {
    return isKnownNever(toFPClassMask(Mask));
  }

  bool isKnownAlways(FPClassMask Mask) const { return isKnownNever(~Mask); }

  bool isKnownAlways(FPClassTest Mask) const {
    return isKnownAlways(toFPClassMask(Mask));
  }

  bool isUnknown() const { return KnownFPMask == FPClassMask::kfcAllFlags; }

  /// Return true if it's known this can never be a nan.
  bool isKnownNeverNaN() const { return isKnownNever(fcNan); }

  /// Return true if it's known this must always be a nan.
  bool isKnownAlwaysNaN() const { return isKnownAlways(fcNan); }

  /// Return true if it's known this can never be an infinity.
  bool isKnownNeverInfinity() const { return isKnownNever(fcInf); }

  /// Return true if it's known this can never be an infinity or nan
  bool isKnownNeverInfOrNaN() const { return isKnownNever(fcInf | fcNan); }

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

  /// Return true if it's known this can never be interpreted as a zero. This
  /// extends isKnownNeverZero to cover the case where the assumed
  /// floating-point mode for the function interprets denormals as zero.
  LLVM_ABI bool isKnownNeverLogicalZero(DenormalMode Mode) const;

  /// Return true if it's known this can never be interpreted as a negative
  /// zero.
  LLVM_ABI bool isKnownNeverLogicalNegZero(DenormalMode Mode) const;

  /// Return true if it's known this can never be interpreted as a positive
  /// zero.
  LLVM_ABI bool isKnownNeverLogicalPosZero(DenormalMode Mode) const;

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

  /// Return true if it's known this can never be a positive value or a logical
  /// 0.
  ///
  ///      NaN --> true
  ///  x <= +0 --> false
  ///     psub --> true if mode is ieee, false otherwise.
  ///   x > +0 --> true
  bool cannotBeOrderedLessEqZero(DenormalMode Mode) const {
    return isKnownNever(fcNegative) && isKnownNeverLogicalPosZero(Mode);
  }

  /// Return true if it's know this can never be a negative value or a logical
  /// 0.
  ///
  ///      NaN --> true
  ///  x >= -0 --> false
  ///     nsub --> true if mode is ieee, false otherwise.
  ///   x < -0 --> true
  bool cannotBeOrderedGreaterEqZero(DenormalMode Mode) const {
    return isKnownNever(fcPositive) && isKnownNeverLogicalNegZero(Mode);
  }

  /// Return the known classes of \p Src after passing through an input that
  /// may flush subnormals to zero according to \p Mode.
  LLVM_ABI static KnownFPClass applyInputDenormalMode(const KnownFPClass &Src,
                                                      DenormalMode Mode);

  /// Return the known classes of \p Src after producing an output that may
  /// flush subnormals to zero according to \p Mode.
  LLVM_ABI static KnownFPClass applyOutputDenormalMode(const KnownFPClass &Src,
                                                       DenormalMode Mode);

  KnownFPClass intersectWith(const KnownFPClass &RHS) const {
    return KnownFPClass(KnownFPMask | RHS.KnownFPMask);
  }

  KnownFPClass unionWith(const KnownFPClass &RHS) const {
    return KnownFPClass(KnownFPMask & RHS.KnownFPMask);
  }

  KnownFPClass &operator|=(const KnownFPClass &RHS) {
    KnownFPMask |= RHS.KnownFPMask;
    return *this;
  }

  void knownNot(FPClassMask RuleOut) { KnownFPMask &= ~RuleOut; }

  void knownNot(FPClassTest RuleOut) { knownNot(toFPClassMask(RuleOut)); }

  void fneg() {
    FPClassMask Known = FPClassMask::kfcNone;

    if ((KnownFPMask & FPClassMask::kfcNegQNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosQNan;
    if ((KnownFPMask & FPClassMask::kfcNegSNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosSNan;
    if ((KnownFPMask & FPClassMask::kfcNegInf) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosInf;
    if ((KnownFPMask & FPClassMask::kfcNegNormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosNormal;
    if ((KnownFPMask & FPClassMask::kfcNegSubnormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosSubnormal;
    if ((KnownFPMask & FPClassMask::kfcNegZero) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosZero;
    if ((KnownFPMask & FPClassMask::kfcPosZero) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegZero;
    if ((KnownFPMask & FPClassMask::kfcPosSubnormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegSubnormal;
    if ((KnownFPMask & FPClassMask::kfcPosNormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegNormal;
    if ((KnownFPMask & FPClassMask::kfcPosInf) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegInf;
    if ((KnownFPMask & FPClassMask::kfcPosSNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegSNan;
    if ((KnownFPMask & FPClassMask::kfcPosQNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcNegQNan;

    KnownFPMask = Known;
  }

  static KnownFPClass fneg(const KnownFPClass &Src) {
    KnownFPClass Known = Src;
    Known.fneg();
    return Known;
  }

  void fabs() {
    FPClassMask Known = FPClassMask::kfcNone;

    if ((KnownFPMask & FPClassMask::kfcQNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosQNan;
    if ((KnownFPMask & FPClassMask::kfcSNan) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosSNan;
    if ((KnownFPMask & FPClassMask::kfcInf) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosInf;
    if ((KnownFPMask & FPClassMask::kfcNormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosNormal;
    if ((KnownFPMask & FPClassMask::kfcSubnormal) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosSubnormal;
    if ((KnownFPMask & FPClassMask::kfcZero) != FPClassMask::kfcNone)
      Known |= FPClassMask::kfcPosZero;

    KnownFPMask = Known;
  }

  static KnownFPClass fabs(const KnownFPClass &Src) {
    KnownFPClass Known = Src;
    Known.fabs();
    return Known;
  }

  // Enum of min/max intrinsics to avoid dependency on IR.
  enum class MinMaxKind {
    minimum,
    maximum,
    minimumnum,
    maximumnum,
    minnum,
    maxnum
  };

  LLVM_ABI static KnownFPClass
  minMaxLike(const KnownFPClass &LHS, const KnownFPClass &RHS, MinMaxKind Kind,
             DenormalMode DenormMode = DenormalMode::getDynamic());

  /// Apply the canonicalize intrinsic to this value. This is essentially a
  /// stronger form of propagateCanonicalizingSrc.
  LLVM_ABI static KnownFPClass
  canonicalize(const KnownFPClass &Src,
               DenormalMode DenormMode = DenormalMode::getDynamic());

  /// Report known values for a bitcast into a float with provided semantics.
  LLVM_ABI static KnownFPClass bitcast(const fltSemantics &FltSemantics,
                                       const KnownBits &Bits);

  /// Report known bits for a float with provided semantics.
  LLVM_ABI KnownBits toKnownBits(const fltSemantics &FltSemantics) const;

  /// Report known values for fadd
  LLVM_ABI static KnownFPClass
  fadd(const KnownFPClass &LHS, const KnownFPClass &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fadd x, x
  LLVM_ABI static KnownFPClass
  fadd_self(const KnownFPClass &Src,
            DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fsub
  LLVM_ABI static KnownFPClass
  fsub(const KnownFPClass &LHS, const KnownFPClass &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fmul
  LLVM_ABI static KnownFPClass
  fmul(const KnownFPClass &LHS, const KnownFPClass &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  // Special case of fmul x, x.
  static KnownFPClass square(const KnownFPClass &Src,
                             DenormalMode Mode = DenormalMode::getDynamic()) {
    KnownFPClass Known = fmul(Src, Src, Mode);

    // X * X is always non-negative or a NaN.
    Known.knownNot(fcNegative);
    Known.propagateNonNaN(Src);
    return Known;
  }

  LLVM_ABI static KnownFPClass
  fmul(const KnownFPClass &LHS, const APFloat &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fdiv
  LLVM_ABI static KnownFPClass
  fdiv(const KnownFPClass &LHS, const KnownFPClass &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fdiv x, x
  LLVM_ABI static KnownFPClass
  fdiv_self(const KnownFPClass &Src,
            DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for frem
  LLVM_ABI static KnownFPClass
  frem(const KnownFPClass &LHS, const KnownFPClass &RHS,
       DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for frem x, x
  LLVM_ABI static KnownFPClass
  frem_self(const KnownFPClass &Src,
            DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fma
  LLVM_ABI static KnownFPClass
  fma(const KnownFPClass &LHS, const KnownFPClass &RHS,
      const KnownFPClass &Addend,
      DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for fma squared, squared, addend
  LLVM_ABI static KnownFPClass
  fma_square(const KnownFPClass &Squared, const KnownFPClass &Addend,
             DenormalMode Mode = DenormalMode::getDynamic());

  /// Propagate known class for sqrt
  LLVM_ABI static KnownFPClass
  sqrt(const KnownFPClass &Src, DenormalMode Mode = DenormalMode::getDynamic());

  /// Propagate known class for log/log2/log10
  LLVM_ABI static KnownFPClass
  log(const KnownFPClass &Src, DenormalMode Mode = DenormalMode::getDynamic());

  /// Report known values for exp, exp2 and exp10.
  /// This function assumes that exp10(-1.0) = +0.1 and exp10(+1.0) = +10.0 are
  /// both finite normal values, which holds for formats with a corresponding
  /// LLVM IR type (e.g. does not hold for Float4E2M1FN).
  LLVM_ABI static KnownFPClass exp(const KnownFPClass &Src);

  /// Report known values for sin
  LLVM_ABI static KnownFPClass sin(const KnownFPClass &Src);

  /// Report known values for cos
  LLVM_ABI static KnownFPClass cos(const KnownFPClass &Src);

  /// Report known values for tan
  LLVM_ABI static KnownFPClass tan(const KnownFPClass &Src);

  /// Report known values for sinh
  LLVM_ABI static KnownFPClass sinh(const KnownFPClass &Src);

  /// Report known values for cosh
  LLVM_ABI static KnownFPClass cosh(const KnownFPClass &Src);

  /// Report known values for tanh
  LLVM_ABI static KnownFPClass tanh(const KnownFPClass &Src);

  /// Report known values for asin
  LLVM_ABI static KnownFPClass asin(const KnownFPClass &Src);

  /// Report known values for acos
  LLVM_ABI static KnownFPClass acos(const KnownFPClass &Src);

  /// Report known values for atan
  LLVM_ABI static KnownFPClass atan(const KnownFPClass &Src);

  /// Report known values for atan2
  LLVM_ABI static KnownFPClass
  atan2(const KnownFPClass &LHS, const KnownFPClass &RHS,
        DenormalMode Mode = DenormalMode::getDynamic());

  /// Return true if the sign bit must be 0, ignoring the sign of nans.
  bool signBitIsZeroOrNaN() const { return isKnownNever(fcNegative); }

  /// Assume the sign bit is zero.
  void signBitMustBeZero() { KnownFPMask &= FPClassMask::kfcPosSignBit; }

  /// Assume the sign bit is one.
  void signBitMustBeOne() { KnownFPMask &= FPClassMask::kfcNegSignBit; }

  void copysign(const KnownFPClass &Sign) {
    // Don't know anything about the sign of the source. Expand the possible set
    // to its opposite sign pair.

    if ((KnownFPMask & FPClassMask::kfcQNan) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcQNan;
    if ((KnownFPMask & FPClassMask::kfcSNan) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcSNan;
    if ((KnownFPMask & FPClassMask::kfcInf) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcInf;
    if ((KnownFPMask & FPClassMask::kfcNormal) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcNormal;
    if ((KnownFPMask & FPClassMask::kfcSubnormal) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcSubnormal;
    if ((KnownFPMask & FPClassMask::kfcZero) != FPClassMask::kfcNone)
      KnownFPMask |= FPClassMask::kfcZero;

    if (Sign.getSignBit() && !*Sign.getSignBit())
      KnownFPMask &= FPClassMask::kfcPosSignBit;
    if (Sign.getSignBit() && *Sign.getSignBit())
      KnownFPMask &= FPClassMask::kfcNegSignBit;
  }

  static KnownFPClass copysign(const KnownFPClass &KnownMag,
                               const KnownFPClass &KnownSign) {
    KnownFPClass Known = KnownMag;
    Known.copysign(KnownSign);
    return Known;
  }

  // Propagate knowledge that an operation cannot introduce a signaling NaN.
  void propagateNonSNaN(const KnownFPClass &Src) {
    if (Src.isKnownNever(fcSNan))
      knownNot(fcSNan);
  }

  // Propagate knowledge that an operation cannot introduce a signaling NaN.
  void propagateNonSNaN(const KnownFPClass &LHS, const KnownFPClass &RHS) {
    if (LHS.isKnownNever(fcSNan) && RHS.isKnownNever(fcSNan))
      knownNot(fcSNan);
  }

  // Propagate knowledge that a non-NaN source implies the result can also not
  // be a NaN. For unconstrained operations, signaling nans are not guaranteed
  // to be quieted but cannot be introduced.
  void propagateNonNaN(const KnownFPClass &Src) {
    propagateNonSNaN(Src);
    if (Src.isKnownNever(fcNan))
      knownNot(fcNan);
  }

  void propagateNonNaN(const KnownFPClass &LHS, const KnownFPClass &RHS) {
    propagateNonSNaN(LHS, RHS);
    if (LHS.isKnownNeverNaN() && RHS.isKnownNeverNaN())
      knownNot(fcNan);
  }

  // Propagate knowledge for operations whose result sign is the xor of the
  // operand signs, such as multiply and divide. This only rules out possible
  // non-NaN sign classes. NaNs do not have a constrained sign class here.
  void propagateXorSign(const KnownFPClass &LHS, const KnownFPClass &RHS,
                        DenormalMode Mode) {
    const bool LHSCannotHavePositiveInput =
        LHS.isKnownNever(fcPositive) && LHS.isKnownNeverLogicalPosZero(Mode);
    const bool RHSCannotHavePositiveInput =
        RHS.isKnownNever(fcPositive) && RHS.isKnownNeverLogicalPosZero(Mode);
    if ((LHS.isKnownNever(fcNegative) && RHS.isKnownNever(fcNegative)) ||
        (LHSCannotHavePositiveInput && RHSCannotHavePositiveInput))
      knownNot(fcNegative);

    if ((LHSCannotHavePositiveInput && RHS.isKnownNever(fcNegative)) ||
        (LHS.isKnownNever(fcNegative) && RHSCannotHavePositiveInput)) {
      knownNot(fcPosInf | fcPosNormal | fcPosSubnormal);
      if (!Mode.outputsMayBePositiveZero())
        knownNot(fcPosZero);
    }
  }

  /// Propagate knowledge from a source value that could be a denormal or
  /// zero. We have to be conservative since output flushing is not guaranteed,
  /// so known-never-zero may not hold.
  ///
  /// This assumes a copy-like operation and will replace any currently known
  /// information.
  LLVM_ABI void propagateDenormal(const KnownFPClass &Src, DenormalMode Mode);

  /// Report known classes if \p Src is evaluated through a potentially
  /// canonicalizing operation. We can assume signaling nans will not be
  /// introduced, but cannot assume a denormal will be flushed under FTZ/DAZ.
  ///
  /// This assumes a copy-like operation and will replace any currently known
  /// information.
  LLVM_ABI void propagateCanonicalizingSrc(const KnownFPClass &Src,
                                           DenormalMode Mode);

  /// Propagate known class for fpext.
  LLVM_ABI static KnownFPClass fpext(const KnownFPClass &KnownSrc,
                                     const fltSemantics &DstTy,
                                     const fltSemantics &SrcTy);

  /// Propagate known class for fptrunc.
  LLVM_ABI static KnownFPClass fptrunc(const KnownFPClass &KnownSrc);

  /// Propagate known class for rounding intrinsics (trunc, floor, ceil, rint,
  /// nearbyint, round, roundeven). This is trunc if \p IsTrunc. \p
  /// IsMultiUnitFPType if this is for a multi-unit floating-point type.
  LLVM_ABI static KnownFPClass
  roundToIntegral(const KnownFPClass &Src, bool IsTrunc, bool IsMultiUnitFPType,
                  DenormalMode Mode = DenormalMode::getDynamic());

  /// Propagate known class for mantissa component of frexp
  LLVM_ABI static KnownFPClass
  frexp_mant(const KnownFPClass &Src,
             DenormalMode Mode = DenormalMode::getDynamic());

  /// Propagate known class for ldexp, assuming the exponent is known to be
  /// within [\p ConstantRangeMin, \p ConstantRangeMax]
  ///
  // TODO: This really ought to use ConstantRange, but it's in IR not Support.
  LLVM_ABI static KnownFPClass
  ldexp(const KnownFPClass &Src, const APInt &ConstantRangeMin,
        const APInt &ConstantRangeMax, const fltSemantics &Flt,
        DenormalMode Mode = DenormalMode::getDynamic());
  LLVM_ABI static KnownFPClass
  ldexp(const KnownFPClass &Src, const KnownBits &ExpBits,
        const fltSemantics &Flt,
        DenormalMode Mode = DenormalMode::getDynamic());

  /// Propagate known class for pow
  LLVM_ABI static KnownFPClass pow(const KnownFPClass &LHS,
                                   const KnownFPClass &RHS);

  /// Propagate known class for powi
  LLVM_ABI static KnownFPClass powi(const KnownFPClass &Src,
                                    const KnownBits &N);

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
