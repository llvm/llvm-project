//===- KnownFPClassTest.cpp - KnownFPClass tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownFPClass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void expectConstant(const char *SemanticsName, const char *ValueName,
                           const fltSemantics &Semantics, APInt ValueBits,
                           FPClassTest PositiveClass, bool Negative) {
  if (Negative)
    ValueBits.setBit(Semantics.sizeInBits - 1);

  SCOPED_TRACE(testing::Message()
               << SemanticsName << ' ' << (Negative ? "negative " : "positive ")
               << ValueName);
  KnownFPClass Known =
      KnownFPClass::bitcast(Semantics, KnownBits::makeConstant(ValueBits));
  FPClassTest ExpectedClass =
      Negative ? llvm::fneg(PositiveClass) : PositiveClass;
  EXPECT_EQ(ExpectedClass, Known.getKnownFPClasses());
  EXPECT_EQ(Negative, Known.getSignBit());
}

TEST(KnownFPClassTest, BitcastExhaustiveIEEEHalf) {
  const fltSemantics &Semantics = APFloat::IEEEhalf();

  for (uint64_t RawBits = 0; RawBits != (1u << 16); ++RawBits) {
    APInt ValueBits(16, RawBits);
    KnownFPClass Known =
        KnownFPClass::bitcast(Semantics, KnownBits::makeConstant(ValueBits));
    KnownFPClass Expected(APFloat(Semantics, ValueBits));

    ASSERT_EQ(Expected.getKnownFPClasses(), Known.getKnownFPClasses())
        << RawBits;
    ASSERT_EQ(Expected.getSignBit(), Known.getSignBit()) << RawBits;
  }
}

TEST(KnownFPClassTest, BitcastConflict) {
  const fltSemantics &Semantics = APFloat::IEEEsingle();
  KnownBits Bits(Semantics.sizeInBits);
  Bits.setAllConflict();

  ASSERT_TRUE(Bits.hasConflict());
  KnownFPClass Known = KnownFPClass::bitcast(Semantics, Bits);
  EXPECT_EQ(fcAllFlags, Known.getKnownFPClasses());
  EXPECT_EQ(std::nullopt, Known.getSignBit());
}

TEST(KnownFPClassTest, BitcastPartialConflict) {
  const fltSemantics &Semantics = APFloat::IEEEsingle();
  KnownBits Bits(Semantics.sizeInBits);
  Bits.Zero.setAllBits();
  Bits.One.setBit(0);

  ASSERT_TRUE(Bits.hasConflict());
  KnownFPClass Known = KnownFPClass::bitcast(Semantics, Bits);
  EXPECT_EQ(fcAllFlags, Known.getKnownFPClasses());
  EXPECT_EQ(std::nullopt, Known.getSignBit());
}

TEST(KnownFPClassTest, BitcastConstant) {
  struct SemanticsCase {
    const char *Name;
    const fltSemantics *Semantics;
  };

  for (const SemanticsCase &TestCase :
       {SemanticsCase{"ieee_binary16", &APFloat::IEEEhalf()},
        SemanticsCase{"bfloat16", &APFloat::BFloat()},
        SemanticsCase{"ieee_binary32", &APFloat::IEEEsingle()},
        SemanticsCase{"ieee_binary64", &APFloat::IEEEdouble()},
        SemanticsCase{"ieee_binary128", &APFloat::IEEEquad()},
        SemanticsCase{"x87float80", &APFloat::x87DoubleExtended()}}) {
    const fltSemantics &Semantics = *TestCase.Semantics;
    const unsigned BitWidth = Semantics.sizeInBits;
    const APInt AllOnesPayload = APInt::getAllOnes(BitWidth);
    APInt SNaNLSBSetPayload =
        APFloat::getInf(Semantics).bitcastToAPInt() | APInt(BitWidth, 1);
    APFloat MaxSubnormal = APFloat::getSmallestNormalized(Semantics);
    ASSERT_EQ(APFloat::opOK, MaxSubnormal.next(/*nextDown=*/true));

    for (bool Negative : {false, true}) {
      expectConstant(TestCase.Name, "0.0", Semantics,
                     APFloat::getZero(Semantics).bitcastToAPInt(), fcPosZero,
                     Negative);
      expectConstant(TestCase.Name, "min_subnormal", Semantics,
                     APFloat::getSmallest(Semantics).bitcastToAPInt(),
                     fcPosSubnormal, Negative);
      expectConstant(TestCase.Name, "max_subnormal", Semantics,
                     MaxSubnormal.bitcastToAPInt(), fcPosSubnormal, Negative);
      expectConstant(TestCase.Name, "min_normal", Semantics,
                     APFloat::getSmallestNormalized(Semantics).bitcastToAPInt(),
                     fcPosNormal, Negative);
      expectConstant(TestCase.Name, "1.0", Semantics,
                     APFloat::getOne(Semantics).bitcastToAPInt(), fcPosNormal,
                     Negative);
      expectConstant(TestCase.Name, "max_normal", Semantics,
                     APFloat::getLargest(Semantics).bitcastToAPInt(),
                     fcPosNormal, Negative);
      expectConstant(TestCase.Name, "inf", Semantics,
                     APFloat::getInf(Semantics).bitcastToAPInt(), fcPosInf,
                     Negative);

      // An sNaN has a clear quiet bit and a non-zero payload.
      expectConstant(TestCase.Name, "snan_lsb_set_payload", Semantics,
                     SNaNLSBSetPayload, fcSNan, Negative);

      expectConstant(
          TestCase.Name, "snan_all_ones_payload", Semantics,
          APFloat::getSNaN(Semantics, false, &AllOnesPayload).bitcastToAPInt(),
          fcSNan, Negative);

      // A qNaN has a set quiet bit. The remaining payload bits may be zero.
      expectConstant(TestCase.Name, "qnan_no_payload", Semantics,
                     APFloat::getQNaN(Semantics).bitcastToAPInt(), fcQNan,
                     Negative);

      expectConstant(
          TestCase.Name, "qnan_all_ones_payload", Semantics,
          APFloat::getQNaN(Semantics, false, &AllOnesPayload).bitcastToAPInt(),
          fcQNan, Negative);
    }
  }
}

TEST(KnownFPClassTest, BitcastUnsupported) {
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

  for (unsigned I = 0; I != APFloat::S_MaxSemantics + 1; ++I) {
    APFloat::Semantics SemanticsKind = static_cast<APFloat::Semantics>(I);
    const fltSemantics &Semantics = APFloat::EnumToSemantics(SemanticsKind);

    for (const APInt &ValueBits : {APInt::getZero(Semantics.sizeInBits),
                                   APInt::getAllOnes(Semantics.sizeInBits)}) {
      SCOPED_TRACE(testing::Message()
                   << "Semantics = " << I << ", bits = " << ValueBits);
      KnownFPClass Known =
          KnownFPClass::bitcast(Semantics, KnownBits::makeConstant(ValueBits));
      if (IsSupported(Semantics)) {
        // We should be able to make at least one deduction for "Supported"
        // types.
        EXPECT_FALSE(Known.isUnknown());
      } else {
        EXPECT_TRUE(Known.isUnknown());
      }
    }
  }
}

TEST(KnownFPClassTest, ToKnownBitsUnsupported) {
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

  for (unsigned I = 0; I != APFloat::S_MaxSemantics + 1; ++I) {
    APFloat::Semantics SemanticsKind = static_cast<APFloat::Semantics>(I);
    const fltSemantics &Semantics = APFloat::EnumToSemantics(SemanticsKind);

    for (FPClassTest FPClass : {fcPosZero, fcPosNormal}) {
      SCOPED_TRACE(testing::Message()
                   << "Semantics = " << I << ", class = " << FPClass);
      KnownBits Known = KnownFPClass(FPClass, false).toKnownBits(Semantics);
      if (IsSupported(Semantics)) {
        // The following expectations are true for all "supported" types.
        EXPECT_TRUE(Known.isNonNegative());
        if (FPClass == fcPosZero)
          EXPECT_TRUE(Known.isZero());
      } else {
        EXPECT_TRUE(Known.isUnknown());
      }
    }
  }
}

static APInt makeX87Bits(uint16_t Exponent, bool IntegerBit,
                         uint64_t Fraction) {
  return (APInt(80, Exponent) << 64) | (APInt(80, IntegerBit) << 63) |
         APInt(80, Fraction);
}

TEST(KnownFPClassTest, BitcastNonCanonicalX87) {
  const fltSemantics &Semantics = APFloat::x87DoubleExtended();
  constexpr uint64_t QuietBit = UINT64_C(1) << 62;
  constexpr uint64_t AllOnesPayload = (UINT64_C(1) << 62) - 1;
  constexpr uint64_t AllOnesFraction = (UINT64_C(1) << 63) - 1;
  constexpr uint16_t InfNanExponent = UINT16_C(0x7FFF);
  constexpr uint16_t MaxFiniteExponent = InfNanExponent - 1;

  for (bool Negative : {false, true}) {
    // An exponent of zero with a set integer bit is a pseudo-denormal, which
    // APFloat classifies as normal.
    expectConstant("x87float80", "pseudo_denormal_zero_fraction", Semantics,
                   makeX87Bits(0, true, 0), fcPosNormal, Negative);
    expectConstant("x87float80", "pseudo_denormal_lsb_set_fraction", Semantics,
                   makeX87Bits(0, true, 1), fcPosNormal, Negative);
    expectConstant("x87float80", "pseudo_denormal_all_ones_fraction", Semantics,
                   makeX87Bits(0, true, AllOnesFraction), fcPosNormal,
                   Negative);

    // An all-ones exponent with a clear integer bit is a pseudo-infinity or
    // pseudo-NaN, which APFloat classifies as sNaN/qNaN.
    expectConstant("x87float80", "pseudo_infinity", Semantics,
                   makeX87Bits(InfNanExponent, false, 0), fcSNan, Negative);
    expectConstant("x87float80", "pseudo_snan_lsb_set_payload", Semantics,
                   makeX87Bits(InfNanExponent, false, 1), fcSNan, Negative);
    expectConstant("x87float80", "pseudo_snan_all_ones_payload", Semantics,
                   makeX87Bits(InfNanExponent, false, AllOnesPayload), fcSNan,
                   Negative);
    expectConstant("x87float80", "pseudo_qnan", Semantics,
                   makeX87Bits(InfNanExponent, false, QuietBit), fcQNan,
                   Negative);
    expectConstant(
        "x87float80", "pseudo_qnan_all_ones_payload", Semantics,
        makeX87Bits(InfNanExponent, false, QuietBit | AllOnesPayload), fcQNan,
        Negative);

    // A nonzero, non-all-ones exponent with a clear integer bit is an
    // unnormal, which APFloat classifies as sNaN/qNaN.
    expectConstant("x87float80", "unnormal_zero_fraction", Semantics,
                   makeX87Bits(1, false, 0), fcSNan, Negative);
    expectConstant("x87float80", "unnormal_snan_lsb_set_payload", Semantics,
                   makeX87Bits(1, false, 1), fcSNan, Negative);
    expectConstant("x87float80", "unnormal_snan_all_ones_payload", Semantics,
                   makeX87Bits(1, false, AllOnesPayload), fcSNan, Negative);
    expectConstant("x87float80", "unnormal_qnan", Semantics,
                   makeX87Bits(1, false, QuietBit), fcQNan, Negative);
    expectConstant("x87float80", "unnormal_qnan_all_ones_payload", Semantics,
                   makeX87Bits(1, false, QuietBit | AllOnesPayload), fcQNan,
                   Negative);

    expectConstant("x87float80", "unnormal_zero_fraction", Semantics,
                   makeX87Bits(MaxFiniteExponent, false, 0), fcSNan, Negative);
    expectConstant("x87float80", "unnormal_snan_lsb_set_payload", Semantics,
                   makeX87Bits(MaxFiniteExponent, false, 1), fcSNan, Negative);
    expectConstant("x87float80", "unnormal_snan_all_ones_payload", Semantics,
                   makeX87Bits(MaxFiniteExponent, false, AllOnesPayload),
                   fcSNan, Negative);
    expectConstant("x87float80", "unnormal_qnan", Semantics,
                   makeX87Bits(MaxFiniteExponent, false, QuietBit), fcQNan,
                   Negative);
    expectConstant(
        "x87float80", "unnormal_qnan_all_ones_payload", Semantics,
        makeX87Bits(MaxFiniteExponent, false, QuietBit | AllOnesPayload),
        fcQNan, Negative);
  }
}

} // end anonymous namespace
