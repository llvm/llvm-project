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
  EXPECT_EQ(ExpectedClass, Known.KnownFPClasses);
  EXPECT_EQ(Negative, Known.getSignBit());
}

TEST(KnownFPClassTest, BitcastExhaustiveIEEEHalf) {
  const fltSemantics &Semantics = APFloat::IEEEhalf();

  for (uint64_t RawBits = 0; RawBits != (1u << 16); ++RawBits) {
    APInt ValueBits(16, RawBits);
    KnownFPClass Known =
        KnownFPClass::bitcast(Semantics, KnownBits::makeConstant(ValueBits));
    KnownFPClass Expected(APFloat(Semantics, ValueBits));

    ASSERT_EQ(Expected.KnownFPClasses, Known.KnownFPClasses) << RawBits;
    ASSERT_EQ(Expected.getSignBit(), Known.getSignBit()) << RawBits;
  }
}

TEST(KnownFPClassTest, BitcastConflict) {
  const fltSemantics &Semantics = APFloat::IEEEsingle();
  KnownBits Bits(Semantics.sizeInBits);
  Bits.setAllConflict();

  ASSERT_TRUE(Bits.hasConflict());
  KnownFPClass Known = KnownFPClass::bitcast(Semantics, Bits);
  EXPECT_EQ(fcAllFlags, Known.KnownFPClasses);
  EXPECT_EQ(std::nullopt, Known.getSignBit());
}

TEST(KnownFPClassTest, BitcastPartialConflict) {
  const fltSemantics &Semantics = APFloat::IEEEsingle();
  KnownBits Bits(Semantics.sizeInBits);
  Bits.Zero.setAllBits();
  Bits.One.setBit(0);

  ASSERT_TRUE(Bits.hasConflict());
  KnownFPClass Known = KnownFPClass::bitcast(Semantics, Bits);
  EXPECT_EQ(fcAllFlags, Known.KnownFPClasses);
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
        SemanticsCase{"ieee_binary128", &APFloat::IEEEquad()}}) {
    const fltSemantics &Semantics = *TestCase.Semantics;
    const unsigned BitWidth = Semantics.sizeInBits;
    const unsigned MantissaBits = Semantics.precision - 1;
    const APInt ExponentMask =
        APInt::getBitsSet(BitWidth, MantissaBits, BitWidth - 1);
    const APInt MantissaMask = APInt::getLowBitsSet(BitWidth, MantissaBits);
    const APInt QuietBit = APInt::getOneBitSet(BitWidth, MantissaBits - 1);

    for (bool Negative : {false, true}) {
      expectConstant(TestCase.Name, "0.0", Semantics, APInt::getZero(BitWidth),
                     fcPosZero, Negative);
      expectConstant(TestCase.Name, "min_subnormal", Semantics,
                     APInt(BitWidth, 1), fcPosSubnormal, Negative);
      expectConstant(TestCase.Name, "max_subnormal", Semantics, MantissaMask,
                     fcPosSubnormal, Negative);
      expectConstant(TestCase.Name, "min_normal", Semantics,
                     APInt::getOneBitSet(BitWidth, MantissaBits), fcPosNormal,
                     Negative);
      expectConstant(TestCase.Name, "1.0", Semantics,
                     APFloat::getOne(Semantics).bitcastToAPInt(), fcPosNormal,
                     Negative);
      expectConstant(TestCase.Name, "max_normal", Semantics,
                     APFloat::getLargest(Semantics).bitcastToAPInt(),
                     fcPosNormal, Negative);
      expectConstant(TestCase.Name, "inf", Semantics, ExponentMask, fcPosInf,
                     Negative);

      // An sNaN has a clear quiet bit and a non-zero payload.
      expectConstant(TestCase.Name, "snan_mostly_zero", Semantics,
                     ExponentMask | APInt(BitWidth, 1), fcSNan, Negative);

      // A qNaN has a set quiet bit. The remaining payload bits may be zero.
      expectConstant(TestCase.Name, "qnan_mostly_zero", Semantics,
                     ExponentMask | QuietBit, fcQNan, Negative);

      expectConstant(TestCase.Name, "snan_mostly_one", Semantics,
                     ExponentMask | (MantissaMask & ~QuietBit), fcSNan,
                     Negative);
      expectConstant(TestCase.Name, "qnan_mostly_one", Semantics,
                     ExponentMask | MantissaMask, fcQNan, Negative);
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

} // end anonymous namespace
