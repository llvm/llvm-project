//===- unittests/ADT/FixedPointTest.cpp -- fixed point number tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFixedPoint.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"

using llvm::APFixedPoint;
using llvm::APFloat;
using llvm::APInt;
using llvm::APSInt;
using llvm::FixedPointSemantics;

namespace {

FixedPointSemantics Saturated(FixedPointSemantics Sema) {
  Sema.setSaturated(true);
  return Sema;
}

FixedPointSemantics getSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/7, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/15, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getLAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/31, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/7, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/15, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getLFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/31, /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/8, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/16, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getULAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/32, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/8, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getUFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/16, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getULFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/32, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getPadUSAccumSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/7, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUAccumSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/15, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadULAccumSema() {
  return FixedPointSemantics(/*width=*/64, /*scale=*/31, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUSFractSema() {
  return FixedPointSemantics(/*width=*/8, /*scale=*/7, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadUFractSema() {
  return FixedPointSemantics(/*width=*/16, /*scale=*/15, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getPadULFractSema() {
  return FixedPointSemantics(/*width=*/32, /*scale=*/31, /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/true);
}

FixedPointSemantics getU8Neg10() {
  return FixedPointSemantics(/*width=*/8, /*lsb=*/FixedPointSemantics::Lsb{-10},
                             /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getS16Neg18() {
  return FixedPointSemantics(/*width=*/16,
                             /*lsb=*/FixedPointSemantics::Lsb{-18},
                             /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getU8Pos4() {
  return FixedPointSemantics(/*width=*/8, /*lsb=*/FixedPointSemantics::Lsb{4},
                             /*isSigned=*/false,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

FixedPointSemantics getS32Pos2() {
  return FixedPointSemantics(/*width=*/32, /*lsb=*/FixedPointSemantics::Lsb{2},
                             /*isSigned=*/true,
                             /*isSaturated=*/false,
                             /*hasUnsignedPadding=*/false);
}

void CheckUnpaddedMax(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMax(Sema).getValue(),
            APSInt::getMaxValue(Sema.getWidth(), !Sema.isSigned()));
}

void CheckPaddedMax(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMax(Sema).getValue(),
            APSInt::getMaxValue(Sema.getWidth(), !Sema.isSigned()) >> 1);
}

void CheckMin(const FixedPointSemantics &Sema) {
  ASSERT_EQ(APFixedPoint::getMin(Sema).getValue(),
            APSInt::getMinValue(Sema.getWidth(), !Sema.isSigned()));
}

TEST(FixedPointTest, getMax) {
  CheckUnpaddedMax(getSAccumSema());
  CheckUnpaddedMax(getAccumSema());
  CheckUnpaddedMax(getLAccumSema());
  CheckUnpaddedMax(getUSAccumSema());
  CheckUnpaddedMax(getUAccumSema());
  CheckUnpaddedMax(getULAccumSema());
  CheckUnpaddedMax(getSFractSema());
  CheckUnpaddedMax(getFractSema());
  CheckUnpaddedMax(getLFractSema());
  CheckUnpaddedMax(getUSFractSema());
  CheckUnpaddedMax(getUFractSema());
  CheckUnpaddedMax(getULFractSema());
  CheckUnpaddedMax(getU8Neg10());
  CheckUnpaddedMax(getS16Neg18());
  CheckUnpaddedMax(getU8Pos4());
  CheckUnpaddedMax(getS32Pos2());

  CheckPaddedMax(getPadUSAccumSema());
  CheckPaddedMax(getPadUAccumSema());
  CheckPaddedMax(getPadULAccumSema());
  CheckPaddedMax(getPadUSFractSema());
  CheckPaddedMax(getPadUFractSema());
  CheckPaddedMax(getPadULFractSema());
}

TEST(FixedPointTest, getMin) {
  CheckMin(getSAccumSema());
  CheckMin(getAccumSema());
  CheckMin(getLAccumSema());
  CheckMin(getUSAccumSema());
  CheckMin(getUAccumSema());
  CheckMin(getULAccumSema());
  CheckMin(getSFractSema());
  CheckMin(getFractSema());
  CheckMin(getLFractSema());
  CheckMin(getUSFractSema());
  CheckMin(getUFractSema());
  CheckMin(getULFractSema());
  CheckMin(getU8Neg10());
  CheckMin(getS16Neg18());
  CheckMin(getU8Pos4());
  CheckMin(getS32Pos2());

  CheckMin(getPadUSAccumSema());
  CheckMin(getPadUAccumSema());
  CheckMin(getPadULAccumSema());
  CheckMin(getPadUSFractSema());
  CheckMin(getPadUFractSema());
  CheckMin(getPadULFractSema());
}

int64_t relativeShr(int64_t Int, int64_t Shift) {
  return (Shift > 0) ? Int >> Shift : Int << -Shift;
}

void CheckIntPart(const FixedPointSemantics &Sema, int64_t IntPart) {
  int64_t FullFactPart =
      (Sema.getLsbWeight() > 0) ? 0 : (1ULL << (-Sema.getLsbWeight() - 1));

  // Value with a fraction
  APFixedPoint ValWithFract(
      APInt(Sema.getWidth(),
            relativeShr(IntPart, Sema.getLsbWeight()) + FullFactPart,
            Sema.isSigned(), /*implicitTrunc=*/true),
      Sema);
  ASSERT_EQ(ValWithFract.getIntPart(), IntPart);

  // Just fraction
  APFixedPoint JustFract(APInt(Sema.getWidth(), FullFactPart, Sema.isSigned(),
                               /*implicitTrunc=*/true),
                         Sema);
  ASSERT_EQ(JustFract.getIntPart(), 0);

  // Whole number
  APFixedPoint WholeNum(APInt(Sema.getWidth(),
                              relativeShr(IntPart, Sema.getLsbWeight()),
                              Sema.isSigned(), /*implicitTrunc=*/true),
                        Sema);
  ASSERT_EQ(WholeNum.getIntPart(), IntPart);

  // Negative
  if (Sema.isSigned()) {
    APFixedPoint Negative(APInt(Sema.getWidth(),
                                relativeShr(IntPart, Sema.getLsbWeight()),
                                Sema.isSigned(), /*implicitTrunc=*/true),
                          Sema);
    ASSERT_EQ(Negative.getIntPart(), IntPart);
  }
}

void CheckIntPartMin(const FixedPointSemantics &Sema, int64_t Expected) {
  EXPECT_TRUE(APSInt::compareValues(APFixedPoint::getMin(Sema).getIntPart(),
                                    APSInt::get(Expected)) == 0);
}

void CheckIntPartMax(const FixedPointSemantics &Sema, uint64_t Expected) {
  EXPECT_TRUE(APSInt::compareValues(APFixedPoint::getMax(Sema).getIntPart(),
                                    APSInt::getUnsigned(Expected)) == 0);
}

void CheckIntPartRes(const FixedPointSemantics &Sema, int64_t Representation,
                     uint64_t Result) {
  APFixedPoint Val(Representation, Sema);
  ASSERT_EQ(Val.getIntPart().getZExtValue(), Result) ;
}

TEST(FixedPoint, getIntPart) {
  // Normal values
  CheckIntPart(getSAccumSema(), 2);
  CheckIntPart(getAccumSema(), 2);
  CheckIntPart(getLAccumSema(), 2);
  CheckIntPart(getUSAccumSema(), 2);
  CheckIntPart(getUAccumSema(), 2);
  CheckIntPart(getULAccumSema(), 2);
  CheckIntPart(getU8Pos4(), 32);
  CheckIntPart(getS32Pos2(), 32);

  // Zero
  CheckIntPart(getSAccumSema(), 0);
  CheckIntPart(getAccumSema(), 0);
  CheckIntPart(getLAccumSema(), 0);
  CheckIntPart(getUSAccumSema(), 0);
  CheckIntPart(getUAccumSema(), 0);
  CheckIntPart(getULAccumSema(), 0);

  CheckIntPart(getSFractSema(), 0);
  CheckIntPart(getFractSema(), 0);
  CheckIntPart(getLFractSema(), 0);
  CheckIntPart(getUSFractSema(), 0);
  CheckIntPart(getUFractSema(), 0);
  CheckIntPart(getULFractSema(), 0);

  CheckIntPart(getS16Neg18(), 0);
  CheckIntPart(getU8Neg10(), 0);
  CheckIntPart(getU8Pos4(), 0);
  CheckIntPart(getS32Pos2(), 0);

  // Min
  CheckIntPartMin(getSAccumSema(), -256);
  CheckIntPartMin(getAccumSema(), -65536);
  CheckIntPartMin(getLAccumSema(), -4294967296);

  CheckIntPartMin(getSFractSema(), -1);
  CheckIntPartMin(getFractSema(), -1);
  CheckIntPartMin(getLFractSema(), -1);

  CheckIntPartMin(getS32Pos2(), -8589934592);

  // Max
  CheckIntPartMax(getSAccumSema(), 255);
  CheckIntPartMax(getAccumSema(), 65535);
  CheckIntPartMax(getLAccumSema(), 4294967295);
  CheckIntPartMax(getUSAccumSema(), 255);
  CheckIntPartMax(getUAccumSema(), 65535);
  CheckIntPartMax(getULAccumSema(), 4294967295);

  CheckIntPartMax(getU8Pos4(), 255 << 4);
  CheckIntPartMax(getS32Pos2(), 2147483647ull << 2);

  CheckIntPartMax(getSFractSema(), 0);
  CheckIntPartMax(getFractSema(), 0);
  CheckIntPartMax(getLFractSema(), 0);
  CheckIntPartMax(getUSFractSema(), 0);
  CheckIntPartMax(getUFractSema(), 0);
  CheckIntPartMax(getULFractSema(), 0);

  // Padded
  // Normal Values
  CheckIntPart(getPadUSAccumSema(), 2);
  CheckIntPart(getPadUAccumSema(), 2);
  CheckIntPart(getPadULAccumSema(), 2);

  // Zero
  CheckIntPart(getPadUSAccumSema(), 0);
  CheckIntPart(getPadUAccumSema(), 0);
  CheckIntPart(getPadULAccumSema(), 0);

  CheckIntPart(getPadUSFractSema(), 0);
  CheckIntPart(getPadUFractSema(), 0);
  CheckIntPart(getPadULFractSema(), 0);

  // Max
  CheckIntPartMax(getPadUSAccumSema(), 255);
  CheckIntPartMax(getPadUAccumSema(), 65535);
  CheckIntPartMax(getPadULAccumSema(), 4294967295);

  CheckIntPartMax(getPadUSFractSema(), 0);
  CheckIntPartMax(getPadUFractSema(), 0);
  CheckIntPartMax(getPadULFractSema(), 0);

  // Rounded Towards Zero
  CheckIntPartRes(getSFractSema(), -127, 0);
  CheckIntPartRes(getFractSema(), -32767, 0);
  CheckIntPartRes(getLFractSema(), -2147483647, 0);
  CheckIntPartRes(getS16Neg18(), -32768, 0);
}

TEST(FixedPoint, compare) {
  // Equality
  // With fractional part (2.5)
  // Across sizes
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(81920, getAccumSema()));
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(5368709120, getLAccumSema()));
  ASSERT_EQ(APFixedPoint(0, getSAccumSema()), APFixedPoint(0, getLAccumSema()));

  ASSERT_EQ(APFixedPoint(0, getS16Neg18()), APFixedPoint(0, getU8Neg10()));
  ASSERT_EQ(APFixedPoint(256, getS16Neg18()), APFixedPoint(1, getU8Neg10()));
  ASSERT_EQ(APFixedPoint(32512, getS16Neg18()),
            APFixedPoint(127, getU8Neg10()));
  ASSERT_EQ(APFixedPoint(4, getS32Pos2()), APFixedPoint(1, getU8Pos4()));
  ASSERT_EQ(APFixedPoint(1020, getS32Pos2()), APFixedPoint(255, getU8Pos4()));

  // Across types (0.5)
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(64, getSFractSema()));
  ASSERT_EQ(APFixedPoint(16384, getAccumSema()),
            APFixedPoint(16384, getFractSema()));
  ASSERT_EQ(APFixedPoint(1073741824, getLAccumSema()),
            APFixedPoint(1073741824, getLFractSema()));

  // Across widths and types (0.5)
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(16384, getFractSema()));
  ASSERT_EQ(APFixedPoint(64, getSAccumSema()),
            APFixedPoint(1073741824, getLFractSema()));

  // Across saturation
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(81920, Saturated(getAccumSema())));

  // Across signs
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(640, getUSAccumSema()));
  ASSERT_EQ(APFixedPoint(-320, getSAccumSema()),
            APFixedPoint(-81920, getAccumSema()));

  // Across padding
  ASSERT_EQ(APFixedPoint(320, getSAccumSema()),
            APFixedPoint(320, getPadUSAccumSema()));
  ASSERT_EQ(APFixedPoint(640, getUSAccumSema()),
            APFixedPoint(320, getPadUSAccumSema()));

  // Less than
  ASSERT_LT(APFixedPoint(-1, getSAccumSema()), APFixedPoint(0, getAccumSema()));
  ASSERT_LT(APFixedPoint(-1, getSAccumSema()),
            APFixedPoint(0, getUAccumSema()));
  ASSERT_LT(APFixedPoint(0, getSAccumSema()), APFixedPoint(1, getAccumSema()));
  ASSERT_LT(APFixedPoint(0, getSAccumSema()), APFixedPoint(1, getUAccumSema()));
  ASSERT_LT(APFixedPoint(0, getUSAccumSema()), APFixedPoint(1, getAccumSema()));
  ASSERT_LT(APFixedPoint(0, getUSAccumSema()),
            APFixedPoint(1, getUAccumSema()));
  ASSERT_LT(APFixedPoint(65280, getS16Neg18()),
            APFixedPoint(255, getU8Neg10()));

  // Greater than
  ASSERT_GT(APFixedPoint(0, getAccumSema()), APFixedPoint(-1, getSAccumSema()));
  ASSERT_GT(APFixedPoint(0, getUAccumSema()),
            APFixedPoint(-1, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getAccumSema()), APFixedPoint(0, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getUAccumSema()), APFixedPoint(0, getSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getAccumSema()), APFixedPoint(0, getUSAccumSema()));
  ASSERT_GT(APFixedPoint(1, getUAccumSema()),
            APFixedPoint(0, getUSAccumSema()));
}

// Check that a fixed point value in one sema is the same in another sema
void CheckUnsaturatedConversion(FixedPointSemantics Src,
                                FixedPointSemantics Dst, int64_t TestVal) {
  int64_t ScaledVal = TestVal;
  bool IsNegative = ScaledVal < 0;
  if (IsNegative)
    ScaledVal = -ScaledVal;

  if (Dst.getLsbWeight() < Src.getLsbWeight()) {
    ScaledVal <<= (Src.getLsbWeight() - Dst.getLsbWeight());
  } else {
    ScaledVal >>= (Dst.getLsbWeight() - Src.getLsbWeight());
  }

  if (IsNegative)
    ScaledVal = -ScaledVal;

  APFixedPoint Fixed(TestVal, Src);
  APFixedPoint Expected(ScaledVal, Dst);
  ASSERT_EQ(Fixed.convert(Dst), Expected);
}

// Check the value in a given fixed point sema overflows to the saturated min
// for another sema
void CheckSaturatedConversionMin(FixedPointSemantics Src,
                                 FixedPointSemantics Dst, int64_t TestVal) {
  APFixedPoint Fixed(TestVal, Src);
  ASSERT_EQ(Fixed.convert(Dst), APFixedPoint::getMin(Dst));
}

// Check the value in a given fixed point sema overflows to the saturated max
// for another sema
void CheckSaturatedConversionMax(FixedPointSemantics Src,
                                 FixedPointSemantics Dst, int64_t TestVal) {
  APFixedPoint Fixed(TestVal, Src);
  ASSERT_EQ(Fixed.convert(Dst), APFixedPoint::getMax(Dst));
}

// Check one signed _Accum sema converted to other sema for different values.
void CheckSignedAccumConversionsAgainstOthers(FixedPointSemantics Src,
                                              int64_t OneVal) {
  int64_t NormalVal = (OneVal * 2) + (OneVal / 2); // 2.5
  int64_t HalfVal = (OneVal / 2);                  // 0.5

  // +Accums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getULAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), NormalVal);

  // -Accums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), -NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), -NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getUSAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getUAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getULAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUSAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUAccumSema()), -NormalVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadULAccumSema()), -NormalVal);

  // +Accums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getULFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), HalfVal);

  // -Accums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), -HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), -HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getUSFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getUFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getULFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUSFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadUFractSema()), -HalfVal);
  CheckSaturatedConversionMin(Src, Saturated(getPadULFractSema()), -HalfVal);

  // 0 to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getLAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getUAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getULAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), 0);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), 0);

  // 0 to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getFractSema(), 0);
  CheckUnsaturatedConversion(Src, getLFractSema(), 0);
  CheckUnsaturatedConversion(Src, getUSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getUFractSema(), 0);
  CheckUnsaturatedConversion(Src, getULFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), 0);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), 0);
}

// Check one unsigned _Accum sema converted to other sema for different
// values.
void CheckUnsignedAccumConversionsAgainstOthers(FixedPointSemantics Src,
                                                int64_t OneVal) {
  int64_t NormalVal = (OneVal * 2) + (OneVal / 2); // 2.5
  int64_t HalfVal = (OneVal / 2);                  // 0.5

  // +UAccums to Accums
  CheckUnsaturatedConversion(Src, getSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getLAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getULAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUSAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadUAccumSema(), NormalVal);
  CheckUnsaturatedConversion(Src, getPadULAccumSema(), NormalVal);

  // +UAccums to Fracts
  CheckUnsaturatedConversion(Src, getSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getLFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getULFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUSFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadUFractSema(), HalfVal);
  CheckUnsaturatedConversion(Src, getPadULFractSema(), HalfVal);
}

TEST(FixedPoint, AccumConversions) {
  // Normal conversions
  CheckSignedAccumConversionsAgainstOthers(getSAccumSema(), 128);
  CheckUnsignedAccumConversionsAgainstOthers(getUSAccumSema(), 256);
  CheckSignedAccumConversionsAgainstOthers(getAccumSema(), 32768);
  CheckUnsignedAccumConversionsAgainstOthers(getUAccumSema(), 65536);
  CheckSignedAccumConversionsAgainstOthers(getLAccumSema(), 2147483648);
  CheckUnsignedAccumConversionsAgainstOthers(getULAccumSema(), 4294967296);

  CheckUnsignedAccumConversionsAgainstOthers(getPadUSAccumSema(), 128);
  CheckUnsignedAccumConversionsAgainstOthers(getPadUAccumSema(), 32768);
  CheckUnsignedAccumConversionsAgainstOthers(getPadULAccumSema(), 2147483648);
}

TEST(FixedPoint, AccumConversionOverflow) {
  // To SAccum max limit (65536)
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getPadUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getAccumSema()),
                              281474976710656);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getUAccumSema()),
                              281474976710656);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getPadUAccumSema()),
                              281474976710656);

  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getUAccumSema()),
                              140737488355328);
  CheckSaturatedConversionMax(getPadULAccumSema(),
                              Saturated(getPadUAccumSema()), 140737488355328);

  // To SAccum min limit (-65536)
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getAccumSema()),
                              -140737488355328);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getUAccumSema()),
                              -140737488355328);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getPadUAccumSema()),
                              -140737488355328);
}

TEST(FixedPoint, SAccumConversionOverflow) {
  // To SAccum max limit (256)
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getAccumSema(), Saturated(getPadUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getUSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getUAccumSema(), Saturated(getPadUSAccumSema()),
                              16777216);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getLAccumSema(), Saturated(getPadUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getSAccumSema()),
                              1099511627776);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getUSAccumSema()),
                              1099511627776);
  CheckSaturatedConversionMax(getULAccumSema(), Saturated(getPadUSAccumSema()),
                              1099511627776);

  CheckSaturatedConversionMax(getPadUAccumSema(), Saturated(getSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getPadUAccumSema(), Saturated(getUSAccumSema()),
                              8388608);
  CheckSaturatedConversionMax(getPadUAccumSema(),
                              Saturated(getPadUSAccumSema()), 8388608);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getPadULAccumSema(), Saturated(getUSAccumSema()),
                              549755813888);
  CheckSaturatedConversionMax(getPadULAccumSema(),
                              Saturated(getPadUSAccumSema()), 549755813888);

  // To SAccum min limit (-256)
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getUSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getAccumSema(), Saturated(getPadUSAccumSema()),
                              -8388608);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getSAccumSema()),
                              -549755813888);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getUSAccumSema()),
                              -549755813888);
  CheckSaturatedConversionMin(getLAccumSema(), Saturated(getPadUSAccumSema()),
                              -549755813888);
}

TEST(FixedPoint, GetValueSignAfterConversion) {
  APFixedPoint Fixed(255 << 7, getSAccumSema());
  ASSERT_TRUE(Fixed.getValue().isSigned());
  APFixedPoint UFixed = Fixed.convert(getUSAccumSema());
  ASSERT_TRUE(UFixed.getValue().isUnsigned());
  ASSERT_EQ(UFixed.getValue(), APSInt::getUnsigned(255 << 8).extOrTrunc(16));
}

TEST(FixedPoint, ModularWrapAround) {
  // Positive to negative
  APFixedPoint Val = APFixedPoint(1ULL << 7, getSAccumSema());
  ASSERT_EQ(Val.convert(getLFractSema()).getValue(), -(1ULL << 31));

  Val = APFixedPoint(1ULL << 23, getAccumSema());
  ASSERT_EQ(Val.convert(getSAccumSema()).getValue(), -(1ULL << 15));

  Val = APFixedPoint(1ULL << 47, getLAccumSema());
  ASSERT_EQ(Val.convert(getAccumSema()).getValue(), -(1ULL << 31));

  // Negative to positive
  Val = APFixedPoint(/*-1.5*/ -192, getSAccumSema());
  ASSERT_EQ(Val.convert(getLFractSema()).getValue(), 1ULL << 30);

  Val = APFixedPoint(-(257 << 15), getAccumSema());
  ASSERT_EQ(Val.convert(getSAccumSema()).getValue(), 255 << 7);

  Val = APFixedPoint(-(65537ULL << 31), getLAccumSema());
  ASSERT_EQ(Val.convert(getAccumSema()).getValue(), 65535 << 15);

  // Signed to unsigned
  Val = APFixedPoint(-(1 << 7), getSAccumSema());
  ASSERT_EQ(Val.convert(getUSAccumSema()).getValue(), 255 << 8);

  Val = APFixedPoint(-(1 << 15), getAccumSema());
  ASSERT_EQ(Val.convert(getUAccumSema()).getValue(), 65535ULL << 16);

  Val = APFixedPoint(-(1ULL << 31), getLAccumSema());
  ASSERT_EQ(Val.convert(getULAccumSema()).getValue().getZExtValue(),
            4294967295ULL << 32);
}

enum OvfKind { MinSat, MaxSat };

void CheckFloatToFixedConversion(APFloat &Val, const FixedPointSemantics &Sema,
                                 int64_t ExpectedNonSat) {
  bool Ovf;
  ASSERT_EQ(APFixedPoint::getFromFloatValue(Val, Sema, &Ovf).getValue(),
            ExpectedNonSat);
  ASSERT_EQ(Ovf, false);
  ASSERT_EQ(
      APFixedPoint::getFromFloatValue(Val, Saturated(Sema), &Ovf).getValue(),
      ExpectedNonSat);
  ASSERT_EQ(Ovf, false);
}

void CheckFloatToFixedConversion(APFloat &Val, const FixedPointSemantics &Sema,
                                 OvfKind ExpectedOvf) {
  bool Ovf;
  (void)APFixedPoint::getFromFloatValue(Val, Sema, &Ovf);
  ASSERT_EQ(Ovf, true);
  ASSERT_EQ(
      APFixedPoint::getFromFloatValue(Val, Saturated(Sema), &Ovf).getValue(),
      (ExpectedOvf == MinSat ? APFixedPoint::getMin(Sema)
                             : APFixedPoint::getMax(Sema))
          .getValue());
  ASSERT_EQ(Ovf, false);
}

TEST(FixedPoint, toString) {
  ASSERT_EQ(APFixedPoint::getMax(getS16Neg18()).toString(),
            "0.124996185302734375");
  ASSERT_EQ(APFixedPoint::getMin(getS16Neg18())
                .add(APFixedPoint(1, getS16Neg18()))
                .toString(),
            "-0.124996185302734375");
  ASSERT_EQ(APFixedPoint::getMin(getS16Neg18()).toString(), "-0.125");
  ASSERT_EQ(APFixedPoint::getMax(getU8Neg10()).toString(), "0.2490234375");
  ASSERT_EQ(APFixedPoint::getMin(getU8Neg10()).toString(), "0.0");
  ASSERT_EQ(APFixedPoint::getMax(getS32Pos2()).toString(), "8589934588.0");
  ASSERT_EQ(APFixedPoint::getMin(getS32Pos2())
                .add(APFixedPoint(1, getS32Pos2()))
                .toString(),
            "-8589934588.0");
  ASSERT_EQ(APFixedPoint::getMin(getS32Pos2()).toString(), "-8589934592.0");
  ASSERT_EQ(APFixedPoint::getMax(getU8Pos4()).toString(), "4080.0");
  ASSERT_EQ(APFixedPoint::getMin(getU8Pos4()).toString(), "0.0");
}

TEST(FixedPoint, FloatToFixed) {
  APFloat Val(0.0f);

  // Simple exact fraction
  Val = APFloat(0.75f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), 3ULL << 5);
  CheckFloatToFixedConversion(Val, getAccumSema(),  3ULL << 13);
  CheckFloatToFixedConversion(Val, getLAccumSema(), 3ULL << 29);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), 3ULL << 6);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  3ULL << 14);
  CheckFloatToFixedConversion(Val, getULAccumSema(), 3ULL << 30);

  CheckFloatToFixedConversion(Val, getSFractSema(), 3ULL << 5);
  CheckFloatToFixedConversion(Val, getFractSema(),  3ULL << 13);
  CheckFloatToFixedConversion(Val, getLFractSema(), 3ULL << 29);

  CheckFloatToFixedConversion(Val, getUSFractSema(), 3ULL << 6);
  CheckFloatToFixedConversion(Val, getUFractSema(),  3ULL << 14);
  CheckFloatToFixedConversion(Val, getULFractSema(), 3ULL << 30);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MaxSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(),  0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MaxSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  // Simple negative exact fraction
  Val = APFloat(-0.75f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), -3ULL << 5);
  CheckFloatToFixedConversion(Val, getAccumSema(),  -3ULL << 13);
  CheckFloatToFixedConversion(Val, getLAccumSema(), -3ULL << 29);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), MinSat);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  MinSat);
  CheckFloatToFixedConversion(Val, getULAccumSema(), MinSat);

  CheckFloatToFixedConversion(Val, getSFractSema(), -3ULL << 5);
  CheckFloatToFixedConversion(Val, getFractSema(),  -3ULL << 13);
  CheckFloatToFixedConversion(Val, getLFractSema(), -3ULL << 29);

  CheckFloatToFixedConversion(Val, getUSFractSema(), MinSat);
  CheckFloatToFixedConversion(Val, getUFractSema(),  MinSat);
  CheckFloatToFixedConversion(Val, getULFractSema(), MinSat);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MinSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(),  0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MinSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  // Highly precise fraction
  Val = APFloat(0.999999940395355224609375f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), 0x7FULL);
  CheckFloatToFixedConversion(Val, getAccumSema(),  0x7FFFULL);
  CheckFloatToFixedConversion(Val, getLAccumSema(), 0xFFFFFFULL << 7);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), 0xFFULL);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  0xFFFFULL);
  CheckFloatToFixedConversion(Val, getULAccumSema(), 0xFFFFFFULL << 8);

  CheckFloatToFixedConversion(Val, getSFractSema(), 0x7FULL);
  CheckFloatToFixedConversion(Val, getFractSema(),  0x7FFFULL);
  CheckFloatToFixedConversion(Val, getLFractSema(), 0xFFFFFFULL << 7);

  CheckFloatToFixedConversion(Val, getUSFractSema(), 0xFFULL);
  CheckFloatToFixedConversion(Val, getUFractSema(),  0xFFFFULL);
  CheckFloatToFixedConversion(Val, getULFractSema(), 0xFFFFFFULL << 8);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MaxSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MaxSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  // Integral and fraction
  Val = APFloat(17.99609375f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), 0x11FFULL >> 1);
  CheckFloatToFixedConversion(Val, getAccumSema(),  0x11FFULL << 7);
  CheckFloatToFixedConversion(Val, getLAccumSema(), 0x11FFULL << 23);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), 0x11FFULL);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  0x11FFULL << 8);
  CheckFloatToFixedConversion(Val, getULAccumSema(), 0x11FFULL << 24);

  CheckFloatToFixedConversion(Val, getSFractSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getFractSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getLFractSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getUSFractSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getUFractSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getULFractSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MaxSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 1);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MaxSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 1 << 2);

  // Negative integral and fraction
  Val = APFloat(-17.99609375f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), -0x11FELL >> 1);
  CheckFloatToFixedConversion(Val, getAccumSema(),  -0x11FFULL << 7);
  CheckFloatToFixedConversion(Val, getLAccumSema(), -0x11FFULL << 23);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), MinSat);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  MinSat);
  CheckFloatToFixedConversion(Val, getULAccumSema(), MinSat);

  CheckFloatToFixedConversion(Val, getSFractSema(), MinSat);
  CheckFloatToFixedConversion(Val, getFractSema(),  MinSat);
  CheckFloatToFixedConversion(Val, getLFractSema(), MinSat);

  CheckFloatToFixedConversion(Val, getUSFractSema(), MinSat);
  CheckFloatToFixedConversion(Val, getUFractSema(),  MinSat);
  CheckFloatToFixedConversion(Val, getULFractSema(), MinSat);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MinSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), MinSat);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MinSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), -4);

  // Very large value
  Val = APFloat(1.0e38f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getAccumSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getLAccumSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getULAccumSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getSFractSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getFractSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getLFractSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getUSFractSema(), MaxSat);
  CheckFloatToFixedConversion(Val, getUFractSema(),  MaxSat);
  CheckFloatToFixedConversion(Val, getULFractSema(), MaxSat);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MaxSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), MaxSat);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MaxSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), MaxSat);

  // Very small value
  Val = APFloat(1.0e-38f);
  CheckFloatToFixedConversion(Val, getSAccumSema(), 0);
  CheckFloatToFixedConversion(Val, getAccumSema(),  0);
  CheckFloatToFixedConversion(Val, getLAccumSema(), 0);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), 0);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  0);
  CheckFloatToFixedConversion(Val, getULAccumSema(), 0);

  CheckFloatToFixedConversion(Val, getSFractSema(), 0);
  CheckFloatToFixedConversion(Val, getFractSema(),  0);
  CheckFloatToFixedConversion(Val, getLFractSema(), 0);

  CheckFloatToFixedConversion(Val, getUSFractSema(), 0);
  CheckFloatToFixedConversion(Val, getUFractSema(),  0);
  CheckFloatToFixedConversion(Val, getULFractSema(), 0);

  CheckFloatToFixedConversion(Val, getU8Neg10(), 0);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), 0);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  // Half conversion
  Val = APFloat(0.99951171875f);
  bool Ignored;
  Val.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);

  CheckFloatToFixedConversion(Val, getSAccumSema(), 0x7FULL);
  CheckFloatToFixedConversion(Val, getAccumSema(),  0x7FFULL << 4);
  CheckFloatToFixedConversion(Val, getLAccumSema(), 0x7FFULL << 20);

  CheckFloatToFixedConversion(Val, getUSAccumSema(), 0xFFULL);
  CheckFloatToFixedConversion(Val, getUAccumSema(),  0xFFEULL << 4);
  CheckFloatToFixedConversion(Val, getULAccumSema(), 0xFFEULL << 20);

  CheckFloatToFixedConversion(Val, getSFractSema(), 0x7FULL);
  CheckFloatToFixedConversion(Val, getFractSema(),  0x7FFULL << 4);
  CheckFloatToFixedConversion(Val, getLFractSema(), 0x7FFULL << 20);

  CheckFloatToFixedConversion(Val, getUSFractSema(), 0xFFULL);
  CheckFloatToFixedConversion(Val, getUFractSema(),  0xFFEULL << 4);
  CheckFloatToFixedConversion(Val, getULFractSema(), 0xFFEULL << 20);

  CheckFloatToFixedConversion(Val, getU8Neg10(), MaxSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), MaxSat);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  Val = APFloat(0.124996185302734375);
  CheckFloatToFixedConversion(Val, getU8Neg10(), 0x7f);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), 0x7fff);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);

  Val = APFloat(-0.124996185302734375);
  CheckFloatToFixedConversion(Val, getU8Neg10(), MinSat);
  CheckFloatToFixedConversion(Val, getU8Pos4(), 0);
  CheckFloatToFixedConversion(Val, getS16Neg18(), -0x7fff);
  CheckFloatToFixedConversion(Val, getS32Pos2(), 0);
}

void CheckFixedToFloatConversion(int64_t Val, const FixedPointSemantics &Sema,
                                 float Result) {
  APFixedPoint FXVal(Val, Sema);
  APFloat APRes(Result);
  ASSERT_EQ(FXVal.convertToFloat(APFloat::IEEEsingle()), APRes);
}

void CheckFixedToHalfConversion(int64_t Val, const FixedPointSemantics &Sema,
                                float Result) {
  APFixedPoint FXVal(Val, Sema);
  APFloat APRes(Result);
  bool Ignored;
  APRes.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Ignored);
  ASSERT_EQ(FXVal.convertToFloat(APFloat::IEEEhalf()), APRes);
}

TEST(FixedPoint, FixedToFloat) {
  int64_t Val = 0x1ULL;
  CheckFixedToFloatConversion(Val, getSAccumSema(), 0.0078125f);
  CheckFixedToFloatConversion(Val, getFractSema(),  0.000030517578125f);
  CheckFixedToFloatConversion(Val, getAccumSema(),  0.000030517578125f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              0.0000000004656612873077392578125f);

  CheckFixedToFloatConversion(Val, getUSAccumSema(), 0.00390625f);
  CheckFixedToFloatConversion(Val, getUFractSema(),  0.0000152587890625f);
  CheckFixedToFloatConversion(Val, getUAccumSema(),  0.0000152587890625f);
  CheckFixedToFloatConversion(Val, getULFractSema(),
                              0.00000000023283064365386962890625f);

  CheckFixedToFloatConversion(Val, getU8Neg10(), 0.0009765625f);
  CheckFixedToFloatConversion(Val, getU8Pos4(), 16.0f);
  CheckFixedToFloatConversion(Val, getS16Neg18(), 0.000003814697265625f);
  CheckFixedToFloatConversion(Val, getS32Pos2(), 4.0f);

  Val = 0x7FULL;
  CheckFixedToFloatConversion(Val, getSAccumSema(), 0.9921875f);
  CheckFixedToFloatConversion(Val, getFractSema(),  0.003875732421875f);
  CheckFixedToFloatConversion(Val, getAccumSema(),  0.003875732421875f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              0.0000000591389834880828857421875f);

  CheckFixedToFloatConversion(Val, getUSAccumSema(), 0.49609375f);
  CheckFixedToFloatConversion(Val, getUFractSema(),  0.0019378662109375f);
  CheckFixedToFloatConversion(Val, getUAccumSema(),  0.0019378662109375f);
  CheckFixedToFloatConversion(Val, getULFractSema(),
                              0.00000002956949174404144287109375f);

  CheckFixedToFloatConversion(Val, getU8Neg10(), 0.1240234375f);
  CheckFixedToFloatConversion(Val, getU8Pos4(), 2032.0f);
  CheckFixedToFloatConversion(Val, getS16Neg18(), 0.000484466552734375f);
  CheckFixedToFloatConversion(Val, getS32Pos2(), 508.0f);

  Val = -0x1ULL;
  CheckFixedToFloatConversion(Val, getSAccumSema(), -0.0078125f);
  CheckFixedToFloatConversion(Val, getFractSema(),  -0.000030517578125f);
  CheckFixedToFloatConversion(Val, getAccumSema(),  -0.000030517578125f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              -0.0000000004656612873077392578125f);

  CheckFixedToFloatConversion(Val, getU8Neg10(), 0.249023437f);
  CheckFixedToFloatConversion(Val, getU8Pos4(), 4080.0f);
  CheckFixedToFloatConversion(Val, getS16Neg18(), -0.000003814697265625f);
  CheckFixedToFloatConversion(Val, getS32Pos2(), -4.0f);

  CheckFixedToFloatConversion(-0x80ULL,       getSAccumSema(), -1.0f);
  CheckFixedToFloatConversion(-0x8000ULL,     getFractSema(),  -1.0f);
  CheckFixedToFloatConversion(-0x8000ULL,     getAccumSema(),  -1.0f);
  CheckFixedToFloatConversion(-0x80000000ULL, getLFractSema(), -1.0f);

  Val = 0xAFAULL;
  CheckFixedToFloatConversion(Val, getSAccumSema(), 21.953125f);
  CheckFixedToFloatConversion(Val, getFractSema(),  0.08575439453125f);
  CheckFixedToFloatConversion(Val, getAccumSema(),  0.08575439453125f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              0.000001308508217334747314453125f);

  CheckFixedToFloatConversion(Val, getUSAccumSema(), 10.9765625f);
  CheckFixedToFloatConversion(Val, getUFractSema(),  0.042877197265625f);
  CheckFixedToFloatConversion(Val, getUAccumSema(),  0.042877197265625f);
  CheckFixedToFloatConversion(Val, getULFractSema(),
                              0.0000006542541086673736572265625f);

  CheckFixedToFloatConversion(Val, getS16Neg18(), 0.01071929931640625f);
  CheckFixedToFloatConversion(Val, getS32Pos2(), 11240.0f);

  Val = -0xAFAULL;
  CheckFixedToFloatConversion(Val, getSAccumSema(), -21.953125f);
  CheckFixedToFloatConversion(Val, getFractSema(),  -0.08575439453125f);
  CheckFixedToFloatConversion(Val, getAccumSema(),  -0.08575439453125f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              -0.000001308508217334747314453125f);

  CheckFixedToFloatConversion(Val, getS16Neg18(), -0.01071929931640625f);
  CheckFixedToFloatConversion(Val, getS32Pos2(), -11240.0f);

  Val = 0x40000080ULL;
  CheckFixedToFloatConversion(Val, getAccumSema(),  32768.00390625f);
  CheckFixedToFloatConversion(Val, getLFractSema(),
                              0.500000059604644775390625f);

  CheckFixedToFloatConversion(Val, getUAccumSema(),  16384.001953125f);
  CheckFixedToFloatConversion(Val, getULFractSema(),
                              0.2500000298023223876953125f);

  CheckFixedToFloatConversion(Val, getS32Pos2(), 4294967808.0f);

  Val = 0x40000040ULL;
  CheckFixedToFloatConversion(Val, getAccumSema(),  32768.0f);
  CheckFixedToFloatConversion(Val, getLFractSema(), 0.5f);

  CheckFixedToFloatConversion(Val, getUAccumSema(),  16384.0f);
  CheckFixedToFloatConversion(Val, getULFractSema(), 0.25f);

  CheckFixedToFloatConversion(Val, getS32Pos2(), 4294967552.0f);

  Val = 0x7FF0ULL;
  CheckFixedToHalfConversion(Val, getAccumSema(), 0.99951171875f);
  CheckFixedToHalfConversion(Val, getLFractSema(), 0.000015251338481903076171875f);

  CheckFixedToHalfConversion(Val, getUAccumSema(), 0.499755859375f);
  CheckFixedToHalfConversion(Val, getULFractSema(), 0.0000076256692409515380859375f);

  CheckFixedToFloatConversion(Val, getS32Pos2(), 131008.0f);
}

void CheckAdd(const APFixedPoint &Lhs, const APFixedPoint &Rhs,
              const APFixedPoint &Res) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.add(Rhs, &Overflow);
  ASSERT_FALSE(Overflow);
  ASSERT_EQ(Result.getSemantics(), Res.getSemantics());
  ASSERT_EQ(Result, Res);
}

void CheckAddOverflow(const APFixedPoint &Lhs, const APFixedPoint &Rhs) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.add(Rhs, &Overflow);
  ASSERT_TRUE(Overflow);
}

TEST(FixedPoint, add) {
  CheckAdd(APFixedPoint(1, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(2, getS32Pos2()));
  CheckAdd(APFixedPoint(1, getS16Neg18()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(2, getS16Neg18()));
  CheckAdd(APFixedPoint(1, getU8Neg10()), APFixedPoint(1, getU8Neg10()),
           APFixedPoint(2, getU8Neg10()));
  CheckAdd(APFixedPoint(1, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(2, getU8Pos4()));

  CheckAdd(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(12, getS32Pos2()));
  CheckAdd(APFixedPoint(11, getS16Neg18()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(12, getS16Neg18()));
  CheckAdd(APFixedPoint(11, getU8Neg10()), APFixedPoint(1, getU8Neg10()),
           APFixedPoint(12, getU8Neg10()));
  CheckAdd(APFixedPoint(11, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(12, getU8Pos4()));

  CheckAdd(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(11534337,
                        FixedPointSemantics(52, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckAdd(
      APFixedPoint(11, getU8Neg10()), APFixedPoint(-9472, getS16Neg18()),
      APFixedPoint(-6656, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                              true, false, false)));
  CheckAddOverflow(
      APFixedPoint::getMax(getU8Neg10()), APFixedPoint::getMax(getS16Neg18()));
  CheckAdd(APFixedPoint::getMin(getU8Neg10()),
           APFixedPoint::getMin(getS16Neg18()),
           APFixedPoint::getMin(getS16Neg18())
               .convert(FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckAddOverflow(APFixedPoint::getMin(getS32Pos2()),
                   APFixedPoint::getMin(getS16Neg18()));
}

void CheckMul(const APFixedPoint &Lhs, const APFixedPoint &Rhs,
              const APFixedPoint &Res) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.mul(Rhs, &Overflow);
  ASSERT_FALSE(Overflow);
  ASSERT_EQ(Result.getSemantics(), Res.getSemantics());
  ASSERT_EQ(Result, Res);
}

void CheckMulOverflow(const APFixedPoint &Lhs, const APFixedPoint &Rhs) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.mul(Rhs, &Overflow);
  ASSERT_TRUE(Overflow);
}

TEST(FixedPoint, mul) {
  CheckMul(APFixedPoint(1, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(4, getS32Pos2()));
  CheckMul(APFixedPoint(1, getS16Neg18()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(0, getS16Neg18()));
  CheckMul(APFixedPoint(1, getU8Neg10()), APFixedPoint(1, getU8Neg10()),
           APFixedPoint(0, getU8Neg10()));
  CheckMul(APFixedPoint(1, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(16, getU8Pos4()));

  CheckMul(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(44, getS32Pos2()));
  CheckMul(APFixedPoint(11, getS16Neg18()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(0, getS16Neg18()));
  CheckMul(APFixedPoint(11, getU8Neg10()), APFixedPoint(1, getU8Neg10()),
           APFixedPoint(0, getU8Neg10()));
  CheckMul(APFixedPoint(11, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(176, getU8Pos4()));

  CheckMul(APFixedPoint(512, getS16Neg18()), APFixedPoint(512, getS16Neg18()),
           APFixedPoint(1, getS16Neg18()));
  CheckMul(APFixedPoint(32, getU8Neg10()), APFixedPoint(32, getU8Neg10()),
           APFixedPoint(1, getU8Neg10()));

  CheckMul(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(44,
                        FixedPointSemantics(52, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckMul(
      APFixedPoint(11, getU8Neg10()), APFixedPoint(-9472, getS16Neg18()),
      APFixedPoint(-102, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                             true, false, false)));
  CheckMul(
      APFixedPoint::getMax(getU8Neg10()), APFixedPoint::getMax(getS16Neg18()),
      APFixedPoint(8159, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                             true, false, false)));
  CheckMul(
      APFixedPoint::getMin(getU8Neg10()), APFixedPoint::getMin(getS16Neg18()),
      APFixedPoint(0, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                          true, false, false)));
  CheckMul(APFixedPoint::getMin(getS32Pos2()),
           APFixedPoint::getMin(getS16Neg18()),
           APFixedPoint(281474976710656,
                        FixedPointSemantics(52, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckMulOverflow(APFixedPoint::getMax(getS32Pos2()), APFixedPoint::getMax(getU8Pos4()));
  CheckMulOverflow(APFixedPoint::getMin(getS32Pos2()), APFixedPoint::getMax(getU8Pos4()));
}

void CheckDiv(const APFixedPoint &Lhs, const APFixedPoint &Rhs,
              const APFixedPoint &Expected) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.div(Rhs, &Overflow);
  ASSERT_FALSE(Overflow);
  ASSERT_EQ(Result.getSemantics(), Expected.getSemantics());
  ASSERT_EQ(Result, Expected);
}

void CheckDivOverflow(const APFixedPoint &Lhs, const APFixedPoint &Rhs) {
  bool Overflow = false;
  APFixedPoint Result = Lhs.div(Rhs, &Overflow);
  ASSERT_TRUE(Overflow);
}

TEST(FixedPoint, div) {
  CheckDiv(APFixedPoint(1, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(0, getS32Pos2()));
  CheckDivOverflow(APFixedPoint(1, getS16Neg18()), APFixedPoint(1, getS16Neg18()));
  CheckDivOverflow(APFixedPoint(1, getU8Neg10()), APFixedPoint(1, getU8Neg10()));
  CheckDiv(APFixedPoint(1, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(0, getU8Pos4()));

  CheckDiv(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS32Pos2()),
           APFixedPoint(2, getS32Pos2()));
  CheckDiv(APFixedPoint(11, getU8Pos4()), APFixedPoint(1, getU8Pos4()),
           APFixedPoint(0, getU8Pos4()));

  CheckDiv(APFixedPoint(11, getS32Pos2()), APFixedPoint(1, getS16Neg18()),
           APFixedPoint(3023656976384,
                        FixedPointSemantics(52, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckDiv(APFixedPoint(11, getU8Neg10()), APFixedPoint(-11264, getS16Neg18()),
           APFixedPoint::getMin(FixedPointSemantics(
               17, FixedPointSemantics::Lsb{-18}, true, false, false)));
  CheckDiv(APFixedPoint(11, getU8Neg10()), APFixedPoint(11265, getS16Neg18()),
           APFixedPoint(0xfffa,
                        FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                            true, false, false)));
  CheckDivOverflow(APFixedPoint(11, getU8Neg10()),
                   APFixedPoint(11264, getS16Neg18()));

  CheckDivOverflow(APFixedPoint(11, getU8Neg10()),
                   APFixedPoint(-9472, getS16Neg18()));
  CheckDivOverflow(APFixedPoint::getMax(getU8Neg10()),
                   APFixedPoint::getMax(getS16Neg18()));
  CheckDiv(
      APFixedPoint::getMin(getU8Neg10()), APFixedPoint::getMin(getS16Neg18()),
      APFixedPoint(0, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                          true, false, false)));
  CheckDiv(
      APFixedPoint(1, getU8Neg10()), APFixedPoint::getMin(getS16Neg18()),
      APFixedPoint(-2048, FixedPointSemantics(17, FixedPointSemantics::Lsb{-18},
                                              true, false, false)));
}

TEST(FixedPoint, semanticsSerialization) {
  auto roundTrip = [](FixedPointSemantics FPS) -> bool {
    uint32_t I = FPS.toOpaqueInt();
    FixedPointSemantics FPS2 = FixedPointSemantics::getFromOpaqueInt(I);
    return FPS == FPS2;
  };

  ASSERT_TRUE(roundTrip(getS32Pos2()));
  ASSERT_TRUE(roundTrip(getU8Pos4()));
  ASSERT_TRUE(roundTrip(getS16Neg18()));
  ASSERT_TRUE(roundTrip(getU8Neg10()));
  ASSERT_TRUE(roundTrip(getPadULFractSema()));
  ASSERT_TRUE(roundTrip(getPadUFractSema()));
  ASSERT_TRUE(roundTrip(getPadUSFractSema()));
  ASSERT_TRUE(roundTrip(getPadULAccumSema()));
  ASSERT_TRUE(roundTrip(getPadUAccumSema()));
  ASSERT_TRUE(roundTrip(getPadUSAccumSema()));
  ASSERT_TRUE(roundTrip(getULFractSema()));
  ASSERT_TRUE(roundTrip(getUFractSema()));
  ASSERT_TRUE(roundTrip(getUSFractSema()));
  ASSERT_TRUE(roundTrip(getULAccumSema()));
  ASSERT_TRUE(roundTrip(getUAccumSema()));
  ASSERT_TRUE(roundTrip(getUSAccumSema()));
  ASSERT_TRUE(roundTrip(getLFractSema()));
  ASSERT_TRUE(roundTrip(getFractSema()));
  ASSERT_TRUE(roundTrip(getSFractSema()));
  ASSERT_TRUE(roundTrip(getLAccumSema()));
  ASSERT_TRUE(roundTrip(getAccumSema()));
  ASSERT_TRUE(roundTrip(getSAccumSema()));
}

} // namespace
