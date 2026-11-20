//===- InferIntRangeInterfaceTest.cpp - Unit Tests for InferIntRange... --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/APInt.h"
#include <array>
#include <limits>

#include <gtest/gtest.h>

using namespace mlir;

TEST(IntRangeAttrs, BasicConstructors) {
  APInt zero = APInt::getZero(64);
  APInt two(64, 2);
  APInt three(64, 3);
  ConstantIntRanges boundedAbove(zero, two, zero, three);
  EXPECT_EQ(boundedAbove.umin(), zero);
  EXPECT_EQ(boundedAbove.umax(), two);
  EXPECT_EQ(boundedAbove.smin(), zero);
  EXPECT_EQ(boundedAbove.smax(), three);
}

TEST(IntRangeAttrs, FromUnsigned) {
  APInt zero = APInt::getZero(64);
  APInt maxInt = APInt::getSignedMaxValue(64);
  APInt minInt = APInt::getSignedMinValue(64);
  APInt minIntPlusOne = minInt + 1;

  ConstantIntRanges canPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, maxInt);
  EXPECT_EQ(canPortToSigned.smin(), zero);
  EXPECT_EQ(canPortToSigned.smax(), maxInt);

  ConstantIntRanges cantPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, minInt);
  EXPECT_EQ(cantPortToSigned.smin(), minInt);
  EXPECT_EQ(cantPortToSigned.smax(), maxInt);

  ConstantIntRanges signedNegative =
      ConstantIntRanges::fromUnsigned(minInt, minIntPlusOne);
  EXPECT_EQ(signedNegative.smin(), minInt);
  EXPECT_EQ(signedNegative.smax(), minIntPlusOne);
}

TEST(IntRangeAttrs, FromSigned) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt negOne = zero - 1;
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt intMin = APInt::getSignedMinValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges noUnsignedBound =
      ConstantIntRanges::fromSigned(negOne, one);
  EXPECT_EQ(noUnsignedBound.umin(), zero);
  EXPECT_EQ(noUnsignedBound.umax(), uintMax);

  ConstantIntRanges positive = ConstantIntRanges::fromSigned(one, intMax);
  EXPECT_EQ(positive.umin(), one);
  EXPECT_EQ(positive.umax(), intMax);

  ConstantIntRanges negative = ConstantIntRanges::fromSigned(intMin, negOne);
  EXPECT_EQ(negative.umin(), intMin);
  EXPECT_EQ(negative.umax(), negOne);

  ConstantIntRanges preserved = ConstantIntRanges::fromSigned(zero, one);
  EXPECT_EQ(preserved.umin(), zero);
  EXPECT_EQ(preserved.umax(), one);
}

TEST(IntRangeAttrs, Join) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt two = zero + 2;
  APInt intMin = APInt::getSignedMinValue(64);
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges maximal(zero, uintMax, intMin, intMax);
  ConstantIntRanges zeroOne(zero, one, zero, one);

  EXPECT_EQ(zeroOne.rangeUnion(maximal), maximal);
  EXPECT_EQ(maximal.rangeUnion(zeroOne), maximal);

  EXPECT_EQ(zeroOne.rangeUnion(zeroOne), zeroOne);

  ConstantIntRanges oneTwo(one, two, one, two);
  ConstantIntRanges zeroTwo(zero, two, zero, two);
  EXPECT_EQ(zeroOne.rangeUnion(oneTwo), zeroTwo);

  ConstantIntRanges zeroOneUnsignedOnly(zero, one, intMin, intMax);
  ConstantIntRanges zeroOneSignedOnly(zero, uintMax, zero, one);
  EXPECT_EQ(zeroOneUnsignedOnly.rangeUnion(zeroOneSignedOnly), maximal);
}

static ConstantIntRanges signedRange(unsigned width, int64_t lo, int64_t hi) {
  return ConstantIntRanges::fromSigned(APInt(width, lo, true),
                                       APInt(width, hi, true));
}

TEST(IntRangeInference, SignedDivRemBounds) {
  auto check = [](int64_t lhsMin, int64_t lhsMax, int64_t rhsMin,
                  int64_t rhsMax, int64_t divMin, int64_t divMax,
                  int64_t remMin, int64_t remMax) {
    std::array operands{signedRange(32, lhsMin, lhsMax),
                        signedRange(32, rhsMin, rhsMax)};
    EXPECT_EQ(intrange::inferDivS(operands), signedRange(32, divMin, divMax));
    EXPECT_EQ(intrange::inferRemS(operands), signedRange(32, remMin, remMax));
  };
  check(0, 127, 0, 16, 0, 127, 0, 15);
  check(-127, 0, 0, 16, -127, 0, -15, 0);
  check(0, 127, -16, 0, -127, 0, 0, 15);
  check(-127, 0, -16, 0, 0, 127, -15, 0);
  check(-127, 127, -16, 16, -127, 127, -15, 15);
  check(0, 7, 0, 1024, 0, 7, 0, 7);
  check(-7, 0, -1024, 0, 0, 7, -7, 0);
  check(-7, 7, -1024, 1024, -7, 7, -7, 7);
  check(0, 7, 1, 1024, 0, 7, 0, 7);
  check(-7, 0, -1024, -1, 0, 7, -7, 0);
}

static bool containsSigned(const ConstantIntRanges &range, int64_t value) {
  APInt v(range.smin().getBitWidth(), value, true);
  return range.smin().sle(v) && range.smax().sge(v) && range.umin().ule(v) &&
         range.umax().uge(v);
}

static void checkSignedDivRem(unsigned width, int64_t lhsMin, int64_t lhsMax,
                              int64_t rhsMin, int64_t rhsMax) {
  std::array operands{signedRange(width, lhsMin, lhsMax),
                      signedRange(width, rhsMin, rhsMax)};
  ConstantIntRanges div = intrange::inferDivS(operands);
  ConstantIntRanges rem = intrange::inferRemS(operands);
  for (int64_t lhs = lhsMin; lhs <= lhsMax; ++lhs) {
    for (int64_t rhs = rhsMin; rhs <= rhsMax; ++rhs) {
      if (rhs == 0)
        continue;
      // Signed division overflow produces poison.
      if (!(lhs == -(int64_t{1} << (width - 1)) && rhs == -1)) {
        ASSERT_TRUE(containsSigned(div, lhs / rhs))
            << width << ": [" << lhsMin << ", " << lhsMax << "] / [" << rhsMin
            << ", " << rhsMax << "] at " << lhs << ", " << rhs;
      }
      ASSERT_TRUE(containsSigned(rem, lhs % rhs))
          << width << ": [" << lhsMin << ", " << lhsMax << "] % [" << rhsMin
          << ", " << rhsMax << "] at " << lhs << ", " << rhs;
    }
  }
}

TEST(IntRangeInference, SignedDivRemExhaustive) {
  for (unsigned width = 1; width <= 4; ++width) {
    int64_t min = -(int64_t{1} << (width - 1));
    int64_t max = -min - 1;
    for (int64_t lhsMin = min; lhsMin <= max; ++lhsMin)
      for (int64_t lhsMax = lhsMin; lhsMax <= max; ++lhsMax)
        for (int64_t rhsMin = min; rhsMin <= max; ++rhsMin)
          for (int64_t rhsMax = rhsMin; rhsMax <= max; ++rhsMax)
            checkSignedDivRem(width, lhsMin, lhsMax, rhsMin, rhsMax);
  }
}

TEST(IntRangeInference, SignedDivRemPoisonDivisor) {
  for (unsigned width : {1, 4, 32, 64}) {
    ConstantIntRanges lhs = ConstantIntRanges::maxRange(width);
    for (ConstantIntRanges rhs :
         {signedRange(width, 0, 0), signedRange(width, 0, -1)}) {
      std::array operands{lhs, rhs};
      EXPECT_EQ(intrange::inferDivS(operands), lhs);
      EXPECT_EQ(intrange::inferRemS(operands), lhs);
    }
  }
}

TEST(IntRangeInference, RoundedSignedDivZeroEndpoint) {
  std::array positive{signedRange(32, 1, 7), signedRange(32, 0, 3)};
  EXPECT_EQ(intrange::inferCeilDivS(positive), signedRange(32, 1, 7));
  EXPECT_EQ(intrange::inferFloorDivS(positive), signedRange(32, 0, 7));
  std::array negative{signedRange(32, -7, -1), signedRange(32, 0, 3)};
  EXPECT_EQ(intrange::inferCeilDivS(negative), signedRange(32, -7, 0));
  EXPECT_EQ(intrange::inferFloorDivS(negative), signedRange(32, -7, -1));
}

TEST(IntRangeInference, SignedRemMinDivisor) {
  for (unsigned width : {1, 4, 32, 64}) {
    APInt min = APInt::getSignedMinValue(width);
    APInt zero = APInt::getZero(width);
    std::array operands{ConstantIntRanges::fromSigned(min, zero),
                        ConstantIntRanges::fromSigned(min, zero)};
    EXPECT_EQ(intrange::inferRemS(operands),
              ConstantIntRanges::fromSigned(min + 1, zero));
  }
}

static void checkRoundedSignedDiv(unsigned width, int64_t lhsMin,
                                  int64_t lhsMax, int64_t rhsMin,
                                  int64_t rhsMax) {
  std::array operands{signedRange(width, lhsMin, lhsMax),
                      signedRange(width, rhsMin, rhsMax)};
  ConstantIntRanges ceil = intrange::inferCeilDivS(operands);
  ConstantIntRanges floor = intrange::inferFloorDivS(operands);
  for (int64_t lhs = lhsMin; lhs <= lhsMax; ++lhs) {
    for (int64_t rhs = rhsMin; rhs <= rhsMax; ++rhs) {
      if (rhs == 0 || (lhs == -(int64_t{1} << (width - 1)) && rhs == -1))
        continue;
      int64_t quotient = lhs / rhs;
      int64_t remainder = lhs % rhs;
      int64_t ceilValue = quotient + (remainder != 0 && (lhs < 0) == (rhs < 0));
      // Index folding and lowering negate INT_MIN before division.
      if (lhs == -(int64_t{1} << (width - 1)) && rhs > 1)
        ceilValue = -quotient;
      int64_t floorValue =
          quotient - (remainder != 0 && (lhs < 0) != (rhs < 0));
      ASSERT_TRUE(containsSigned(ceil, ceilValue))
          << width << ": [" << lhsMin << ", " << lhsMax << "] ceildiv ["
          << rhsMin << ", " << rhsMax << "] at " << lhs << ", " << rhs;
      ASSERT_TRUE(containsSigned(floor, floorValue))
          << width << ": [" << lhsMin << ", " << lhsMax << "] floordiv ["
          << rhsMin << ", " << rhsMax << "] at " << lhs << ", " << rhs;
    }
  }
}

TEST(IntRangeInference, RoundedSignedDivExhaustive) {
  for (unsigned width = 1; width <= 4; ++width) {
    int64_t min = -(int64_t{1} << (width - 1));
    int64_t max = -min - 1;
    for (int64_t lhsMin = min; lhsMin <= max; ++lhsMin)
      for (int64_t lhsMax = lhsMin; lhsMax <= max; ++lhsMax)
        for (int64_t rhsMin = min; rhsMin <= max; ++rhsMin)
          for (int64_t rhsMax = rhsMin; rhsMax <= max; ++rhsMax)
            checkRoundedSignedDiv(width, lhsMin, lhsMax, rhsMin, rhsMax);
  }
}

TEST(IntRangeInference, SignedCeilDivMinDividend) {
  for (unsigned width : {4, 32, 64}) {
    APInt min = APInt::getSignedMinValue(width);
    APInt two(width, 2);
    std::array singleton{ConstantIntRanges::constant(min),
                         ConstantIntRanges::constant(two)};
    EXPECT_EQ(intrange::inferCeilDivS(singleton),
              ConstantIntRanges::constant(-min.sdiv(two)));

    for (int64_t lower : {0, 1}) {
      std::array divisors{ConstantIntRanges::constant(min),
                          signedRange(width, lower, 3)};
      EXPECT_EQ(intrange::inferCeilDivS(divisors),
                ConstantIntRanges::fromSigned(min, -min.sdiv(two)));
    }

    std::array interval{ConstantIntRanges::fromSigned(min, min + 1),
                        ConstantIntRanges::constant(two)};
    EXPECT_EQ(
        intrange::inferCeilDivS(interval),
        ConstantIntRanges::fromSigned((min + 1).sdiv(two), -min.sdiv(two)));
  }
}
