//===- InferIntRangeInterfaceTest.cpp - Unit Tests for InferIntRange... --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/APInt.h"
#include <array>
#include <limits>

#include <gtest/gtest.h>

using namespace mlir;

TEST(IntRangeInference, AffineDivisionIndexClamping) {
  MLIRContext context;
  AffineExpr lhs = getAffineDimExpr(0, &context);
  AffineExpr rhs = getAffineSymbolExpr(0, &context);
  auto full = ConstantIntRanges::maxRange(64);
  auto poison = ConstantIntRanges::empty(64);
  auto negative = ConstantIntRanges::constant(APInt(64, -4294967295LL, true));
  for (AffineExpr expr : {lhs.floorDiv(rhs), lhs.ceilDiv(rhs)}) {
    EXPECT_EQ(intrange::inferAffineExpr(expr, {full}, {negative}), full);
    EXPECT_EQ(intrange::inferAffineExpr(expr, {poison}, {negative}), poison);
    EXPECT_EQ(intrange::inferAffineExpr(expr, {full}, {poison}), poison);
  }
}

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

TEST(IntRangeInference, ZeroDivisor) {
  using BinaryFn = ConstantIntRanges (*)(ArrayRef<ConstantIntRanges>);
  BinaryFn functions[] = {intrange::inferDivS,      intrange::inferDivU,
                          intrange::inferCeilDivS,  intrange::inferCeilDivU,
                          intrange::inferFloorDivS, intrange::inferRemS,
                          intrange::inferRemU};
  for (unsigned width : {1, 4, 32, 64, 129}) {
    auto zero = ConstantIntRanges::constant(APInt::getZero(width));
    auto empty = ConstantIntRanges::empty(width);
    for (const auto &lhs :
         {ConstantIntRanges::maxRange(width), zero,
          ConstantIntRanges::constant(APInt::getSignedMinValue(width))}) {
      for (const auto &rhs : {zero, empty}) {
        std::array operands{lhs, rhs};
        for (BinaryFn fn : functions)
          EXPECT_EQ(fn(operands), empty);
      }
    }
  }
}

TEST(IntRangeInference, ZeroAndNonzeroDivisor) {
  for (unsigned width : {2, 4, 32, 64, 129}) {
    auto one = ConstantIntRanges::constant(APInt(width, 1));
    auto zero = ConstantIntRanges::constant(APInt(width, 0));
    std::array operands{one, zero.rangeUnion(one)};
    EXPECT_EQ(intrange::inferDivS(operands), one);
    EXPECT_EQ(intrange::inferDivU(operands), one);
    EXPECT_EQ(intrange::inferCeilDivS(operands), one);
    EXPECT_EQ(intrange::inferCeilDivU(operands), one);
    EXPECT_EQ(intrange::inferFloorDivS(operands), one);
    EXPECT_EQ(intrange::inferRemS(operands), zero);
    EXPECT_EQ(intrange::inferRemU(operands), zero);
  }
}

TEST(IntRangeInference, IndexZeroDivisor) {
  using BinaryFn = ConstantIntRanges (*)(ArrayRef<ConstantIntRanges>);
  BinaryFn functions[] = {intrange::inferDivS,      intrange::inferDivU,
                          intrange::inferCeilDivS,  intrange::inferCeilDivU,
                          intrange::inferFloorDivS, intrange::inferRemS,
                          intrange::inferRemU};
  auto lhs = ConstantIntRanges::constant(APInt(64, 7));
  std::array zeroOperands{lhs, ConstantIntRanges::constant(APInt(64, 0))};
  std::array zeroOnlyAt32{lhs,
                          ConstantIntRanges::constant(APInt(64, 1ULL << 32))};
  for (BinaryFn fn : functions) {
    EXPECT_EQ(intrange::inferIndexOp(fn, zeroOperands, intrange::CmpMode::Both),
              ConstantIntRanges::empty(64));
    EXPECT_EQ(intrange::inferIndexOp(fn, zeroOnlyAt32, intrange::CmpMode::Both),
              fn(zeroOnlyAt32));
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

TEST(IntRangeAttrs, InitializedEmptyRanges) {
  IntegerValueRange pending;
  EXPECT_TRUE(pending.isUninitialized());

  for (unsigned width : {1, 8, 64, 129}) {
    auto empty = ConstantIntRanges::empty(width);
    auto full = ConstantIntRanges::maxRange(width);
    auto zero = ConstantIntRanges::constant(APInt(width, 0));
    EXPECT_FALSE(IntegerValueRange(empty).isUninitialized());
    EXPECT_TRUE(empty.isEmpty());
    EXPECT_TRUE(empty.isSignedEmpty());
    EXPECT_TRUE(empty.isUnsignedEmpty());
    EXPECT_FALSE(empty.hasBounds());
    EXPECT_EQ(empty.getBitWidth(), width);
    EXPECT_TRUE(full.hasBounds());
    EXPECT_FALSE(empty.getConstantValue());
    EXPECT_EQ(empty.rangeUnion(zero), zero);
    EXPECT_EQ(zero.rangeUnion(empty), zero);
    EXPECT_EQ(empty.intersection(full), empty);
    EXPECT_EQ(full.intersection(empty), empty);
    EXPECT_EQ(IntegerValueRange::join(pending, IntegerValueRange(empty)),
              IntegerValueRange(empty));
  }
}

TEST(IntRangeAttrs, EmptyIntersection) {
  for (unsigned width : {1, 8, 64, 129}) {
    APInt zero(width, 0), one(width, 1);
    auto empty = ConstantIntRanges::empty(width);
    EXPECT_EQ(ConstantIntRanges::fromUnsigned(one, zero), empty);
    EXPECT_EQ(ConstantIntRanges::fromSigned(zero, APInt::getAllOnes(width)),
              empty);
    EXPECT_EQ(ConstantIntRanges(one, zero, zero, zero), empty);
    EXPECT_EQ(ConstantIntRanges(zero, zero, zero, APInt::getAllOnes(width)),
              empty);
    EXPECT_EQ(ConstantIntRanges::constant(zero).intersection(
                  ConstantIntRanges::constant(one)),
              empty);
    APInt unsignedMax = APInt::getMaxValue(width);
    APInt signedMin = APInt::getSignedMinValue(width);
    APInt signedMax = APInt::getSignedMaxValue(width);
    ConstantIntRanges negative(zero, unsignedMax, signedMin,
                               APInt::getAllOnes(width));
    ConstantIntRanges nonnegative(zero, unsignedMax, zero, signedMax);
    EXPECT_EQ(negative.intersection(nonnegative), empty);
    ConstantIntRanges unsignedZero(zero, zero, signedMin, signedMax);
    ConstantIntRanges unsignedNonzero(one, unsignedMax, signedMin, signedMax);
    EXPECT_EQ(unsignedZero.intersection(unsignedNonzero), empty);
  }
}

TEST(IntRangeInference, EmptyInputs) {
  using BinaryFn = ConstantIntRanges (*)(ArrayRef<ConstantIntRanges>);
  BinaryFn functions[] = {intrange::inferDivS,      intrange::inferDivU,
                          intrange::inferCeilDivS,  intrange::inferCeilDivU,
                          intrange::inferFloorDivS, intrange::inferRemS,
                          intrange::inferRemU,      intrange::inferMaxS,
                          intrange::inferMaxU,      intrange::inferMinS,
                          intrange::inferMinU,      intrange::inferAnd,
                          intrange::inferOr,        intrange::inferXor,
                          intrange::inferShrS,      intrange::inferShrU};
  for (unsigned width : {1, 8, 64, 129}) {
    auto empty = ConstantIntRanges::empty(width);
    auto zero = ConstantIntRanges::constant(APInt(width, 0));
    for (auto operands : {std::array{empty, zero}, std::array{zero, empty}}) {
      for (BinaryFn fn : functions)
        EXPECT_EQ(fn(operands), empty);
      EXPECT_EQ(intrange::inferAdd(operands), empty);
      EXPECT_EQ(intrange::inferSub(operands), empty);
      EXPECT_EQ(intrange::inferMul(operands), empty);
      EXPECT_EQ(intrange::inferShl(operands), empty);
      EXPECT_FALSE(intrange::evaluatePred(intrange::CmpPredicate::eq,
                                          operands[0], operands[1]));
    }
    EXPECT_EQ(intrange::extRange(empty, width + 1),
              ConstantIntRanges::empty(width + 1));
    EXPECT_EQ(intrange::extSIRange(empty, width + 1),
              ConstantIntRanges::empty(width + 1));
    EXPECT_EQ(intrange::extUIRange(empty, width + 1),
              ConstantIntRanges::empty(width + 1));
    if (width > 1)
      EXPECT_EQ(intrange::truncRange(empty, 1), ConstantIntRanges::empty(1));
  }
}

TEST(IntRangeInference, IndexExtensionContainsBothInterpretations) {
  auto minusOne = ConstantIntRanges::constant(APInt::getAllOnes(32));
  auto extended = intrange::extRange(minusOne, 64);
  ASSERT_TRUE(extended.hasBounds());
  EXPECT_EQ(extended.smin(), APInt(64, -1, true));
  EXPECT_EQ(extended.smax(), APInt(64, -1, true));
  EXPECT_EQ(extended.umin(), APInt(64, 4294967295));
  EXPECT_EQ(extended.umax(), APInt(64, 4294967295));
}

TEST(IntRangeInference, OverflowFlagsDefinedResults) {
  using namespace intrange;
  for (unsigned width = 1; width <= 3; ++width) {
    int limit = 1 << width;
    auto asSigned = [=](int x) { return x < limit / 2 ? x : x - limit; };
    for (int aMin = 0; aMin < limit; ++aMin)
      for (int aMax = aMin; aMax < limit; ++aMax)
        for (int bMin = 0; bMin < limit; ++bMin)
          for (int bMax = bMin; bMax < limit; ++bMax) {
            std::array operands{ConstantIntRanges::fromUnsigned(
                                    APInt(width, aMin), APInt(width, aMax)),
                                ConstantIntRanges::fromUnsigned(
                                    APInt(width, bMin), APInt(width, bMax))};
            for (auto flags :
                 {OverflowFlags::None, OverflowFlags::Nsw, OverflowFlags::Nuw,
                  OverflowFlags::Nsw | OverflowFlags::Nuw}) {
              std::array ranges{
                  inferAdd(operands, flags), inferSub(operands, flags),
                  inferMul(operands, flags), inferShl(operands, flags)};
              for (int a = aMin; a <= aMax; ++a)
                for (int b = bMin; b <= bMax; ++b) {
                  int sa = asSigned(a), sb = asSigned(b);
                  std::array signedResults{sa + sb, sa - sb, sa * sb,
                                           sa * (1 << b)};
                  std::array unsignedResults{a + b, a - b, a * b, a * (1 << b)};
                  for (unsigned op = 0; op < ranges.size(); ++op) {
                    if (op == 3 && unsigned(b) >= width)
                      continue;
                    int sr = signedResults[op], ur = unsignedResults[op];
                    if (any(flags & OverflowFlags::Nsw) &&
                        (sr < -limit / 2 || sr >= limit / 2))
                      continue;
                    if (any(flags & OverflowFlags::Nuw) &&
                        (ur < 0 || ur >= limit))
                      continue;
                    ASSERT_TRUE(ranges[op].hasBounds());
                    EXPECT_TRUE(
                        containsSigned(ranges[op], asSigned(ur & (limit - 1))));
                  }
                }
            }
          }
  }
}

TEST(IntRangeAttrs, IndependentIntervalEmptiness) {
  for (unsigned width = 1; width <= 3; ++width) {
    unsigned limit = 1u << width;
    for (unsigned umin = 0; umin < limit; ++umin)
      for (unsigned umax = 0; umax < limit; ++umax)
        for (unsigned smin = 0; smin < limit; ++smin)
          for (unsigned smax = 0; smax < limit; ++smax) {
            APInt lo(width, smin), hi(width, smax);
            bool hasBounds = umin <= umax && lo.sle(hi);
            ConstantIntRanges range(APInt(width, umin), APInt(width, umax), lo,
                                    hi);
            EXPECT_EQ(range.hasBounds(), hasBounds);
            EXPECT_EQ(range.isEmpty(), !hasBounds);
            EXPECT_EQ(range.isSignedEmpty(), !hasBounds);
            EXPECT_EQ(range.isUnsignedEmpty(), !hasBounds);
          }
  }
}

TEST(IntRangeInference, ExtendIndependentIndexBounds) {
  auto negative =
      ConstantIntRanges::fromSigned(APInt(32, -4, true), APInt(32, -1, true));
  auto extended = intrange::extRange(negative, 64);
  ASSERT_TRUE(extended.hasBounds());
  EXPECT_EQ(extended.umin(), APInt(64, 4294967292ULL));
  EXPECT_EQ(extended.umax(), APInt(64, 4294967295ULL));
  EXPECT_EQ(extended.smin(), APInt(64, -4, true));
  EXPECT_EQ(extended.smax(), APInt(64, -1, true));
  EXPECT_EQ(intrange::truncRange(extended, 32), negative);
  EXPECT_EQ(extended.intersection(ConstantIntRanges::maxRange(64)), extended);
  EXPECT_EQ(extended.rangeUnion(ConstantIntRanges::empty(64)), extended);
}
