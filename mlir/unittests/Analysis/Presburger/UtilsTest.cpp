//===- Utils.cpp - Tests for Utils file ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>

using namespace mlir;
using namespace presburger;

static DivisionRepr
parseDivisionRepr(unsigned numVars, unsigned numDivs,
                  ArrayRef<ArrayRef<DynamicAPInt>> dividends,
                  ArrayRef<DynamicAPInt> divisors) {
  DivisionRepr repr(numVars, numDivs);
  for (unsigned i = 0, rows = dividends.size(); i < rows; ++i)
    repr.setDiv(i, dividends[i], divisors[i]);
  return repr;
}

static void checkEqual(DivisionRepr &a, DivisionRepr &b) {
  EXPECT_EQ(a.getNumVars(), b.getNumVars());
  EXPECT_EQ(a.getNumDivs(), b.getNumDivs());
  for (unsigned i = 0, rows = a.getNumDivs(); i < rows; ++i) {
    EXPECT_EQ(a.hasRepr(i), b.hasRepr(i));
    if (!a.hasRepr(i))
      continue;
    EXPECT_TRUE(a.getDenom(i) == b.getDenom(i));
    EXPECT_TRUE(a.getDividend(i).equals(b.getDividend(i)));
  }
}

TEST(UtilsTest, ParseAndCompareDivisionReprTest) {
  auto merge = [](unsigned i, unsigned j) -> bool { return true; };
  DivisionRepr a = parseDivisionRepr(1, 1, {{DynamicAPInt(1), DynamicAPInt(2)}},
                                     {DynamicAPInt(2)}),
               b = parseDivisionRepr(1, 1, {{DynamicAPInt(1), DynamicAPInt(2)}},
                                     {DynamicAPInt(2)}),
               c = parseDivisionRepr(
                   2, 2,
                   {{DynamicAPInt(0), DynamicAPInt(1), DynamicAPInt(2)},
                    {DynamicAPInt(0), DynamicAPInt(1), DynamicAPInt(2)}},
                   {DynamicAPInt(2), DynamicAPInt(2)});
  c.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(a, c);
}

TEST(UtilsTest, DivisionReprNormalizeTest) {
  auto merge = [](unsigned i, unsigned j) -> bool { return true; };
  DivisionRepr a = parseDivisionRepr(
                   2, 1, {{DynamicAPInt(1), DynamicAPInt(2), DynamicAPInt(-1)}},
                   {DynamicAPInt(2)}),
               b = parseDivisionRepr(
                   2, 1,
                   {{DynamicAPInt(16), DynamicAPInt(32), DynamicAPInt(-16)}},
                   {DynamicAPInt(32)}),
               c = parseDivisionRepr(1, 1,
                                     {{DynamicAPInt(12), DynamicAPInt(-4)}},
                                     {DynamicAPInt(8)}),
               d = parseDivisionRepr(
                   2, 2,
                   {{DynamicAPInt(1), DynamicAPInt(2), DynamicAPInt(-1)},
                    {DynamicAPInt(4), DynamicAPInt(8), DynamicAPInt(-4)}},
                   {DynamicAPInt(2), DynamicAPInt(8)});
  b.removeDuplicateDivs(merge);
  c.removeDuplicateDivs(merge);
  d.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(c, d);
}

TEST(UtilsTest, convolution) {
  std::vector<Fraction> aVals({1, 2, 3, 4});
  std::vector<Fraction> bVals({7, 3, 1, 6});
  ArrayRef<Fraction> a(aVals);
  ArrayRef<Fraction> b(bVals);

  std::vector<Fraction> conv = multiplyPolynomials(a, b);

  EXPECT_EQ(conv, std::vector<Fraction>({7, 17, 28, 45, 27, 22, 24}));

  aVals = {3, 6, 0, 2, 5};
  bVals = {2, 0, 6};
  a = aVals;
  b = bVals;

  conv = multiplyPolynomials(a, b);
  EXPECT_EQ(conv, std::vector<Fraction>({6, 12, 18, 40, 10, 12, 30}));
}

// A zero expected divisor denotes failure of the fixed-width overload.
static void checkFixedWidthFloorDiv(ArrayRef<DynamicAPInt> inputDividend,
                                    const DynamicAPInt &inputDivisor,
                                    ArrayRef<int64_t> expectedDividend = {},
                                    unsigned expectedDivisor = 0) {
  IntegerRelation cst(PresburgerSpace::getSetSpace(inputDividend.size() - 1));
  unsigned pos = cst.addLocalFloorDiv(inputDividend, inputDivisor);
  SmallVector<bool, 8> foundRepr(cst.getNumVars(), true);
  foundRepr[pos] = false;

  // The arbitrary-precision representation must remain available even when
  // the fixed-width overload cannot represent it.
  SmallVector<DynamicAPInt, 8> wideDividend(cst.getNumCols());
  DynamicAPInt wideDivisor;
  ASSERT_EQ(
      computeSingleVarRepr(cst, foundRepr, pos, wideDividend, wideDivisor).kind,
      ReprKind::Inequality);

  SmallVector<int64_t, 8> dividend = {17, 23};
  unsigned divisor = 19;
  MaybeLocalRepr repr =
      computeSingleVarRepr(cst, foundRepr, pos, dividend, divisor);
  if (expectedDivisor == 0) {
    EXPECT_EQ(repr.kind, ReprKind::None);
    EXPECT_FALSE(repr);
    EXPECT_THAT(dividend, testing::ElementsAre(17, 23));
    EXPECT_EQ(divisor, 19u);
    return;
  }
  EXPECT_EQ(repr.kind, ReprKind::Inequality);
  EXPECT_THAT(dividend, testing::ElementsAreArray(expectedDividend));
  EXPECT_EQ(divisor, expectedDivisor);
}

TEST(UtilsTest, ComputeSingleVarReprDividendOverflow) {
  DynamicAPInt aboveMax = DynamicAPInt(std::numeric_limits<int64_t>::max()) + 1;
  DynamicAPInt belowMin = DynamicAPInt(std::numeric_limits<int64_t>::min()) - 1;
  for (const DynamicAPInt &value : {aboveMax, belowMin}) {
    checkFixedWidthFloorDiv({value, DynamicAPInt(0)}, DynamicAPInt(1));
    checkFixedWidthFloorDiv({DynamicAPInt(1), value}, DynamicAPInt(1));
  }
}

TEST(UtilsTest, ComputeSingleVarReprDivisorOverflow) {
  DynamicAPInt aboveUnsignedMax =
      DynamicAPInt(std::numeric_limits<unsigned>::max()) + 1;
  DynamicAPInt aboveSignedMax =
      DynamicAPInt(std::numeric_limits<int64_t>::max()) + 1;
  for (const DynamicAPInt &divisor : {aboveUnsignedMax, aboveSignedMax})
    checkFixedWidthFloorDiv({DynamicAPInt(1), DynamicAPInt(0)}, divisor);
}

TEST(UtilsTest, ComputeSingleVarReprBoundaries) {
  for (int64_t value : {std::numeric_limits<int64_t>::min(),
                        std::numeric_limits<int64_t>::max()}) {
    checkFixedWidthFloorDiv({DynamicAPInt(value), DynamicAPInt(0)},
                            DynamicAPInt(1), {value, 0, 0}, 1);
    checkFixedWidthFloorDiv({DynamicAPInt(1), DynamicAPInt(value)},
                            DynamicAPInt(1), {1, 0, value}, 1);
  }
  for (unsigned divisor : {1u, std::numeric_limits<unsigned>::max()})
    checkFixedWidthFloorDiv({DynamicAPInt(1), DynamicAPInt(0)},
                            DynamicAPInt(divisor), {1, 0, 0}, divisor);
}

TEST(UtilsTest, ComputeSingleVarReprAfterNormalization) {
  DynamicAPInt scale = DynamicAPInt(std::numeric_limits<int64_t>::max()) + 1;
  checkFixedWidthFloorDiv({scale, scale}, scale, {1, 0, 1}, 1);
  checkFixedWidthFloorDiv({-scale, -scale}, scale, {-1, 0, -1}, 1);
}

TEST(UtilsTest, ComputeSingleVarReprDiscoveryFailure) {
  IntegerRelation cst(PresburgerSpace::getSetSpace(1, 0, 1));
  SmallVector<bool, 8> foundRepr = {false, false};
  auto checkFailure = [&]() {
    SmallVector<int64_t, 8> dividend = {17, 23};
    unsigned divisor = 19;
    MaybeLocalRepr repr =
        computeSingleVarRepr(cst, foundRepr, 1, dividend, divisor);
    EXPECT_EQ(repr.kind, ReprKind::None);
    EXPECT_THAT(dividend, testing::ElementsAre(17, 23));
    EXPECT_EQ(divisor, 19u);
  };
  checkFailure();

  // A candidate can populate the temporary outputs before being rejected for
  // depending on an unknown variable. Do not convert those outputs either.
  DynamicAPInt large = DynamicAPInt(std::numeric_limits<int64_t>::max()) + 1;
  cst.addEquality({large, DynamicAPInt(-1), DynamicAPInt(0)});
  checkFailure();
}

TEST(UtilsTest, ComputeSingleVarReprEquality) {
  IntegerRelation cst(PresburgerSpace::getSetSpace(1, 0, 1));
  cst.addEquality({1, -2, 3});
  SmallVector<bool, 8> foundRepr = {true, false};
  SmallVector<int64_t, 8> dividend;
  unsigned divisor = 0;
  MaybeLocalRepr repr =
      computeSingleVarRepr(cst, foundRepr, 1, dividend, divisor);
  ASSERT_EQ(repr.kind, ReprKind::Equality);
  EXPECT_EQ(repr.repr.equalityIdx, 0u);
  EXPECT_THAT(dividend, testing::ElementsAre(1, 0, 3));
  EXPECT_EQ(divisor, 2u);
}
