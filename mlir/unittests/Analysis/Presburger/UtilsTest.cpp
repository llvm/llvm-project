//===- Utils.cpp - Tests for Utils file ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
