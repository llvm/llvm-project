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

static DivisionRepr parseDivisionRepr(unsigned numVars, unsigned numDivs,
                                      ArrayRef<ArrayRef<MPInt>> dividends,
                                      ArrayRef<MPInt> divisors) {
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
  DivisionRepr a = parseDivisionRepr(1, 1, {{MPInt(1), MPInt(2)}}, {MPInt(2)}),
               b = parseDivisionRepr(1, 1, {{MPInt(1), MPInt(2)}}, {MPInt(2)}),
               c = parseDivisionRepr(2, 2,
                                     {{MPInt(0), MPInt(1), MPInt(2)},
                                      {MPInt(0), MPInt(1), MPInt(2)}},
                                     {MPInt(2), MPInt(2)});
  c.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(a, c);
}

TEST(UtilsTest, DivisionReprNormalizeTest) {
  auto merge = [](unsigned i, unsigned j) -> bool { return true; };
  DivisionRepr a = parseDivisionRepr(2, 1, {{MPInt(1), MPInt(2), MPInt(-1)}},
                                     {MPInt(2)}),
               b = parseDivisionRepr(2, 1, {{MPInt(16), MPInt(32), MPInt(-16)}},
                                     {MPInt(32)}),
               c = parseDivisionRepr(1, 1, {{MPInt(12), MPInt(-4)}},
                                     {MPInt(8)}),
               d = parseDivisionRepr(2, 2,
                                     {{MPInt(1), MPInt(2), MPInt(-1)},
                                      {MPInt(4), MPInt(8), MPInt(-4)}},
                                     {MPInt(2), MPInt(8)});
  b.removeDuplicateDivs(merge);
  c.removeDuplicateDivs(merge);
  d.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(c, d);
}
