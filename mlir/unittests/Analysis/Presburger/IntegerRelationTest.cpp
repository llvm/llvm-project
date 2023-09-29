//===- IntegerRelationTest.cpp - Tests for IntegerRelation class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "Parser.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(IntegerRelationTest, getDomainAndRangeSet) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, xr)[N] : (xr - x - 10 == 0, xr >= 0, N - xr >= 0)", 1);

  IntegerPolyhedron domainSet = rel.getDomainSet();

  IntegerPolyhedron expectedDomainSet =
      parseIntegerPolyhedron("(x)[N] : (x + 10 >= 0, N - x - 10 >= 0)");

  EXPECT_TRUE(domainSet.isEqual(expectedDomainSet));

  IntegerPolyhedron rangeSet = rel.getRangeSet();

  IntegerPolyhedron expectedRangeSet =
      parseIntegerPolyhedron("(x)[N] : (x >= 0, N - x >= 0)");

  EXPECT_TRUE(rangeSet.isEqual(expectedRangeSet));
}

TEST(IntegerRelationTest, inverse) {
  IntegerRelation rel =
      parseRelationFromSet("(x, y, z)[N, M] : (z - x - y == 0, x >= 0, N - x "
                           ">= 0, y >= 0, M - y >= 0)",
                           2);

  IntegerRelation inverseRel =
      parseRelationFromSet("(z, x, y)[N, M]  : (x >= 0, N - x >= 0, y >= 0, M "
                           "- y >= 0, x + y - z == 0)",
                           1);

  rel.inverse();

  EXPECT_TRUE(rel.isEqual(inverseRel));
}

TEST(IntegerRelationTest, intersectDomainAndRange) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
      ">= 0, x + y + z floordiv 7 == 0)",
      1);

  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(x)[N, M] : (x >= 0, M - x - 1 >= 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, x >= 0, M - x - 1 >= 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectDomain(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }

  {
    IntegerPolyhedron poly = parseIntegerPolyhedron(
        "(y, z)[N, M] : (y >= 0, M - y - 1 >= 0, y + z == 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, y >= 0, M - y - 1 >= 0, y + z == 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectRange(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }
}

TEST(IntegerRelationTest, applyDomainAndRange) {

  {
    IntegerRelation map1 = parseRelationFromSet(
        "(x, y, a, b)[N] : (a - x - N == 0, b - y + N == 0)", 2);
    IntegerRelation map2 =
        parseRelationFromSet("(x, y, a)[N] : (a - x - y == 0)", 2);

    map1.applyRange(map2);

    IntegerRelation map3 =
        parseRelationFromSet("(x, y, a)[N] : (a - x - y == 0)", 2);

    EXPECT_TRUE(map1.isEqual(map3));
  }

  {
    IntegerRelation map1 = parseRelationFromSet(
        "(x, y, a, b)[N] : (a - x + N == 0, b - y - N == 0)", 2);
    IntegerRelation map2 =
        parseRelationFromSet("(x, y, a, b)[N] : (a - N == 0, b - N == 0)", 2);

    IntegerRelation map3 =
        parseRelationFromSet("(x, y, a, b)[N] : (x - N == 0, y - N == 0)", 2);

    map1.applyDomain(map2);

    EXPECT_TRUE(map1.isEqual(map3));
  }
}

TEST(IntegerRelationTest, symbolicLexmin) {
  SymbolicLexOpt lexmin =
      parseRelationFromSet("(a, x)[b] : (x - a >= 0, x - b >= 0)", 1)
          .findSymbolicIntegerLexMin();

  PWMAFunction expectedLexmin = parsePWMAF({
      {"(a)[b] : (a - b >= 0)", "(a)[b] -> (a)"},     // a
      {"(a)[b] : (b - a - 1 >= 0)", "(a)[b] -> (b)"}, // b
  });
  EXPECT_TRUE(lexmin.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmin.lexopt.isEqual(expectedLexmin));
}

TEST(IntegerRelationTest, symbolicLexmax) {
  SymbolicLexOpt lexmax1 =
      parseRelationFromSet("(a, x)[b] : (a - x >= 0, b - x >= 0)", 1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax1 = parsePWMAF({
      {"(a)[b] : (a - b >= 0)", "(a)[b] -> (b)"},
      {"(a)[b] : (b - a - 1 >= 0)", "(a)[b] -> (a)"},
  });

  SymbolicLexOpt lexmax2 =
      parseRelationFromSet("(i, j)[N] : (i >= 0, j >= 0, N - i - j >= 0)", 1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax2 = parsePWMAF({
      {"(i)[N] : (i >= 0, N - i >= 0)", "(i)[N] -> (N - i)"},
  });

  SymbolicLexOpt lexmax3 =
      parseRelationFromSet("(x, y)[N] : (x >= 0, 2 * N - x >= 0, y >= 0, x - y "
                           "+ 2 * N >= 0, 4 * N - x - y >= 0)",
                           1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax3 =
      parsePWMAF({{"(x)[N] : (x >= 0, 2 * N - x >= 0, x - N - 1 >= 0)",
                   "(x)[N] -> (4 * N - x)"},
                  {"(x)[N] : (x >= 0, 2 * N - x >= 0, -x + N >= 0)",
                   "(x)[N] -> (x + 2 * N)"}});

  EXPECT_TRUE(lexmax1.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax1.lexopt.isEqual(expectedLexmax1));
  EXPECT_TRUE(lexmax2.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax2.lexopt.isEqual(expectedLexmax2));
  EXPECT_TRUE(lexmax3.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax3.lexopt.isEqual(expectedLexmax3));
}
