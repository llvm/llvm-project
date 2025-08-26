//===- IntegerRelationTest.cpp - Tests for IntegerRelation class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "Parser.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;
using ::testing::ElementsAre;

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

TEST(IntegerRelationTest, swapVar) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 2, 0);

  int identifiers[6] = {0, 1, 2, 3, 4};

  // Attach identifiers to domain identifiers.
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));

  // Attach identifiers to range identifiers.
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));

  // Attach identifiers to symbol identifiers.
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));

  IntegerRelation rel =
      parseRelationFromSet("(x, y, z)[N, M] : (z - x - y == 0, x >= 0, N - x "
                           ">= 0, y >= 0, M - y >= 0)",
                           2);
  rel.setSpace(space);
  // Swap (Domain 0, Range 0)
  rel.swapVar(0, 2);
  // Swap (Domain 1, Symbol 1)
  rel.swapVar(1, 4);

  PresburgerSpace swappedSpace = rel.getSpace();

  EXPECT_TRUE(swappedSpace.getId(VarKind::Domain, 0)
                  .isEqual(space.getId(VarKind::Range, 0)));
  EXPECT_TRUE(swappedSpace.getId(VarKind::Domain, 1)
                  .isEqual(space.getId(VarKind::Symbol, 1)));
  EXPECT_TRUE(swappedSpace.getId(VarKind::Range, 0)
                  .isEqual(space.getId(VarKind::Domain, 0)));
  EXPECT_TRUE(swappedSpace.getId(VarKind::Symbol, 1)
                  .isEqual(space.getId(VarKind::Domain, 1)));
}

TEST(IntegerRelationTest, mergeAndAlignSymbols) {
  IntegerRelation rel =
      parseRelationFromSet("(x, y, z, a, b, c)[N, Q] : (a - x - y == 0, "
                           "x >= 0, N - b >= 0, y >= 0, Q - y >= 0)",
                           3);
  IntegerRelation otherRel = parseRelationFromSet(
      "(x, y, z, a, b)[N, M, P] : (z - x - y == 0, x >= 0, N - x "
      ">= 0, y >= 0, M - y >= 0, 2 * P - 3 * a + 2 * b == 0)",
      3);
  PresburgerSpace space = PresburgerSpace::getRelationSpace(3, 3, 2, 0);

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(3, 2, 3, 0);

  // Attach identifiers.
  int identifiers[7] = {0, 1, 2, 3, 4, 5, 6};
  int otherIdentifiers[8] = {10, 11, 12, 13, 14, 15, 16, 17};

  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  // Note the common identifier.
  space.setId(VarKind::Domain, 2, Identifier(&otherIdentifiers[2]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Range, 1, Identifier(&identifiers[3]));
  space.setId(VarKind::Range, 2, Identifier(&identifiers[4]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[5]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[6]));

  otherSpace.setId(VarKind::Domain, 0, Identifier(&otherIdentifiers[0]));
  otherSpace.setId(VarKind::Domain, 1, Identifier(&otherIdentifiers[1]));
  otherSpace.setId(VarKind::Domain, 2, Identifier(&otherIdentifiers[2]));
  otherSpace.setId(VarKind::Range, 0, Identifier(&otherIdentifiers[3]));
  otherSpace.setId(VarKind::Range, 1, Identifier(&otherIdentifiers[4]));
  // Note the common identifier.
  otherSpace.setId(VarKind::Symbol, 0, Identifier(&identifiers[6]));
  otherSpace.setId(VarKind::Symbol, 1, Identifier(&otherIdentifiers[5]));
  otherSpace.setId(VarKind::Symbol, 2, Identifier(&otherIdentifiers[7]));

  rel.setSpace(space);
  otherRel.setSpace(otherSpace);
  rel.mergeAndAlignSymbols(otherRel);

  space = rel.getSpace();
  otherSpace = otherRel.getSpace();

  // Check if merge and align is successful.
  // Check symbol var identifiers.
  EXPECT_EQ(4u, space.getNumSymbolVars());
  EXPECT_EQ(4u, otherSpace.getNumSymbolVars());
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[6]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&otherIdentifiers[7]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 0), Identifier(&identifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 1), Identifier(&identifiers[6]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 2),
            Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 3),
            Identifier(&otherIdentifiers[7]));
  // Check that domain and range var identifiers are not affected.
  EXPECT_EQ(3u, space.getNumDomainVars());
  EXPECT_EQ(3u, space.getNumRangeVars());
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Domain, 1), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Domain, 2), Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Range, 1), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Range, 2), Identifier(&identifiers[4]));
  EXPECT_EQ(3u, otherSpace.getNumDomainVars());
  EXPECT_EQ(2u, otherSpace.getNumRangeVars());
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 0),
            Identifier(&otherIdentifiers[0]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 1),
            Identifier(&otherIdentifiers[1]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 2),
            Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 0),
            Identifier(&otherIdentifiers[3]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 1),
            Identifier(&otherIdentifiers[4]));
}

// Check that mergeAndAlignSymbols unions symbol variables when they are
// disjoint.
TEST(IntegerRelationTest, mergeAndAlignDisjointSymbols) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[A, B, C, D] : (x + A - C - y + D - z >= 0)", 2);
  IntegerRelation otherRel = parseRelationFromSet(
      "(u, v, a, b)[E, F, G, H] : (E - u + v == 0, v - G - H >= 0)", 2);
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 4, 0);

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(2, 2, 4, 0);

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  int otherIdentifiers[8] = {'u', 'v', 'a', 'b', 'E', 'F', 'G', 'H'};

  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  space.setId(VarKind::Symbol, 2, Identifier(&identifiers[5]));
  space.setId(VarKind::Symbol, 3, Identifier(&identifiers[6]));

  otherSpace.setId(VarKind::Domain, 0, Identifier(&otherIdentifiers[0]));
  otherSpace.setId(VarKind::Domain, 1, Identifier(&otherIdentifiers[1]));
  otherSpace.setId(VarKind::Range, 0, Identifier(&otherIdentifiers[2]));
  otherSpace.setId(VarKind::Range, 1, Identifier(&otherIdentifiers[3]));
  otherSpace.setId(VarKind::Symbol, 0, Identifier(&otherIdentifiers[4]));
  otherSpace.setId(VarKind::Symbol, 1, Identifier(&otherIdentifiers[5]));
  otherSpace.setId(VarKind::Symbol, 2, Identifier(&otherIdentifiers[6]));
  otherSpace.setId(VarKind::Symbol, 3, Identifier(&otherIdentifiers[7]));

  rel.setSpace(space);
  otherRel.setSpace(otherSpace);
  rel.mergeAndAlignSymbols(otherRel);

  space = rel.getSpace();
  otherSpace = otherRel.getSpace();

  // Check if merge and align is successful.
  // Check symbol var identifiers.
  EXPECT_EQ(8u, space.getNumSymbolVars());
  EXPECT_EQ(8u, otherSpace.getNumSymbolVars());
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&identifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[6]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 4), Identifier(&otherIdentifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 5), Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 6), Identifier(&otherIdentifiers[6]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 7), Identifier(&otherIdentifiers[7]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 2), Identifier(&identifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 3), Identifier(&identifiers[6]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 4),
            Identifier(&otherIdentifiers[4]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 5),
            Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 6),
            Identifier(&otherIdentifiers[6]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 7),
            Identifier(&otherIdentifiers[7]));
  // Check that domain and range var identifiers are not affected.
  EXPECT_EQ(2u, space.getNumDomainVars());
  EXPECT_EQ(1u, space.getNumRangeVars());
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Domain, 1), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(2u, otherSpace.getNumDomainVars());
  EXPECT_EQ(2u, otherSpace.getNumRangeVars());
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 0),
            Identifier(&otherIdentifiers[0]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 1),
            Identifier(&otherIdentifiers[1]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 0),
            Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 1),
            Identifier(&otherIdentifiers[3]));
}

// Check that mergeAndAlignSymbols is correct when a suffix of identifiers is
// shared; i.e. identifiers are [A, B, C, D] and [E, F, C, D].
TEST(IntegerRelationTest, mergeAndAlignCommonSuffixSymbols) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[A, B, C, D] : (x + A - C - y + D - z >= 0)", 2);
  IntegerRelation otherRel = parseRelationFromSet(
      "(u, v, a, b)[E, F, C, D] : (E - u + v == 0, v - C - D >= 0)", 2);
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 4, 0);

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(2, 2, 4, 0);

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  int otherIdentifiers[6] = {'u', 'v', 'a', 'b', 'E', 'F'};

  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  space.setId(VarKind::Symbol, 2, Identifier(&identifiers[5]));
  space.setId(VarKind::Symbol, 3, Identifier(&identifiers[6]));

  otherSpace.setId(VarKind::Domain, 0, Identifier(&otherIdentifiers[0]));
  otherSpace.setId(VarKind::Domain, 1, Identifier(&otherIdentifiers[1]));
  otherSpace.setId(VarKind::Range, 0, Identifier(&otherIdentifiers[2]));
  otherSpace.setId(VarKind::Range, 1, Identifier(&otherIdentifiers[3]));
  otherSpace.setId(VarKind::Symbol, 0, Identifier(&otherIdentifiers[4]));
  otherSpace.setId(VarKind::Symbol, 1, Identifier(&otherIdentifiers[5]));
  // Note common identifiers
  otherSpace.setId(VarKind::Symbol, 2, Identifier(&identifiers[5]));
  otherSpace.setId(VarKind::Symbol, 3, Identifier(&identifiers[6]));

  rel.setSpace(space);
  otherRel.setSpace(otherSpace);
  rel.mergeAndAlignSymbols(otherRel);

  space = rel.getSpace();
  otherSpace = otherRel.getSpace();

  // Check if merge and align is successful.
  // Check symbol var identifiers.
  EXPECT_EQ(6u, space.getNumSymbolVars());
  EXPECT_EQ(6u, otherSpace.getNumSymbolVars());
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&identifiers[5]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[6]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 4), Identifier(&otherIdentifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 5), Identifier(&otherIdentifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 2), Identifier(&identifiers[5]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 3), Identifier(&identifiers[6]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 4),
            Identifier(&otherIdentifiers[4]));
  EXPECT_EQ(otherSpace.getId(VarKind::Symbol, 5),
            Identifier(&otherIdentifiers[5]));
  // Check that domain and range var identifiers are not affected.
  EXPECT_EQ(2u, space.getNumDomainVars());
  EXPECT_EQ(1u, space.getNumRangeVars());
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Domain, 1), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(2u, otherSpace.getNumDomainVars());
  EXPECT_EQ(2u, otherSpace.getNumRangeVars());
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 0),
            Identifier(&otherIdentifiers[0]));
  EXPECT_EQ(otherSpace.getId(VarKind::Domain, 1),
            Identifier(&otherIdentifiers[1]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 0),
            Identifier(&otherIdentifiers[2]));
  EXPECT_EQ(otherSpace.getId(VarKind::Range, 1),
            Identifier(&otherIdentifiers[3]));
}

TEST(IntegerRelationTest, setId) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[A, B, C, D] : (x + A - C - y + D - z >= 0)", 2);
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 4, 0);

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  space.setId(VarKind::Symbol, 2, Identifier(&identifiers[5]));
  space.setId(VarKind::Symbol, 3, Identifier(&identifiers[6]));
  rel.setSpace(space);

  int newIdentifiers[3] = {1, 2, 3};
  rel.setId(VarKind::Domain, 1, Identifier(&newIdentifiers[0]));
  rel.setId(VarKind::Range, 0, Identifier(&newIdentifiers[1]));
  rel.setId(VarKind::Symbol, 2, Identifier(&newIdentifiers[2]));

  space = rel.getSpace();
  // Check that new identifiers are set correctly.
  EXPECT_EQ(space.getId(VarKind::Domain, 1), Identifier(&newIdentifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&newIdentifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&newIdentifiers[2]));
  // Check that old identifier are not changed.
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[6]));
}

TEST(IntegerRelationTest, convertVarKind) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(3, 3, 0);

  // Attach identifiers.
  int identifiers[6] = {0, 1, 2, 3, 4, 5};
  space.setId(VarKind::SetDim, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::SetDim, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::SetDim, 2, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  space.setId(VarKind::Symbol, 2, Identifier(&identifiers[5]));

  // Cannot call parseIntegerRelation to test convertVarKind as
  // parseIntegerRelation uses convertVarKind.
  IntegerRelation rel = parseIntegerPolyhedron(
      // 0  1  2  3  4  5
      "(x, y, a)[U, V, W] : (x - U == 0, y + a - W == 0, U - V >= 0,"
      "y - a >= 0)");
  rel.setSpace(space);

  // Make a few kind conversions.
  rel.convertVarKind(VarKind::Symbol, 1, 2, VarKind::Domain, 0);
  rel.convertVarKind(VarKind::Range, 2, 3, VarKind::Domain, 0);
  rel.convertVarKind(VarKind::Range, 0, 2, VarKind::Symbol, 1);
  rel.convertVarKind(VarKind::Domain, 1, 2, VarKind::Range, 0);
  rel.convertVarKind(VarKind::Domain, 0, 1, VarKind::Range, 1);

  space = rel.getSpace();

  // Expected rel.
  IntegerRelation expectedRel = parseIntegerPolyhedron(
      "(V, a)[U, x, y, W] : (x - U == 0, y + a - W == 0, U - V >= 0,"
      "y - a >= 0)");
  expectedRel.setSpace(space);

  EXPECT_TRUE(rel.isEqual(expectedRel));

  EXPECT_EQ(space.getId(VarKind::SetDim, 0), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::SetDim, 1), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[5]));
}

TEST(IntegerRelationTest, convertVarKindToLocal) {
  // Convert all range variables to local variables.
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[N, M] : (x - y >= 0, y - N >= 0, 3 - z >= 0, 2 * M - 5 >= 0)",
      1);
  PresburgerSpace space = rel.getSpace();
  // Attach identifiers.
  char identifiers[5] = {'x', 'y', 'z', 'N', 'M'};
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 1, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  rel.setSpace(space);
  rel.convertToLocal(VarKind::Range, 0, rel.getNumRangeVars());
  IntegerRelation expectedRel =
      parseRelationFromSet("(x)[N, M] : (x - N >= 0, 2 * M - 5 >= 0)", 1);
  EXPECT_TRUE(rel.isEqual(expectedRel));
  space = rel.getSpace();
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));

  // Convert all domain variables to local variables.
  IntegerRelation rel2 = parseRelationFromSet(
      "(x, y, z)[N, M] : (x - y >= 0, y - N >= 0, 3 - z >= 0, 2 * M - 5 >= 0)",
      2);
  space = rel2.getSpace();
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  rel2.setSpace(space);
  rel2.convertToLocal(VarKind::Domain, 0, rel2.getNumDomainVars());
  expectedRel =
      parseIntegerPolyhedron("(z)[N, M] : (3 - z >= 0, 2 * M - 5 >= 0)");
  EXPECT_TRUE(rel2.isEqual(expectedRel));
  space = rel2.getSpace();
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));

  // Convert a prefix of range variables to local variables.
  IntegerRelation rel3 = parseRelationFromSet(
      "(x, y, z)[N, M] : (x - y >= 0, y - N >= 0, 3 - z >= 0, 2 * M - 5 >= 0)",
      1);
  space = rel3.getSpace();
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 1, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  rel3.setSpace(space);
  rel3.convertToLocal(VarKind::Range, 0, 1);
  expectedRel = parseRelationFromSet(
      "(x, z)[N, M] : (x - N >= 0, 3 - z >= 0, 2 * M - 5 >= 0)", 1);
  EXPECT_TRUE(rel3.isEqual(expectedRel));
  space = rel3.getSpace();
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));

  // Convert a suffix of domain variables to local variables.
  IntegerRelation rel4 = parseRelationFromSet(
      "(x, y, z)[N, M] : (x - y >= 0, y - N >= 0, 3 - z >= 0, 2 * M - 5 >= 0)",
      2);
  space = rel4.getSpace();
  space.setId(VarKind::Domain, 0, Identifier(&identifiers[0]));
  space.setId(VarKind::Domain, 1, Identifier(&identifiers[1]));
  space.setId(VarKind::Range, 0, Identifier(&identifiers[2]));
  space.setId(VarKind::Symbol, 0, Identifier(&identifiers[3]));
  space.setId(VarKind::Symbol, 1, Identifier(&identifiers[4]));
  rel4.setSpace(space);
  rel4.convertToLocal(VarKind::Domain, rel4.getNumDomainVars() - 1,
                      rel4.getNumDomainVars());
  // expectedRel same as before.
  EXPECT_TRUE(rel4.isEqual(expectedRel));
  space = rel4.getSpace();
  EXPECT_EQ(space.getId(VarKind::Domain, 0), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Range, 0), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[4]));
}

TEST(IntegerRelationTest, rangeProduct) {
  IntegerRelation r1 = parseRelationFromSet(
      "(i, j, k) : (2*i + 3*k == 0, i >= 0, j >= 0, k >= 0)", 2);
  IntegerRelation r2 = parseRelationFromSet(
      "(i, j, l) : (4*i + 6*j + 9*l == 0, i >= 0, j >= 0, l >= 0)", 2);

  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected =
      parseRelationFromSet("(i, j, k, l) : (2*i + 3*k == 0, 4*i + 6*j + 9*l == "
                           "0, i >= 0, j >= 0, k >= 0, l >= 0)",
                           2);

  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductMultdimRange) {
  IntegerRelation r1 =
      parseRelationFromSet("(i, k) : (2*i + 3*k == 0, i >= 0, k >= 0)", 1);
  IntegerRelation r2 = parseRelationFromSet(
      "(i, l, m) : (4*i + 6*m + 9*l == 0, i >= 0, l >= 0, m >= 0)", 1);

  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected =
      parseRelationFromSet("(i, k, l, m) : (2*i + 3*k == 0, 4*i + 6*m + 9*l == "
                           "0, i >= 0, k >= 0, l >= 0, m >= 0)",
                           1);

  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductMultdimRangeSwapped) {
  IntegerRelation r1 = parseRelationFromSet(
      "(i, l, m) : (4*i + 6*m + 9*l == 0, i >= 0, l >= 0, m >= 0)", 1);
  IntegerRelation r2 =
      parseRelationFromSet("(i, k) : (2*i + 3*k == 0, i >= 0, k >= 0)", 1);

  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected =
      parseRelationFromSet("(i, l, m, k) : (2*i + 3*k == 0, 4*i + 6*m + 9*l == "
                           "0, i >= 0, k >= 0, l >= 0, m >= 0)",
                           1);

  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductEmptyDomain) {
  IntegerRelation r1 =
      parseRelationFromSet("(i, j) : (4*i + 9*j == 0, i >= 0, j >= 0)", 0);
  IntegerRelation r2 =
      parseRelationFromSet("(k, l) : (2*k + 3*l == 0, k >= 0, l >= 0)", 0);
  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected =
      parseRelationFromSet("(i, j, k, l) : (2*k + 3*l == 0, 4*i + 9*j == "
                           "0, i >= 0, j >= 0, k >= 0, l >= 0)",
                           0);
  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductEmptyRange) {
  IntegerRelation r1 =
      parseRelationFromSet("(i, j) : (4*i + 9*j == 0, i >= 0, j >= 0)", 2);
  IntegerRelation r2 =
      parseRelationFromSet("(i, j) : (2*i + 3*j == 0, i >= 0, j >= 0)", 2);
  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected =
      parseRelationFromSet("(i, j) : (2*i + 3*j == 0, 4*i + 9*j == "
                           "0, i >= 0, j >= 0)",
                           2);
  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductEmptyDomainAndRange) {
  IntegerRelation r1 = parseRelationFromSet("() : ()", 0);
  IntegerRelation r2 = parseRelationFromSet("() : ()", 0);
  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected = parseRelationFromSet("() : ()", 0);
  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, rangeProductSymbols) {
  IntegerRelation r1 = parseRelationFromSet(
      "(i, j)[s] : (2*i + 3*j + s == 0, i >= 0, j >= 0)", 1);
  IntegerRelation r2 = parseRelationFromSet(
      "(i, l)[s] : (3*i + 4*l + s == 0, i >= 0, l >= 0)", 1);

  IntegerRelation rangeProd = r1.rangeProduct(r2);
  IntegerRelation expected = parseRelationFromSet(
      "(i, j, l)[s] : (2*i + 3*j + s == 0, 3*i + 4*l + s == "
      "0, i >= 0, j >= 0, l >= 0)",
      1);

  EXPECT_TRUE(expected.isEqual(rangeProd));
}

TEST(IntegerRelationTest, getVarKindRange) {
  IntegerRelation r1 = parseRelationFromSet(
      "(i1, i2, i3, i4, i5) : (i1 >= 0, i2 >= 0, i3 >= 0, i4 >= 0, i5 >= 0)",
      2);
  SmallVector<unsigned> actual;
  for (unsigned var : r1.iterVarKind(VarKind::Range)) {
    actual.push_back(var);
  }
  EXPECT_THAT(actual, ElementsAre(2, 3, 4));
}

TEST(IntegerRelationTest, addLocalModulo) {
  IntegerRelation rel = parseRelationFromSet("(x) : (x >= 0, 100 - x >= 0)", 1);
  unsigned result = rel.addLocalModulo({1, 0}, 32); // x % 32
  rel.convertVarKind(VarKind::Local,
                     result - rel.getVarKindOffset(VarKind::Local),
                     rel.getNumVarKind(VarKind::Local), VarKind::Range);
  for (unsigned x = 0; x <= 100; ++x) {
    EXPECT_TRUE(rel.containsPointNoLocal({x, x % 32}));
  }
}
