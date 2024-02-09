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
  space.resetIds();

  int identifiers[6] = {0, 1, 2, 3, 4};

  // Attach identifiers to domain identifiers.
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);

  // Attach identifiers to range identifiers.
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);

  // Attach identifiers to symbol identifiers.
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);

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
  space.resetIds();

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(3, 2, 3, 0);
  otherSpace.resetIds();

  // Attach identifiers.
  int identifiers[7] = {0, 1, 2, 3, 4, 5, 6};
  int otherIdentifiers[8] = {10, 11, 12, 13, 14, 15, 16, 17};

  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  // Note the common identifier.
  space.getId(VarKind::Domain, 2) = Identifier(&otherIdentifiers[2]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Range, 1) = Identifier(&identifiers[3]);
  space.getId(VarKind::Range, 2) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[5]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[6]);

  otherSpace.getId(VarKind::Domain, 0) = Identifier(&otherIdentifiers[0]);
  otherSpace.getId(VarKind::Domain, 1) = Identifier(&otherIdentifiers[1]);
  otherSpace.getId(VarKind::Domain, 2) = Identifier(&otherIdentifiers[2]);
  otherSpace.getId(VarKind::Range, 0) = Identifier(&otherIdentifiers[3]);
  otherSpace.getId(VarKind::Range, 1) = Identifier(&otherIdentifiers[4]);
  // Note the common identifier.
  otherSpace.getId(VarKind::Symbol, 0) = Identifier(&identifiers[6]);
  otherSpace.getId(VarKind::Symbol, 1) = Identifier(&otherIdentifiers[5]);
  otherSpace.getId(VarKind::Symbol, 2) = Identifier(&otherIdentifiers[7]);

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
  space.resetIds();

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(2, 2, 4, 0);
  otherSpace.resetIds();

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  int otherIdentifiers[8] = {'u', 'v', 'a', 'b', 'E', 'F', 'G', 'H'};

  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);
  space.getId(VarKind::Symbol, 3) = Identifier(&identifiers[6]);

  otherSpace.getId(VarKind::Domain, 0) = Identifier(&otherIdentifiers[0]);
  otherSpace.getId(VarKind::Domain, 1) = Identifier(&otherIdentifiers[1]);
  otherSpace.getId(VarKind::Range, 0) = Identifier(&otherIdentifiers[2]);
  otherSpace.getId(VarKind::Range, 1) = Identifier(&otherIdentifiers[3]);
  otherSpace.getId(VarKind::Symbol, 0) = Identifier(&otherIdentifiers[4]);
  otherSpace.getId(VarKind::Symbol, 1) = Identifier(&otherIdentifiers[5]);
  otherSpace.getId(VarKind::Symbol, 2) = Identifier(&otherIdentifiers[6]);
  otherSpace.getId(VarKind::Symbol, 3) = Identifier(&otherIdentifiers[7]);

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
  space.resetIds();

  PresburgerSpace otherSpace = PresburgerSpace::getRelationSpace(2, 2, 4, 0);
  otherSpace.resetIds();

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  int otherIdentifiers[6] = {'u', 'v', 'a', 'b', 'E', 'F'};

  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);
  space.getId(VarKind::Symbol, 3) = Identifier(&identifiers[6]);

  otherSpace.getId(VarKind::Domain, 0) = Identifier(&otherIdentifiers[0]);
  otherSpace.getId(VarKind::Domain, 1) = Identifier(&otherIdentifiers[1]);
  otherSpace.getId(VarKind::Range, 0) = Identifier(&otherIdentifiers[2]);
  otherSpace.getId(VarKind::Range, 1) = Identifier(&otherIdentifiers[3]);
  otherSpace.getId(VarKind::Symbol, 0) = Identifier(&otherIdentifiers[4]);
  otherSpace.getId(VarKind::Symbol, 1) = Identifier(&otherIdentifiers[5]);
  // Note common identifiers
  otherSpace.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);
  otherSpace.getId(VarKind::Symbol, 3) = Identifier(&identifiers[6]);

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
  space.resetIds();

  // Attach identifiers.
  int identifiers[7] = {'x', 'y', 'z', 'A', 'B', 'C', 'D'};
  space.getId(VarKind::Domain, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::Domain, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::Range, 0) = Identifier(&identifiers[2]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);
  space.getId(VarKind::Symbol, 3) = Identifier(&identifiers[6]);
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
