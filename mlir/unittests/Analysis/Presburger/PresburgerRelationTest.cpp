//===- PresburgerRelationTest.cpp - Tests for PresburgerRelation class ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "Parser.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir;
using namespace presburger;

TEST(PresburgerRelationTest, intersectDomainAndRange) {
  {
    PresburgerRelation rel = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (x + N, y - N)
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0)",
         // (x, y) -> (x + y, x - y)
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0)",
         // (x, y) -> (x - y, y - x)}
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0)"},
        2);

    PresburgerSet set =
        parsePresburgerSet({// (2x, x)
                            "(a, b)[N] : (a - 2 * b == 0)",
                            // (x, -x)
                            "(a, b)[N] : (a + b == 0)",
                            // (N, N)
                            "(a, b)[N] : (a - N == 0, b - N == 0)"});

    PresburgerRelation expectedRel = parsePresburgerRelationFromPresburgerSet(
        {"(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, x - 2 * y == 0)",
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, x + y == 0)",
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, x - N == 0, y - N "
         "== 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, x - 2 * y == 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, x + y == 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, x - N == 0, y - N "
         "== 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, x - 2 * y == 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, x + y == 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, x - N == 0, y - N "
         "== 0)"},
        2);

    PresburgerRelation computedRel = rel.intersectDomain(set);
    EXPECT_TRUE(computedRel.isEqual(expectedRel));
  }

  {
    PresburgerRelation rel = parsePresburgerRelationFromPresburgerSet(
        {// (x)[N] -> (x + N, x - N)
         "(x, a, b)[N] : (a - x - N == 0, b - x + N == 0)",
         // (x)[N] -> (x, -x)
         "(x, a, b)[N] : (a - x == 0, b + x == 0)",
         // (x)[N] -> (N - x, 2 * x)}
         "(x, a, b)[N] : (a - N + x == 0, b - 2 * x == 0)"},
        1);

    PresburgerSet set =
        parsePresburgerSet({// (2x, x)
                            "(a, b)[N] : (a - 2 * b == 0)",
                            // (x, -x)
                            "(a, b)[N] : (a + b == 0)",
                            // (N, N)
                            "(a, b)[N] : (a - N == 0, b - N == 0)"});

    PresburgerRelation expectedRel = parsePresburgerRelationFromPresburgerSet(
        {"(x, a, b)[N] : (a - x - N == 0, b - x + N == 0, a - 2 * b == 0)",
         "(x, a, b)[N] : (a - x - N == 0, b - x + N == 0, a + b == 0)",
         "(x, a, b)[N] : (a - x - N == 0, b - x + N == 0, a - N == 0, b - N "
         "== 0)",
         "(x, a, b)[N] : (a - x == 0, b + x == 0, a - 2 * b == 0)",
         "(x, a, b)[N] : (a - x == 0, b + x == 0, a + b == 0)",
         "(x, a, b)[N] : (a - x == 0, b + x == 0, a - N == 0, b - N "
         "== 0)",
         "(x, a, b)[N] : (a - N + x == 0, b - 2 * x == 0, a - 2 * b == 0)",
         "(x, a, b)[N] : (a - N + x == 0, b - 2 * x == 0, a + b == 0)",
         "(x, a, b)[N] : (a - N + x == 0, b - 2 * x == 0, a - N == 0, b - N "
         "== 0)"},
        1);

    PresburgerRelation computedRel = rel.intersectRange(set);
    EXPECT_TRUE(computedRel.isEqual(expectedRel));
  }
}

TEST(PresburgerRelationTest, applyDomainAndRange) {
  {
    PresburgerRelation map1 = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (y, x)
         "(x, y, a, b)[N] : (y - a == 0, x - b == 0)",
         // (x, y) -> (x + N, y - N)
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0)"},
        2);
    PresburgerRelation map2 = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (x - y)
         "(x, y, r)[N] : (x - y - r == 0)",
         // (x, y) -> N
         "(x, y, r)[N] : (N - r == 0)"},
        2);

    map1.applyDomain(map2);

    PresburgerRelation map3 = parsePresburgerRelationFromPresburgerSet(
        {// (y - x) -> (x, y)
         "(r, x, y)[N] : (y - x - r == 0)",
         // (x - y - 2N) -> (x, y)
         "(r, x, y)[N] : (x - y - 2 * N - r == 0)",
         // (x, y) -> N
         "(r, x, y)[N] : (N - r == 0)"},
        1);

    EXPECT_TRUE(map1.isEqual(map3));
  }
}

TEST(PresburgerRelationTest, inverse) {
  {
    PresburgerRelation rel = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (-y, -x)
         "(x, y, a, b)[N] : (y + a == 0, x + b == 0)",
         // (x, y) -> (x + N, y - N)
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0)"},
        2);

    rel.inverse();

    PresburgerRelation inverseRel = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (-y, -x)
         "(x, y, a, b)[N] : (y + a == 0, x + b == 0)",
         // (x, y) -> (x - N, y + N)
         "(x, y, a, b)[N] : (x - N - a == 0, y + N - b == 0)"},
        2);

    EXPECT_TRUE(rel.isEqual(inverseRel));
  }
}

TEST(PresburgerRelationTest, symbolicLexOpt) {
  PresburgerRelation rel1 = parsePresburgerRelationFromPresburgerSet(
      {"(x, y)[N, M] : (x >= 0, y >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1>= 0, "
       "2 * N - x >= 0, 2 * N - y >= 0)",
       "(x, y)[N, M] : (x >= 0, y >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1>= 0, "
       "x - N >= 0, M - x >= 0, y - 2 * N >= 0, M - y >= 0)"},
      1);

  SymbolicLexOpt lexmin1 = rel1.findSymbolicIntegerLexMin();

  PWMAFunction expectedLexMin1 = parsePWMAF({
      {"(x)[N, M] : (x >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1 >= 0, "
       "2 * N - x >= 0)",
       "(x)[N, M] -> (0)"},
      {"(x)[N, M] : (x >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1 >= 0, "
       "x - 2 * N- 1 >= 0, M - x >= 0)",
       "(x)[N, M] -> (2 * N)"},
  });

  SymbolicLexOpt lexmax1 = rel1.findSymbolicIntegerLexMax();

  PWMAFunction expectedLexMax1 = parsePWMAF({
      {"(x)[N, M] : (x >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1 >= 0, "
       "N - 1 - x  >= 0)",
       "(x)[N, M] -> (2 * N)"},
      {"(x)[N, M] : (x >= 0, N - 1 >= 0, M >= 0, M - 2 * N - 1 >= 0, "
       "x - N >= 0, M - x >= 0)",
       "(x)[N, M] -> (M)"},
  });

  EXPECT_TRUE(lexmin1.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmin1.lexopt.isEqual(expectedLexMin1));
  EXPECT_TRUE(lexmax1.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax1.lexopt.isEqual(expectedLexMax1));

  PresburgerRelation rel2 = parsePresburgerRelationFromPresburgerSet(
      // x or y or z
      // lexmin = (x, 0, 1 - x)
      // lexmax = (x, 1, 1)
      {"(x, y, z) : (x >= 0, y >= 0, z >= 0, 1 - x >= 0, 1 - y >= 0, "
       "1 - z >= 0, x + y + z - 1 >= 0)",
       // (x or y) and (y or z) and (z or x)
       // lexmin = (x, 1 - x, 1)
       // lexmax = (x, 1, 1)
       "(x, y, z) : (x >= 0, y >= 0, z >= 0, 1 - x >= 0, 1 - y >= 0, "
       "1 - z >= 0, x + y - 1 >= 0, y + z - 1 >= 0, z + x - 1 >= 0)",
       // x => (not y) or (not z)
       // lexmin = (x, 0, 0)
       // lexmax = (x, 1, 1 - x)
       "(x, y, z) : (x >= 0, y >= 0, z >= 0, 1 - x >= 0, 1 - y >= 0, "
       "1 - z >= 0, 2 - x - y - z >= 0)"},
      1);

  SymbolicLexOpt lexmin2 = rel2.findSymbolicIntegerLexMin();

  PWMAFunction expectedLexMin2 =
      parsePWMAF({{"(x) : (x >= 0, 1 - x >= 0)", "(x) -> (0, 0)"}});

  SymbolicLexOpt lexmax2 = rel2.findSymbolicIntegerLexMax();

  PWMAFunction expectedLexMax2 =
      parsePWMAF({{"(x) : (x >= 0, 1 - x >= 0)", "(x) -> (1, 1)"}});

  EXPECT_TRUE(lexmin2.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmin2.lexopt.isEqual(expectedLexMin2));
  EXPECT_TRUE(lexmax2.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax2.lexopt.isEqual(expectedLexMax2));

  PresburgerRelation rel3 = parsePresburgerRelationFromPresburgerSet(
      // (x => u or v or w) and (x or v) and (x or (not w))
      // lexmin = (x, 0, 0, 1 - x)
      // lexmax = (x, 1, 1 - x, x)
      {"(x, u, v, w) : (x >= 0, u >= 0, v >= 0, w >= 0, 1 - x >= 0, "
       "1 - u >= 0, 1 - v >= 0, 1 - w >= 0, -x + u + v + w >= 0, "
       "x + v - 1 >= 0, x - w >= 0)",
       // x => (u => (v => w)) and (x or (not v)) and (x or (not w))
       // lexmin = (x, 0, 0, x)
       // lexmax = (x, 1, x, x)
       "(x, u, v, w) : (x >= 0, u >= 0, v >= 0, w >= 0, 1 - x >= 0, "
       "1 - u >= 0, 1 - v >= 0, 1 - w >= 0, -x - u - v + w + 2 >= 0, "
       "x - v >= 0, x - w >= 0)",
       // (x or (u or (not v))) and ((not x) or ((not u) or w))
       // and (x or (not v)) and (x or (not w))
       // lexmin = (x, 0, 0, x)
       // lexmax = (x, 1, x, x)
       "(x, u, v, w) : (x >= 0, u >= 0, v >= 0, w >= 0, 1 - x >= 0, "
       "1 - u >= 0, 1 - v >= 0, 1 - w >= 0, x + u - v >= 0, x - u + w >= 0, "
       "x - v >= 0, x - w >= 0)"},
      1);

  SymbolicLexOpt lexmin3 = rel3.findSymbolicIntegerLexMin();

  PWMAFunction expectedLexMin3 =
      parsePWMAF({{"(x) : (x >= 0, 1 - x >= 0)", "(x) -> (0, 0, 0)"}});

  SymbolicLexOpt lexmax3 = rel3.findSymbolicIntegerLexMax();

  PWMAFunction expectedLexMax3 =
      parsePWMAF({{"(x) : (x >= 0, 1 - x >= 0)", "(x) -> (1, 1, x)"}});

  EXPECT_TRUE(lexmin3.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmin3.lexopt.isEqual(expectedLexMin3));
  EXPECT_TRUE(lexmax3.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax3.lexopt.isEqual(expectedLexMax3));
}

TEST(PresburgerRelationTest, getDomainAndRangeSet) {
  PresburgerRelation rel = parsePresburgerRelationFromPresburgerSet(
      {// (x, y) -> (x + N, y - N)
       "(x, y, a, b)[N] : (a >= 0, b >= 0, N - a >= 0, N - b >= 0, x - a + N "
       "== 0, y - b - N == 0)",
       // (x, y) -> (- y, - x)
       "(x, y, a, b)[N] : (a >= 0, b >= 0, 2 * N - a >= 0, 2 * N - b >= 0, a + "
       "y == 0, b + x == 0)"},
      2);

  PresburgerSet domainSet = rel.getDomainSet();

  PresburgerSet expectedDomainSet = parsePresburgerSet(
      {"(x, y)[N] : (x + N >= 0, -x >= 0, y - N >= 0, 2 * N - y >= 0)",
       "(x, y)[N] : (x + 2 * N >= 0, -x >= 0, y + 2 * N >= 0, -y >= 0)"});

  EXPECT_TRUE(domainSet.isEqual(expectedDomainSet));

  PresburgerSet rangeSet = rel.getRangeSet();

  PresburgerSet expectedRangeSet = parsePresburgerSet(
      {"(x, y)[N] : (x >= 0, 2 * N - x >= 0, y >= 0, 2 * N - y >= 0)"});

  EXPECT_TRUE(rangeSet.isEqual(expectedRangeSet));
}

TEST(PresburgerRelationTest, convertVarKind) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3, 0);

  IntegerRelation disj1 = parseRelationFromSet(
                      "(x, y, a)[U, V, W] : (x - U == 0, y + a - W == 0,"
                      "U - V >= 0, y - a >= 0)",
                      2),
                  disj2 = parseRelationFromSet(
                      "(x, y, a)[U, V, W] : (x + y - U == 0, x - a + V == 0,"
                      "V - U >= 0, y + a >= 0)",
                      2);

  PresburgerRelation rel(disj1);
  rel.unionInPlace(disj2);

  // Make a few kind conversions.
  rel.convertVarKind(VarKind::Domain, 0, 1, VarKind::Range, 0);
  rel.convertVarKind(VarKind::Symbol, 1, 2, VarKind::Domain, 1);
  rel.convertVarKind(VarKind::Symbol, 0, 1, VarKind::Range, 1);

  // Expected rel.
  disj1.convertVarKind(VarKind::Domain, 0, 1, VarKind::Range, 0);
  disj1.convertVarKind(VarKind::Symbol, 1, 3, VarKind::Domain, 1);
  disj1.convertVarKind(VarKind::Symbol, 0, 1, VarKind::Range, 1);
  disj2.convertVarKind(VarKind::Domain, 0, 1, VarKind::Range, 0);
  disj2.convertVarKind(VarKind::Symbol, 1, 3, VarKind::Domain, 1);
  disj2.convertVarKind(VarKind::Symbol, 0, 1, VarKind::Range, 1);

  PresburgerRelation expectedRel(disj1);
  expectedRel.unionInPlace(disj2);

  // Check if var counts are correct.
  EXPECT_EQ(rel.getNumDomainVars(), 3u);
  EXPECT_EQ(rel.getNumRangeVars(), 3u);
  EXPECT_EQ(rel.getNumSymbolVars(), 0u);

  // Check if identifiers are transferred correctly.
  EXPECT_TRUE(expectedRel.isEqual(rel));
}
