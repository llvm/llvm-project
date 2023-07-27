//===- PresburgerRelationTest.cpp - Tests for PresburgerRelation class ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "Parser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir;
using namespace presburger;

static PresburgerRelation
parsePresburgerRelationFromPresburgerSet(ArrayRef<StringRef> strs,
                                         unsigned numDomain) {
  assert(!strs.empty() && "strs should not be empty");

  IntegerRelation rel = parseIntegerPolyhedron(strs[0]);
  rel.convertVarKind(VarKind::SetDim, 0, numDomain, VarKind::Domain);
  PresburgerRelation result(rel);
  for (unsigned i = 1, e = strs.size(); i < e; ++i) {
    rel = parseIntegerPolyhedron(strs[i]);
    rel.convertVarKind(VarKind::SetDim, 0, numDomain, VarKind::Domain);
    result.unionInPlace(rel);
  }
  return result;
}

TEST(PresburgerRelationTest, intersectDomainAndRange) {
  PresburgerRelation rel = parsePresburgerRelationFromPresburgerSet(
      {// (x, y) -> (x + N, y - N)
       "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0)",
       // (x, y) -> (x + y, x - y)
       "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0)",
       // (x, y) -> (x - y, y - x)}
       "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0)"},
      2);

  {
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
    PresburgerSet set =
        parsePresburgerSet({// (2x, x)
                            "(a, b)[N] : (a - 2 * b == 0)",
                            // (x, -x)
                            "(a, b)[N] : (a + b == 0)",
                            // (N, N)
                            "(a, b)[N] : (a - N == 0, b - N == 0)"});

    PresburgerRelation expectedRel = parsePresburgerRelationFromPresburgerSet(
        {"(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, a - 2 * b == 0)",
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, a + b == 0)",
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0, a - N == 0, b - N "
         "== 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, a - 2 * b == 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, a + b == 0)",
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0, a - N == 0, b - N "
         "== 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, a - 2 * b == 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, a + b == 0)",
         "(x, y, a, b)[N] : (a - x + y == 0, b - y + x == 0, a - N == 0, b - N "
         "== 0)"},
        2);

    PresburgerRelation computedRel = rel.intersectRange(set);
    EXPECT_TRUE(computedRel.isEqual(expectedRel));
  }
}

TEST(PresburgerRelationTest, applyDomainAndRange) {
  {
    PresburgerRelation map1 = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (x + N, y - N)
         "(x, y, a, b)[N] : (x - a + N == 0, y - b - N == 0)",
         // (x, y) -> (y, x)
         "(x, y, a, b)[N] : (a - y == 0, b - x == 0)",
         // (x, y) -> (x + y, x - y)
         "(x, y, a, b)[N] : (a - x - y == 0, b - x + y == 0)"},
        2);
    PresburgerRelation map2 = parsePresburgerRelationFromPresburgerSet(
        {// (x, y) -> (x + y)
         "(x, y, r)[N] : (r - x - y == 0)",
         // (x, y) -> (N)
         "(x, y, r)[N] : (r - N == 0)",
         // (x, y) -> (y - x)
         "(x, y, r)[N] : (r + x - y == 0)"},
        2);

    map1.applyRange(map2);

    PresburgerRelation map3 = parsePresburgerRelationFromPresburgerSet(
        {
            // (x, y) -> (x + y)
            "(x, y, r)[N] : (r - x - y == 0)",
            // (x, y) -> (N)
            "(x, y, r)[N] : (r - N == 0)",
            // (x, y) -> (y - x - 2N)
            "(x, y, r)[N] : (r - y + x + 2 * N == 0)",
            // (x, y) -> (x - y)
            "(x, y, r)[N] : (r - x + y == 0)",
            // (x, y) -> (2x)
            "(x, y, r)[N] : (r - 2 * x == 0)",
            // (x, y) -> (-2y)
            "(x, y, r)[N] : (r + 2 * y == 0)",
        },
        2);

    EXPECT_TRUE(map1.isEqual(map3));
  }

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
