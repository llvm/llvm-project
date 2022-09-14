//===- SetTest.cpp - Tests for PresburgerSet ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for PresburgerSet. The tests for union,
// intersection, subtract, and complement work by computing the operation on
// two sets and checking, for a set of points, that the resulting set contains
// the point iff the result is supposed to contain it. The test for isEqual just
// checks if the result for two sets matches the expected result.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

/// Compute the union of s and t, and check that each of the given points
/// belongs to the union iff it belongs to at least one of s and t.
static void testUnionAtPoints(const PresburgerSet &s, const PresburgerSet &t,
                              ArrayRef<SmallVector<int64_t, 4>> points) {
  PresburgerSet unionSet = s.unionSet(t);
  for (const SmallVector<int64_t, 4> &point : points) {
    bool inS = s.containsPoint(point);
    bool inT = t.containsPoint(point);
    bool inUnion = unionSet.containsPoint(point);
    EXPECT_EQ(inUnion, inS || inT);
  }
}

/// Compute the intersection of s and t, and check that each of the given points
/// belongs to the intersection iff it belongs to both s and t.
static void testIntersectAtPoints(const PresburgerSet &s,
                                  const PresburgerSet &t,
                                  ArrayRef<SmallVector<int64_t, 4>> points) {
  PresburgerSet intersection = s.intersect(t);
  for (const SmallVector<int64_t, 4> &point : points) {
    bool inS = s.containsPoint(point);
    bool inT = t.containsPoint(point);
    bool inIntersection = intersection.containsPoint(point);
    EXPECT_EQ(inIntersection, inS && inT);
  }
}

/// Compute the set difference s \ t, and check that each of the given points
/// belongs to the difference iff it belongs to s and does not belong to t.
static void testSubtractAtPoints(const PresburgerSet &s, const PresburgerSet &t,
                                 ArrayRef<SmallVector<int64_t, 4>> points) {
  PresburgerSet diff = s.subtract(t);
  for (const SmallVector<int64_t, 4> &point : points) {
    bool inS = s.containsPoint(point);
    bool inT = t.containsPoint(point);
    bool inDiff = diff.containsPoint(point);
    if (inT)
      EXPECT_FALSE(inDiff);
    else
      EXPECT_EQ(inDiff, inS);
  }
}

/// Compute the complement of s, and check that each of the given points
/// belongs to the complement iff it does not belong to s.
static void testComplementAtPoints(const PresburgerSet &s,
                                   ArrayRef<SmallVector<int64_t, 4>> points) {
  PresburgerSet complement = s.complement();
  complement.complement();
  for (const SmallVector<int64_t, 4> &point : points) {
    bool inS = s.containsPoint(point);
    bool inComplement = complement.containsPoint(point);
    if (inS)
      EXPECT_FALSE(inComplement);
    else
      EXPECT_TRUE(inComplement);
  }
}

/// Construct a PresburgerSet having `numDims` dimensions and no symbols from
/// the given list of IntegerPolyhedron. Each Poly in `polys` should also have
/// `numDims` dimensions and no symbols, although it can have any number of
/// local ids.
static PresburgerSet makeSetFromPoly(unsigned numDims,
                                     ArrayRef<IntegerPolyhedron> polys) {
  PresburgerSet set =
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(numDims));
  for (const IntegerPolyhedron &poly : polys)
    set.unionInPlace(poly);
  return set;
}

TEST(SetTest, containsPoint) {
  PresburgerSet setA = parsePresburgerSetFromPolyStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"});
  for (unsigned x = 0; x <= 21; ++x) {
    if ((2 <= x && x <= 8) || (10 <= x && x <= 20))
      EXPECT_TRUE(setA.containsPoint({x}));
    else
      EXPECT_FALSE(setA.containsPoint({x}));
  }

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} union
  // a square with opposite corners (2, 2) and (10, 10).
  PresburgerSet setB = parsePresburgerSetFromPolyStrings(
      2, {"(x,y) : (x + y - 4 >= 0, -x - y + 32 >= 0, "
          "x - y - 2 >= 0, -x + y + 16 >= 0)",
          "(x,y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, -y + 10 >= 0)"});

  for (unsigned x = 1; x <= 25; ++x) {
    for (unsigned y = -6; y <= 16; ++y) {
      if (4 <= x + y && x + y <= 32 && 2 <= x - y && x - y <= 16)
        EXPECT_TRUE(setB.containsPoint({x, y}));
      else if (2 <= x && x <= 10 && 2 <= y && y <= 10)
        EXPECT_TRUE(setB.containsPoint({x, y}));
      else
        EXPECT_FALSE(setB.containsPoint({x, y}));
    }
  }

  // The PresburgerSet has only one id, x, so we supply one value.
  EXPECT_TRUE(PresburgerSet(parsePoly("(x) : (x - 2*(x floordiv 2) == 0)"))
                  .containsPoint({0}));
}

TEST(SetTest, Union) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"});

  // Universe union set.
  testUnionAtPoints(PresburgerSet::getUniverse(PresburgerSpace::getSetSpace(1)),
                    set, {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set union set.
  testUnionAtPoints(PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(1)),
                    set, {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set union Universe.
  testUnionAtPoints(PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(1)),
                    PresburgerSet::getUniverse(PresburgerSpace::getSetSpace(1)),
                    {{1}, {2}, {0}, {-1}});

  // Universe union empty set.
  testUnionAtPoints(PresburgerSet::getUniverse(PresburgerSpace::getSetSpace(1)),
                    PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(1)),
                    {{1}, {2}, {0}, {-1}});

  // empty set union empty set.
  testUnionAtPoints(PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))),
                    PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))),
                    {{1}, {2}, {0}, {-1}});
}

TEST(SetTest, Intersect) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"});

  // Universe intersection set.
  testIntersectAtPoints(
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))), set,
      {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set intersection set.
  testIntersectAtPoints(
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))), set,
      {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set intersection Universe.
  testIntersectAtPoints(
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))),
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))),
      {{1}, {2}, {0}, {-1}});

  // Universe intersection empty set.
  testIntersectAtPoints(
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))),
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))),
      {{1}, {2}, {0}, {-1}});

  // Universe intersection Universe.
  testIntersectAtPoints(
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))),
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))),
      {{1}, {2}, {0}, {-1}});
}

TEST(SetTest, Subtract) {
  // The interval [2, 8] minus the interval [10, 20].
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(1, {"(x) : (x - 2 >= 0, -x + 8 >= 0)"}),
      parsePresburgerSetFromPolyStrings(1,
                                        {"(x) : (x - 10 >= 0, -x + 20 >= 0)"}),
      {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // Universe minus [2, 8] U [10, 20]
  testSubtractAtPoints(parsePresburgerSetFromPolyStrings(1, {"(x) : ()"}),
                       parsePresburgerSetFromPolyStrings(
                           1, {"(x) : (x - 2 >= 0, -x + 8 >= 0)",
                               "(x) : (x - 10 >= 0, -x + 20 >= 0)"}),
                       {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // ((-infinity, 0] U [3, 4] U [6, 7]) - ([2, 3] U [5, 6])
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(1, {"(x) : (-x >= 0)",
                                            "(x) : (x - 3 >= 0, -x + 4 >= 0)",
                                            "(x) : (x - 6 >= 0, -x + 7 >= 0)"}),
      parsePresburgerSetFromPolyStrings(1, {"(x) : (x - 2 >= 0, -x + 3 >= 0)",
                                            "(x) : (x - 5 >= 0, -x + 6 >= 0)"}),
      {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});

  // Expected result is {[x, y] : x > y}, i.e., {[x, y] : x >= y + 1}.
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(2, {"(x, y) : (x - y >= 0)"}),
      parsePresburgerSetFromPolyStrings(2, {"(x, y) : (x + y >= 0)"}),
      {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, -10) and (7, 100).
  // This splits the former rectangle into two halves, (2, 2) to (5, 10) and
  // (7, 2) to (10, 10).
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(
          2,
          {
              "(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, -y + 10 >= 0)",
          }),
      parsePresburgerSetFromPolyStrings(
          2,
          {
              "(x, y) : (x - 5 >= 0, y + 10 >= 0, -x + 7 >= 0, -y + 100 >= 0)",
          }),
      {{1, 2},  {2, 2},  {4, 2},  {5, 2},  {7, 2},  {8, 2},  {11, 2},
       {1, 1},  {2, 1},  {4, 1},  {5, 1},  {7, 1},  {8, 1},  {11, 1},
       {1, 10}, {2, 10}, {4, 10}, {5, 10}, {7, 10}, {8, 10}, {11, 10},
       {1, 11}, {2, 11}, {4, 11}, {5, 11}, {7, 11}, {8, 11}, {11, 11}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, 4) and (7, 8).
  // This creates a hole in the middle of the former rectangle, and the
  // resulting set can be represented as a union of four rectangles.
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(
          2, {"(x, y) : (x - 2 >= 0, y -2 >= 0, -x + 10 >= 0, -y + 10 >= 0)"}),
      parsePresburgerSetFromPolyStrings(
          2,
          {
              "(x, y) : (x - 5 >= 0, y - 4 >= 0, -x + 7 >= 0, -y + 8 >= 0)",
          }),
      {{1, 1},
       {2, 2},
       {10, 10},
       {11, 11},
       {5, 4},
       {7, 4},
       {5, 8},
       {7, 8},
       {4, 4},
       {8, 4},
       {4, 8},
       {8, 8}});

  // The second set is a superset of the first one, since on the line x + y = 0,
  // y <= 1 is equivalent to x >= -1. So the result is empty.
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(2, {"(x, y) : (x >= 0, x + y == 0)"}),
      parsePresburgerSetFromPolyStrings(2,
                                        {"(x, y) : (-y + 1 >= 0, x + y == 0)"}),
      {{0, 0},
       {1, -1},
       {2, -2},
       {-1, 1},
       {-2, 2},
       {1, 1},
       {-1, -1},
       {-1, 1},
       {1, -1}});

  // The result should be {0} U {2}.
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(1, {"(x) : (x >= 0, -x + 2 >= 0)"}),
      parsePresburgerSetFromPolyStrings(1, {"(x) : (x - 1 == 0)"}),
      {{-1}, {0}, {1}, {2}, {3}});

  // Sets with lots of redundant inequalities to test the redundancy heuristic.
  // (the heuristic is for the subtrahend, the second set which is the one being
  // subtracted)

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} minus
  // a triangle with vertices {(2, 2), (10, 2), (10, 10)}.
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(
          2,
          {
              "(x, y) : (x + y - 4 >= 0, -x - y + 32 >= 0, x - y - 2 >= 0, "
              "-x + y + 16 >= 0)",
          }),
      parsePresburgerSetFromPolyStrings(
          2, {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, "
              "-y + 10 >= 0, x + y - 2 >= 0, -x - y + 30 >= 0, x - y >= 0, "
              "-x + y + 10 >= 0)"}),
      {{1, 2},  {2, 2},   {3, 2},   {4, 2},  {1, 1},   {2, 1},   {3, 1},
       {4, 1},  {2, 0},   {3, 0},   {4, 0},  {5, 0},   {10, 2},  {11, 2},
       {10, 1}, {10, 10}, {10, 11}, {10, 9}, {11, 10}, {10, -6}, {11, -6},
       {24, 8}, {24, 7},  {17, 15}, {16, 15}});

  // ((-infinity, -5] U [3, 3] U [4, 4] U [5, 5]) - ([-2, -10] U [3, 4] U [6,
  // 7])
  testSubtractAtPoints(
      parsePresburgerSetFromPolyStrings(
          1, {"(x) : (-x - 5 >= 0)", "(x) : (x - 3 == 0)", "(x) : (x - 4 == 0)",
              "(x) : (x - 5 == 0)"}),
      parsePresburgerSetFromPolyStrings(
          1, {"(x) : (-x - 2 >= 0, x - 10 >= 0, -x >= 0, -x + 10 >= 0, "
              "x - 100 >= 0, x - 50 >= 0)",
              "(x) : (x - 3 >= 0, -x + 4 >= 0, x + 1 >= 0, "
              "x + 7 >= 0, -x + 10 >= 0)",
              "(x) : (x - 6 >= 0, -x + 7 >= 0, x + 1 >= 0, x - 3 >= 0, "
              "-x + 5 >= 0)"}),
      {{-6},
       {-5},
       {-4},
       {-9},
       {-10},
       {-11},
       {0},
       {1},
       {2},
       {3},
       {4},
       {5},
       {6},
       {7},
       {8}});
}

TEST(SetTest, Complement) {
  // Complement of universe.
  testComplementAtPoints(
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // Complement of empty set.
  testComplementAtPoints(
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1))),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  testComplementAtPoints(
      parsePresburgerSetFromPolyStrings(2, {"(x,y) : (x - 2 >= 0, y - 2 >= 0, "
                                            "-x + 10 >= 0, -y + 10 >= 0)"}),
      {{1, 1},
       {2, 1},
       {1, 2},
       {2, 2},
       {2, 3},
       {3, 2},
       {10, 10},
       {10, 11},
       {11, 10},
       {2, 10},
       {2, 11},
       {1, 10}});
}

TEST(SetTest, isEqual) {
  // set = [2, 8] U [10, 20].
  PresburgerSet universe =
      PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1)));
  PresburgerSet emptySet =
      PresburgerSet::getEmpty(PresburgerSpace::getSetSpace((1)));
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"});

  // universe != emptySet.
  EXPECT_FALSE(universe.isEqual(emptySet));
  // emptySet != universe.
  EXPECT_FALSE(emptySet.isEqual(universe));
  // emptySet == emptySet.
  EXPECT_TRUE(emptySet.isEqual(emptySet));
  // universe == universe.
  EXPECT_TRUE(universe.isEqual(universe));

  // universe U emptySet == universe.
  EXPECT_TRUE(universe.unionSet(emptySet).isEqual(universe));
  // universe U universe == universe.
  EXPECT_TRUE(universe.unionSet(universe).isEqual(universe));
  // emptySet U emptySet == emptySet.
  EXPECT_TRUE(emptySet.unionSet(emptySet).isEqual(emptySet));
  // universe U emptySet != emptySet.
  EXPECT_FALSE(universe.unionSet(emptySet).isEqual(emptySet));
  // universe U universe != emptySet.
  EXPECT_FALSE(universe.unionSet(universe).isEqual(emptySet));
  // emptySet U emptySet != universe.
  EXPECT_FALSE(emptySet.unionSet(emptySet).isEqual(universe));

  // set \ set == emptySet.
  EXPECT_TRUE(set.subtract(set).isEqual(emptySet));
  // set == set.
  EXPECT_TRUE(set.isEqual(set));
  // set U (universe \ set) == universe.
  EXPECT_TRUE(set.unionSet(set.complement()).isEqual(universe));
  // set U (universe \ set) != set.
  EXPECT_FALSE(set.unionSet(set.complement()).isEqual(set));
  // set != set U (universe \ set).
  EXPECT_FALSE(set.isEqual(set.unionSet(set.complement())));

  // square is one unit taller than rect.
  PresburgerSet square = parsePresburgerSetFromPolyStrings(
      2, {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 9 >= 0, -y + 9 >= 0)"});
  PresburgerSet rect = parsePresburgerSetFromPolyStrings(
      2, {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 9 >= 0, -y + 8 >= 0)"});
  EXPECT_FALSE(square.isEqual(rect));
  PresburgerSet universeRect = square.unionSet(square.complement());
  PresburgerSet universeSquare = rect.unionSet(rect.complement());
  EXPECT_TRUE(universeRect.isEqual(universeSquare));
  EXPECT_FALSE(universeRect.isEqual(rect));
  EXPECT_FALSE(universeSquare.isEqual(square));
  EXPECT_FALSE(rect.complement().isEqual(square.complement()));
}

void expectEqual(const PresburgerSet &s, const PresburgerSet &t) {
  EXPECT_TRUE(s.isEqual(t));
}

void expectEqual(const IntegerPolyhedron &s, const IntegerPolyhedron &t) {
  EXPECT_TRUE(s.isEqual(t));
}

void expectEmpty(const PresburgerSet &s) { EXPECT_TRUE(s.isIntegerEmpty()); }

TEST(SetTest, divisions) {
  // evens = {x : exists q, x = 2q}.
  PresburgerSet evens{parsePoly("(x) : (x - 2 * (x floordiv 2) == 0)")};

  //  odds = {x : exists q, x = 2q + 1}.
  PresburgerSet odds{parsePoly("(x) : (x - 2 * (x floordiv 2) - 1 == 0)")};

  // multiples3 = {x : exists q, x = 3q}.
  PresburgerSet multiples3{parsePoly("(x) : (x - 3 * (x floordiv 3) == 0)")};

  // multiples6 = {x : exists q, x = 6q}.
  PresburgerSet multiples6{parsePoly("(x) : (x - 6 * (x floordiv 6) == 0)")};

  // evens /\ odds = empty.
  expectEmpty(PresburgerSet(evens).intersect(PresburgerSet(odds)));
  // evens U odds = universe.
  expectEqual(evens.unionSet(odds),
              PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))));
  expectEqual(evens.complement(), odds);
  expectEqual(odds.complement(), evens);
  // even multiples of 3 = multiples of 6.
  expectEqual(multiples3.intersect(evens), multiples6);

  PresburgerSet setA{parsePoly("(x) : (-x >= 0)")};
  PresburgerSet setB{parsePoly("(x) : (x floordiv 2 - 4 >= 0)")};
  EXPECT_TRUE(setA.subtract(setB).isEqual(setA));
}

void convertSuffixDimsToLocals(IntegerPolyhedron &poly, unsigned numLocals) {
  poly.convertVarKind(VarKind::SetDim, poly.getNumDimVars() - numLocals,
                      poly.getNumDimVars(), VarKind::Local);
}

inline IntegerPolyhedron parsePolyAndMakeLocals(StringRef str,
                                                unsigned numLocals) {
  IntegerPolyhedron poly = parsePoly(str);
  convertSuffixDimsToLocals(poly, numLocals);
  return poly;
}

TEST(SetTest, divisionsDefByEq) {
  // evens = {x : exists q, x = 2q}.
  PresburgerSet evens{
      parsePolyAndMakeLocals("(x, y) : (x - 2 * y == 0)", /*numLocals=*/1)};

  //  odds = {x : exists q, x = 2q + 1}.
  PresburgerSet odds{
      parsePolyAndMakeLocals("(x, y) : (x - 2 * y - 1 == 0)", /*numLocals=*/1)};

  // multiples3 = {x : exists q, x = 3q}.
  PresburgerSet multiples3{
      parsePolyAndMakeLocals("(x, y) : (x - 3 * y == 0)", /*numLocals=*/1)};

  // multiples6 = {x : exists q, x = 6q}.
  PresburgerSet multiples6{
      parsePolyAndMakeLocals("(x, y) : (x - 6 * y == 0)", /*numLocals=*/1)};

  // evens /\ odds = empty.
  expectEmpty(PresburgerSet(evens).intersect(PresburgerSet(odds)));
  // evens U odds = universe.
  expectEqual(evens.unionSet(odds),
              PresburgerSet::getUniverse(PresburgerSpace::getSetSpace((1))));
  expectEqual(evens.complement(), odds);
  expectEqual(odds.complement(), evens);
  // even multiples of 3 = multiples of 6.
  expectEqual(multiples3.intersect(evens), multiples6);

  PresburgerSet evensDefByIneq{
      parsePoly("(x) : (x - 2 * (x floordiv 2) == 0)")};
  expectEqual(evens, PresburgerSet(evensDefByIneq));
}

TEST(SetTest, divisionNonDivLocals) {
  // This is a tetrahedron with vertices at
  // (1/3, 0, 0), (2/3, 0, 0), (2/3, 0, 1000), and (1000, 1000, 1000).
  //
  // The only integer point in this is at (1000, 1000, 1000).
  // We project this to the xy plane.
  IntegerPolyhedron tetrahedron =
      parsePolyAndMakeLocals("(x, y, z) : (y >= 0, z - y >= 0, 3000*x - 2998*y "
                             "- 1000 - z >= 0, -1500*x + 1499*y + 1000 >= 0)",
                             /*numLocals=*/1);

  // This is a triangle with vertices at (1/3, 0), (2/3, 0) and (1000, 1000).
  // The only integer point in this is at (1000, 1000).
  //
  // It also happens to be the projection of the above onto the xy plane.
  IntegerPolyhedron triangle = parsePoly("(x,y) : (y >= 0, "
                                         "3000 * x - 2999 * y - 1000 >= 0, "
                                         "-3000 * x + 2998 * y + 2000 >= 0)");
  EXPECT_TRUE(triangle.containsPoint({1000, 1000}));
  EXPECT_FALSE(triangle.containsPoint({1001, 1001}));
  expectEqual(triangle, tetrahedron);

  convertSuffixDimsToLocals(triangle, 1);
  IntegerPolyhedron line = parsePoly("(x) : (x - 1000 == 0)");
  expectEqual(line, triangle);

  // Triangle with vertices (0, 0), (5, 0), (15, 5).
  // Projected on x, it becomes [0, 13] U {15} as it becomes too narrow towards
  // the apex and so does not have have any integer point at x = 14.
  // At x = 15, the apex is an integer point.
  PresburgerSet triangle2{parsePolyAndMakeLocals("(x,y) : (y >= 0, "
                                                 "x - 3*y >= 0, "
                                                 "2*y - x + 5 >= 0)",
                                                 /*numLocals=*/1)};
  PresburgerSet zeroToThirteen{parsePoly("(x) : (13 - x >= 0, x >= 0)")};
  PresburgerSet fifteen{parsePoly("(x) : (x - 15 == 0)")};
  expectEqual(triangle2.subtract(zeroToThirteen), fifteen);
}

TEST(SetTest, subtractDuplicateDivsRegression) {
  // Previously, subtracting sets with duplicate divs might result in crashes
  // due to existing divs being removed when merging local ids, due to being
  // identified as being duplicates for the first time.
  IntegerPolyhedron setA(PresburgerSpace::getSetSpace(1));
  setA.addLocalFloorDiv({1, 0}, 2);
  setA.addLocalFloorDiv({1, 0, 0}, 2);
  EXPECT_TRUE(setA.isEqual(setA));
}

/// Coalesce `set` and check that the `newSet` is equal to `set` and that
/// `expectedNumPoly` matches the number of Poly in the coalesced set.
void expectCoalesce(size_t expectedNumPoly, const PresburgerSet &set) {
  PresburgerSet newSet = set.coalesce();
  EXPECT_TRUE(set.isEqual(newSet));
  EXPECT_TRUE(expectedNumPoly == newSet.getNumDisjuncts());
}

TEST(SetTest, coalesceNoPoly) {
  PresburgerSet set = makeSetFromPoly(0, {});
  expectCoalesce(0, set);
}

TEST(SetTest, coalesceContainedOneDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : (x >= 0, -x + 4 >= 0)", "(x) : (x - 1 >= 0, -x + 2 >= 0)"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstEmpty) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ( x >= 0, -x - 1 >= 0)", "(x) : ( x - 1 >= 0, -x + 2 >= 0)"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSecondEmpty) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : (x - 1 >= 0, -x + 2 >= 0)", "(x) : (x >= 0, -x - 1 >= 0)"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceBothEmpty) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : (x - 3 >= 0, -x - 1 >= 0)", "(x) : (x >= 0, -x - 1 >= 0)"});
  expectCoalesce(0, set);
}

TEST(SetTest, coalesceFirstUniv) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ()", "(x) : ( x >= 0, -x + 1 >= 0)"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSecondUniv) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ( x >= 0, -x + 1 >= 0)", "(x) : ()"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceBothUniv) {
  PresburgerSet set =
      parsePresburgerSetFromPolyStrings(1, {"(x) : ()", "(x) : ()"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstUnivSecondEmpty) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ()", "(x) : ( x >= 0, -x - 1 >= 0)"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstEmptySecondUniv) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ( x >= 0, -x - 1 >= 0)", "(x) : ()"});
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCutOneDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {
             "(x) : ( x >= 0, -x + 3 >= 0)",
             "(x) : ( x - 2 >= 0, -x + 4 >= 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSeparateOneDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ( x >= 0, -x + 2 >= 0)", "(x) : ( x - 3 >= 0, -x + 4 >= 0)"});
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceAdjEq) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {"(x) : ( x == 0)", "(x) : ( x - 1 == 0)"});
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedTwoDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 3 >= 0)",
             "(x,y) : (x >= 0, -x + 3 >= 0, y - 2 >= 0, -y + 3 >= 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCutTwoDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 2 >= 0)",
             "(x,y) : (x >= 0, -x + 3 >= 0, y - 1 >= 0, -y + 3 >= 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceEqStickingOut) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 2 >= 0, y >= 0, -y + 2 >= 0)",
             "(x,y) : (y - 1 == 0, x >= 0, -x + 3 >= 0)",
         });
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceSeparateTwoDim) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 1 >= 0)",
             "(x,y) : (x >= 0, -x + 3 >= 0, y - 2 >= 0, -y + 3 >= 0)",
         });
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedEq) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 3 >= 0, x - y == 0)",
             "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCuttingEq) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x + 1 >= 0, -x + 1 >= 0, x - y == 0)",
             "(x,y) : (x >= 0, -x + 2 >= 0, x - y == 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSeparateEq) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x - 3 >= 0, -x + 4 >= 0, x - y == 0)",
             "(x,y) : (x >= 0, -x + 1 >= 0, x - y == 0)",
         });
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedEqAsIneq) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x >= 0, -x + 3 >= 0, x - y >= 0, -x + y >= 0)",
             "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceContainedEqComplex) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      2, {
             "(x,y) : (x - 2 == 0, x - y == 0)",
             "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
         });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceThreeContained) {
  PresburgerSet set =
      parsePresburgerSetFromPolyStrings(1, {
                                               "(x) : (x >= 0, -x + 1 >= 0)",
                                               "(x) : (x >= 0, -x + 2 >= 0)",
                                               "(x) : (x >= 0, -x + 3 >= 0)",
                                           });
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceDoubleIncrement) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {
             "(x) : (x == 0)",
             "(x) : (x - 2 == 0)",
             "(x) : (x + 2 == 0)",
             "(x) : (x - 2 >= 0, -x + 3 >= 0)",
         });
  expectCoalesce(3, set);
}

TEST(SetTest, coalesceLastCoalesced) {
  PresburgerSet set = parsePresburgerSetFromPolyStrings(
      1, {
             "(x) : (x == 0)",
             "(x) : (x - 1 >= 0, -x + 3 >= 0)",
             "(x) : (x + 2 == 0)",
             "(x) : (x - 2 >= 0, -x + 4 >= 0)",
         });
  expectCoalesce(3, set);
}

TEST(SetTest, coalesceDiv) {
  PresburgerSet set =
      parsePresburgerSetFromPolyStrings(1, {
                                               "(x) : (x floordiv 2 == 0)",
                                               "(x) : (x floordiv 2 - 1 == 0)",
                                           });
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceDivOtherContained) {
  PresburgerSet set =
      parsePresburgerSetFromPolyStrings(1, {
                                               "(x) : (x floordiv 2 == 0)",
                                               "(x) : (x == 0)",
                                               "(x) : (x >= 0, -x + 1 >= 0)",
                                           });
  expectCoalesce(2, set);
}

static void
expectComputedVolumeIsValidOverapprox(const PresburgerSet &set,
                                      Optional<int64_t> trueVolume,
                                      Optional<int64_t> resultBound) {
  expectComputedVolumeIsValidOverapprox(set.computeVolume(), trueVolume,
                                        resultBound);
}

TEST(SetTest, computeVolume) {
  // Diamond with vertices at (0, 0), (5, 5), (5, 5), (10, 0).
  PresburgerSet diamond(
      parsePoly("(x, y) : (x + y >= 0, -x - y + 10 >= 0, x - y >= 0, -x + y + "
                "10 >= 0)"));
  expectComputedVolumeIsValidOverapprox(diamond,
                                        /*trueVolume=*/61ull,
                                        /*resultBound=*/121ull);

  // Diamond with vertices at (-5, 0), (0, -5), (0, 5), (5, 0).
  PresburgerSet shiftedDiamond(parsePoly(
      "(x, y) : (x + y + 5 >= 0, -x - y + 5 >= 0, x - y + 5 >= 0, -x + y + "
      "5 >= 0)"));
  expectComputedVolumeIsValidOverapprox(shiftedDiamond,
                                        /*trueVolume=*/61ull,
                                        /*resultBound=*/121ull);

  // Diamond with vertices at (-5, 0), (5, -10), (5, 10), (15, 0).
  PresburgerSet biggerDiamond(parsePoly(
      "(x, y) : (x + y + 5 >= 0, -x - y + 15 >= 0, x - y + 5 >= 0, -x + y + "
      "15 >= 0)"));
  expectComputedVolumeIsValidOverapprox(biggerDiamond,
                                        /*trueVolume=*/221ull,
                                        /*resultBound=*/441ull);

  // There is some overlap between diamond and shiftedDiamond.
  expectComputedVolumeIsValidOverapprox(diamond.unionSet(shiftedDiamond),
                                        /*trueVolume=*/104ull,
                                        /*resultBound=*/242ull);

  // biggerDiamond subsumes both the small ones.
  expectComputedVolumeIsValidOverapprox(
      diamond.unionSet(shiftedDiamond).unionSet(biggerDiamond),
      /*trueVolume=*/221ull,
      /*resultBound=*/683ull);

  // Unbounded polytope.
  PresburgerSet unbounded(parsePoly("(x, y) : (2*x - y >= 0, y - 3*x >= 0)"));
  expectComputedVolumeIsValidOverapprox(unbounded, /*trueVolume=*/{},
                                        /*resultBound=*/{});

  // Union of unbounded with bounded is unbounded.
  expectComputedVolumeIsValidOverapprox(unbounded.unionSet(diamond),
                                        /*trueVolume=*/{},
                                        /*resultBound=*/{});
}

// The last `numToProject` dims will be projected out, i.e., converted to
// locals.
void testComputeReprAtPoints(IntegerPolyhedron poly,
                             ArrayRef<SmallVector<int64_t, 4>> points,
                             unsigned numToProject) {
  poly.convertVarKind(VarKind::SetDim, poly.getNumDimVars() - numToProject,
                      poly.getNumDimVars(), VarKind::Local);
  PresburgerRelation repr = poly.computeReprWithOnlyDivLocals();
  EXPECT_TRUE(repr.hasOnlyDivLocals());
  EXPECT_TRUE(repr.getSpace().isCompatible(poly.getSpace()));
  for (const SmallVector<int64_t, 4> &point : points) {
    EXPECT_EQ(poly.containsPointNoLocal(point).has_value(),
              repr.containsPoint(point));
  }
}

void testComputeRepr(IntegerPolyhedron poly, const PresburgerSet &expected,
                     unsigned numToProject) {
  poly.convertVarKind(VarKind::SetDim, poly.getNumDimVars() - numToProject,
                      poly.getNumDimVars(), VarKind::Local);
  PresburgerRelation repr = poly.computeReprWithOnlyDivLocals();
  EXPECT_TRUE(repr.hasOnlyDivLocals());
  EXPECT_TRUE(repr.getSpace().isCompatible(poly.getSpace()));
  EXPECT_TRUE(repr.isEqual(expected));
}

TEST(SetTest, computeReprWithOnlyDivLocals) {
  testComputeReprAtPoints(parsePoly("(x, y) : (x - 2*y == 0)"),
                          {{1, 0}, {2, 1}, {3, 0}, {4, 2}, {5, 3}},
                          /*numToProject=*/0);
  testComputeReprAtPoints(parsePoly("(x, e) : (x - 2*e == 0)"),
                          {{1}, {2}, {3}, {4}, {5}}, /*numToProject=*/1);

  // Tests to check that the space is preserved.
  testComputeReprAtPoints(parsePoly("(x, y)[z, w] : ()"), {},
                          /*numToProject=*/1);
  testComputeReprAtPoints(parsePoly("(x, y)[z, w] : (z - (w floordiv 2) == 0)"),
                          {},
                          /*numToProject=*/1);

  // Bezout's lemma: if a, b are constants,
  // the set of values that ax + by can take is all multiples of gcd(a, b).
  testComputeRepr(
      parsePoly("(x, e, f) : (x - 15*e - 21*f == 0)"),
      PresburgerSet(parsePoly({"(x) : (x - 3*(x floordiv 3) == 0)"})),
      /*numToProject=*/2);
}

TEST(SetTest, subtractOutputSizeRegression) {
  PresburgerSet set1 =
      parsePresburgerSetFromPolyStrings(1, {"(i) : (i >= 0, 10 - i >= 0)"});
  PresburgerSet set2 =
      parsePresburgerSetFromPolyStrings(1, {"(i) : (i - 5 >= 0)"});

  PresburgerSet set3 =
      parsePresburgerSetFromPolyStrings(1, {"(i) : (i >= 0, 4 - i >= 0)"});

  PresburgerSet result = set1.subtract(set2);

  EXPECT_TRUE(result.isEqual(set3));

  // Previously, the subtraction result was producing an extra empty set, which
  // is correct, but bad for output size.
  EXPECT_EQ(result.getNumDisjuncts(), 1u);

  PresburgerSet subtractSelf = set1.subtract(set1);
  EXPECT_TRUE(subtractSelf.isIntegerEmpty());
  // Previously, the subtraction result was producing several unnecessary empty
  // sets, which is correct, but bad for output size.
  EXPECT_EQ(subtractSelf.getNumDisjuncts(), 0u);
}
