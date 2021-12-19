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

#include "mlir/Analysis/PresburgerSet.h"
#include "./AffineStructuresParser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

/// Parses a FlatAffineConstraints from a StringRef. It is expected that the
/// string represents a valid IntegerSet, otherwise it will violate a gtest
/// assertion.
static FlatAffineConstraints parseFAC(StringRef str, MLIRContext *context) {
  FailureOr<FlatAffineConstraints> fac = parseIntegerSetToFAC(str, context);

  EXPECT_TRUE(succeeded(fac));

  return *fac;
}

/// Parse a list of StringRefs to FlatAffineConstraints and combine them into a
/// PresburgerSet be using the union operation. It is expected that the strings
/// are all valid IntegerSet representation and that all of them have the same
/// number of dimensions as is specified by the numDims argument.
static PresburgerSet parsePresburgerSetFromFACStrings(unsigned numDims,
                                                      ArrayRef<StringRef> strs,
                                                      MLIRContext *context) {
  PresburgerSet set = PresburgerSet::getEmptySet(numDims);
  for (StringRef str : strs)
    set.unionFACInPlace(parseFAC(str, context));
  return set;
}

/// Compute the union of s and t, and check that each of the given points
/// belongs to the union iff it belongs to at least one of s and t.
static void testUnionAtPoints(PresburgerSet s, PresburgerSet t,
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
static void testIntersectAtPoints(PresburgerSet s, PresburgerSet t,
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
static void testSubtractAtPoints(PresburgerSet s, PresburgerSet t,
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
static void testComplementAtPoints(PresburgerSet s,
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
/// the given list of FlatAffineConstraints. Each FAC in `facs` should also have
/// `numDims` dimensions and no symbols, although it can have any number of
/// local ids.
static PresburgerSet makeSetFromFACs(unsigned numDims,
                                     ArrayRef<FlatAffineConstraints> facs) {
  PresburgerSet set = PresburgerSet::getEmptySet(numDims);
  for (const FlatAffineConstraints &fac : facs)
    set.unionFACInPlace(fac);
  return set;
}

TEST(SetTest, containsPoint) {
  MLIRContext context;

  PresburgerSet setA = parsePresburgerSetFromFACStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"},
      &context);
  for (unsigned x = 0; x <= 21; ++x) {
    if ((2 <= x && x <= 8) || (10 <= x && x <= 20))
      EXPECT_TRUE(setA.containsPoint({x}));
    else
      EXPECT_FALSE(setA.containsPoint({x}));
  }

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} union
  // a square with opposite corners (2, 2) and (10, 10).
  PresburgerSet setB = parsePresburgerSetFromFACStrings(
      2,
      {"(x,y) : (x + y - 4 >= 0, -x - y + 32 >= 0, "
       "x - y - 2 >= 0, -x + y + 16 >= 0)",
       "(x,y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, -y + 10 >= 0)"},
      &context);

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
}

TEST(SetTest, Union) {
  MLIRContext context;

  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"},
      &context);

  // Universe union set.
  testUnionAtPoints(PresburgerSet::getUniverse(1), set,
                    {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set union set.
  testUnionAtPoints(PresburgerSet::getEmptySet(1), set,
                    {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set union Universe.
  testUnionAtPoints(PresburgerSet::getEmptySet(1),
                    PresburgerSet::getUniverse(1), {{1}, {2}, {0}, {-1}});

  // Universe union empty set.
  testUnionAtPoints(PresburgerSet::getUniverse(1),
                    PresburgerSet::getEmptySet(1), {{1}, {2}, {0}, {-1}});

  // empty set union empty set.
  testUnionAtPoints(PresburgerSet::getEmptySet(1),
                    PresburgerSet::getEmptySet(1), {{1}, {2}, {0}, {-1}});
}

TEST(SetTest, Intersect) {
  MLIRContext context;

  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"},
      &context);

  // Universe intersection set.
  testIntersectAtPoints(PresburgerSet::getUniverse(1), set,
                        {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set intersection set.
  testIntersectAtPoints(PresburgerSet::getEmptySet(1), set,
                        {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // empty set intersection Universe.
  testIntersectAtPoints(PresburgerSet::getEmptySet(1),
                        PresburgerSet::getUniverse(1), {{1}, {2}, {0}, {-1}});

  // Universe intersection empty set.
  testIntersectAtPoints(PresburgerSet::getUniverse(1),
                        PresburgerSet::getEmptySet(1), {{1}, {2}, {0}, {-1}});

  // Universe intersection Universe.
  testIntersectAtPoints(PresburgerSet::getUniverse(1),
                        PresburgerSet::getUniverse(1), {{1}, {2}, {0}, {-1}});
}

TEST(SetTest, Subtract) {
  MLIRContext context;
  // The interval [2, 8] minus the interval [10, 20].
  testSubtractAtPoints(parsePresburgerSetFromFACStrings(
                           1, {"(x) : (x - 2 >= 0, -x + 8 >= 0)"}, &context),
                       parsePresburgerSetFromFACStrings(
                           1, {"(x) : (x - 10 >= 0, -x + 20 >= 0)"}, &context),
                       {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // Universe minus [2, 8] U [10, 20]
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(1, {"(x) : ()"}, &context),
      parsePresburgerSetFromFACStrings(1,
                                       {"(x) : (x - 2 >= 0, -x + 8 >= 0)",
                                        "(x) : (x - 10 >= 0, -x + 20 >= 0)"},
                                       &context),
      {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // ((-infinity, 0] U [3, 4] U [6, 7]) - ([2, 3] U [5, 6])
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(1,
                                       {"(x) : (-x >= 0)",
                                        "(x) : (x - 3 >= 0, -x + 4 >= 0)",
                                        "(x) : (x - 6 >= 0, -x + 7 >= 0)"},
                                       &context),
      parsePresburgerSetFromFACStrings(1,
                                       {"(x) : (x - 2 >= 0, -x + 3 >= 0)",
                                        "(x) : (x - 5 >= 0, -x + 6 >= 0)"},
                                       &context),
      {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});

  // Expected result is {[x, y] : x > y}, i.e., {[x, y] : x >= y + 1}.
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(2, {"(x, y) : (x - y >= 0)"}, &context),
      parsePresburgerSetFromFACStrings(2, {"(x, y) : (x + y >= 0)"}, &context),
      {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, -10) and (7, 100).
  // This splits the former rectangle into two halves, (2, 2) to (5, 10) and
  // (7, 2) to (10, 10).
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(
          2,
          {
              "(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, -y + 10 >= 0)",
          },
          &context),
      parsePresburgerSetFromFACStrings(
          2,
          {
              "(x, y) : (x - 5 >= 0, y + 10 >= 0, -x + 7 >= 0, -y + 100 >= 0)",
          },
          &context),
      {{1, 2},  {2, 2},  {4, 2},  {5, 2},  {7, 2},  {8, 2},  {11, 2},
       {1, 1},  {2, 1},  {4, 1},  {5, 1},  {7, 1},  {8, 1},  {11, 1},
       {1, 10}, {2, 10}, {4, 10}, {5, 10}, {7, 10}, {8, 10}, {11, 10},
       {1, 11}, {2, 11}, {4, 11}, {5, 11}, {7, 11}, {8, 11}, {11, 11}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, 4) and (7, 8).
  // This creates a hole in the middle of the former rectangle, and the
  // resulting set can be represented as a union of four rectangles.
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(
          2, {"(x, y) : (x - 2 >= 0, y -2 >= 0, -x + 10 >= 0, -y + 10 >= 0)"},
          &context),
      parsePresburgerSetFromFACStrings(
          2,
          {
              "(x, y) : (x - 5 >= 0, y - 4 >= 0, -x + 7 >= 0, -y + 8 >= 0)",
          },
          &context),
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
  testSubtractAtPoints(parsePresburgerSetFromFACStrings(
                           2, {"(x, y) : (x >= 0, x + y == 0)"}, &context),
                       parsePresburgerSetFromFACStrings(
                           2, {"(x, y) : (-y + 1 >= 0, x + y == 0)"}, &context),
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
      parsePresburgerSetFromFACStrings(1, {"(x) : (x >= 0, -x + 2 >= 0)"},
                                       &context),
      parsePresburgerSetFromFACStrings(1, {"(x) : (x - 1 == 0)"}, &context),
      {{-1}, {0}, {1}, {2}, {3}});

  // Sets with lots of redundant inequalities to test the redundancy heuristic.
  // (the heuristic is for the subtrahend, the second set which is the one being
  // subtracted)

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} minus
  // a triangle with vertices {(2, 2), (10, 2), (10, 10)}.
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(
          2,
          {
              "(x, y) : (x + y - 4 >= 0, -x - y + 32 >= 0, x - y - 2 >= 0, "
              "-x + y + 16 >= 0)",
          },
          &context),
      parsePresburgerSetFromFACStrings(
          2,
          {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 10 >= 0, "
           "-y + 10 >= 0, x + y - 2 >= 0, -x - y + 30 >= 0, x - y >= 0, "
           "-x + y + 10 >= 0)"},
          &context),
      {{1, 2},  {2, 2},   {3, 2},   {4, 2},  {1, 1},   {2, 1},   {3, 1},
       {4, 1},  {2, 0},   {3, 0},   {4, 0},  {5, 0},   {10, 2},  {11, 2},
       {10, 1}, {10, 10}, {10, 11}, {10, 9}, {11, 10}, {10, -6}, {11, -6},
       {24, 8}, {24, 7},  {17, 15}, {16, 15}});

  // ((-infinity, -5] U [3, 3] U [4, 4] U [5, 5]) - ([-2, -10] U [3, 4] U [6,
  // 7])
  testSubtractAtPoints(
      parsePresburgerSetFromFACStrings(
          1,
          {"(x) : (-x - 5 >= 0)", "(x) : (x - 3 == 0)", "(x) : (x - 4 == 0)",
           "(x) : (x - 5 == 0)"},
          &context),
      parsePresburgerSetFromFACStrings(
          1,
          {"(x) : (-x - 2 >= 0, x - 10 >= 0, -x >= 0, -x + 10 >= 0, "
           "x - 100 >= 0, x - 50 >= 0)",
           "(x) : (x - 3 >= 0, -x + 4 >= 0, x + 1 >= 0, "
           "x + 7 >= 0, -x + 10 >= 0)",
           "(x) : (x - 6 >= 0, -x + 7 >= 0, x + 1 >= 0, x - 3 >= 0, "
           "-x + 5 >= 0)"},
          &context),
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

  MLIRContext context;
  // Complement of universe.
  testComplementAtPoints(
      PresburgerSet::getUniverse(1),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // Complement of empty set.
  testComplementAtPoints(
      PresburgerSet::getEmptySet(1),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  testComplementAtPoints(
      parsePresburgerSetFromFACStrings(2,
                                       {"(x,y) : (x - 2 >= 0, y - 2 >= 0, "
                                        "-x + 10 >= 0, -y + 10 >= 0)"},
                                       &context),
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

  MLIRContext context;
  // set = [2, 8] U [10, 20].
  PresburgerSet universe = PresburgerSet::getUniverse(1);
  PresburgerSet emptySet = PresburgerSet::getEmptySet(1);
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1,
      {"(x) : (x - 2 >= 0, -x + 8 >= 0)", "(x) : (x - 10 >= 0, -x + 20 >= 0)"},
      &context);

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
  PresburgerSet square = parsePresburgerSetFromFACStrings(
      2, {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 9 >= 0, -y + 9 >= 0)"},
      &context);
  PresburgerSet rect = parsePresburgerSetFromFACStrings(
      2, {"(x, y) : (x - 2 >= 0, y - 2 >= 0, -x + 9 >= 0, -y + 8 >= 0)"},
      &context);
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

void expectEmpty(PresburgerSet s) { EXPECT_TRUE(s.isIntegerEmpty()); }

TEST(SetTest, divisions) {
  MLIRContext context;

  // evens = {x : exists q, x = 2q}.
  PresburgerSet evens{
      parseFAC("(x) : (x - 2 * (x floordiv 2) == 0)", &context)};

  //  odds = {x : exists q, x = 2q + 1}.
  PresburgerSet odds{
      parseFAC("(x) : (x - 2 * (x floordiv 2) - 1 == 0)", &context)};

  // multiples3 = {x : exists q, x = 3q}.
  PresburgerSet multiples3{
      parseFAC("(x) : (x - 3 * (x floordiv 3) == 0)", &context)};

  // multiples6 = {x : exists q, x = 6q}.
  PresburgerSet multiples6{
      parseFAC("(x) : (x - 6 * (x floordiv 6) == 0)", &context)};

  // evens /\ odds = empty.
  expectEmpty(PresburgerSet(evens).intersect(PresburgerSet(odds)));
  // evens U odds = universe.
  expectEqual(evens.unionSet(odds), PresburgerSet::getUniverse(1));
  expectEqual(evens.complement(), odds);
  expectEqual(odds.complement(), evens);
  // even multiples of 3 = multiples of 6.
  expectEqual(multiples3.intersect(evens), multiples6);

  PresburgerSet setA{parseFAC("(x) : (-x >= 0)", &context)};
  PresburgerSet setB{parseFAC("(x) : (x floordiv 2 - 4 >= 0)", &context)};
  EXPECT_TRUE(setA.subtract(setB).isEqual(setA));
}

/// Coalesce `set` and check that the `newSet` is equal to `set and that
/// `expectedNumFACs` matches the number of FACs in the coalesced set.
/// If one of the two
void expectCoalesce(size_t expectedNumFACs, const PresburgerSet set) {
  PresburgerSet newSet = set.coalesce();
  EXPECT_TRUE(set.isEqual(newSet));
  EXPECT_TRUE(expectedNumFACs == newSet.getNumFACs());
}

TEST(SetTest, coalesceNoFAC) {
  PresburgerSet set = makeSetFromFACs(0, {});
  expectCoalesce(0, set);
}

TEST(SetTest, coalesceContainedOneDim) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : (x >= 0, -x + 4 >= 0)", "(x) : (x - 1 >= 0, -x + 2 >= 0)"},
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstEmpty) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ( x >= 0, -x - 1 >= 0)", "(x) : ( x - 1 >= 0, -x + 2 >= 0)"},
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSecondEmpty) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : (x - 1 >= 0, -x + 2 >= 0)", "(x) : (x >= 0, -x - 1 >= 0)"},
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceBothEmpty) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : (x - 3 >= 0, -x - 1 >= 0)", "(x) : (x >= 0, -x - 1 >= 0)"},
      &context);
  expectCoalesce(0, set);
}

TEST(SetTest, coalesceFirstUniv) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ()", "(x) : ( x >= 0, -x + 1 >= 0)"}, &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceSecondUniv) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ( x >= 0, -x + 1 >= 0)", "(x) : ()"}, &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceBothUniv) {
  MLIRContext context;
  PresburgerSet set =
      parsePresburgerSetFromFACStrings(1, {"(x) : ()", "(x) : ()"}, &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstUnivSecondEmpty) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ()", "(x) : ( x >= 0, -x - 1 >= 0)"}, &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceFirstEmptySecondUniv) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ( x >= 0, -x - 1 >= 0)", "(x) : ()"}, &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCutOneDim) {
  MLIRContext context;
  PresburgerSet set =
      parsePresburgerSetFromFACStrings(1,
                                       {
                                           "(x) : ( x >= 0, -x + 3 >= 0)",
                                           "(x) : ( x - 2 >= 0, -x + 4 >= 0)",
                                       },
                                       &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceSeparateOneDim) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      1, {"(x) : ( x >= 0, -x + 2 >= 0)", "(x) : ( x - 3 >= 0, -x + 4 >= 0)"},
      &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedTwoDim) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 3 >= 0)",
          "(x,y) : (x >= 0, -x + 3 >= 0, y - 2 >= 0, -y + 3 >= 0)",
      },
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCutTwoDim) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 2 >= 0)",
          "(x,y) : (x >= 0, -x + 3 >= 0, y - 1 >= 0, -y + 3 >= 0)",
      },
      &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceSeparateTwoDim) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x >= 0, -x + 3 >= 0, y >= 0, -y + 1 >= 0)",
          "(x,y) : (x >= 0, -x + 3 >= 0, y - 2 >= 0, -y + 3 >= 0)",
      },
      &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedEq) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x >= 0, -x + 3 >= 0, x - y == 0)",
          "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
      },
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceCuttingEq) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x - 1 >= 0, -x + 3 >= 0, x - y == 0)",
          "(x,y) : (x >= 0, -x + 2 >= 0, x - y == 0)",
      },
      &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceSeparateEq) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x - 3 >= 0, -x + 4 >= 0, x - y == 0)",
          "(x,y) : (x >= 0, -x + 1 >= 0, x - y == 0)",
      },
      &context);
  expectCoalesce(2, set);
}

TEST(SetTest, coalesceContainedEqAsIneq) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x >= 0, -x + 3 >= 0, x - y >= 0, -x + y >= 0)",
          "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
      },
      &context);
  expectCoalesce(1, set);
}

TEST(SetTest, coalesceContainedEqComplex) {
  MLIRContext context;
  PresburgerSet set = parsePresburgerSetFromFACStrings(
      2,
      {
          "(x,y) : (x - 2 == 0, x - y == 0)",
          "(x,y) : (x - 1 >= 0, -x + 2 >= 0, x - y == 0)",
      },
      &context);
  expectCoalesce(1, set);
}

} // namespace mlir
