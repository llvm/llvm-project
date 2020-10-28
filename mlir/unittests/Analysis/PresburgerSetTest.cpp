//===- SetTest.cpp - Tests for PresburgerSet ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for PresburgerSet. Each test works by computing
// an operation (union, intersection, subtract, or complement) on two sets
// and checking, for a set of points, that the resulting set contains the point
// iff the result is supposed to contain it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/PresburgerSet.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

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
/// belongs to the intersection iff it belongs to both of s and t.
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

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints
makeFACFromConstraints(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + 1, dims);
  for (const SmallVector<int64_t, 4> &eq : eqs)
    fac.addEquality(eq);
  for (const SmallVector<int64_t, 4> &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

static FlatAffineConstraints
makeFACFromIneqs(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs) {
  return makeFACFromConstraints(dims, ineqs, {});
}

static PresburgerSet makeSetFromFACs(unsigned dims,
                                     ArrayRef<FlatAffineConstraints> facs) {
  PresburgerSet set = PresburgerSet::getEmptySet(dims);
  for (const FlatAffineConstraints &fac : facs)
    set.unionFACInPlace(fac);
  return set;
}

TEST(SetTest, containsPoint) {
  PresburgerSet setA =
      makeSetFromFACs(1, {
                             makeFACFromIneqs(1, {{1, -2},    // x >= 2.
                                                  {-1, 8}}),  // x <= 8.
                             makeFACFromIneqs(1, {{1, -10},   // x >= 10.
                                                  {-1, 20}}), // x <= 20.
                         });
  for (unsigned x = 0; x <= 21; ++x) {
    if ((2 <= x && x <= 8) || (10 <= x && x <= 20))
      EXPECT_TRUE(setA.containsPoint({x}));
    else
      EXPECT_FALSE(setA.containsPoint({x}));
  }

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} union
  // a square with opposite corners (2, 2) and (10, 10).
  PresburgerSet setB =
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 1, -2},   // x + y >= 4.
                                               {-1, -1, 30}, // x + y <= 32.
                                               {1, -1, 0},   // x - y >= 2.
                                               {-1, 1, 10},  // x - y <= 16.
                                           }),
                          makeFACFromIneqs(2, {
                                                  {1, 0, -2},  // x >= 2.
                                                  {0, 1, -2},  // y >= 2.
                                                  {-1, 0, 10}, // x <= 10.
                                                  {0, -1, 10}  // y <= 10.
                                              })});

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
  PresburgerSet set =
      makeSetFromFACs(1, {
                             makeFACFromIneqs(1, {{1, -2},    // x >= 2.
                                                  {-1, 8}}),  // x <= 8.
                             makeFACFromIneqs(1, {{1, -10},   // x >= 10.
                                                  {-1, 20}}), // x <= 20.
                         });

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
  PresburgerSet set =
      makeSetFromFACs(1, {
                             makeFACFromIneqs(1, {{1, -2},    // x >= 2.
                                                  {-1, 8}}),  // x <= 8.
                             makeFACFromIneqs(1, {{1, -10},   // x >= 10.
                                                  {-1, 20}}), // x <= 20.
                         });

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
  // The interval [2, 8] minus
  // the interval [10, 20].
  testSubtractAtPoints(
      makeSetFromFACs(1, {makeFACFromIneqs(1, {})}),
      makeSetFromFACs(1,
                      {
                          makeFACFromIneqs(1, {{1, -2},    // x >= 2.
                                               {-1, 8}}),  // x <= 8.
                          makeFACFromIneqs(1, {{1, -10},   // x >= 10.
                                               {-1, 20}}), // x <= 20.
                      }),
      {{1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // ((-infinity, 0] U [3, 4] U [6, 7]) - ([2, 3] U [5, 6])
  testSubtractAtPoints(
      makeSetFromFACs(1,
                      {
                          makeFACFromIneqs(1,
                                           {
                                               {-1, 0} // x <= 0.
                                           }),
                          makeFACFromIneqs(1,
                                           {
                                               {1, -3}, // x >= 3.
                                               {-1, 4}  // x <= 4.
                                           }),
                          makeFACFromIneqs(1,
                                           {
                                               {1, -6}, // x >= 6.
                                               {-1, 7}  // x <= 7.
                                           }),
                      }),
      makeSetFromFACs(1, {makeFACFromIneqs(1,
                                           {
                                               {1, -2}, // x >= 2.
                                               {-1, 3}, // x <= 3.
                                           }),
                          makeFACFromIneqs(1,
                                           {
                                               {1, -5}, // x >= 5.
                                               {-1, 6}  // x <= 6.
                                           })}),
      {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});

  // Expected result is {[x, y] : x > y}, i.e., {[x, y] : x >= y + 1}.
  testSubtractAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, -1, 0} // x >= y.
                                           })}),
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 1, 0} // x >= -y.
                                           })}),
      {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, -10) and (7, 100).
  // This splits the former rectangle into two halves, (2, 2) to (5, 10) and
  // (7, 2) to (10, 10).
  testSubtractAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 0, -2},  // x >= 2.
                                               {0, 1, -2},  // y >= 2.
                                               {-1, 0, 10}, // x <= 10.
                                               {0, -1, 10}  // y <= 10.
                                           })}),
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 0, -5},   // x >= 5.
                                               {0, 1, 10},   // y >= -10.
                                               {-1, 0, 7},   // x <= 7.
                                               {0, -1, 100}, // y <= 100.
                                           })}),
      {{1, 2},  {2, 2},  {4, 2},  {5, 2},  {7, 2},  {8, 2},  {11, 2},
       {1, 1},  {2, 1},  {4, 1},  {5, 1},  {7, 1},  {8, 1},  {11, 1},
       {1, 10}, {2, 10}, {4, 10}, {5, 10}, {7, 10}, {8, 10}, {11, 10},
       {1, 11}, {2, 11}, {4, 11}, {5, 11}, {7, 11}, {8, 11}, {11, 11}});

  // A rectangle with corners at (2, 2) and (10, 10), minus
  // a rectangle with corners at (5, 4) and (7, 8).
  // This creates a hole in the middle of the former rectangle, and the
  // resulting set can be represented as a union of four rectangles.
  testSubtractAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 0, -2},  // x >= 2.
                                               {0, 1, -2},  // y >= 2.
                                               {-1, 0, 10}, // x <= 10.
                                               {0, -1, 10}  // y <= 10.
                                           })}),
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 0, -5}, // x >= 5.
                                               {0, 1, -4}, // y >= 4.
                                               {-1, 0, 7}, // x <= 7.
                                               {0, -1, 8}, // y <= 8.
                                           })}),
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
      makeSetFromFACs(2, {makeFACFromConstraints(2,
                                                 {
                                                     {1, 0, 0} // x >= 0.
                                                 },
                                                 {
                                                     {1, 1, 0} // x + y = 0.
                                                 })}),
      makeSetFromFACs(2, {makeFACFromConstraints(2,
                                                 {
                                                     {0, -1, 1} // y <= 1.
                                                 },
                                                 {
                                                     {1, 1, 0} // x + y = 0.
                                                 })}),
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
      makeSetFromFACs(1,
                      {
                          makeFACFromIneqs(1, {{1, 0},    // x >= 0.
                                               {-1, 2}}), // x <= 2.
                      }),
      makeSetFromFACs(1,
                      {
                          makeFACFromConstraints(1, {},
                                                 {
                                                     {1, -1} // x = 1.
                                                 }),
                      }),
      {{-1}, {0}, {1}, {2}, {3}});

  // Sets with lots of redundant inequalities to test the redundancy heuristic.
  // (the heuristic is for the subtrahend, the second set which is the one being
  // subtracted)

  // A parallelogram with vertices {(3, 1), (10, -6), (24, 8), (17, 15)} minus
  // a triangle with vertices {(2, 2), (10, 2), (10, 10)}.
  testSubtractAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 1, -2},   // x + y >= 4.
                                               {-1, -1, 30}, // x + y <= 32.
                                               {1, -1, 0},   // x - y >= 2.
                                               {-1, 1, 10},  // x - y <= 16.
                                           })}),
      makeSetFromFACs(
          2, {makeFACFromIneqs(2,
                               {
                                   {1, 0, -2},   // x >= 2. [redundant]
                                   {0, 1, -2},   // y >= 2.
                                   {-1, 0, 10},  // x <= 10.
                                   {0, -1, 10},  // y <= 10. [redundant]
                                   {1, 1, -2},   // x + y >= 2. [redundant]
                                   {-1, -1, 30}, // x + y <= 30. [redundant]
                                   {1, -1, 0},   // x - y >= 0.
                                   {-1, 1, 10},  // x - y <= 10.
                               })}),
      {{1, 2},  {2, 2},   {3, 2},   {4, 2},  {1, 1},   {2, 1},   {3, 1},
       {4, 1},  {2, 0},   {3, 0},   {4, 0},  {5, 0},   {10, 2},  {11, 2},
       {10, 1}, {10, 10}, {10, 11}, {10, 9}, {11, 10}, {10, -6}, {11, -6},
       {24, 8}, {24, 7},  {17, 15}, {16, 15}});

  testSubtractAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 1, -2},   // x + y >= 4.
                                               {-1, -1, 30}, // x + y <= 32.
                                               {1, -1, 0},   // x - y >= 2.
                                               {-1, 1, 10},  // x - y <= 16.
                                           })}),
      makeSetFromFACs(
          2, {makeFACFromIneqs(2,
                               {
                                   {1, 0, -2},   // x >= 2. [redundant]
                                   {0, 1, -2},   // y >= 2.
                                   {-1, 0, 10},  // x <= 10.
                                   {0, -1, 10},  // y <= 10. [redundant]
                                   {1, 1, -2},   // x + y >= 2. [redundant]
                                   {-1, -1, 30}, // x + y <= 30. [redundant]
                                   {1, -1, 0},   // x - y >= 0.
                                   {-1, 1, 10},  // x - y <= 10.
                               })}),
      {{1, 2},  {2, 2},   {3, 2},   {4, 2},  {1, 1},   {2, 1},   {3, 1},
       {4, 1},  {2, 0},   {3, 0},   {4, 0},  {5, 0},   {10, 2},  {11, 2},
       {10, 1}, {10, 10}, {10, 11}, {10, 9}, {11, 10}, {10, -6}, {11, -6},
       {24, 8}, {24, 7},  {17, 15}, {16, 15}});

  // ((-infinity, -5] U [3, 3] U [4, 4] U [5, 5]) - ([-2, -10] U [3, 4] U [6,
  // 7])
  testSubtractAtPoints(
      makeSetFromFACs(1,
                      {
                          makeFACFromIneqs(1,
                                           {
                                               {-1, -5}, // x <= -5.
                                           }),
                          makeFACFromConstraints(1, {},
                                                 {
                                                     {1, -3} // x = 3.
                                                 }),
                          makeFACFromConstraints(1, {},
                                                 {
                                                     {1, -4} // x = 4.
                                                 }),
                          makeFACFromConstraints(1, {},
                                                 {
                                                     {1, -5} // x = 5.
                                                 }),
                      }),
      makeSetFromFACs(
          1,
          {
              makeFACFromIneqs(1,
                               {
                                   {-1, -2},  // x <= -2.
                                   {1, -10},  // x >= -10.
                                   {-1, 0},   // x <= 0. [redundant]
                                   {-1, 10},  // x <= 10. [redundant]
                                   {1, -100}, // x >= -100. [redundant]
                                   {1, -50}   // x >= -50. [redundant]
                               }),
              makeFACFromIneqs(1,
                               {
                                   {1, -3}, // x >= 3.
                                   {-1, 4}, // x <= 4.
                                   {1, 1},  // x >= -1. [redundant]
                                   {1, 7},  // x >= -7. [redundant]
                                   {-1, 10} // x <= 10. [redundant]
                               }),
              makeFACFromIneqs(1,
                               {
                                   {1, -6}, // x >= 6.
                                   {-1, 7}, // x <= 7.
                                   {1, 1},  // x >= -1. [redundant]
                                   {1, -3}, // x >= -3. [redundant]
                                   {-1, 5}  // x <= 5. [redundant]
                               }),
          }),
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
      PresburgerSet::getUniverse(1),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  // Complement of empty set.
  testComplementAtPoints(
      PresburgerSet::getEmptySet(1),
      {{-1}, {-2}, {-8}, {1}, {2}, {8}, {9}, {10}, {20}, {21}});

  testComplementAtPoints(
      makeSetFromFACs(2, {makeFACFromIneqs(2,
                                           {
                                               {1, 0, -2},  // x >= 2.
                                               {0, 1, -2},  // y >= 2.
                                               {-1, 0, 10}, // x <= 10.
                                               {0, -1, 10}  // y <= 10.
                                           })}),
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

} // namespace mlir
