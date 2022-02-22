//===- IntegerPolyhedron.cpp - Tests for IntegerPolyhedron class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "./Utils.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

namespace mlir {
using namespace presburger_utils;
using testing::ElementsAre;

enum class TestFunction { Sample, Empty };

/// Construct a IntegerPolyhedron from a set of inequality and
/// equality constraints.
static IntegerPolyhedron
makeSetFromConstraints(unsigned ids, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned syms = 0) {
  IntegerPolyhedron set(ineqs.size(), eqs.size(), ids + 1, ids - syms, syms,
                        /*numLocals=*/0);
  for (const auto &eq : eqs)
    set.addEquality(eq);
  for (const auto &ineq : ineqs)
    set.addInequality(ineq);
  return set;
}

static void dump(ArrayRef<int64_t> vec) {
  for (int64_t x : vec)
    llvm::errs() << x << ' ';
  llvm::errs() << '\n';
}

/// If fn is TestFunction::Sample (default):
///
///   If hasSample is true, check that findIntegerSample returns a valid sample
///   for the IntegerPolyhedron poly. Also check that getIntegerLexmin finds a
///   non-empty lexmin.
///
///   If hasSample is false, check that findIntegerSample returns None and
///   getIntegerLexMin returns Empty.
///
/// If fn is TestFunction::Empty, check that isIntegerEmpty returns the
/// opposite of hasSample.
static void checkSample(bool hasSample, const IntegerPolyhedron &poly,
                        TestFunction fn = TestFunction::Sample) {
  Optional<SmallVector<int64_t, 8>> maybeSample;
  MaybeOptimum<SmallVector<int64_t, 8>> maybeLexMin;
  switch (fn) {
  case TestFunction::Sample:
    maybeSample = poly.findIntegerSample();
    maybeLexMin = poly.findIntegerLexMin();

    if (!hasSample) {
      EXPECT_FALSE(maybeSample.hasValue());
      if (maybeSample.hasValue()) {
        llvm::errs() << "findIntegerSample gave sample: ";
        dump(*maybeSample);
      }

      EXPECT_TRUE(maybeLexMin.isEmpty());
      if (maybeLexMin.isBounded()) {
        llvm::errs() << "getIntegerLexMin gave sample: ";
        dump(*maybeLexMin);
      }
    } else {
      ASSERT_TRUE(maybeSample.hasValue());
      EXPECT_TRUE(poly.containsPoint(*maybeSample));

      ASSERT_FALSE(maybeLexMin.isEmpty());
      if (maybeLexMin.isUnbounded())
        EXPECT_TRUE(Simplex(poly).isUnbounded());
      if (maybeLexMin.isBounded())
        EXPECT_TRUE(poly.containsPoint(*maybeLexMin));
    }
    break;
  case TestFunction::Empty:
    EXPECT_EQ(!hasSample, poly.isIntegerEmpty());
    break;
  }
}

/// Check sampling for all the permutations of the dimensions for the given
/// constraint set. Since the GBR algorithm progresses dimension-wise, different
/// orderings may cause the algorithm to proceed differently. At least some of
///.these permutations should make it past the heuristics and test the
/// implementation of the GBR algorithm itself.
/// Use TestFunction fn to test.
static void checkPermutationsSample(bool hasSample, unsigned nDim,
                                    ArrayRef<SmallVector<int64_t, 4>> ineqs,
                                    ArrayRef<SmallVector<int64_t, 4>> eqs,
                                    TestFunction fn = TestFunction::Sample) {
  SmallVector<unsigned, 4> perm(nDim);
  std::iota(perm.begin(), perm.end(), 0);
  auto permute = [&perm](ArrayRef<int64_t> coeffs) {
    SmallVector<int64_t, 4> permuted;
    for (unsigned id : perm)
      permuted.push_back(coeffs[id]);
    permuted.push_back(coeffs.back());
    return permuted;
  };
  do {
    SmallVector<SmallVector<int64_t, 4>, 4> permutedIneqs, permutedEqs;
    for (const auto &ineq : ineqs)
      permutedIneqs.push_back(permute(ineq));
    for (const auto &eq : eqs)
      permutedEqs.push_back(permute(eq));

    checkSample(hasSample,
                makeSetFromConstraints(nDim, permutedIneqs, permutedEqs), fn);
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(IntegerPolyhedronTest, removeInequality) {
  IntegerPolyhedron set =
      makeSetFromConstraints(1, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, {});

  set.removeInequalityRange(0, 0);
  EXPECT_EQ(set.getNumInequalities(), 5u);

  set.removeInequalityRange(1, 3);
  EXPECT_EQ(set.getNumInequalities(), 3u);
  EXPECT_THAT(set.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getInequality(1), ElementsAre(3, 3));
  EXPECT_THAT(set.getInequality(2), ElementsAre(4, 4));

  set.removeInequality(1);
  EXPECT_EQ(set.getNumInequalities(), 2u);
  EXPECT_THAT(set.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getInequality(1), ElementsAre(4, 4));
}

TEST(IntegerPolyhedronTest, removeEquality) {
  IntegerPolyhedron set =
      makeSetFromConstraints(1, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});

  set.removeEqualityRange(0, 0);
  EXPECT_EQ(set.getNumEqualities(), 5u);

  set.removeEqualityRange(1, 3);
  EXPECT_EQ(set.getNumEqualities(), 3u);
  EXPECT_THAT(set.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getEquality(1), ElementsAre(3, 3));
  EXPECT_THAT(set.getEquality(2), ElementsAre(4, 4));

  set.removeEquality(1);
  EXPECT_EQ(set.getNumEqualities(), 2u);
  EXPECT_THAT(set.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(set.getEquality(1), ElementsAre(4, 4));
}

TEST(IntegerPolyhedronTest, clearConstraints) {
  IntegerPolyhedron set = makeSetFromConstraints(1, {}, {});

  set.addInequality({1, 0});
  EXPECT_EQ(set.atIneq(0, 0), 1);
  EXPECT_EQ(set.atIneq(0, 1), 0);

  set.clearConstraints();

  set.addInequality({1, 0});
  EXPECT_EQ(set.atIneq(0, 0), 1);
  EXPECT_EQ(set.atIneq(0, 1), 0);
}

TEST(IntegerPolyhedronTest, removeIdRange) {
  IntegerPolyhedron set(3, 2, 1);

  set.addInequality({10, 11, 12, 20, 21, 30, 40});
  set.removeId(IntegerPolyhedron::IdKind::Symbol, 1);
  EXPECT_THAT(set.getInequality(0),
              testing::ElementsAre(10, 11, 12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::SetDim, 0, 2);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::Local, 1, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeIdRange(IntegerPolyhedron::IdKind::Local, 0, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 40));
}

TEST(IntegerPolyhedronTest, FindSampleTest) {
  // Bounded sets with only inequalities.

  MLIRContext context;

  // 0 <= 7x <= 5
  checkSample(true, parsePoly("(x) : (7 * x >= 0, -7 * x + 5 >= 0)", &context));

  // 1 <= 5x and 5x <= 4 (no solution).
  checkSample(false,
              parsePoly("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)", &context));

  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  checkSample(true,
              parsePoly("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)", &context));

  // Bounded sets with equalities.
  // x >= 8 and 40 >= y and x = y.
  checkSample(true, parsePoly("(x,y) : (x - 8 >= 0, -y + 40 >= 0, x - y == 0)",
                              &context));

  // x <= 10 and y <= 10 and 10 <= z and x + 2y = 3z.
  // solution: x = y = z = 10.
  checkSample(true, parsePoly("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                              "z - 10 >= 0, x + 2 * y - 3 * z == 0)",
                              &context));

  // x <= 10 and y <= 10 and 11 <= z and x + 2y = 3z.
  // This implies x + 2y >= 33 and x + 2y <= 30, which has no solution.
  checkSample(false, parsePoly("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                               "z - 11 >= 0, x + 2 * y - 3 * z == 0)",
                               &context));

  // 0 <= r and r <= 3 and 4q + r = 7.
  // Solution: q = 1, r = 3.
  checkSample(
      true,
      parsePoly("(q,r) : (r >= 0, -r + 3 >= 0, 4 * q + r - 7 == 0)", &context));

  // 4q + r = 7 and r = 0.
  // Solution: q = 1, r = 3.
  checkSample(false,
              parsePoly("(q,r) : (4 * q + r - 7 == 0, r == 0)", &context));

  // The next two sets are large sets that should take a long time to sample
  // with a naive branch and bound algorithm but can be sampled efficiently with
  // the GBR algorithm.
  //
  // This is a triangle with vertices at (1/3, 0), (2/3, 0) and (10000, 10000).
  checkSample(true, parsePoly("(x,y) : (y >= 0, "
                              "300000 * x - 299999 * y - 100000 >= 0, "
                              "-300000 * x + 299998 * y + 200000 >= 0)",
                              &context));

  // This is a tetrahedron with vertices at
  // (1/3, 0, 0), (2/3, 0, 0), (2/3, 0, 10000), and (10000, 10000, 10000).
  // The first three points form a triangular base on the xz plane with the
  // apex at the fourth point, which is the only integer point.
  checkPermutationsSample(
      true, 3,
      {
          {0, 1, 0, 0},  // y >= 0
          {0, -1, 1, 0}, // z >= y
          {300000, -299998, -1,
           -100000},                    // -300000x + 299998y + 100000 + z <= 0.
          {-150000, 149999, 0, 100000}, // -150000x + 149999y + 100000 >= 0.
      },
      {});

  // Same thing with some spurious extra dimensions equated to constants.
  checkSample(
      true,
      parsePoly("(a,b,c,d,e) : (b + d - e >= 0, -b + c - d + e >= 0, "
                "300000 * a - 299998 * b - c - 9 * d + 21 * e - 112000 >= 0, "
                "-150000 * a + 149999 * b - 15 * d + 47 * e + 68000 >= 0, "
                "d - e == 0, d + e - 2000 == 0)",
                &context));

  // This is a tetrahedron with vertices at
  // (1/3, 0, 0), (2/3, 0, 0), (2/3, 0, 100), (100, 100 - 1/3, 100).
  checkPermutationsSample(false, 3,
                          {
                              {0, 1, 0, 0},
                              {0, -300, 299, 0},
                              {300 * 299, -89400, -299, -100 * 299},
                              {-897, 894, 0, 598},
                          },
                          {});

  // Two tests involving equalities that are integer empty but not rational
  // empty.

  // This is a line segment from (0, 1/3) to (100, 100 + 1/3).
  checkSample(
      false,
      parsePoly("(x,y) : (x >= 0, -x + 100 >= 0, 3 * x - 3 * y + 1 == 0)",
                &context));

  // A thin parallelogram. 0 <= x <= 100 and x + 1/3 <= y <= x + 2/3.
  checkSample(false,
              parsePoly("(x,y) : (x >= 0, -x + 100 >= 0, "
                        "3 * x - 3 * y + 2 >= 0, -3 * x + 3 * y - 1 >= 0)",
                        &context));

  checkSample(true, parsePoly("(x,y) : (2 * x >= 0, -2 * x + 99 >= 0, "
                              "2 * y >= 0, -2 * y + 99 >= 0)",
                              &context));

  // 2D cone with apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  checkSample(true, parsePoly("(x,y) : (300000 * x - 299999 * y - 100000 >= 0, "
                              "-300000 * x + 299998 * y + 200000 >= 0)",
                              &context));

  // Cartesian product of a tetrahedron and a 2D cone.
  // The tetrahedron has vertices at
  // (1/3, 0, 0), (2/3, 0, 0), (2/3, 0, 10000), and (10000, 10000, 10000).
  // The first three points form a triangular base on the xz plane with the
  // apex at the fourth point, which is the only integer point.
  // The cone has apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  checkPermutationsSample(
      true /* not empty */, 5,
      {
          // Tetrahedron contraints:
          {0, 1, 0, 0, 0, 0},  // y >= 0
          {0, -1, 1, 0, 0, 0}, // z >= y
                               // -300000x + 299998y + 100000 + z <= 0.
          {300000, -299998, -1, 0, 0, -100000},
          // -150000x + 149999y + 100000 >= 0.
          {-150000, 149999, 0, 0, 0, 100000},

          // Triangle constraints:
          // 300000p - 299999q >= 100000
          {0, 0, 0, 300000, -299999, -100000},
          // -300000p + 299998q + 200000 >= 0
          {0, 0, 0, -300000, 299998, 200000},
      },
      {});

  // Cartesian product of same tetrahedron as above and {(p, q) : 1/3 <= p <=
  // 2/3}. Since the second set is empty, the whole set is too.
  checkPermutationsSample(
      false /* empty */, 5,
      {
          // Tetrahedron contraints:
          {0, 1, 0, 0, 0, 0},  // y >= 0
          {0, -1, 1, 0, 0, 0}, // z >= y
                               // -300000x + 299998y + 100000 + z <= 0.
          {300000, -299998, -1, 0, 0, -100000},
          // -150000x + 149999y + 100000 >= 0.
          {-150000, 149999, 0, 0, 0, 100000},

          // Second set constraints:
          // 3p >= 1
          {0, 0, 0, 3, 0, -1},
          // 3p <= 2
          {0, 0, 0, -3, 0, 2},
      },
      {});

  // Cartesian product of same tetrahedron as above and
  // {(p, q, r) : 1 <= p <= 2 and p = 3q + 3r}.
  // Since the second set is empty, the whole set is too.
  checkPermutationsSample(
      false /* empty */, 5,
      {
          // Tetrahedron contraints:
          {0, 1, 0, 0, 0, 0, 0},  // y >= 0
          {0, -1, 1, 0, 0, 0, 0}, // z >= y
                                  // -300000x + 299998y + 100000 + z <= 0.
          {300000, -299998, -1, 0, 0, 0, -100000},
          // -150000x + 149999y + 100000 >= 0.
          {-150000, 149999, 0, 0, 0, 0, 100000},

          // Second set constraints:
          // p >= 1
          {0, 0, 0, 1, 0, 0, -1},
          // p <= 2
          {0, 0, 0, -1, 0, 0, 2},
      },
      {
          {0, 0, 0, 1, -3, -3, 0}, // p = 3q + 3r
      });

  // Cartesian product of a tetrahedron and a 2D cone.
  // The tetrahedron is empty and has vertices at
  // (1/3, 0, 0), (2/3, 0, 0), (2/3, 0, 100), and (100, 100 - 1/3, 100).
  // The cone has apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  // Since the tetrahedron is empty, the Cartesian product is too.
  checkPermutationsSample(false /* empty */, 5,
                          {
                              // Tetrahedron contraints:
                              {0, 1, 0, 0, 0, 0},
                              {0, -300, 299, 0, 0, 0},
                              {300 * 299, -89400, -299, 0, 0, -100 * 299},
                              {-897, 894, 0, 0, 0, 598},

                              // Triangle constraints:
                              // 300000p - 299999q >= 100000
                              {0, 0, 0, 300000, -299999, -100000},
                              // -300000p + 299998q + 200000 >= 0
                              {0, 0, 0, -300000, 299998, 200000},
                          },
                          {});

  // Cartesian product of same tetrahedron as above and
  // {(p, q) : 1/3 <= p <= 2/3}.
  checkPermutationsSample(false /* empty */, 5,
                          {
                              // Tetrahedron contraints:
                              {0, 1, 0, 0, 0, 0},
                              {0, -300, 299, 0, 0, 0},
                              {300 * 299, -89400, -299, 0, 0, -100 * 299},
                              {-897, 894, 0, 0, 0, 598},

                              // Second set constraints:
                              // 3p >= 1
                              {0, 0, 0, 3, 0, -1},
                              // 3p <= 2
                              {0, 0, 0, -3, 0, 2},
                          },
                          {});

  checkSample(true, parsePoly("(x, y, z) : (2 * x - 1 >= 0, x - y - 1 == 0, "
                              "y - z == 0)",
                              &context));

  // Regression tests for the computation of dual coefficients.
  checkSample(false, parsePoly("(x, y, z) : ("
                               "6*x - 4*y + 9*z + 2 >= 0,"
                               "x + 5*y + z + 5 >= 0,"
                               "-4*x + y + 2*z - 1 >= 0,"
                               "-3*x - 2*y - 7*z - 1 >= 0,"
                               "-7*x - 5*y - 9*z - 1 >= 0)",
                               &context));
  checkSample(true, parsePoly("(x, y, z) : ("
                              "3*x + 3*y + 3 >= 0,"
                              "-4*x - 8*y - z + 4 >= 0,"
                              "-7*x - 4*y + z + 1 >= 0,"
                              "2*x - 7*y - 8*z - 7 >= 0,"
                              "9*x + 8*y - 9*z - 7 >= 0)",
                              &context));
}

TEST(IntegerPolyhedronTest, IsIntegerEmptyTest) {

  MLIRContext context;

  // 1 <= 5x and 5x <= 4 (no solution).
  EXPECT_TRUE(parsePoly("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)", &context)
                  .isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  EXPECT_FALSE(parsePoly("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)", &context)
                   .isIntegerEmpty());

  // Unbounded sets.
  EXPECT_TRUE(parsePoly("(x,y,z) : (2 * y - 1 >= 0, -2 * y + 1 >= 0, "
                        "2 * z - 1 >= 0, 2 * x - 1 == 0)",
                        &context)
                  .isIntegerEmpty());

  EXPECT_FALSE(parsePoly("(x,y,z) : (2 * x - 1 >= 0, -3 * x + 3 >= 0, "
                         "5 * z - 6 >= 0, -7 * z + 17 >= 0, 3 * y - 2 >= 0)",
                         &context)
                   .isIntegerEmpty());

  EXPECT_FALSE(
      parsePoly("(x,y,z) : (2 * x - 1 >= 0, x - y - 1 == 0, y - z == 0)",
                &context)
          .isIntegerEmpty());

  // IntegerPolyhedron::isEmpty() does not detect the following sets to be
  // empty.

  // 3x + 7y = 1 and 0 <= x, y <= 10.
  // Since x and y are non-negative, 3x + 7y can never be 1.
  EXPECT_TRUE(parsePoly("(x,y) : (x >= 0, -x + 10 >= 0, y >= 0, -y + 10 >= 0, "
                        "3 * x + 7 * y - 1 == 0)",
                        &context)
                  .isIntegerEmpty());

  // 2x = 3y and y = x - 1 and x + y = 6z + 2 and 0 <= x, y <= 100.
  // Substituting y = x - 1 in 3y = 2x, we obtain x = 3 and hence y = 2.
  // Since x + y = 5 cannot be equal to 6z + 2 for any z, the set is empty.
  EXPECT_TRUE(
      parsePoly("(x,y,z) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
                "2 * x - 3 * y == 0, x - y - 1 == 0, x + y - 6 * z - 2 == 0)",
                &context)
          .isIntegerEmpty());

  // 2x = 3y and y = x - 1 + 6z and x + y = 6q + 2 and 0 <= x, y <= 100.
  // 2x = 3y implies x is a multiple of 3 and y is even.
  // Now y = x - 1 + 6z implies y = 2 mod 3. In fact, since y is even, we have
  // y = 2 mod 6. Then since x = y + 1 + 6z, we have x = 3 mod 6, implying
  // x + y = 5 mod 6, which contradicts x + y = 6q + 2, so the set is empty.
  EXPECT_TRUE(
      parsePoly(
          "(x,y,z,q) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
          "2 * x - 3 * y == 0, x - y + 6 * z - 1 == 0, x + y - 6 * q - 2 == 0)",
          &context)
          .isIntegerEmpty());

  // Set with symbols.
  EXPECT_FALSE(parsePoly("(x)[s] : (x + s >= 0, x - s == 0)", &context)
                   .isIntegerEmpty());
}

TEST(IntegerPolyhedronTest, removeRedundantConstraintsTest) {
  MLIRContext context;

  IntegerPolyhedron poly =
      parsePoly("(x) : (x - 2 >= 0, -x + 2 >= 0, x - 2 == 0)", &context);
  poly.removeRedundantConstraints();

  // Both inequalities are redundant given the equality. Both have been removed.
  EXPECT_EQ(poly.getNumInequalities(), 0u);
  EXPECT_EQ(poly.getNumEqualities(), 1u);

  IntegerPolyhedron poly2 =
      parsePoly("(x,y) : (x - 3 >= 0, y - 2 >= 0, x - y == 0)", &context);
  poly2.removeRedundantConstraints();

  // The second inequality is redundant and should have been removed. The
  // remaining inequality should be the first one.
  EXPECT_EQ(poly2.getNumInequalities(), 1u);
  EXPECT_THAT(poly2.getInequality(0), ElementsAre(1, 0, -3));
  EXPECT_EQ(poly2.getNumEqualities(), 1u);

  IntegerPolyhedron poly3 =
      parsePoly("(x,y,z) : (x - y == 0, x - z == 0, y - z == 0)", &context);
  poly3.removeRedundantConstraints();

  // One of the three equalities can be removed.
  EXPECT_EQ(poly3.getNumInequalities(), 0u);
  EXPECT_EQ(poly3.getNumEqualities(), 2u);

  IntegerPolyhedron poly4 =
      parsePoly("(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) : ("
                "b - 1 >= 0,"
                "-b + 500 >= 0,"
                "-16 * d + f >= 0,"
                "f - 1 >= 0,"
                "-f + 998 >= 0,"
                "16 * d - f + 15 >= 0,"
                "-16 * e + g >= 0,"
                "g - 1 >= 0,"
                "-g + 998 >= 0,"
                "16 * e - g + 15 >= 0,"
                "h >= 0,"
                "-h + 1 >= 0,"
                "j - 1 >= 0,"
                "-j + 500 >= 0,"
                "-f + 16 * l + 15 >= 0,"
                "f - 16 * l >= 0,"
                "-16 * m + o >= 0,"
                "o - 1 >= 0,"
                "-o + 998 >= 0,"
                "16 * m - o + 15 >= 0,"
                "p >= 0,"
                "-p + 1 >= 0,"
                "-g - h + 8 * q + 8 >= 0,"
                "-o - p + 8 * q + 8 >= 0,"
                "o + p - 8 * q - 1 >= 0,"
                "g + h - 8 * q - 1 >= 0,"
                "-f + n >= 0,"
                "f - n >= 0,"
                "k - 10 >= 0,"
                "-k + 10 >= 0,"
                "i - 13 >= 0,"
                "-i + 13 >= 0,"
                "c - 10 >= 0,"
                "-c + 10 >= 0,"
                "a - 13 >= 0,"
                "-a + 13 >= 0"
                ")",
                &context);

  // The above is a large set of constraints without any redundant constraints,
  // as verified by the Fourier-Motzkin based removeRedundantInequalities.
  unsigned nIneq = poly4.getNumInequalities();
  unsigned nEq = poly4.getNumEqualities();
  poly4.removeRedundantInequalities();
  ASSERT_EQ(poly4.getNumInequalities(), nIneq);
  ASSERT_EQ(poly4.getNumEqualities(), nEq);
  // Now we test that removeRedundantConstraints does not find any constraints
  // to be redundant either.
  poly4.removeRedundantConstraints();
  EXPECT_EQ(poly4.getNumInequalities(), nIneq);
  EXPECT_EQ(poly4.getNumEqualities(), nEq);

  IntegerPolyhedron poly5 = parsePoly(
      "(x,y) : (128 * x + 127 >= 0, -x + 7 >= 0, -128 * x + y >= 0, y >= 0)",
      &context);
  // 128x + 127 >= 0  implies that 128x >= 0, since x has to be an integer.
  // (This should be caught by GCDTightenInqualities().)
  // So -128x + y >= 0 and 128x + 127 >= 0 imply y >= 0 since we have
  // y >= 128x >= 0.
  poly5.removeRedundantConstraints();
  EXPECT_EQ(poly5.getNumInequalities(), 3u);
  SmallVector<int64_t, 8> redundantConstraint = {0, 1, 0};
  for (unsigned i = 0; i < 3; ++i) {
    // Ensure that the removed constraint was the redundant constraint [3].
    EXPECT_NE(poly5.getInequality(i), ArrayRef<int64_t>(redundantConstraint));
  }
}

TEST(IntegerPolyhedronTest, addConstantUpperBound) {
  IntegerPolyhedron poly(2);
  poly.addBound(IntegerPolyhedron::UB, 0, 1);
  EXPECT_EQ(poly.atIneq(0, 0), -1);
  EXPECT_EQ(poly.atIneq(0, 1), 0);
  EXPECT_EQ(poly.atIneq(0, 2), 1);

  poly.addBound(IntegerPolyhedron::UB, {1, 2, 3}, 1);
  EXPECT_EQ(poly.atIneq(1, 0), -1);
  EXPECT_EQ(poly.atIneq(1, 1), -2);
  EXPECT_EQ(poly.atIneq(1, 2), -2);
}

TEST(IntegerPolyhedronTest, addConstantLowerBound) {
  IntegerPolyhedron poly(2);
  poly.addBound(IntegerPolyhedron::LB, 0, 1);
  EXPECT_EQ(poly.atIneq(0, 0), 1);
  EXPECT_EQ(poly.atIneq(0, 1), 0);
  EXPECT_EQ(poly.atIneq(0, 2), -1);

  poly.addBound(IntegerPolyhedron::LB, {1, 2, 3}, 1);
  EXPECT_EQ(poly.atIneq(1, 0), 1);
  EXPECT_EQ(poly.atIneq(1, 1), 2);
  EXPECT_EQ(poly.atIneq(1, 2), 2);
}

/// Check if the expected division representation of local variables matches the
/// computed representation. The expected division representation is given as
/// a vector of expressions set in `expectedDividends` and the corressponding
/// denominator in `expectedDenominators`. The `denominators` and `dividends`
/// obtained through `getLocalRepr` function is verified against the
/// `expectedDenominators` and `expectedDividends` respectively.
static void checkDivisionRepresentation(
    IntegerPolyhedron &poly,
    const std::vector<SmallVector<int64_t, 8>> &expectedDividends,
    const SmallVectorImpl<unsigned> &expectedDenominators) {

  std::vector<SmallVector<int64_t, 8>> dividends;
  SmallVector<unsigned, 4> denominators;

  poly.getLocalReprs(dividends, denominators);

  // Check that the `denominators` and `expectedDenominators` match.
  EXPECT_TRUE(expectedDenominators == denominators);

  // Check that the `dividends` and `expectedDividends` match. If the
  // denominator for a division is zero, we ignore its dividend.
  EXPECT_TRUE(dividends.size() == expectedDividends.size());
  for (unsigned i = 0, e = dividends.size(); i < e; ++i) {
    if (denominators[i] != 0) {
      EXPECT_TRUE(expectedDividends[i] == dividends[i]);
    }
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprSimple) {
  IntegerPolyhedron poly(1);

  poly.addLocalFloorDiv({1, 4}, 10);
  poly.addLocalFloorDiv({1, 0, 100}, 10);

  std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0, 4},
                                                    {1, 0, 0, 100}};
  SmallVector<unsigned, 8> denoms = {10, 10};

  // Check if floordivs can be computed when no other inequalities exist
  // and floor divs do not depend on each other.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprConstantFloorDiv) {
  IntegerPolyhedron poly(4);

  poly.addInequality({1, 0, 3, 1, 2});
  poly.addInequality({1, 2, -8, 1, 10});
  poly.addEquality({1, 2, -4, 1, 10});

  poly.addLocalFloorDiv({0, 0, 0, 0, 100}, 30);
  poly.addLocalFloorDiv({0, 0, 0, 0, 0, 206}, 101);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0, 0, 0, 0, 3},
                                                    {0, 0, 0, 0, 0, 0, 2}};
  SmallVector<unsigned, 8> denoms = {1, 1};

  // Check if floordivs with constant numerator can be computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprRecursive) {
  IntegerPolyhedron poly(4);
  poly.addInequality({1, 0, 3, 1, 2});
  poly.addInequality({1, 2, -8, 1, 10});
  poly.addEquality({1, 2, -4, 1, 10});

  poly.addLocalFloorDiv({0, -2, 7, 2, 10}, 3);
  poly.addLocalFloorDiv({3, 0, 9, 2, 2, 10}, 5);
  poly.addLocalFloorDiv({0, 1, -123, 2, 0, -4, 10}, 3);

  poly.addInequality({1, 2, -2, 1, -5, 0, 6, 100});
  poly.addInequality({1, 2, -8, 1, 3, 7, 0, -9});

  std::vector<SmallVector<int64_t, 8>> divisions = {
      {0, -2, 7, 2, 0, 0, 0, 10},
      {3, 0, 9, 2, 2, 0, 0, 10},
      {0, 1, -123, 2, 0, -4, 0, 10}};

  SmallVector<unsigned, 8> denoms = {3, 5, 3};

  // Check if floordivs which may depend on other floordivs can be computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprTightUpperBound) {
  MLIRContext context;

  {
    IntegerPolyhedron poly = parsePoly("(i) : (i mod 3 - 1 >= 0)", &context);

    // The set formed by the poly is:
    //        3q - i + 2 >= 0             <-- Division lower bound
    //       -3q + i - 1 >= 0
    //       -3q + i     >= 0             <-- Division upper bound
    // We remove redundant constraints to get the set:
    //        3q - i + 2 >= 0             <-- Division lower bound
    //       -3q + i - 1 >= 0             <-- Tighter division upper bound
    // thus, making the upper bound tighter.
    poly.removeRedundantConstraints();

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0}};
    SmallVector<unsigned, 8> denoms = {3};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(poly, divisions, denoms);
  }

  {
    IntegerPolyhedron poly = parsePoly(
        "(i, j, q) : (4*q - i - j + 2 >= 0, -4*q + i + j >= 0)", &context);
    // Convert `q` to a local variable.
    poly.convertDimToLocal(2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 1}};
    SmallVector<unsigned, 8> denoms = {4};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprFromEquality) {
  MLIRContext context;
  {
    IntegerPolyhedron poly =
        parsePoly("(i, j, q) : (-4*q + i + j == 0)", &context);
    // Convert `q` to a local variable.
    poly.convertDimToLocal(2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{-1, -1, 0, 0}};
    SmallVector<unsigned, 8> denoms = {4};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
  {
    IntegerPolyhedron poly =
        parsePoly("(i, j, q) : (4*q - i - j == 0)", &context);
    // Convert `q` to a local variable.
    poly.convertDimToLocal(2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{-1, -1, 0, 0}};
    SmallVector<unsigned, 8> denoms = {4};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
  {
    IntegerPolyhedron poly =
        parsePoly("(i, j, q) : (3*q + i + j - 2 == 0)", &context);
    // Convert `q` to a local variable.
    poly.convertDimToLocal(2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, -2}};
    SmallVector<unsigned, 8> denoms = {3};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprFromEqualityAndInequality) {
  MLIRContext context;
  {
    IntegerPolyhedron poly =
        parsePoly("(i, j, q, k) : (-3*k + i + j == 0, 4*q - "
                  "i - j + 2 >= 0, -4*q + i + j >= 0)",
                  &context);
    // Convert `q` and `k` to local variables.
    poly.convertDimToLocal(2, 4);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 0, 1},
                                                      {-1, -1, 0, 0, 0}};
    SmallVector<unsigned, 8> denoms = {4, 3};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprNoRepr) {
  MLIRContext context;
  IntegerPolyhedron poly =
      parsePoly("(x, q) : (x - 3 * q >= 0, -x + 3 * q + 3 >= 0)", &context);
  // Convert q to a local variable.
  poly.convertDimToLocal(1, 2);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0}};
  SmallVector<unsigned, 8> denoms = {0};

  // Check that no division is computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprNegConstNormalize) {
  MLIRContext context;
  IntegerPolyhedron poly = parsePoly(
      "(x, q) : (-1 - 3*x - 6 * q >= 0, 6 + 3*x + 6*q >= 0)", &context);
  // Convert q to a local variable.
  poly.convertDimToLocal(1, 2);

  // q = floor((-1/3 - x)/2)
  //   = floor((1/3) + (-1 - x)/2)
  //   = floor((-1 - x)/2).
  std::vector<SmallVector<int64_t, 8>> divisions = {{-1, 0, -1}};
  SmallVector<unsigned, 8> denoms = {2};
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, simplifyLocalsTest) {
  // (x) : (exists y: 2x + y = 1 and y = 2).
  IntegerPolyhedron poly(1, 0, 1);
  poly.addEquality({2, 1, -1});
  poly.addEquality({0, 1, -2});

  EXPECT_TRUE(poly.isEmpty());

  // (x) : (exists y, z, w: 3x + y = 1 and 2y = z and 3y = w and z = w).
  IntegerPolyhedron poly2(1, 0, 3);
  poly2.addEquality({3, 1, 0, 0, -1});
  poly2.addEquality({0, 2, -1, 0, 0});
  poly2.addEquality({0, 3, 0, -1, 0});
  poly2.addEquality({0, 0, 1, -1, 0});

  EXPECT_TRUE(poly2.isEmpty());

  // (x) : (exists y: x >= y + 1 and 2x + y = 0 and y >= -1).
  IntegerPolyhedron poly3(1, 0, 1);
  poly3.addInequality({1, -1, -1});
  poly3.addInequality({0, 1, 1});
  poly3.addEquality({2, 1, 0});

  EXPECT_TRUE(poly3.isEmpty());
}

TEST(IntegerPolyhedronTest, mergeDivisionsSimple) {
  {
    // (x) : (exists z, y  = [x / 2] : x = 3y and x + z + 1 >= 0).
    IntegerPolyhedron poly1(1, 0, 1);
    poly1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3y.
    poly1.addInequality({1, 1, 0, 1});    // x + z + 1 >= 0.

    // (x) : (exists y = [x / 2], z : x = 5y).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    poly2.addEquality({1, -5, 0});     // x = 5y.
    poly2.appendLocalId();             // Add local id z.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 1 division should be matched + 2 unmatched local ids.
    EXPECT_EQ(poly1.getNumLocalIds(), 3u);
    EXPECT_EQ(poly2.getNumLocalIds(), 3u);
  }

  {
    // (x) : (exists z = [x / 5], y = [x / 2] : x = 3y).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({1, 0}, 5);    // z = [x / 5].
    poly1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3y.

    // (x) : (exists y = [x / 2], z = [x / 5]: x = 5z).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly2.addLocalFloorDiv({1, 0, 0}, 5); // z = [x / 5].
    poly2.addEquality({1, 0, -5, 0});     // x = 5z.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 2u);
    EXPECT_EQ(poly2.getNumLocalIds(), 2u);
  }

  {
    // Division Normalization test.
    // (x) : (exists z, y  = [x / 2] : x = 3y and x + z + 1 >= 0).
    IntegerPolyhedron poly1(1, 0, 1);
    // This division would be normalized.
    poly1.addLocalFloorDiv({3, 0, 0}, 6); // y = [3x / 6] -> [x/2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3z.
    poly1.addInequality({1, 1, 0, 1});    // x + y + 1 >= 0.

    // (x) : (exists y = [x / 2], z : x = 5y).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    poly2.addEquality({1, -5, 0});     // x = 5y.
    poly2.appendLocalId();             // Add local id z.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // One division should be matched + 2 unmatched local ids.
    EXPECT_EQ(poly1.getNumLocalIds(), 3u);
    EXPECT_EQ(poly2.getNumLocalIds(), 3u);
  }
}

TEST(IntegerPolyhedronTest, mergeDivisionsNestedDivsions) {
  {
    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z >= x).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z <= x).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly2.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 2u);
    EXPECT_EQ(poly2.getNumLocalIds(), 2u);
  }

  {
    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z >= x).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    poly1.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    poly1.addInequality({-1, 1, 1, 0, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z <= x).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    poly2.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    poly2.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    poly2.addInequality({1, -1, -1, 0, 0});  // y + z <= x.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 3 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 3u);
    EXPECT_EQ(poly2.getNumLocalIds(), 3u);
  }
  {
    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z >= x).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({2, 0}, 4);    // y = [2x / 4] -> [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z <= x).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    // This division would be normalized.
    poly2.addLocalFloorDiv({3, 3, 0}, 9); // z = [3x + 3y / 9] -> [x + y / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 2u);
    EXPECT_EQ(poly2.getNumLocalIds(), 2u);
  }
}

TEST(IntegerPolyhedronTest, mergeDivisionsConstants) {
  {
    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z >= x).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    poly1.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
    IntegerPolyhedron poly2(1);
    poly2.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    poly2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 2u);
    EXPECT_EQ(poly2.getNumLocalIds(), 2u);
  }
  {
    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z >= x).
    IntegerPolyhedron poly1(1);
    poly1.addLocalFloorDiv({1, 1}, 2); // y = [x + 1 / 2].
    // Normalization test.
    poly1.addLocalFloorDiv({3, 0, 6}, 9); // z = [3x + 6 / 9] -> [x + 2 / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
    IntegerPolyhedron poly2(1);
    // Normalization test.
    poly2.addLocalFloorDiv({2, 2}, 4);    // y = [2x + 2 / 4] -> [x + 1 / 2].
    poly2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalIds(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalIds(), poly2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalIds(), 2u);
    EXPECT_EQ(poly2.getNumLocalIds(), 2u);
  }
}

TEST(IntegerPolyhedronTest, negativeDividends) {
  // (x) : (exists y = [-x + 1 / 2], z = [-x - 2 / 3]: y + z >= x).
  IntegerPolyhedron poly1(1);
  poly1.addLocalFloorDiv({-1, 1}, 2); // y = [x + 1 / 2].
  // Normalization test with negative dividends
  poly1.addLocalFloorDiv({-3, 0, -6}, 9); // z = [3x + 6 / 9] -> [x + 2 / 3].
  poly1.addInequality({-1, 1, 1, 0});     // y + z >= x.

  // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
  IntegerPolyhedron poly2(1);
  // Normalization test.
  poly2.addLocalFloorDiv({-2, 2}, 4);     // y = [-2x + 2 / 4] -> [-x + 1 / 2].
  poly2.addLocalFloorDiv({-1, 0, -2}, 3); // z = [-x - 2 / 3].
  poly2.addInequality({1, -1, -1, 0});    // y + z <= x.

  poly1.mergeLocalIds(poly2);

  // Merging triggers normalization.
  std::vector<SmallVector<int64_t, 8>> divisions = {{-1, 0, 0, 1},
                                                    {-1, 0, 0, -2}};
  SmallVector<unsigned, 8> denoms = {2, 3};
  checkDivisionRepresentation(poly1, divisions, denoms);
}

void expectRationalLexMin(const IntegerPolyhedron &poly,
                          ArrayRef<Fraction> min) {
  auto lexMin = poly.findRationalLexMin();
  ASSERT_TRUE(lexMin.isBounded());
  EXPECT_EQ(ArrayRef<Fraction>(*lexMin), min);
}

void expectNoRationalLexMin(OptimumKind kind, const IntegerPolyhedron &poly) {
  ASSERT_NE(kind, OptimumKind::Bounded)
      << "Use expectRationalLexMin for bounded min";
  EXPECT_EQ(poly.findRationalLexMin().getKind(), kind);
}

TEST(IntegerPolyhedronTest, getRationalLexMin) {
  MLIRContext context;
  expectRationalLexMin(
      parsePoly("(x, y, z) : (x + 10 >= 0, y + 40 >= 0, z + 30 >= 0)",
                &context),
      {{-10, 1}, {-40, 1}, {-30, 1}});
  expectRationalLexMin(
      parsePoly(
          "(x, y, z) : (2*x + 7 >= 0, 3*y - 5 >= 0, 8*z + 10 >= 0, 9*z >= 0)",
          &context),
      {{-7, 2}, {5, 3}, {0, 1}});
  expectRationalLexMin(
      parsePoly(
          "(x, y) : (3*x + 2*y + 10 >= 0, -3*y + 10 >= 0, 4*x - 7*y - 10 >= 0)",
          &context),
      {{-50, 29}, {-70, 29}});

  // Test with some locals. This is basically x >= 11, 0 <= x - 2e <= 1.
  // It'll just choose x = 11, e = 5.5 since it's rational lexmin.
  expectRationalLexMin(
      parsePoly(
          "(x, y) : (x - 2*(x floordiv 2) == 0, y - 2*x >= 0, x - 11 >= 0)",
          &context),
      {{11, 1}, {22, 1}});

  expectRationalLexMin(parsePoly("(x, y) : (3*x + 2*y + 10 >= 0,"
                                 "-4*x + 7*y + 10 >= 0, -3*y + 10 >= 0)",
                                 &context),
                       {{-50, 9}, {10, 3}});

  // Cartesian product of above with itself.
  expectRationalLexMin(
      parsePoly("(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0,"
                "-3*y + 10 >= 0, 3*z + 2*w + 10 >= 0, -4*z + 7*w + 10 >= 0,"
                "-3*w + 10 >= 0)",
                &context),
      {{-50, 9}, {10, 3}, {-50, 9}, {10, 3}});

  // Same as above but for the constraints on z and w, we express "10" in terms
  // of x and y. We know that x and y still have to take the values
  // -50/9 and 10/3 since their constraints are the same and their values are
  // minimized first. Accordingly, the values -9x - 12y,  -9x - 0y - 10,
  // and -9x - 15y + 10 are all equal to 10.
  expectRationalLexMin(
      parsePoly(
          "(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0, "
          "-3*y + 10 >= 0, 3*z + 2*w - 9*x - 12*y >= 0,"
          "-4*z + 7*w + - 9*x - 9*y - 10 >= 0, -3*w - 9*x - 15*y + 10 >= 0)",
          &context),
      {{-50, 9}, {10, 3}, {-50, 9}, {10, 3}});

  // Same as above with one constraint removed, making the lexmin unbounded.
  expectNoRationalLexMin(
      OptimumKind::Unbounded,
      parsePoly("(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0,"
                "-3*y + 10 >= 0, 3*z + 2*w - 9*x - 12*y >= 0,"
                "-4*z + 7*w + - 9*x - 9*y - 10>= 0)",
                &context));

  // Again, the lexmin is unbounded.
  expectNoRationalLexMin(
      OptimumKind::Unbounded,
      parsePoly("(x, y, z) : (2*x + 5*y + 8*z - 10 >= 0,"
                "2*x + 10*y + 8*z - 10 >= 0, 2*x + 5*y + 10*z - 10 >= 0)",
                &context));

  // The set is empty.
  expectNoRationalLexMin(OptimumKind::Empty,
                         parsePoly("(x) : (2*x >= 0, -x - 1 >= 0)", &context));
}

void expectIntegerLexMin(const IntegerPolyhedron &poly, ArrayRef<int64_t> min) {
  auto lexMin = poly.findIntegerLexMin();
  ASSERT_TRUE(lexMin.isBounded());
  EXPECT_EQ(ArrayRef<int64_t>(*lexMin), min);
}

void expectNoIntegerLexMin(OptimumKind kind, const IntegerPolyhedron &poly) {
  ASSERT_NE(kind, OptimumKind::Bounded)
      << "Use expectRationalLexMin for bounded min";
  EXPECT_EQ(poly.findRationalLexMin().getKind(), kind);
}

TEST(IntegerPolyhedronTest, getIntegerLexMin) {
  MLIRContext context;
  expectIntegerLexMin(parsePoly("(x, y, z) : (2*x + 13 >= 0, 4*y - 3*x - 2  >= "
                                "0, 11*z + 5*y - 3*x + 7 >= 0)",
                                &context),
                      {-6, -4, 0});
  // Similar to above but no lower bound on z.
  expectNoIntegerLexMin(OptimumKind::Unbounded,
                        parsePoly("(x, y, z) : (2*x + 13 >= 0, 4*y - 3*x - 2  "
                                  ">= 0, -11*z + 5*y - 3*x + 7 >= 0)",
                                  &context));
}

static void
expectComputedVolumeIsValidOverapprox(const IntegerPolyhedron &poly,
                                      Optional<uint64_t> trueVolume,
                                      Optional<uint64_t> resultBound) {
  expectComputedVolumeIsValidOverapprox(poly.computeVolume(), trueVolume,
                                        resultBound);
}

TEST(IntegerPolyhedronTest, computeVolume) {
  MLIRContext context;

  // 0 <= x <= 3 + 1/3, -5.5 <= y <= 2 + 3/5, 3 <= z <= 1.75.
  // i.e. 0 <= x <= 3, -5 <= y <= 2, 3 <= z <= 3 + 1/4.
  // So volume is 4 * 8 * 1 = 32.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y, z) : (x >= 0, -3*x + 10 >= 0, 2*y + 11 >= 0,"
                "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)",
                &context),
      /*trueVolume=*/32ull, /*resultBound=*/32ull);

  // Same as above but y has bounds 2 + 1/5 <= y <= 2 + 3/5. So the volume is
  // zero.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y, z) : (x >= 0, -3*x + 10 >= 0, 5*y - 11 >= 0,"
                "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)",
                &context),
      /*trueVolume=*/0ull, /*resultBound=*/0ull);

  // Now x is unbounded below but y still has no integer values.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y, z) : (-3*x + 10 >= 0, 5*y - 11 >= 0,"
                "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)",
                &context),
      /*trueVolume=*/0ull, /*resultBound=*/0ull);

  // A diamond shape, 0 <= x + y <= 10, 0 <= x - y <= 10,
  // with vertices at (0, 0), (5, 5), (5, 5), (10, 0).
  // x and y can take 11 possible values so result computed is 11*11 = 121.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y) : (x + y >= 0, -x - y + 10 >= 0, x - y >= 0,"
                "-x + y + 10 >= 0)",
                &context),
      /*trueVolume=*/61ull, /*resultBound=*/121ull);

  // Effectively the same diamond as above; constrain the variables to be even
  // and double the constant terms of the constraints. The algorithm can't
  // eliminate locals exactly, so the result is an overapproximation by
  // computing that x and y can take 21 possible values so result is 21*21 =
  // 441.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y) : (x + y >= 0, -x - y + 20 >= 0, x - y >= 0,"
                " -x + y + 20 >= 0, x - 2*(x floordiv 2) == 0,"
                "y - 2*(y floordiv 2) == 0)",
                &context),
      /*trueVolume=*/61ull, /*resultBound=*/441ull);

  // Unbounded polytope.
  expectComputedVolumeIsValidOverapprox(
      parsePoly("(x, y) : (2*x - y >= 0, y - 3*x >= 0)", &context),
      /*trueVolume=*/{}, /*resultBound=*/{});
}

} // namespace mlir
