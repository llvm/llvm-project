//===- IntegerPolyhedron.cpp - Tests for IntegerPolyhedron class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "Utils.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <optional>

using namespace mlir;
using namespace presburger;

using testing::ElementsAre;

enum class TestFunction { Sample, Empty };

/// Construct a IntegerPolyhedron from a set of inequality and
/// equality constraints.
static IntegerPolyhedron
makeSetFromConstraints(unsigned ids, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned syms = 0) {
  IntegerPolyhedron set(
      ineqs.size(), eqs.size(), ids + 1,
      PresburgerSpace::getSetSpace(ids - syms, syms, /*numLocals=*/0));
  for (const auto &eq : eqs)
    set.addEquality(eq);
  for (const auto &ineq : ineqs)
    set.addInequality(ineq);
  return set;
}

static void dump(ArrayRef<DynamicAPInt> vec) {
  for (const DynamicAPInt &x : vec)
    llvm::errs() << x << ' ';
  llvm::errs() << '\n';
}

/// If fn is TestFunction::Sample (default):
///
///   If hasSample is true, check that findIntegerSample returns a valid sample
///   for the IntegerPolyhedron poly. Also check that getIntegerLexmin finds a
///   non-empty lexmin.
///
///   If hasSample is false, check that findIntegerSample returns std::nullopt
///   and findIntegerLexMin returns Empty.
///
/// If fn is TestFunction::Empty, check that isIntegerEmpty returns the
/// opposite of hasSample.
static void checkSample(bool hasSample, const IntegerPolyhedron &poly,
                        TestFunction fn = TestFunction::Sample) {
  std::optional<SmallVector<DynamicAPInt, 8>> maybeSample;
  MaybeOptimum<SmallVector<DynamicAPInt, 8>> maybeLexMin;
  switch (fn) {
  case TestFunction::Sample:
    maybeSample = poly.findIntegerSample();
    maybeLexMin = poly.findIntegerLexMin();

    if (!hasSample) {
      EXPECT_FALSE(maybeSample.has_value());
      if (maybeSample.has_value()) {
        llvm::errs() << "findIntegerSample gave sample: ";
        dump(*maybeSample);
      }

      EXPECT_TRUE(maybeLexMin.isEmpty());
      if (maybeLexMin.isBounded()) {
        llvm::errs() << "findIntegerLexMin gave sample: ";
        dump(*maybeLexMin);
      }
    } else {
      ASSERT_TRUE(maybeSample.has_value());
      EXPECT_TRUE(poly.containsPoint(*maybeSample));

      ASSERT_FALSE(maybeLexMin.isEmpty());
      if (maybeLexMin.isUnbounded()) {
        EXPECT_TRUE(Simplex(poly).isUnbounded());
      }
      if (maybeLexMin.isBounded()) {
        EXPECT_TRUE(poly.containsPointNoLocal(*maybeLexMin));
      }
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
  IntegerPolyhedron set(PresburgerSpace::getSetSpace(3, 2, 1));

  set.addInequality({10, 11, 12, 20, 21, 30, 40});
  set.removeVar(VarKind::Symbol, 1);
  EXPECT_THAT(set.getInequality(0),
              testing::ElementsAre(10, 11, 12, 20, 30, 40));

  set.removeVarRange(VarKind::SetDim, 0, 2);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeVarRange(VarKind::Local, 1, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  set.removeVarRange(VarKind::Local, 0, 1);
  EXPECT_THAT(set.getInequality(0), testing::ElementsAre(12, 20, 40));
}

TEST(IntegerPolyhedronTest, FindSampleTest) {
  // Bounded sets with only inequalities.
  // 0 <= 7x <= 5
  checkSample(true,
              parseIntegerPolyhedron("(x) : (7 * x >= 0, -7 * x + 5 >= 0)"));

  // 1 <= 5x and 5x <= 4 (no solution).
  checkSample(
      false, parseIntegerPolyhedron("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)"));

  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  checkSample(
      true, parseIntegerPolyhedron("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)"));

  // Bounded sets with equalities.
  // x >= 8 and 40 >= y and x = y.
  checkSample(true, parseIntegerPolyhedron(
                        "(x,y) : (x - 8 >= 0, -y + 40 >= 0, x - y == 0)"));

  // x <= 10 and y <= 10 and 10 <= z and x + 2y = 3z.
  // solution: x = y = z = 10.
  checkSample(true,
              parseIntegerPolyhedron("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                                     "z - 10 >= 0, x + 2 * y - 3 * z == 0)"));

  // x <= 10 and y <= 10 and 11 <= z and x + 2y = 3z.
  // This implies x + 2y >= 33 and x + 2y <= 30, which has no solution.
  checkSample(false,
              parseIntegerPolyhedron("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                                     "z - 11 >= 0, x + 2 * y - 3 * z == 0)"));

  // 0 <= r and r <= 3 and 4q + r = 7.
  // Solution: q = 1, r = 3.
  checkSample(true, parseIntegerPolyhedron(
                        "(q,r) : (r >= 0, -r + 3 >= 0, 4 * q + r - 7 == 0)"));

  // 4q + r = 7 and r = 0.
  // Solution: q = 1, r = 3.
  checkSample(false,
              parseIntegerPolyhedron("(q,r) : (4 * q + r - 7 == 0, r == 0)"));

  // The next two sets are large sets that should take a long time to sample
  // with a naive branch and bound algorithm but can be sampled efficiently with
  // the GBR algorithm.
  //
  // This is a triangle with vertices at (1/3, 0), (2/3, 0) and (10000, 10000).
  checkSample(
      true, parseIntegerPolyhedron("(x,y) : (y >= 0, "
                                   "300000 * x - 299999 * y - 100000 >= 0, "
                                   "-300000 * x + 299998 * y + 200000 >= 0)"));

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
  checkSample(true,
              parseIntegerPolyhedron(
                  "(a,b,c,d,e) : (b + d - e >= 0, -b + c - d + e >= 0, "
                  "300000 * a - 299998 * b - c - 9 * d + 21 * e - 112000 >= 0, "
                  "-150000 * a + 149999 * b - 15 * d + 47 * e + 68000 >= 0, "
                  "d - e == 0, d + e - 2000 == 0)"));

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
  checkSample(false,
              parseIntegerPolyhedron(
                  "(x,y) : (x >= 0, -x + 100 >= 0, 3 * x - 3 * y + 1 == 0)"));

  // A thin parallelogram. 0 <= x <= 100 and x + 1/3 <= y <= x + 2/3.
  checkSample(false, parseIntegerPolyhedron(
                         "(x,y) : (x >= 0, -x + 100 >= 0, "
                         "3 * x - 3 * y + 2 >= 0, -3 * x + 3 * y - 1 >= 0)"));

  checkSample(true,
              parseIntegerPolyhedron("(x,y) : (2 * x >= 0, -2 * x + 99 >= 0, "
                                     "2 * y >= 0, -2 * y + 99 >= 0)"));

  // 2D cone with apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  checkSample(true, parseIntegerPolyhedron(
                        "(x,y) : (300000 * x - 299999 * y - 100000 >= 0, "
                        "-300000 * x + 299998 * y + 200000 >= 0)"));

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

  checkSample(true, parseIntegerPolyhedron(
                        "(x, y, z) : (2 * x - 1 >= 0, x - y - 1 == 0, "
                        "y - z == 0)"));

  // Test with a local id.
  checkSample(true, parseIntegerPolyhedron("(x) : (x == 5*(x floordiv 2))"));

  // Regression tests for the computation of dual coefficients.
  checkSample(false, parseIntegerPolyhedron("(x, y, z) : ("
                                            "6*x - 4*y + 9*z + 2 >= 0,"
                                            "x + 5*y + z + 5 >= 0,"
                                            "-4*x + y + 2*z - 1 >= 0,"
                                            "-3*x - 2*y - 7*z - 1 >= 0,"
                                            "-7*x - 5*y - 9*z - 1 >= 0)"));
  checkSample(true, parseIntegerPolyhedron("(x, y, z) : ("
                                           "3*x + 3*y + 3 >= 0,"
                                           "-4*x - 8*y - z + 4 >= 0,"
                                           "-7*x - 4*y + z + 1 >= 0,"
                                           "2*x - 7*y - 8*z - 7 >= 0,"
                                           "9*x + 8*y - 9*z - 7 >= 0)"));
}

TEST(IntegerPolyhedronTest, IsIntegerEmptyTest) {
  // 1 <= 5x and 5x <= 4 (no solution).
  EXPECT_TRUE(parseIntegerPolyhedron("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)")
                  .isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  EXPECT_FALSE(parseIntegerPolyhedron("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)")
                   .isIntegerEmpty());

  // Unbounded sets.
  EXPECT_TRUE(
      parseIntegerPolyhedron("(x,y,z) : (2 * y - 1 >= 0, -2 * y + 1 >= 0, "
                             "2 * z - 1 >= 0, 2 * x - 1 == 0)")
          .isIntegerEmpty());

  EXPECT_FALSE(parseIntegerPolyhedron(
                   "(x,y,z) : (2 * x - 1 >= 0, -3 * x + 3 >= 0, "
                   "5 * z - 6 >= 0, -7 * z + 17 >= 0, 3 * y - 2 >= 0)")
                   .isIntegerEmpty());

  EXPECT_FALSE(parseIntegerPolyhedron(
                   "(x,y,z) : (2 * x - 1 >= 0, x - y - 1 == 0, y - z == 0)")
                   .isIntegerEmpty());

  // IntegerPolyhedron::isEmpty() does not detect the following sets to be
  // empty.

  // 3x + 7y = 1 and 0 <= x, y <= 10.
  // Since x and y are non-negative, 3x + 7y can never be 1.
  EXPECT_TRUE(parseIntegerPolyhedron(
                  "(x,y) : (x >= 0, -x + 10 >= 0, y >= 0, -y + 10 >= 0, "
                  "3 * x + 7 * y - 1 == 0)")
                  .isIntegerEmpty());

  // 2x = 3y and y = x - 1 and x + y = 6z + 2 and 0 <= x, y <= 100.
  // Substituting y = x - 1 in 3y = 2x, we obtain x = 3 and hence y = 2.
  // Since x + y = 5 cannot be equal to 6z + 2 for any z, the set is empty.
  EXPECT_TRUE(parseIntegerPolyhedron(
                  "(x,y,z) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
                  "2 * x - 3 * y == 0, x - y - 1 == 0, x + y - 6 * z - 2 == 0)")
                  .isIntegerEmpty());

  // 2x = 3y and y = x - 1 + 6z and x + y = 6q + 2 and 0 <= x, y <= 100.
  // 2x = 3y implies x is a multiple of 3 and y is even.
  // Now y = x - 1 + 6z implies y = 2 mod 3. In fact, since y is even, we have
  // y = 2 mod 6. Then since x = y + 1 + 6z, we have x = 3 mod 6, implying
  // x + y = 5 mod 6, which contradicts x + y = 6q + 2, so the set is empty.
  EXPECT_TRUE(
      parseIntegerPolyhedron(
          "(x,y,z,q) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
          "2 * x - 3 * y == 0, x - y + 6 * z - 1 == 0, x + y - 6 * q - 2 == 0)")
          .isIntegerEmpty());

  // Set with symbols.
  EXPECT_FALSE(parseIntegerPolyhedron("(x)[s] : (x + s >= 0, x - s == 0)")
                   .isIntegerEmpty());
}

TEST(IntegerPolyhedronTest, removeRedundantConstraintsTest) {
  IntegerPolyhedron poly =
      parseIntegerPolyhedron("(x) : (x - 2 >= 0, -x + 2 >= 0, x - 2 == 0)");
  poly.removeRedundantConstraints();

  // Both inequalities are redundant given the equality. Both have been removed.
  EXPECT_EQ(poly.getNumInequalities(), 0u);
  EXPECT_EQ(poly.getNumEqualities(), 1u);

  IntegerPolyhedron poly2 =
      parseIntegerPolyhedron("(x,y) : (x - 3 >= 0, y - 2 >= 0, x - y == 0)");
  poly2.removeRedundantConstraints();

  // The second inequality is redundant and should have been removed. The
  // remaining inequality should be the first one.
  EXPECT_EQ(poly2.getNumInequalities(), 1u);
  EXPECT_THAT(poly2.getInequality(0), ElementsAre(1, 0, -3));
  EXPECT_EQ(poly2.getNumEqualities(), 1u);

  IntegerPolyhedron poly3 =
      parseIntegerPolyhedron("(x,y,z) : (x - y == 0, x - z == 0, y - z == 0)");
  poly3.removeRedundantConstraints();

  // One of the three equalities can be removed.
  EXPECT_EQ(poly3.getNumInequalities(), 0u);
  EXPECT_EQ(poly3.getNumEqualities(), 2u);

  IntegerPolyhedron poly4 = parseIntegerPolyhedron(
      "(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) : ("
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
      ")");

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

  IntegerPolyhedron poly5 = parseIntegerPolyhedron(
      "(x,y) : (128 * x + 127 >= 0, -x + 7 >= 0, -128 * x + y >= 0, y >= 0)");
  // 128x + 127 >= 0  implies that 128x >= 0, since x has to be an integer.
  // (This should be caught by GCDTightenInqualities().)
  // So -128x + y >= 0 and 128x + 127 >= 0 imply y >= 0 since we have
  // y >= 128x >= 0.
  poly5.removeRedundantConstraints();
  EXPECT_EQ(poly5.getNumInequalities(), 3u);
  SmallVector<DynamicAPInt, 8> redundantConstraint =
      getDynamicAPIntVec({0, 1, 0});
  for (unsigned i = 0; i < 3; ++i) {
    // Ensure that the removed constraint was the redundant constraint [3].
    EXPECT_NE(poly5.getInequality(i),
              ArrayRef<DynamicAPInt>(redundantConstraint));
  }
}

TEST(IntegerPolyhedronTest, addConstantUpperBound) {
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(2));
  poly.addBound(BoundType::UB, 0, 1);
  EXPECT_EQ(poly.atIneq(0, 0), -1);
  EXPECT_EQ(poly.atIneq(0, 1), 0);
  EXPECT_EQ(poly.atIneq(0, 2), 1);

  poly.addBound(BoundType::UB, {1, 2, 3}, 1);
  EXPECT_EQ(poly.atIneq(1, 0), -1);
  EXPECT_EQ(poly.atIneq(1, 1), -2);
  EXPECT_EQ(poly.atIneq(1, 2), -2);
}

TEST(IntegerPolyhedronTest, addConstantLowerBound) {
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(2));
  poly.addBound(BoundType::LB, 0, 1);
  EXPECT_EQ(poly.atIneq(0, 0), 1);
  EXPECT_EQ(poly.atIneq(0, 1), 0);
  EXPECT_EQ(poly.atIneq(0, 2), -1);

  poly.addBound(BoundType::LB, {1, 2, 3}, 1);
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
    ArrayRef<int64_t> expectedDenominators) {
  DivisionRepr divs = poly.getLocalReprs();

  // Check that the `denominators` and `expectedDenominators` match.
  EXPECT_EQ(ArrayRef<DynamicAPInt>(getDynamicAPIntVec(expectedDenominators)),
            divs.getDenoms());

  // Check that the `dividends` and `expectedDividends` match. If the
  // denominator for a division is zero, we ignore its dividend.
  EXPECT_TRUE(divs.getNumDivs() == expectedDividends.size());
  for (unsigned i = 0, e = divs.getNumDivs(); i < e; ++i) {
    if (divs.hasRepr(i)) {
      for (unsigned j = 0, f = divs.getNumVars() + 1; j < f; ++j) {
        EXPECT_TRUE(expectedDividends[i][j] == divs.getDividend(i)[j]);
      }
    }
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprSimple) {
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(1));

  poly.addLocalFloorDiv({1, 4}, 10);
  poly.addLocalFloorDiv({1, 0, 100}, 10);

  std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0, 4},
                                                    {1, 0, 0, 100}};
  SmallVector<int64_t, 8> denoms = {10, 10};

  // Check if floordivs can be computed when no other inequalities exist
  // and floor divs do not depend on each other.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprConstantFloorDiv) {
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(4));

  poly.addInequality({1, 0, 3, 1, 2});
  poly.addInequality({1, 2, -8, 1, 10});
  poly.addEquality({1, 2, -4, 1, 10});

  poly.addLocalFloorDiv({0, 0, 0, 0, 100}, 30);
  poly.addLocalFloorDiv({0, 0, 0, 0, 0, 206}, 101);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0, 0, 0, 0, 3},
                                                    {0, 0, 0, 0, 0, 0, 2}};
  SmallVector<int64_t, 8> denoms = {1, 1};

  // Check if floordivs with constant numerator can be computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprRecursive) {
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(4));
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

  SmallVector<int64_t, 8> denoms = {3, 5, 3};

  // Check if floordivs which may depend on other floordivs can be computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprTightUpperBound) {
  {
    IntegerPolyhedron poly = parseIntegerPolyhedron("(i) : (i mod 3 - 1 >= 0)");

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
    SmallVector<int64_t, 8> denoms = {3};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(poly, divisions, denoms);
  }

  {
    IntegerPolyhedron poly = parseIntegerPolyhedron(
        "(i, j, q) : (4*q - i - j + 2 >= 0, -4*q + i + j >= 0)");
    // Convert `q` to a local variable.
    poly.convertToLocal(VarKind::SetDim, 2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 1}};
    SmallVector<int64_t, 8> denoms = {4};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprFromEquality) {
  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(i, j, q) : (-4*q + i + j == 0)");
    // Convert `q` to a local variable.
    poly.convertToLocal(VarKind::SetDim, 2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 0}};
    SmallVector<int64_t, 8> denoms = {4};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(i, j, q) : (4*q - i - j == 0)");
    // Convert `q` to a local variable.
    poly.convertToLocal(VarKind::SetDim, 2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 0}};
    SmallVector<int64_t, 8> denoms = {4};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(i, j, q) : (3*q + i + j - 2 == 0)");
    // Convert `q` to a local variable.
    poly.convertToLocal(VarKind::SetDim, 2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{-1, -1, 0, 2}};
    SmallVector<int64_t, 8> denoms = {3};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprFromEqualityAndInequality) {
  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(i, j, q, k) : (-3*k + i + j == 0, 4*q - "
                               "i - j + 2 >= 0, -4*q + i + j >= 0)");
    // Convert `q` and `k` to local variables.
    poly.convertToLocal(VarKind::SetDim, 2, 4);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 0, 1},
                                                      {1, 1, 0, 0, 0}};
    SmallVector<int64_t, 8> denoms = {4, 3};

    checkDivisionRepresentation(poly, divisions, denoms);
  }
}

TEST(IntegerPolyhedronTest, computeLocalReprNoRepr) {
  IntegerPolyhedron poly =
      parseIntegerPolyhedron("(x, q) : (x - 3 * q >= 0, -x + 3 * q + 3 >= 0)");
  // Convert q to a local variable.
  poly.convertToLocal(VarKind::SetDim, 1, 2);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0}};
  SmallVector<int64_t, 8> denoms = {0};

  // Check that no division is computed.
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, computeLocalReprNegConstNormalize) {
  IntegerPolyhedron poly = parseIntegerPolyhedron(
      "(x, q) : (-1 - 3*x - 6 * q >= 0, 6 + 3*x + 6*q >= 0)");
  // Convert q to a local variable.
  poly.convertToLocal(VarKind::SetDim, 1, 2);

  // q = floor((-1/3 - x)/2)
  //   = floor((1/3) + (-1 - x)/2)
  //   = floor((-1 - x)/2).
  std::vector<SmallVector<int64_t, 8>> divisions = {{-1, 0, -1}};
  SmallVector<int64_t, 8> denoms = {2};
  checkDivisionRepresentation(poly, divisions, denoms);
}

TEST(IntegerPolyhedronTest, simplifyLocalsTest) {
  // (x) : (exists y: 2x + y = 1 and y = 2).
  IntegerPolyhedron poly(PresburgerSpace::getSetSpace(1, 0, 1));
  poly.addEquality({2, 1, -1});
  poly.addEquality({0, 1, -2});

  EXPECT_TRUE(poly.isEmpty());

  // (x) : (exists y, z, w: 3x + y = 1 and 2y = z and 3y = w and z = w).
  IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1, 0, 3));
  poly2.addEquality({3, 1, 0, 0, -1});
  poly2.addEquality({0, 2, -1, 0, 0});
  poly2.addEquality({0, 3, 0, -1, 0});
  poly2.addEquality({0, 0, 1, -1, 0});

  EXPECT_TRUE(poly2.isEmpty());

  // (x) : (exists y: x >= y + 1 and 2x + y = 0 and y >= -1).
  IntegerPolyhedron poly3(PresburgerSpace::getSetSpace(1, 0, 1));
  poly3.addInequality({1, -1, -1});
  poly3.addInequality({0, 1, 1});
  poly3.addEquality({2, 1, 0});

  EXPECT_TRUE(poly3.isEmpty());
}

TEST(IntegerPolyhedronTest, mergeDivisionsSimple) {
  {
    // (x) : (exists z, y  = [x / 2] : x = 3y and x + z + 1 >= 0).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1, 0, 1));
    poly1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3y.
    poly1.addInequality({1, 1, 0, 1});    // x + z + 1 >= 0.

    // (x) : (exists y = [x / 2], z : x = 5y).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    poly2.addEquality({1, -5, 0});     // x = 5y.
    poly2.appendVar(VarKind::Local);   // Add local id z.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 1 division should be matched + 2 unmatched local ids.
    EXPECT_EQ(poly1.getNumLocalVars(), 3u);
    EXPECT_EQ(poly2.getNumLocalVars(), 3u);
  }

  {
    // (x) : (exists z = [x / 5], y = [x / 2] : x = 3y).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({1, 0}, 5);    // z = [x / 5].
    poly1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3y.

    // (x) : (exists y = [x / 2], z = [x / 5]: x = 5z).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly2.addLocalFloorDiv({1, 0, 0}, 5); // z = [x / 5].
    poly2.addEquality({1, 0, -5, 0});     // x = 5z.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 2u);
    EXPECT_EQ(poly2.getNumLocalVars(), 2u);
  }

  {
    // Division Normalization test.
    // (x) : (exists z, y  = [x / 2] : x = 3y and x + z + 1 >= 0).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1, 0, 1));
    // This division would be normalized.
    poly1.addLocalFloorDiv({3, 0, 0}, 6); // y = [3x / 6] -> [x/2].
    poly1.addEquality({1, 0, -3, 0});     // x = 3z.
    poly1.addInequality({1, 1, 0, 1});    // x + y + 1 >= 0.

    // (x) : (exists y = [x / 2], z : x = 5y).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    poly2.addEquality({1, -5, 0});     // x = 5y.
    poly2.appendVar(VarKind::Local);   // Add local id z.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // One division should be matched + 2 unmatched local ids.
    EXPECT_EQ(poly1.getNumLocalVars(), 3u);
    EXPECT_EQ(poly2.getNumLocalVars(), 3u);
  }
}

TEST(IntegerPolyhedronTest, mergeDivisionsNestedDivsions) {
  {
    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z >= x).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z <= x).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    poly2.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 2u);
    EXPECT_EQ(poly2.getNumLocalVars(), 2u);
  }

  {
    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z >= x).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    poly1.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    poly1.addInequality({-1, 1, 1, 0, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z <= x).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    poly2.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    poly2.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    poly2.addInequality({1, -1, -1, 0, 0});  // y + z <= x.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 3 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 3u);
    EXPECT_EQ(poly2.getNumLocalVars(), 3u);
  }
  {
    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z >= x).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({2, 0}, 4);    // y = [2x / 4] -> [x / 2].
    poly1.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z <= x).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    // This division would be normalized.
    poly2.addLocalFloorDiv({3, 3, 0}, 9); // z = [3x + 3y / 9] -> [x + y / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 2u);
    EXPECT_EQ(poly2.getNumLocalVars(), 2u);
  }
}

TEST(IntegerPolyhedronTest, mergeDivisionsConstants) {
  {
    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z >= x).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    poly1.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    poly2.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    poly2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 2u);
    EXPECT_EQ(poly2.getNumLocalVars(), 2u);
  }
  {
    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z >= x).
    IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
    poly1.addLocalFloorDiv({1, 1}, 2); // y = [x + 1 / 2].
    // Normalization test.
    poly1.addLocalFloorDiv({3, 0, 6}, 9); // z = [3x + 6 / 9] -> [x + 2 / 3].
    poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
    IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
    // Normalization test.
    poly2.addLocalFloorDiv({2, 2}, 4);    // y = [2x + 2 / 4] -> [x + 1 / 2].
    poly2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

    poly1.mergeLocalVars(poly2);

    // Local space should be same.
    EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

    // 2 divisions should be matched.
    EXPECT_EQ(poly1.getNumLocalVars(), 2u);
    EXPECT_EQ(poly2.getNumLocalVars(), 2u);
  }
}

TEST(IntegerPolyhedronTest, mergeDivisionsDuplicateInSameSet) {
  // (x) : (exists y = [x + 1 / 3], z = [x + 1 / 3]: y + z >= x).
  IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
  poly1.addLocalFloorDiv({1, 1}, 3);    // y = [x + 1 / 2].
  poly1.addLocalFloorDiv({1, 0, 1}, 3); // z = [x + 1 / 3].
  poly1.addInequality({-1, 1, 1, 0});   // y + z >= x.

  // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
  IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
  poly2.addLocalFloorDiv({1, 1}, 3);    // y = [x + 1 / 3].
  poly2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
  poly2.addInequality({1, -1, -1, 0});  // y + z <= x.

  poly1.mergeLocalVars(poly2);

  // Local space should be same.
  EXPECT_EQ(poly1.getNumLocalVars(), poly2.getNumLocalVars());

  // 1 divisions should be matched.
  EXPECT_EQ(poly1.getNumLocalVars(), 3u);
  EXPECT_EQ(poly2.getNumLocalVars(), 3u);
}

TEST(IntegerPolyhedronTest, negativeDividends) {
  // (x) : (exists y = [-x + 1 / 2], z = [-x - 2 / 3]: y + z >= x).
  IntegerPolyhedron poly1(PresburgerSpace::getSetSpace(1));
  poly1.addLocalFloorDiv({-1, 1}, 2); // y = [x + 1 / 2].
  // Normalization test with negative dividends
  poly1.addLocalFloorDiv({-3, 0, -6}, 9); // z = [3x + 6 / 9] -> [x + 2 / 3].
  poly1.addInequality({-1, 1, 1, 0});     // y + z >= x.

  // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
  IntegerPolyhedron poly2(PresburgerSpace::getSetSpace(1));
  // Normalization test.
  poly2.addLocalFloorDiv({-2, 2}, 4);     // y = [-2x + 2 / 4] -> [-x + 1 / 2].
  poly2.addLocalFloorDiv({-1, 0, -2}, 3); // z = [-x - 2 / 3].
  poly2.addInequality({1, -1, -1, 0});    // y + z <= x.

  poly1.mergeLocalVars(poly2);

  // Merging triggers normalization.
  std::vector<SmallVector<int64_t, 8>> divisions = {{-1, 0, 0, 1},
                                                    {-1, 0, 0, -2}};
  SmallVector<int64_t, 8> denoms = {2, 3};
  checkDivisionRepresentation(poly1, divisions, denoms);
}

static void expectRationalLexMin(const IntegerPolyhedron &poly,
                                 ArrayRef<Fraction> min) {
  auto lexMin = poly.findRationalLexMin();
  ASSERT_TRUE(lexMin.isBounded());
  EXPECT_EQ(ArrayRef<Fraction>(*lexMin), min);
}

static void expectNoRationalLexMin(OptimumKind kind,
                                   const IntegerPolyhedron &poly) {
  ASSERT_NE(kind, OptimumKind::Bounded)
      << "Use expectRationalLexMin for bounded min";
  EXPECT_EQ(poly.findRationalLexMin().getKind(), kind);
}

TEST(IntegerPolyhedronTest, findRationalLexMin) {
  expectRationalLexMin(
      parseIntegerPolyhedron(
          "(x, y, z) : (x + 10 >= 0, y + 40 >= 0, z + 30 >= 0)"),
      {{-10, 1}, {-40, 1}, {-30, 1}});
  expectRationalLexMin(
      parseIntegerPolyhedron(
          "(x, y, z) : (2*x + 7 >= 0, 3*y - 5 >= 0, 8*z + 10 >= 0, 9*z >= 0)"),
      {{-7, 2}, {5, 3}, {0, 1}});
  expectRationalLexMin(
      parseIntegerPolyhedron("(x, y) : (3*x + 2*y + 10 >= 0, -3*y + 10 >= "
                             "0, 4*x - 7*y - 10 >= 0)"),
      {{-50, 29}, {-70, 29}});

  // Test with some locals. This is basically x >= 11, 0 <= x - 2e <= 1.
  // It'll just choose x = 11, e = 5.5 since it's rational lexmin.
  expectRationalLexMin(
      parseIntegerPolyhedron(
          "(x, y) : (x - 2*(x floordiv 2) == 0, y - 2*x >= 0, x - 11 >= 0)"),
      {{11, 1}, {22, 1}});

  expectRationalLexMin(
      parseIntegerPolyhedron("(x, y) : (3*x + 2*y + 10 >= 0,"
                             "-4*x + 7*y + 10 >= 0, -3*y + 10 >= 0)"),
      {{-50, 9}, {10, 3}});

  // Cartesian product of above with itself.
  expectRationalLexMin(
      parseIntegerPolyhedron(
          "(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0,"
          "-3*y + 10 >= 0, 3*z + 2*w + 10 >= 0, -4*z + 7*w + 10 >= 0,"
          "-3*w + 10 >= 0)"),
      {{-50, 9}, {10, 3}, {-50, 9}, {10, 3}});

  // Same as above but for the constraints on z and w, we express "10" in terms
  // of x and y. We know that x and y still have to take the values
  // -50/9 and 10/3 since their constraints are the same and their values are
  // minimized first. Accordingly, the values -9x - 12y,  -9x - 0y - 10,
  // and -9x - 15y + 10 are all equal to 10.
  expectRationalLexMin(
      parseIntegerPolyhedron(
          "(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0, "
          "-3*y + 10 >= 0, 3*z + 2*w - 9*x - 12*y >= 0,"
          "-4*z + 7*w + - 9*x - 9*y - 10 >= 0, -3*w - 9*x - 15*y + 10 >= 0)"),
      {{-50, 9}, {10, 3}, {-50, 9}, {10, 3}});

  // Same as above with one constraint removed, making the lexmin unbounded.
  expectNoRationalLexMin(
      OptimumKind::Unbounded,
      parseIntegerPolyhedron(
          "(x, y, z, w) : (3*x + 2*y + 10 >= 0, -4*x + 7*y + 10 >= 0,"
          "-3*y + 10 >= 0, 3*z + 2*w - 9*x - 12*y >= 0,"
          "-4*z + 7*w + - 9*x - 9*y - 10>= 0)"));

  // Again, the lexmin is unbounded.
  expectNoRationalLexMin(
      OptimumKind::Unbounded,
      parseIntegerPolyhedron(
          "(x, y, z) : (2*x + 5*y + 8*z - 10 >= 0,"
          "2*x + 10*y + 8*z - 10 >= 0, 2*x + 5*y + 10*z - 10 >= 0)"));

  // The set is empty.
  expectNoRationalLexMin(
      OptimumKind::Empty,
      parseIntegerPolyhedron("(x) : (2*x >= 0, -x - 1 >= 0)"));
}

static void expectIntegerLexMin(const IntegerPolyhedron &poly,
                                ArrayRef<int64_t> min) {
  MaybeOptimum<SmallVector<DynamicAPInt, 8>> lexMin = poly.findIntegerLexMin();
  ASSERT_TRUE(lexMin.isBounded());
  EXPECT_EQ(*lexMin, getDynamicAPIntVec(min));
}

static void expectNoIntegerLexMin(OptimumKind kind,
                                  const IntegerPolyhedron &poly) {
  ASSERT_NE(kind, OptimumKind::Bounded)
      << "Use expectRationalLexMin for bounded min";
  EXPECT_EQ(poly.findRationalLexMin().getKind(), kind);
}

TEST(IntegerPolyhedronTest, findIntegerLexMin) {
  expectIntegerLexMin(
      parseIntegerPolyhedron("(x, y, z) : (2*x + 13 >= 0, 4*y - 3*x - 2  >= "
                             "0, 11*z + 5*y - 3*x + 7 >= 0)"),
      {-6, -4, 0});
  // Similar to above but no lower bound on z.
  expectNoIntegerLexMin(
      OptimumKind::Unbounded,
      parseIntegerPolyhedron("(x, y, z) : (2*x + 13 >= 0, 4*y - 3*x - 2  "
                             ">= 0, -11*z + 5*y - 3*x + 7 >= 0)"));
}

static void expectSymbolicIntegerLexMin(
    StringRef polyStr,
    ArrayRef<std::pair<StringRef, StringRef>> expectedLexminRepr,
    ArrayRef<StringRef> expectedUnboundedDomainRepr) {
  IntegerPolyhedron poly = parseIntegerPolyhedron(polyStr);

  ASSERT_NE(poly.getNumDimVars(), 0u);
  ASSERT_NE(poly.getNumSymbolVars(), 0u);

  SymbolicLexOpt result = poly.findSymbolicIntegerLexMin();

  if (expectedLexminRepr.empty()) {
    EXPECT_TRUE(result.lexopt.getDomain().isIntegerEmpty());
  } else {
    PWMAFunction expectedLexmin = parsePWMAF(expectedLexminRepr);
    EXPECT_TRUE(result.lexopt.isEqual(expectedLexmin));
  }

  if (expectedUnboundedDomainRepr.empty()) {
    EXPECT_TRUE(result.unboundedDomain.isIntegerEmpty());
  } else {
    PresburgerSet expectedUnboundedDomain =
        parsePresburgerSet(expectedUnboundedDomainRepr);
    EXPECT_TRUE(result.unboundedDomain.isEqual(expectedUnboundedDomain));
  }
}

static void
expectSymbolicIntegerLexMin(StringRef polyStr,
                            ArrayRef<std::pair<StringRef, StringRef>> result) {
  expectSymbolicIntegerLexMin(polyStr, result, {});
}

TEST(IntegerPolyhedronTest, findSymbolicIntegerLexMin) {
  expectSymbolicIntegerLexMin("(x)[a] : (x - a >= 0)",
                              {
                                  {"()[a] : ()", "()[a] -> (a)"},
                              });

  expectSymbolicIntegerLexMin(
      "(x)[a, b] : (x - a >= 0, x - b >= 0)",
      {
          {"()[a, b] : (a - b >= 0)", "()[a, b] -> (a)"},
          {"()[a, b] : (b - a - 1 >= 0)", "()[a, b] -> (b)"},
      });

  expectSymbolicIntegerLexMin(
      "(x)[a, b, c] : (x -a >= 0, x - b >= 0, x - c >= 0)",
      {
          {"()[a, b, c] : (a - b >= 0, a - c >= 0)", "()[a, b, c] -> (a)"},
          {"()[a, b, c] : (b - a - 1 >= 0, b - c >= 0)", "()[a, b, c] -> (b)"},
          {"()[a, b, c] : (c - a - 1 >= 0, c - b - 1 >= 0)",
           "()[a, b, c] -> (c)"},
      });

  expectSymbolicIntegerLexMin("(x, y)[a] : (x - a >= 0, x + y >= 0)",
                              {
                                  {"()[a] : ()", "()[a] -> (a, -a)"},
                              });

  expectSymbolicIntegerLexMin("(x, y)[a] : (x - a >= 0, x + y >= 0, y >= 0)",
                              {
                                  {"()[a] : (a >= 0)", "()[a] -> (a, 0)"},
                                  {"()[a] : (-a - 1 >= 0)", "()[a] -> (a, -a)"},
                              });

  expectSymbolicIntegerLexMin(
      "(x, y)[a, b, c] : (x - a >= 0, y - b >= 0, c - x - y >= 0)",
      {
          {"()[a, b, c] : (c - a - b >= 0)", "()[a, b, c] -> (a, b)"},
      });

  expectSymbolicIntegerLexMin(
      "(x, y, z)[a, b, c] : (c - z >= 0, b - y >= 0, x + y + z - a == 0)",
      {
          {"()[a, b, c] : ()", "()[a, b, c] -> (a - b - c, b, c)"},
      });

  expectSymbolicIntegerLexMin(
      "(x)[a, b] : (a >= 0, b >= 0, x >= 0, a + b + x - 1 >= 0)",
      {
          {"()[a, b] : (a >= 0, b >= 0, a + b - 1 >= 0)", "()[a, b] -> (0)"},
          {"()[a, b] : (a == 0, b == 0)", "()[a, b] -> (1)"},
      });

  expectSymbolicIntegerLexMin(
      "(x)[a, b] : (1 - a >= 0, a >= 0, 1 - b >= 0, b >= 0, 1 - x >= 0, x >= "
      "0, a + b + x - 1 >= 0)",
      {
          {"()[a, b] : (1 - a >= 0, a >= 0, 1 - b >= 0, b >= 0, a + b - 1 >= "
           "0)",
           "()[a, b] -> (0)"},
          {"()[a, b] : (a == 0, b == 0)", "()[a, b] -> (1)"},
      });

  expectSymbolicIntegerLexMin(
      "(x, y, z)[a, b] : (x - a == 0, y - b == 0, x >= 0, y >= 0, z >= 0, x + "
      "y + z - 1 >= 0)",
      {
          {"()[a, b] : (a >= 0, b >= 0, 1 - a - b >= 0)",
           "()[a, b] -> (a, b, 1 - a - b)"},
          {"()[a, b] : (a >= 0, b >= 0, a + b - 2 >= 0)",
           "()[a, b] -> (a, b, 0)"},
      });

  expectSymbolicIntegerLexMin(
      "(x)[a, b] : (x - a == 0, x - b >= 0)",
      {
          {"()[a, b] : (a - b >= 0)", "()[a, b] -> (a)"},
      });

  expectSymbolicIntegerLexMin(
      "(q)[a] : (a - 1 - 3*q == 0, q >= 0)",
      {
          {"()[a] : (a - 1 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (a floordiv 3)"},
      });

  expectSymbolicIntegerLexMin(
      "(r, q)[a] : (a - r - 3*q == 0, q >= 0, 1 - r >= 0, r >= 0)",
      {
          {"()[a] : (a - 0 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (0, a floordiv 3)"},
          {"()[a] : (a - 1 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (1, a floordiv 3)"}, // (1 a floordiv 3)
      });

  expectSymbolicIntegerLexMin(
      "(r, q)[a] : (a - r - 3*q == 0, q >= 0, 2 - r >= 0, r - 1 >= 0)",
      {
          {"()[a] : (a - 1 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (1, a floordiv 3)"},
          {"()[a] : (a - 2 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (2, a floordiv 3)"},
      });

  expectSymbolicIntegerLexMin(
      "(r, q)[a] : (a - r - 3*q == 0, q >= 0, r >= 0)",
      {
          {"()[a] : (a - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (0, a floordiv 3)"},
          {"()[a] : (a - 1 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (1, a floordiv 3)"},
          {"()[a] : (a - 2 - 3*(a floordiv 3) == 0, a >= 0)",
           "()[a] -> (2, a floordiv 3)"},
      });

  expectSymbolicIntegerLexMin(
      "(x, y, z, w)[g] : ("
      // x, y, z, w are boolean variables.
      "1 - x >= 0, x >= 0, 1 - y >= 0, y >= 0,"
      "1 - z >= 0, z >= 0, 1 - w >= 0, w >= 0,"
      // We have some constraints on them:
      "x + y + z - 1 >= 0,"             // x or y or z
      "x + y + w - 1 >= 0,"             // x or y or w
      "1 - x + 1 - y + 1 - w - 1 >= 0," // ~x or ~y or ~w
      // What's the lexmin solution using exactly g true vars?
      "g - x - y - z - w == 0)",
      {
          {"()[g] : (g - 1 == 0)", "()[g] -> (0, 1, 0, 0)"},
          {"()[g] : (g - 2 == 0)", "()[g] -> (0, 0, 1, 1)"},
          {"()[g] : (g - 3 == 0)", "()[g] -> (0, 1, 1, 1)"},
      });

  // Bezout's lemma: if a, b are constants,
  // the set of values that ax + by can take is all multiples of gcd(a, b).
  expectSymbolicIntegerLexMin(
      // If (x, y) is a solution for a given [a, r], then so is (x - 5, y + 2).
      // So the lexmin is unbounded if it exists.
      "(x, y)[a, r] : (a >= 0, r - a + 14*x + 35*y == 0)", {},
      // According to Bezout's lemma, 14x + 35y can take on all multiples
      // of 7 and no other values. So the solution exists iff r - a is a
      // multiple of 7.
      {"()[a, r] : (a >= 0, r - a - 7*((r - a) floordiv 7) == 0)"});

  // The lexmins are unbounded.
  expectSymbolicIntegerLexMin("(x, y)[a] : (9*x - 4*y - 2*a >= 0)", {},
                              {"()[a] : ()"});

  // Test cases adapted from isl.
  expectSymbolicIntegerLexMin(
      // a = 2b - 2(c - b), c - b >= 0.
      // So b is minimized when c = b.
      "(b, c)[a] : (a - 4*b + 2*c == 0, c - b >= 0)",
      {
          {"()[a] : (a - 2*(a floordiv 2) == 0)",
           "()[a] -> (a floordiv 2, a floordiv 2)"},
      });

  expectSymbolicIntegerLexMin(
      // 0 <= b <= 255, 1 <= a - 512b <= 509,
      // b + 8 >= 1 + 16*(b + 8 floordiv 16) // i.e. b % 16 != 8
      "(b)[a] : (255 - b >= 0, b >= 0, a - 512*b - 1 >= 0, 512*b -a + 509 >= "
      "0, b + 7 - 16*((8 + b) floordiv 16) >= 0)",
      {
          {"()[a] : (255 - (a floordiv 512) >= 0, a >= 0, a - 512*(a floordiv "
           "512) - 1 >= 0, 512*(a floordiv 512) - a + 509 >= 0, (a floordiv "
           "512) + 7 - 16*((8 + (a floordiv 512)) floordiv 16) >= 0)",
           "()[a] -> (a floordiv 512)"},
      });

  expectSymbolicIntegerLexMin(
      "(a, b)[K, N, x, y] : (N - K - 2 >= 0, K + 4 - N >= 0, x - 4 >= 0, x + 6 "
      "- 2*N >= 0, K+N - x - 1 >= 0, a - N + 1 >= 0, K+N-1-a >= 0,a + 6 - b - "
      "N >= 0, 2*N - 4 - a >= 0,"
      "2*N - 3*K + a - b >= 0, 4*N - K + 1 - 3*b >= 0, b - N >= 0, a - x - 1 "
      ">= 0)",
      {
          {"()[K, N, x, y] : (x + 6 - 2*N >= 0, 2*N - 5 - x >= 0, x + 1 -3*K + "
           "N >= 0, N + K - 2 - x >= 0, x - 4 >= 0)",
           "()[K, N, x, y] -> (1 + x, N)"},
      });
}

static void
expectComputedVolumeIsValidOverapprox(const IntegerPolyhedron &poly,
                                      std::optional<int64_t> trueVolume,
                                      std::optional<int64_t> resultBound) {
  expectComputedVolumeIsValidOverapprox(poly.computeVolume(), trueVolume,
                                        resultBound);
}

TEST(IntegerPolyhedronTest, computeVolume) {
  // 0 <= x <= 3 + 1/3, -5.5 <= y <= 2 + 3/5, 3 <= z <= 1.75.
  // i.e. 0 <= x <= 3, -5 <= y <= 2, 3 <= z <= 3 + 1/4.
  // So volume is 4 * 8 * 1 = 32.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron(
          "(x, y, z) : (x >= 0, -3*x + 10 >= 0, 2*y + 11 >= 0,"
          "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)"),
      /*trueVolume=*/32ull, /*resultBound=*/32ull);

  // Same as above but y has bounds 2 + 1/5 <= y <= 2 + 3/5. So the volume is
  // zero.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron(
          "(x, y, z) : (x >= 0, -3*x + 10 >= 0, 5*y - 11 >= 0,"
          "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)"),
      /*trueVolume=*/0ull, /*resultBound=*/0ull);

  // Now x is unbounded below but y still has no integer values.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron("(x, y, z) : (-3*x + 10 >= 0, 5*y - 11 >= 0,"
                             "-5*y + 13 >= 0, z - 3 >= 0, -4*z + 13 >= 0)"),
      /*trueVolume=*/0ull, /*resultBound=*/0ull);

  // A diamond shape, 0 <= x + y <= 10, 0 <= x - y <= 10,
  // with vertices at (0, 0), (5, 5), (5, 5), (10, 0).
  // x and y can take 11 possible values so result computed is 11*11 = 121.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron(
          "(x, y) : (x + y >= 0, -x - y + 10 >= 0, x - y >= 0,"
          "-x + y + 10 >= 0)"),
      /*trueVolume=*/61ull, /*resultBound=*/121ull);

  // Effectively the same diamond as above; constrain the variables to be even
  // and double the constant terms of the constraints. The algorithm can't
  // eliminate locals exactly, so the result is an overapproximation by
  // computing that x and y can take 21 possible values so result is 21*21 =
  // 441.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron(
          "(x, y) : (x + y >= 0, -x - y + 20 >= 0, x - y >= 0,"
          " -x + y + 20 >= 0, x - 2*(x floordiv 2) == 0,"
          "y - 2*(y floordiv 2) == 0)"),
      /*trueVolume=*/61ull, /*resultBound=*/441ull);

  // Unbounded polytope.
  expectComputedVolumeIsValidOverapprox(
      parseIntegerPolyhedron("(x, y) : (2*x - y >= 0, y - 3*x >= 0)"),
      /*trueVolume=*/{}, /*resultBound=*/{});
}

static bool containsPointNoLocal(const IntegerPolyhedron &poly,
                                 ArrayRef<int64_t> point) {
  return poly.containsPointNoLocal(getDynamicAPIntVec(point)).has_value();
}

TEST(IntegerPolyhedronTest, containsPointNoLocal) {
  IntegerPolyhedron poly1 =
      parseIntegerPolyhedron("(x) : ((x floordiv 2) - x == 0)");
  EXPECT_TRUE(poly1.containsPointNoLocal({0}));
  EXPECT_FALSE(poly1.containsPointNoLocal({1}));

  IntegerPolyhedron poly2 = parseIntegerPolyhedron(
      "(x) : (x - 2*(x floordiv 2) == 0, x - 4*(x floordiv 4) - 2 == 0)");
  EXPECT_TRUE(containsPointNoLocal(poly2, {6}));
  EXPECT_FALSE(containsPointNoLocal(poly2, {4}));

  IntegerPolyhedron poly3 =
      parseIntegerPolyhedron("(x, y) : (2*x - y >= 0, y - 3*x >= 0)");

  EXPECT_TRUE(poly3.containsPointNoLocal(ArrayRef<int64_t>({0, 0})));
  EXPECT_FALSE(poly3.containsPointNoLocal({1, 0}));
}

TEST(IntegerPolyhedronTest, truncateEqualityRegressionTest) {
  // IntegerRelation::truncate was truncating inequalities to the number of
  // equalities.
  IntegerRelation set(PresburgerSpace::getSetSpace(1));
  IntegerRelation::CountsSnapshot snapshot = set.getCounts();
  set.addEquality({1, 0});
  set.truncate(snapshot);
  EXPECT_EQ(set.getNumEqualities(), 0u);
}

static IntegerPolyhedron universe(unsigned numDims, unsigned numSymbols = 0) {
  return IntegerPolyhedron::getUniverse(
      PresburgerSpace::getSetSpace(numDims, numSymbols));
}

static IntegerPolyhedron empty(unsigned numDims, unsigned numSymbols = 0) {
  return IntegerPolyhedron(IntegerRelation::getEmpty(
      PresburgerSpace::getSetSpace(numDims, numSymbols)));
}

/// Computes the gist of `set` in the context of `context`, checks the gist
/// invariants (see `expectGistInvariants`), checks that gist is idempotent
/// (computing the gist of the result again yields the same constraints), and,
/// if provided, checks that the result is equal to `expected` and has
/// `expectedNumConstraints` constraints. Returns the gist.
static IntegerPolyhedron
expectGist(const IntegerPolyhedron &set, const IntegerPolyhedron &context,
           std::optional<IntegerPolyhedron> expected = std::nullopt,
           std::optional<unsigned> expectedNumConstraints = std::nullopt) {
  IntegerPolyhedron contextCopy = context;
  IntegerPolyhedron gist = set;
  gist.gist(context);
  // The context is never modified.
  EXPECT_TRUE(context.isObviouslyEqual(contextCopy));
  expectGistInvariants(set, context, gist);
  IntegerPolyhedron gistAgain = gist;
  gistAgain.gist(context);
  EXPECT_TRUE(gistAgain.isObviouslyEqual(gist))
      << "gist is not idempotent: " << gist << "became " << gistAgain
      << "in context " << context;
  if (expected)
    EXPECT_TRUE(gist.isEqual(*expected))
        << "gist " << gist << "of set " << set << "in context " << context
        << "is not equal to the expected " << *expected;
  if (expectedNumConstraints)
    EXPECT_EQ(gist.getNumConstraints(), *expectedNumConstraints)
        << "gist " << gist << "of set " << set << "in context " << context;
  return gist;
}

static IntegerPolyhedron
expectGist(StringRef set, StringRef context,
           std::optional<StringRef> expected = std::nullopt,
           std::optional<unsigned> expectedNumConstraints = std::nullopt) {
  std::optional<IntegerPolyhedron> expectedPoly;
  if (expected)
    expectedPoly = parseIntegerPolyhedron(*expected);
  return expectGist(parseIntegerPolyhedron(set),
                    parseIntegerPolyhedron(context), expectedPoly,
                    expectedNumConstraints);
}

TEST(IntegerPolyhedronTest, gistDropsConstraintImpliedByContext) {
  // Constraints of the set that are implied by the context are removed.
  expectGist("(x, y) : (x >= 0, -x + 10 >= 0, y >= 0)", "(x, y) : (x >= 0)",
             "(x, y) : (-x + 10 >= 0, y >= 0)", 2);
}

TEST(IntegerPolyhedronTest, gistDropsConstraintImpliedByContextAndRest) {
  // x + y >= 2 is implied by x >= 1 and y >= 1, but y >= 1 is not implied by
  // x >= 1 and x + y >= 2.
  expectGist("(x, y) : (x + y - 2 >= 0, y - 1 >= 0)", "(x, y) : (x - 1 >= 0)",
             "(x, y) : (y - 1 >= 0)");
}

TEST(IntegerPolyhedronTest, gistKeepsConstraintNotImpliedByContext) {
  // Nothing is redundant.
  expectGist("(x) : (x - 5 >= 0)", "(x) : (x - 1 >= 0)", "(x) : (x - 5 >= 0)",
             1);
}

TEST(IntegerPolyhedronTest, gistDropsShiftedCopyOfContextConstraint) {
  // x >= 0 is a relaxed (shifted) copy of the context constraint x >= 5.
  expectGist(parseIntegerPolyhedron("(x) : (x >= 0)"),
             parseIntegerPolyhedron("(x) : (x - 5 >= 0)"), universe(1), 0);
}

TEST(IntegerPolyhedronTest, gistKeepsTighterShiftedConstraint) {
  expectGist("(x) : (x - 5 >= 0)", "(x) : (x >= 0)", "(x) : (x - 5 >= 0)", 1);
}

TEST(IntegerPolyhedronTest, gistDropsShiftedCopyWithMultipleVars) {
  expectGist("(x, y) : (x - 2*y - 1 >= 0, -x + 2*y + 10 >= 0)",
             "(x, y) : (x - 2*y - 4 >= 0)", "(x, y) : (-x + 2*y + 10 >= 0)", 1);
}

TEST(IntegerPolyhedronTest, gistDropsConstraintImpliedRationally) {
  // 2x >= 1 implies x >= 0 (even rationally, x >= 1/2 > 0).
  expectGist("(x) : (x >= 0, -x + 3 >= 0)", "(x) : (2*x - 1 >= 0)",
             "(x) : (-x + 3 >= 0)", 1);
}

TEST(IntegerPolyhedronTest, gistEqualitiesReducedByContextEqualities) {
  // From the ISL test suite. The set is the single point (1, -1, 3); in the
  // context (a = 1, b + c = 2), only b = -1 is needed.
  expectGist("(a, b, c) : (a - 1 == 0, b + 1 == 0, c - 3 == 0)",
             "(a, b, c) : (a - 1 == 0, b + c - 2 == 0, b + 1 >= 0, "
             "-b + 2 >= 0)",
             "(a, b, c) : (b + 1 == 0)", 1);
}

TEST(IntegerPolyhedronTest, gistEqualityNotImpliedIsKept) {
  // From the ISL test suite. The context implies a <= b, but not a == b;
  // nothing is redundant.
  expectGist("(a, b, c) : (a - b == 0, a - 10 >= 0)",
             "(a, b, c) : (c - a >= 0, b - c >= 0, c - 2 >= 0)",
             "(a, b, c) : (a - b == 0, a - 10 >= 0)", 2);
}

TEST(IntegerPolyhedronTest, gistPromotesImplicitEquality) {
  // From the ISL test suite. Implicit equality: i >= 0 together with the
  // context i <= 0 gives i = 0.
  IntegerPolyhedron gist =
      expectGist("(i, j) : (i >= 0, i + j >= 0)", "(i, j) : (-i >= 0)",
                 "(i, j) : (i == 0, j >= 0)", 2);
  EXPECT_EQ(gist.getNumEqualities(), 1u);
  EXPECT_EQ(gist.getNumInequalities(), 1u);
}

TEST(IntegerPolyhedronTest, gistPromotedEqualityEliminatesLastVariable) {
  // After promoting x - y >= 0 to an equality, the last variable (y) is
  // eliminated from the remaining constraints in favor of the first (x).
  IntegerPolyhedron gist =
      expectGist("(x, y) : (x - y >= 0, x + y >= 0)", "(x, y) : (-x + y >= 0)",
                 "(x, y) : (x - y == 0, x >= 0)", 2);
  EXPECT_THAT(gist.getInequality(0), ElementsAre(1, 0, 0));
}

TEST(IntegerPolyhedronTest, gistEmptyIntersectionYieldsEmpty) {
  // From the ISL test suite. The intersection with the context is empty.
  expectGist(parseIntegerPolyhedron(
                 "(a, b, q, p) : (a == 0, b == 0, q + 5 >= 0, -q + 5 >= 0, "
                 "p >= 0)"),
             parseIntegerPolyhedron("(a, b, q, p) : (b - a - 1 >= 0)"),
             empty(4));
}

TEST(IntegerPolyhedronTest, gistRewritesUsingContextEquality) {
  // The context equality is used to rewrite the constraint. The last variable
  // (y) is eliminated in favor of the first (x), following ISL, so the result
  // is x >= 1 rather than the equivalent (in the context) y >= 1.
  expectGist("(x, y) : (2*x + y - 1 >= 0)", "(x, y) : (x - y == 0)",
             "(x, y) : (x - 1 >= 0)", 1);
}

TEST(IntegerPolyhedronTest, gistContextEqualityOnUninvolvedVarsDoesNotLeak) {
  // Context equalities that only involve variables not involved in the set
  // are not used, so no constraints on y and z are introduced.
  IntegerPolyhedron gist =
      expectGist(parseIntegerPolyhedron("(x, y, z) : (x >= 0)"),
                 parseIntegerPolyhedron("(x, y, z) : (y - z == 0, x - 1 >= 0)"),
                 universe(3), 0);
  EXPECT_EQ(gist.getNumVars(), 3u);
}

TEST(IntegerPolyhedronTest, gistInconsistentAffineHullsYieldEmpty) {
  // The set is empty (a would have to be both odd and even), which is only
  // detected after combining its equalities.
  expectGist(parseIntegerPolyhedron(
                 "(a, b, c) : (2*b - a - 1 == 0, 2*c - a - 2 == 0)"),
             parseIntegerPolyhedron("(a, b, c) : (2*c - a - 2 == 0)"),
             empty(3));
}

/// The set used by the trivial-case tests below.
static IntegerPolyhedron getTrivialCasesSet() {
  return parseIntegerPolyhedron(
      "(x, y)[N] : (x >= 0, -x + N >= 0, y - x >= 0)");
}

TEST(IntegerPolyhedronTest, gistUniverseContextKeepsIrredundantSet) {
  // The universe context does not simplify anything.
  IntegerPolyhedron set = getTrivialCasesSet();
  expectGist(set, universe(2, 1), set, 3);
}

TEST(IntegerPolyhedronTest, gistSelfYieldsUniverse) {
  // A set in its own context is the universe.
  IntegerPolyhedron set = getTrivialCasesSet();
  expectGist(set, set, universe(2, 1), 0);
}

TEST(IntegerPolyhedronTest, gistEmptySetStaysEmpty) {
  IntegerPolyhedron gist = expectGist(empty(2, 1), getTrivialCasesSet());
  EXPECT_TRUE(gist.isObviouslyEmpty());
}

TEST(IntegerPolyhedronTest, gistObviouslyEmptyContextYieldsUniverse) {
  // Anything in an empty context is the universe.
  expectGist(getTrivialCasesSet(), empty(2, 1), universe(2, 1), 0);
}

TEST(IntegerPolyhedronTest, gistEmptySetInEmptyContextYieldsUniverse) {
  // From the ISL test suite.
  expectGist(empty(2, 1), empty(2, 1), universe(2, 1), 0);
}

TEST(IntegerPolyhedronTest, gistUniverseStaysUniverse) {
  expectGist(universe(2, 1), getTrivialCasesSet(), universe(2, 1), 0);
}

TEST(IntegerPolyhedronTest, gistAliasedSelfYieldsUniverse) {
  // A set in its own context, passed as the same object.
  IntegerPolyhedron self = getTrivialCasesSet();
  self.gist(self);
  EXPECT_EQ(self.getNumConstraints(), 0u);
}

TEST(IntegerPolyhedronTest,
     gistUniverseContextEqualsRemoveRedundantConstraints) {
  // In the universe context, gist behaves like removeRedundantConstraints.
  IntegerPolyhedron redundant =
      parseIntegerPolyhedron("(x, y) : (x >= 0, y >= 0, x + y >= 0)");
  IntegerPolyhedron removed = redundant;
  removed.removeRedundantConstraints();
  IntegerPolyhedron gist = expectGist(redundant, universe(2), redundant);
  EXPECT_EQ(gist.getNumConstraints(), removed.getNumConstraints());
  EXPECT_EQ(gist.getNumConstraints(), 2u);
}

TEST(IntegerPolyhedronTest, gistRationallyEmptyContextYieldsEmpty) {
  // Only an obviously empty context yields the universe. A context that is
  // found to be empty later yields the empty set, since the intersection with
  // the context is empty.
  expectGist(parseIntegerPolyhedron("(x) : (x >= 0)"),
             parseIntegerPolyhedron("(x) : (x - 1 >= 0, -x >= 0)"), empty(1));
}

TEST(IntegerPolyhedronTest, gistContextEmptyAfterGcdTighteningYieldsEmpty) {
  // The context is rationally non-empty (x = 1/2) but has no integer points,
  // which GCD tightening of the context reveals.
  expectGist(parseIntegerPolyhedron("(x) : (x >= 0)"),
             parseIntegerPolyhedron("(x) : (2*x - 1 >= 0, -2*x + 1 >= 0)"),
             empty(1));
}

TEST(IntegerPolyhedronTest, gistContextWithInconsistentEqualitiesYieldsEmpty) {
  expectGist(parseIntegerPolyhedron("(x) : (x >= 0)"),
             parseIntegerPolyhedron("(x) : (x == 0, x - 1 == 0)"), empty(1));
}

TEST(IntegerPolyhedronTest, gistSymbolicContextBound) {
  expectGist("(x)[N] : (x >= 0, N - x >= 0)", "(x)[N] : (N - 10 >= 0, x >= 0)",
             "(x)[N] : (N - x >= 0)", 1);
}

TEST(IntegerPolyhedronTest, gistSymbolOnlyContextKeepsSet) {
  expectGist("(x)[N] : (x >= 0, -x + N >= 0)", "(x)[N] : (N >= 0)",
             "(x)[N] : (x >= 0, -x + N >= 0)", 2);
}

TEST(IntegerPolyhedronTest, gistContextEqualityOnSymbolIsSubstituted) {
  // The context equality N = 5 is substituted into the constraints.
  IntegerPolyhedron gist = expectGist("(x)[N] : (x - N >= 0, -x + N + 5 >= 0)",
                                      "(x)[N] : (N - 5 == 0)",
                                      "(x)[N] : (x - 5 >= 0, -x + 10 >= 0)", 2);
  EXPECT_TRUE(isColumnZero(gist, 1));
}

TEST(IntegerPolyhedronTest, gistDropsSharedDivisionDefinition) {
  // From the ISL test suite. The context defines the same division as the
  // set; the set's constraint on it becomes redundant and the local is
  // dropped.
  IntegerPolyhedron gist = expectGist(
      "(x)[n] : (x - n == 0, x - 32*(x floordiv 32) == 0)",
      "(x)[n] : (x - 32*(x floordiv 32) == 0)", "(x)[n] : (x - n == 0)", 1);
  EXPECT_EQ(gist.getNumLocalVars(), 0u);
}

TEST(IntegerPolyhedronTest, gistDropsContextOnlyLocals) {
  // From the ISL test suite. The local of the context is not introduced into
  // the set.
  IntegerPolyhedron gist =
      expectGist("(a, b, c) : (a - 1 >= 0, -a + 15 >= 0)",
                 "(a, b, c) : (a - 1 >= 0, -c + 30 >= 0, "
                 "32*((a - 1) floordiv 16) - 2*a - 2*b + c + 62 >= 0, "
                 "b >= 0)",
                 "(a, b, c) : (-a + 15 >= 0)", 1);
  EXPECT_EQ(gist.getNumLocalVars(), 0u);
}

TEST(IntegerPolyhedronTest, gistKeepsUnrelatedLocalAndZeroColumn) {
  // From the ISL test suite. No constraints on i6 are introduced in the gist,
  // even though the context equality involves i6.
  StringRef set = "(i4, i6)[t1] : (-5*((1530 - 4*t1 - 5*i4) floordiv 20) - t1 "
                  "+ 381 >= 0, -i4 + 1 >= 0)";
  IntegerPolyhedron gist =
      expectGist(set,
                 "(i4, i6)[t1] : (i6 - t1 - 5*((i6 - t1) floordiv 5) == 0, "
                 "-i6 + 6 >= 0, i6 - 3 >= 0)",
                 set);
  EXPECT_TRUE(isColumnZero(gist, 1));
  EXPECT_EQ(gist.getNumLocalVars(), 1u);
}

TEST(IntegerPolyhedronTest, gistDropsOnlySharedDivision) {
  // Only the division shared with the context is dropped.
  IntegerPolyhedron gist =
      expectGist("(n, i, j, k) : (i - 3*(i floordiv 3) - 2 == 0, "
                 "j - 4*(j floordiv 4) - 2 == 0)",
                 "(n, i, j, k) : (j - 4*(j floordiv 4) - 2 == 0)",
                 "(n, i, j, k) : (i - 3*(i floordiv 3) - 2 == 0)");
  EXPECT_EQ(gist.getNumLocalVars(), 1u);
}

TEST(IntegerPolyhedronTest, gistStrideAndSymbolEqualityYieldEmpty) {
  // From the ISL test suite. The intersection with the context is empty:
  // x = 2 mod 3 and x = 3M.
  expectGist(parseIntegerPolyhedron("(x)[M] : (x - 3*(x floordiv 3) - 2 == 0)"),
             parseIntegerPolyhedron("(x)[M] : (x - 3*M == 0)"), empty(1, 1));
}

TEST(IntegerPolyhedronTest, gistRestoresDivisionDefinitionOfUsedLocal) {
  // The division constraints of a local shared with the context are kept if
  // the local is still used, so that the result is meaningful on its own.
  IntegerPolyhedron gist = expectGist("(x) : ((x floordiv 2) - 3 >= 0)",
                                      "(x) : (x - 2*(x floordiv 2) >= 0)",
                                      "(x) : ((x floordiv 2) - 3 >= 0)", 3);
  EXPECT_EQ(gist.getNumLocalVars(), 1u);
}

TEST(IntegerPolyhedronTest, gistRestoresNestedDivisionDefinitions) {
  // Nested divisions: restoring the definition of the outer division makes
  // the inner division used again, so its definition is restored too.
  IntegerPolyhedron gist =
      expectGist("(x) : (((x floordiv 2) floordiv 2) - 1 >= 0)",
                 "(x) : (x - 4*((x floordiv 2) floordiv 2) >= 0)",
                 "(x) : (((x floordiv 2) floordiv 2) - 1 >= 0)", 5);
  EXPECT_EQ(gist.getNumLocalVars(), 2u);
}

TEST(IntegerPolyhedronTest, gistEmptyContextYieldsUniverseWithoutLocals) {
  // An obviously empty context yields the universe without locals.
  IntegerPolyhedron gist =
      expectGist(parseIntegerPolyhedron("(x) : ((x floordiv 2) - 3 >= 0)"),
                 empty(1), universe(1), 0);
  EXPECT_EQ(gist.getNumLocalVars(), 0u);
}

// The following cases, from the ISL test suite, have a division in the set
// that is different from the one in the context. Only the invariants are
// checked.
// TODO: ISL additionally simplifies stride (equality) constraints using the
// equalities of the context (gist_strides), which would give `a - c == 0`,
// `(a - c == 0, a - 3*(a floordiv 3) == 0)` and `x - 2*(x floordiv 2) == 0`
// respectively.
TEST(IntegerPolyhedronTest, gistStrideWithEqualityInvariantsOnly) {
  expectGist("(a, b, c) : (a - 2*(a floordiv 2) == 0, a - c == 0)",
             "(a, b, c) : (b - 2*(b floordiv 2) == 0, b - c == 0)");
}

TEST(IntegerPolyhedronTest, gistPartialStrideWithEqualityInvariantsOnly) {
  expectGist("(a, b, c) : (a - 6*(a floordiv 6) == 0, a - c == 0)",
             "(a, b, c) : (b - 2*(b floordiv 2) == 0, b - c == 0)");
}

TEST(IntegerPolyhedronTest, gistStrideDividingContextStrideInvariantsOnly) {
  expectGist("(x) : (x - 6*(x floordiv 6) == 0)",
             "(x) : (x - 3*(x floordiv 3) == 0)");
}

/// Check that `gist` has the same intersection with `context` as `set`, by
/// checking which integer points in [-2, 12] are contained in each.
static void expectSameIntersection1D(const IntegerPolyhedron &set,
                                     const IntegerPolyhedron &context,
                                     const IntegerPolyhedron &gist) {
  IntegerPolyhedron setAndContext = set.intersect(context);
  IntegerPolyhedron gistAndContext = gist.intersect(context);
  for (int64_t i = -2; i <= 12; ++i)
    EXPECT_EQ(containsPointNoLocal(setAndContext, {i}),
              containsPointNoLocal(gistAndContext, {i}))
        << "at point " << i;
}

/// Returns the set `(i) : exists e0, e1 : 3*e1 - 2*e0 - 1 >= 0,
/// -8*e1 + 5*i - 5*e0 - 1 >= 0, 2*e1 - 2*i + 5*e0 - 1 >= 0` (from the ISL
/// test suite), optionally with the bounds 0 <= i <= 10, where the locals e0
/// and e1 have no division representation.
static IntegerPolyhedron getNonDivLocalsSet(bool bounded) {
  IntegerPolyhedron set = parseIntegerPolyhedron(
      bounded ? "(i, e0, e1) : (3*e1 - 2*e0 - 1 >= 0, -8*e1 + 5*i - 5*e0 - 1 "
                ">= 0, 2*e1 - 2*i + 5*e0 - 1 >= 0, i >= 0, -i + 10 >= 0)"
              : "(i, e0, e1) : (3*e1 - 2*e0 - 1 >= 0, -8*e1 + 5*i - 5*e0 - 1 "
                ">= 0, 2*e1 - 2*i + 5*e0 - 1 >= 0)");
  set.convertToLocal(VarKind::SetDim, 1, 3);
  EXPECT_FALSE(set.hasOnlyDivLocals());
  return set;
}

/// Returns the context `(i) : exists a, b : 2*b - 2*i + 5*a - 1 >= 0,
/// 5*i - 5*a - 8*b - 1 >= 0, 3*b - 2*a - 1 >= 0` (from the ISL test suite),
/// whose locals have no division representation and which implies i >= 1
/// (even rationally).
static IntegerPolyhedron getNonDivLocalsContext() {
  IntegerPolyhedron context = parseIntegerPolyhedron(
      "(i, a, b) : (2*b - 2*i + 5*a - 1 >= 0, 5*i - 5*a - 8*b - 1 >= 0, "
      "3*b - 2*a - 1 >= 0)");
  context.convertToLocal(VarKind::SetDim, 1, 3);
  return context;
}

TEST(IntegerPolyhedronTest, gistNonDivLocalsKept) {
  // From the ISL test suite. Locals without a division representation.
  IntegerPolyhedron set = getNonDivLocalsSet(/*bounded=*/false);
  IntegerPolyhedron context = parseIntegerPolyhedron("(i) : (i >= 0)");
  IntegerPolyhedron gist = set;
  gist.gist(context);
  EXPECT_EQ(gist.getNumConstraints(), 3u);
  EXPECT_EQ(gist.getNumLocalVars(), 2u);
  expectSameIntersection1D(set, context, gist);
}

TEST(IntegerPolyhedronTest, gistNonDivLocalsBoundsDropped) {
  // From the ISL test suite. The bounds on i are implied by the context.
  IntegerPolyhedron set = getNonDivLocalsSet(/*bounded=*/true);
  IntegerPolyhedron context =
      parseIntegerPolyhedron("(i) : (i >= 0, -i + 10 >= 0)");
  IntegerPolyhedron gist = set;
  gist.gist(context);
  EXPECT_EQ(gist.getNumConstraints(), 3u);
  expectSameIntersection1D(set, context, gist);
}

TEST(IntegerPolyhedronTest, gistNonDivLocalContextKeepsStrongerBound) {
  // From the ISL test suite. i >= 100 is not implied by the context.
  expectGist(parseIntegerPolyhedron("(i) : (i - 100 >= 0)"),
             getNonDivLocalsContext(),
             parseIntegerPolyhedron("(i) : (i - 100 >= 0)"), 1);
}

TEST(IntegerPolyhedronTest, gistNonDivLocalContextImpliesBound) {
  // From the ISL test suite. i >= 0 is implied by the context.
  expectGist(parseIntegerPolyhedron("(i) : (i >= 0)"), getNonDivLocalsContext(),
             universe(1), 0);
}

TEST(IntegerPolyhedronTest, gistIntegerOnlyRedundancyWithStrideDropped) {
  // x >= 2 is not implied by x >= 1 rationally, but it is implied for even x.
  expectGist(
      parseIntegerPolyhedron("(x) : (x - 2 >= 0)"),
      parseIntegerPolyhedron("(x) : (x - 2*(x floordiv 2) == 0, x - 1 >= 0)"),
      universe(1), 0);
}

TEST(IntegerPolyhedronTest, gistIntegerOnlyRedundancyDropped) {
  // The rational point (1.2, 1.2) lies in the context but violates
  // x + y <= 2, while all integer points of the context satisfy it.
  expectGist(parseIntegerPolyhedron("(x, y) : (2 - x - y >= 0)"),
             parseIntegerPolyhedron("(x, y) : (x >= 0, y >= 0, "
                                    "6 - 2*x - 3*y >= 0, 6 - 3*x - 2*y >= 0)"),
             universe(2), 0);
}

TEST(IntegerPolyhedronTest, gistIntegerExactBoxEnumeration) {
  // A larger system: check the intersection with the context by enumerating
  // all points of the box [0, 10]^3 that contains the set.
  IntegerPolyhedron set = parseIntegerPolyhedron(
      "(x, y, z) : (x >= 0, -x + 10 >= 0, y >= 0, -y + 10 >= 0, "
      "z >= 0, -z + 10 >= 0, x + y - z >= 0, x - y + z + 5 >= 0, "
      "-x + y + z + 3 >= 0, x + y + z - 4 >= 0)");
  IntegerPolyhedron context =
      parseIntegerPolyhedron("(x, y, z) : (x - 1 >= 0, y - 2 >= 0, z - 1 >= 0, "
                             "-x - y - z + 25 >= 0, 2*x - 3*y + 7 >= 0)");
  IntegerPolyhedron gist = expectGist(set, context);
  IntegerPolyhedron setAndContext = set.intersect(context);
  IntegerPolyhedron gistAndContext = gist.intersect(context);
  for (int64_t x = 0; x <= 10; ++x)
    for (int64_t y = 0; y <= 10; ++y)
      for (int64_t z = 0; z <= 10; ++z)
        EXPECT_EQ(containsPointNoLocal(setAndContext, {x, y, z}),
                  containsPointNoLocal(gistAndContext, {x, y, z}))
            << "at point (" << x << ", " << y << ", " << z << ")";
}

TEST(IntegerPolyhedronTest, gistStrideInCoarserStrideInvariantsOnly) {
  // Stride constraints from the ISL test suite. Only the invariants are
  // checked.
  // TODO: ISL's gist_strides would give `x - 128*(x floordiv 128) == 0`.
  expectGist("(x) : (x - 3200*(x floordiv 3200) == 0)",
             "(x) : (x - 10000*(x floordiv 10000) == 0)");
}

TEST(IntegerPolyhedronTest, gistStrideInFinerStrideUnchanged) {
  // From the ISL test suite; the set is unchanged in ISL too.
  expectGist("(x) : (x - 3200*(x floordiv 3200) == 0)",
             "(x) : (x - 10*(x floordiv 10) == 0)",
             "(x) : (x - 3200*(x floordiv 3200) == 0)");
}
