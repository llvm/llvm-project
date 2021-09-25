//===- AffineStructuresTest.cpp - Tests for AffineStructures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

namespace mlir {

using testing::ElementsAre;

enum class TestFunction { Sample, Empty };

/// If fn is TestFunction::Sample (default):
/// If hasSample is true, check that findIntegerSample returns a valid sample
/// for the FlatAffineConstraints fac.
/// If hasSample is false, check that findIntegerSample returns None.
///
/// If fn is TestFunction::Empty, check that isIntegerEmpty returns the
/// opposite of hasSample.
static void checkSample(bool hasSample, const FlatAffineConstraints &fac,
                        TestFunction fn = TestFunction::Sample) {
  Optional<SmallVector<int64_t, 8>> maybeSample;
  switch (fn) {
  case TestFunction::Sample:
    maybeSample = fac.findIntegerSample();
    if (!hasSample) {
      EXPECT_FALSE(maybeSample.hasValue());
      if (maybeSample.hasValue()) {
        for (auto x : *maybeSample)
          llvm::errs() << x << ' ';
        llvm::errs() << '\n';
      }
    } else {
      ASSERT_TRUE(maybeSample.hasValue());
      EXPECT_TRUE(fac.containsPoint(*maybeSample));
    }
    break;
  case TestFunction::Empty:
    EXPECT_EQ(!hasSample, fac.isIntegerEmpty());
    break;
  }
}

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints
makeFACFromConstraints(unsigned ids, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned syms = 0) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), ids + 1, ids - syms, syms,
                            /*numLocals=*/0);
  for (const auto &eq : eqs)
    fac.addEquality(eq);
  for (const auto &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
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
                makeFACFromConstraints(nDim, permutedIneqs, permutedEqs), fn);
  } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(FlatAffineConstraintsTest, FindSampleTest) {
  // Bounded sets with only inequalities.

  // 0 <= 7x <= 5
  checkSample(true, makeFACFromConstraints(1, {{7, 0}, {-7, 5}}, {}));

  // 1 <= 5x and 5x <= 4 (no solution).
  checkSample(false, makeFACFromConstraints(1, {{5, -1}, {-5, 4}}, {}));

  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  checkSample(true, makeFACFromConstraints(1, {{5, -1}, {-5, 9}}, {}));

  // Bounded sets with equalities.
  // x >= 8 and 40 >= y and x = y.
  checkSample(
      true, makeFACFromConstraints(2, {{1, 0, -8}, {0, -1, 40}}, {{1, -1, 0}}));

  // x <= 10 and y <= 10 and 10 <= z and x + 2y = 3z.
  // solution: x = y = z = 10.
  checkSample(true, makeFACFromConstraints(
                        3, {{-1, 0, 0, 10}, {0, -1, 0, 10}, {0, 0, 1, -10}},
                        {{1, 2, -3, 0}}));

  // x <= 10 and y <= 10 and 11 <= z and x + 2y = 3z.
  // This implies x + 2y >= 33 and x + 2y <= 30, which has no solution.
  checkSample(false, makeFACFromConstraints(
                         3, {{-1, 0, 0, 10}, {0, -1, 0, 10}, {0, 0, 1, -11}},
                         {{1, 2, -3, 0}}));

  // 0 <= r and r <= 3 and 4q + r = 7.
  // Solution: q = 1, r = 3.
  checkSample(true,
              makeFACFromConstraints(2, {{0, 1, 0}, {0, -1, 3}}, {{4, 1, -7}}));

  // 4q + r = 7 and r = 0.
  // Solution: q = 1, r = 3.
  checkSample(false, makeFACFromConstraints(2, {}, {{4, 1, -7}, {0, 1, 0}}));

  // The next two sets are large sets that should take a long time to sample
  // with a naive branch and bound algorithm but can be sampled efficiently with
  // the GBR algorithm.
  //
  // This is a triangle with vertices at (1/3, 0), (2/3, 0) and (10000, 10000).
  checkSample(
      true,
      makeFACFromConstraints(
          2, {{0, 1, 0}, {300000, -299999, -100000}, {-300000, 299998, 200000}},
          {}));

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
              makeFACFromConstraints(
                  5,
                  {
                      {0, 1, 0, 1, -1, 0},
                      {0, -1, 1, -1, 1, 0},
                      {300000, -299998, -1, -9, 21, -112000},
                      {-150000, 149999, 0, -15, 47, 68000},
                  },
                  {{0, 0, 0, 1, -1, 0},       // p = q.
                   {0, 0, 0, 1, 1, -2000}})); // p + q = 20000 => p = q = 10000.

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
  checkSample(false, makeFACFromConstraints(
                         2,
                         {
                             {1, 0, 0},   // x >= 0.
                             {-1, 0, 100} // -x + 100 >= 0, i.e., x <= 100.
                         },
                         {
                             {3, -3, 1} // 3x - 3y + 1 = 0, i.e., y = x + 1/3.
                         }));

  // A thin parallelogram. 0 <= x <= 100 and x + 1/3 <= y <= x + 2/3.
  checkSample(false, makeFACFromConstraints(2,
                                            {
                                                {1, 0, 0},    // x >= 0.
                                                {-1, 0, 100}, // x <= 100.
                                                {3, -3, 2},   // 3x - 3y >= -2.
                                                {-3, 3, -1},  // 3x - 3y <= -1.
                                            },
                                            {}));

  checkSample(true, makeFACFromConstraints(2,
                                           {
                                               {2, 0, 0},   // 2x >= 1.
                                               {-2, 0, 99}, // 2x <= 99.
                                               {0, 2, 0},   // 2y >= 0.
                                               {0, -2, 99}, // 2y <= 99.
                                           },
                                           {}));
  // 2D cone with apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  checkSample(
      true,
      makeFACFromConstraints(
          2, {{300000, -299999, -100000}, {-300000, 299998, 200000}}, {}));

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

  checkSample(true, makeFACFromConstraints(3,
                                           {
                                               {2, 0, 0, -1}, // 2x >= 1
                                           },
                                           {{
                                               {1, -1, 0, -1}, // y = x - 1
                                               {0, 1, -1, 0},  // z = y
                                           }}));
}

TEST(FlatAffineConstraintsTest, IsIntegerEmptyTest) {
  // 1 <= 5x and 5x <= 4 (no solution).
  EXPECT_TRUE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 4}}, {}).isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  EXPECT_FALSE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 9}}, {}).isIntegerEmpty());

  // Unbounded sets.
  EXPECT_TRUE(makeFACFromConstraints(3,
                                     {
                                         {0, 2, 0, -1}, // 2y >= 1
                                         {0, -2, 0, 1}, // 2y <= 1
                                         {0, 0, 2, -1}, // 2z >= 1
                                     },
                                     {{2, 0, 0, -1}} // 2x = 1
                                     )
                  .isIntegerEmpty());

  EXPECT_FALSE(makeFACFromConstraints(3,
                                      {
                                          {2, 0, 0, -1},  // 2x >= 1
                                          {-3, 0, 0, 3},  // 3x <= 3
                                          {0, 0, 5, -6},  // 5z >= 6
                                          {0, 0, -7, 17}, // 7z <= 17
                                          {0, 3, 0, -2},  // 3y >= 2
                                      },
                                      {})
                   .isIntegerEmpty());

  EXPECT_FALSE(makeFACFromConstraints(3,
                                      {
                                          {2, 0, 0, -1}, // 2x >= 1
                                      },
                                      {{
                                          {1, -1, 0, -1}, // y = x - 1
                                          {0, 1, -1, 0},  // z = y
                                      }})
                   .isIntegerEmpty());

  // FlatAffineConstraints::isEmpty() does not detect the following sets to be
  // empty.

  // 3x + 7y = 1 and 0 <= x, y <= 10.
  // Since x and y are non-negative, 3x + 7y can never be 1.
  EXPECT_TRUE(
      makeFACFromConstraints(
          2, {{1, 0, 0}, {-1, 0, 10}, {0, 1, 0}, {0, -1, 10}}, {{3, 7, -1}})
          .isIntegerEmpty());

  // 2x = 3y and y = x - 1 and x + y = 6z + 2 and 0 <= x, y <= 100.
  // Substituting y = x - 1 in 3y = 2x, we obtain x = 3 and hence y = 2.
  // Since x + y = 5 cannot be equal to 6z + 2 for any z, the set is empty.
  EXPECT_TRUE(
      makeFACFromConstraints(3,
                             {
                                 {1, 0, 0, 0},
                                 {-1, 0, 0, 100},
                                 {0, 1, 0, 0},
                                 {0, -1, 0, 100},
                             },
                             {{2, -3, 0, 0}, {1, -1, 0, -1}, {1, 1, -6, -2}})
          .isIntegerEmpty());

  // 2x = 3y and y = x - 1 + 6z and x + y = 6q + 2 and 0 <= x, y <= 100.
  // 2x = 3y implies x is a multiple of 3 and y is even.
  // Now y = x - 1 + 6z implies y = 2 mod 3. In fact, since y is even, we have
  // y = 2 mod 6. Then since x = y + 1 + 6z, we have x = 3 mod 6, implying
  // x + y = 5 mod 6, which contradicts x + y = 6q + 2, so the set is empty.
  EXPECT_TRUE(makeFACFromConstraints(
                  4,
                  {
                      {1, 0, 0, 0, 0},
                      {-1, 0, 0, 0, 100},
                      {0, 1, 0, 0, 0},
                      {0, -1, 0, 0, 100},
                  },
                  {{2, -3, 0, 0, 0}, {1, -1, 6, 0, -1}, {1, 1, 0, -6, -2}})
                  .isIntegerEmpty());

  // Set with symbols.
  FlatAffineConstraints fac6 = makeFACFromConstraints(2,
                                                      {
                                                          {1, 1, 0},
                                                      },
                                                      {
                                                          {1, -1, 0},
                                                      },
                                                      1);
  EXPECT_FALSE(fac6.isIntegerEmpty());
}

TEST(FlatAffineConstraintsTest, removeRedundantConstraintsTest) {
  FlatAffineConstraints fac = makeFACFromConstraints(1,
                                                     {
                                                         {1, -2}, // x >= 2.
                                                         {-1, 2}  // x <= 2.
                                                     },
                                                     {{1, -2}}); // x == 2.
  fac.removeRedundantConstraints();

  // Both inequalities are redundant given the equality. Both have been removed.
  EXPECT_EQ(fac.getNumInequalities(), 0u);
  EXPECT_EQ(fac.getNumEqualities(), 1u);

  FlatAffineConstraints fac2 =
      makeFACFromConstraints(2,
                             {
                                 {1, 0, -3}, // x >= 3.
                                 {0, 1, -2}  // y >= 2 (redundant).
                             },
                             {{1, -1, 0}}); // x == y.
  fac2.removeRedundantConstraints();

  // The second inequality is redundant and should have been removed. The
  // remaining inequality should be the first one.
  EXPECT_EQ(fac2.getNumInequalities(), 1u);
  EXPECT_THAT(fac2.getInequality(0), ElementsAre(1, 0, -3));
  EXPECT_EQ(fac2.getNumEqualities(), 1u);

  FlatAffineConstraints fac3 =
      makeFACFromConstraints(3, {},
                             {{1, -1, 0, 0},   // x == y.
                              {1, 0, -1, 0},   // x == z.
                              {0, 1, -1, 0}}); // y == z.
  fac3.removeRedundantConstraints();

  // One of the three equalities can be removed.
  EXPECT_EQ(fac3.getNumInequalities(), 0u);
  EXPECT_EQ(fac3.getNumEqualities(), 2u);

  FlatAffineConstraints fac4 = makeFACFromConstraints(
      17,
      {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
       {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500},
       {0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
       {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 998},
       {0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
       {0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 998},
       {0, 0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 500},
       {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 15},
       {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16, 0, 1, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 998},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, -1, 0, 0, 15},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1},
       {0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 8, 8},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -8, -1},
       {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -8, -1},
       {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -10},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 10},
       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -13},
       {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 13},
       {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10},
       {0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10},
       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -13},
       {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13}},
      {});

  // The above is a large set of constraints without any redundant constraints,
  // as verified by the Fourier-Motzkin based removeRedundantInequalities.
  unsigned nIneq = fac4.getNumInequalities();
  unsigned nEq = fac4.getNumEqualities();
  fac4.removeRedundantInequalities();
  ASSERT_EQ(fac4.getNumInequalities(), nIneq);
  ASSERT_EQ(fac4.getNumEqualities(), nEq);
  // Now we test that removeRedundantConstraints does not find any constraints
  // to be redundant either.
  fac4.removeRedundantConstraints();
  EXPECT_EQ(fac4.getNumInequalities(), nIneq);
  EXPECT_EQ(fac4.getNumEqualities(), nEq);

  FlatAffineConstraints fac5 =
      makeFACFromConstraints(2,
                             {
                                 {128, 0, 127}, // [0]: 128x >= -127.
                                 {-1, 0, 7},    // [1]: x <= 7.
                                 {-128, 1, 0},  // [2]: y >= 128x.
                                 {0, 1, 0}      // [3]: y >= 0.
                             },
                             {});
  // [0] implies that 128x >= 0, since x has to be an integer. (This should be
  // caught by GCDTightenInqualities().)
  // So [2] and [0] imply [3] since we have y >= 128x >= 0.
  fac5.removeRedundantConstraints();
  EXPECT_EQ(fac5.getNumInequalities(), 3u);
  SmallVector<int64_t, 8> redundantConstraint = {0, 1, 0};
  for (unsigned i = 0; i < 3; ++i) {
    // Ensure that the removed constraint was the redundant constraint [3].
    EXPECT_NE(fac5.getInequality(i), ArrayRef<int64_t>(redundantConstraint));
  }
}

TEST(FlatAffineConstraintsTest, addConstantUpperBound) {
  FlatAffineConstraints fac = makeFACFromConstraints(2, {}, {});
  fac.addBound(FlatAffineConstraints::UB, 0, 1);
  EXPECT_EQ(fac.atIneq(0, 0), -1);
  EXPECT_EQ(fac.atIneq(0, 1), 0);
  EXPECT_EQ(fac.atIneq(0, 2), 1);

  fac.addBound(FlatAffineConstraints::UB, {1, 2, 3}, 1);
  EXPECT_EQ(fac.atIneq(1, 0), -1);
  EXPECT_EQ(fac.atIneq(1, 1), -2);
  EXPECT_EQ(fac.atIneq(1, 2), -2);
}

TEST(FlatAffineConstraintsTest, addConstantLowerBound) {
  FlatAffineConstraints fac = makeFACFromConstraints(2, {}, {});
  fac.addBound(FlatAffineConstraints::LB, 0, 1);
  EXPECT_EQ(fac.atIneq(0, 0), 1);
  EXPECT_EQ(fac.atIneq(0, 1), 0);
  EXPECT_EQ(fac.atIneq(0, 2), -1);

  fac.addBound(FlatAffineConstraints::LB, {1, 2, 3}, 1);
  EXPECT_EQ(fac.atIneq(1, 0), 1);
  EXPECT_EQ(fac.atIneq(1, 1), 2);
  EXPECT_EQ(fac.atIneq(1, 2), 2);
}

TEST(FlatAffineConstraintsTest, removeInequality) {
  FlatAffineConstraints fac =
      makeFACFromConstraints(1, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, {});

  fac.removeInequalityRange(0, 0);
  EXPECT_EQ(fac.getNumInequalities(), 5u);

  fac.removeInequalityRange(1, 3);
  EXPECT_EQ(fac.getNumInequalities(), 3u);
  EXPECT_THAT(fac.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(fac.getInequality(1), ElementsAre(3, 3));
  EXPECT_THAT(fac.getInequality(2), ElementsAre(4, 4));

  fac.removeInequality(1);
  EXPECT_EQ(fac.getNumInequalities(), 2u);
  EXPECT_THAT(fac.getInequality(0), ElementsAre(0, 0));
  EXPECT_THAT(fac.getInequality(1), ElementsAre(4, 4));
}

TEST(FlatAffineConstraintsTest, removeEquality) {
  FlatAffineConstraints fac =
      makeFACFromConstraints(1, {}, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});

  fac.removeEqualityRange(0, 0);
  EXPECT_EQ(fac.getNumEqualities(), 5u);

  fac.removeEqualityRange(1, 3);
  EXPECT_EQ(fac.getNumEqualities(), 3u);
  EXPECT_THAT(fac.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(fac.getEquality(1), ElementsAre(3, 3));
  EXPECT_THAT(fac.getEquality(2), ElementsAre(4, 4));

  fac.removeEquality(1);
  EXPECT_EQ(fac.getNumEqualities(), 2u);
  EXPECT_THAT(fac.getEquality(0), ElementsAre(0, 0));
  EXPECT_THAT(fac.getEquality(1), ElementsAre(4, 4));
}

TEST(FlatAffineConstraintsTest, clearConstraints) {
  FlatAffineConstraints fac = makeFACFromConstraints(1, {}, {});

  fac.addInequality({1, 0});
  EXPECT_EQ(fac.atIneq(0, 0), 1);
  EXPECT_EQ(fac.atIneq(0, 1), 0);

  fac.clearConstraints();

  fac.addInequality({1, 0});
  EXPECT_EQ(fac.atIneq(0, 0), 1);
  EXPECT_EQ(fac.atIneq(0, 1), 0);
}

/// Check if the expected division representation of local variables matches the
/// computed representation. The expected division representation is given as
/// a vector of expressions set in `divisions` and the corressponding
/// denominator in `denoms`. If expected denominator for a variable is
/// non-positive, the local variable is expected to not have a computed
/// representation.
static void checkDivisionRepresentation(
    FlatAffineConstraints &fac,
    const std::vector<SmallVector<int64_t, 8>> &divisions,
    const SmallVector<int64_t, 8> &denoms) {

  assert(divisions.size() == fac.getNumLocalIds() &&
         "Size of expected divisions does not match number of local variables");
  assert(
      denoms.size() == fac.getNumLocalIds() &&
      "Size of expected denominators does not match number of local variables");

  std::vector<llvm::Optional<std::pair<unsigned, unsigned>>> res(
      fac.getNumLocalIds(), llvm::None);
  fac.getLocalReprLbUbPairs(res);

  // Check if all expected divisions are computed.
  for (unsigned i = 0, e = fac.getNumLocalIds(); i < e; ++i)
    if (denoms[i] > 0)
      EXPECT_TRUE(res[i].hasValue());
    else
      EXPECT_FALSE(res[i].hasValue());

  unsigned divOffset = fac.getNumDimAndSymbolIds();
  for (unsigned i = 0, e = fac.getNumLocalIds(); i < e; ++i) {
    if (!res[i])
      continue;

    // Check if the bounds are of the form:
    //      0 <= expr - divisor * id <= divisor - 1
    // Rearranging, we have:
    //       divisor * id - expr + (divisor - 1) >= 0  <-- Lower bound for 'id'
    //      -divisor * id + expr                 >= 0  <-- Upper bound for 'id'
    // where `id = expr floordiv divisor`.
    unsigned ubPos = res[i]->first, lbPos = res[i]->second;
    const SmallVector<int64_t, 8> &expr = divisions[i];

    // Check if lower bound is of the correct form.
    int64_t computedDivisorLb = fac.atIneq(lbPos, i + divOffset);
    EXPECT_EQ(computedDivisorLb, denoms[i]);
    for (unsigned c = 0, f = fac.getNumLocalIds(); c < f; ++c) {
      if (c == i + divOffset)
        continue;
      EXPECT_EQ(fac.atIneq(lbPos, c), -expr[c]);
    }
    // Check if constant term of lower bound matches expected constant term.
    EXPECT_EQ(fac.atIneq(lbPos, fac.getNumCols() - 1),
              -expr.back() + (denoms[i] - 1));

    // Check if upper bound is of the correct form.
    int64_t computedDivisorUb = fac.atIneq(ubPos, i + divOffset);
    EXPECT_EQ(computedDivisorUb, -denoms[i]);
    for (unsigned c = 0, f = fac.getNumLocalIds(); c < f; ++c) {
      if (c == i + divOffset)
        continue;
      EXPECT_EQ(fac.atIneq(ubPos, c), expr[c]);
    }
    // Check if constant term of upper bound matches expected constant term.
    EXPECT_EQ(fac.atIneq(ubPos, fac.getNumCols() - 1), expr.back());
  }
}

TEST(FlatAffineConstraintsTest, computeLocalReprSimple) {
  FlatAffineConstraints fac = makeFACFromConstraints(1, {}, {});

  fac.addLocalFloorDiv({1, 4}, 10);
  fac.addLocalFloorDiv({1, 0, 100}, 10);

  std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0, 4},
                                                    {1, 0, 0, 100}};
  SmallVector<int64_t, 8> denoms = {10, 10};

  // Check if floordivs can be computed when no other inequalities exist
  // and floor divs do not depend on each other.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, computeLocalReprConstantFloorDiv) {
  FlatAffineConstraints fac = makeFACFromConstraints(4, {}, {});

  fac.addInequality({1, 0, 3, 1, 2});
  fac.addInequality({1, 2, -8, 1, 10});
  fac.addEquality({1, 2, -4, 1, 10});

  fac.addLocalFloorDiv({0, 0, 0, 0, 10}, 30);
  fac.addLocalFloorDiv({0, 0, 0, 0, 0, 99}, 101);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0, 0, 0, 0, 10},
                                                    {0, 0, 0, 0, 0, 0, 99}};
  SmallVector<int64_t, 8> denoms = {30, 101};

  // Check if floordivs with constant numerator can be computed.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, computeLocalReprRecursive) {
  FlatAffineConstraints fac = makeFACFromConstraints(4, {}, {});
  fac.addInequality({1, 0, 3, 1, 2});
  fac.addInequality({1, 2, -8, 1, 10});
  fac.addEquality({1, 2, -4, 1, 10});

  fac.addLocalFloorDiv({0, -2, 7, 2, 10}, 3);
  fac.addLocalFloorDiv({3, 0, 9, 2, 2, 10}, 5);
  fac.addLocalFloorDiv({0, 1, -123, 2, 0, -4, 10}, 3);

  fac.addInequality({1, 2, -2, 1, -5, 0, 6, 100});
  fac.addInequality({1, 2, -8, 1, 3, 7, 0, -9});

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, -2, 7, 2, 0, 0, 0, 10},
                                                    {3, 0, 9, 2, 2, 0, 0, 10},
                                                    {0, 1, -123, 2, 0, -4, 10}};
  SmallVector<int64_t, 8> denoms = {3, 5, 3};

  // Check if floordivs which may depend on other floordivs can be computed.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, removeIdRange) {
  FlatAffineConstraints fac(3, 2, 1);

  fac.addInequality({10, 11, 12, 20, 21, 30, 40});
  fac.removeId(FlatAffineConstraints::IdKind::Symbol, 1);
  EXPECT_THAT(fac.getInequality(0),
              testing::ElementsAre(10, 11, 12, 20, 30, 40));

  fac.removeIdRange(FlatAffineConstraints::IdKind::Dimension, 0, 2);
  EXPECT_THAT(fac.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  fac.removeIdRange(FlatAffineConstraints::IdKind::Local, 1, 1);
  EXPECT_THAT(fac.getInequality(0), testing::ElementsAre(12, 20, 30, 40));

  fac.removeIdRange(FlatAffineConstraints::IdKind::Local, 0, 1);
  EXPECT_THAT(fac.getInequality(0), testing::ElementsAre(12, 20, 40));
}

TEST(FlatAffineConstraintsTest, simplifyLocalsTest) {
  // (x) : (exists y: 2x + y = 1 and y = 2).
  FlatAffineConstraints fac(1, 0, 1);
  fac.addEquality({2, 1, -1});
  fac.addEquality({0, 1, -2});

  EXPECT_TRUE(fac.isEmpty());

  // (x) : (exists y, z, w: 3x + y = 1 and 2y = z and 3y = w and z = w).
  FlatAffineConstraints fac2(1, 0, 3);
  fac2.addEquality({3, 1, 0, 0, -1});
  fac2.addEquality({0, 2, -1, 0, 0});
  fac2.addEquality({0, 3, 0, -1, 0});
  fac2.addEquality({0, 0, 1, -1, 0});

  EXPECT_TRUE(fac2.isEmpty());

  // (x) : (exists y: x >= y + 1 and 2x + y = 0 and y >= -1).
  FlatAffineConstraints fac3(1, 0, 1);
  fac3.addInequality({1, -1, -1});
  fac3.addInequality({0, 1, 1});
  fac3.addEquality({2, 1, 0});

  EXPECT_TRUE(fac3.isEmpty());
}

} // namespace mlir
