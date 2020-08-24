//===- AffineStructuresTest.cpp - Tests for AffineStructures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

namespace mlir {

/// Evaluate the value of the given affine expression at the specified point.
/// The expression is a list of coefficients for the dimensions followed by the
/// constant term.
int64_t valueAt(ArrayRef<int64_t> expr, ArrayRef<int64_t> point) {
  assert(expr.size() == 1 + point.size());
  int64_t value = expr.back();
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

/// If 'hasValue' is true, check that findIntegerSample returns a valid sample
/// for the FlatAffineConstraints fac.
///
/// If hasValue is false, check that findIntegerSample does not return None.
void checkSample(bool hasValue, const FlatAffineConstraints &fac) {
  Optional<SmallVector<int64_t, 8>> maybeSample = fac.findIntegerSample();
  if (!hasValue) {
    EXPECT_FALSE(maybeSample.hasValue());
    if (maybeSample.hasValue()) {
      for (auto x : *maybeSample)
        llvm::errs() << x << ' ';
      llvm::errs() << '\n';
    }
  } else {
    ASSERT_TRUE(maybeSample.hasValue());
    for (unsigned i = 0; i < fac.getNumEqualities(); ++i)
      EXPECT_EQ(valueAt(fac.getEquality(i), *maybeSample), 0);
    for (unsigned i = 0; i < fac.getNumInequalities(); ++i)
      EXPECT_GE(valueAt(fac.getInequality(i), *maybeSample), 0);
  }
}

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
FlatAffineConstraints
makeFACFromConstraints(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + 1, dims);
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
void checkPermutationsSample(bool hasValue, unsigned nDim,
                             ArrayRef<SmallVector<int64_t, 4>> ineqs,
                             ArrayRef<SmallVector<int64_t, 4>> eqs) {
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

    checkSample(hasValue,
                makeFACFromConstraints(nDim, permutedIneqs, permutedEqs));
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
  // the GBR algroithm.
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
}

TEST(FlatAffineConstraintsTest, IsIntegerEmptyTest) {
  // 1 <= 5x and 5x <= 4 (no solution).
  EXPECT_TRUE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 4}}, {}).isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  EXPECT_FALSE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 9}}, {}).isIntegerEmpty());

  // An unbounded set, which isIntegerEmpty should detect as unbounded and
  // return without calling findIntegerSample.
  EXPECT_FALSE(makeFACFromConstraints(3,
                                      {
                                          {2, 0, 0, -1},
                                          {-2, 0, 0, 1},
                                          {0, 2, 0, -1},
                                          {0, -2, 0, 1},
                                          {0, 0, 2, -1},
                                      },
                                      {})
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
  EXPECT_THAT(fac2.getInequality(0), testing::ElementsAre(1, 0, -3));
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

} // namespace mlir
