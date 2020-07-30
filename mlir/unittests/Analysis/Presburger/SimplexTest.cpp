//===- SimplexTest.cpp - Tests for Simplex --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

/// Take a snapshot, add constraints making the set empty, and rollback.
/// The set should not be empty after rolling back.
TEST(SimplexTest, emptyRollback) {
  Simplex simplex(2);
  // (u - v) >= 0
  simplex.addInequality({1, -1, 0});
  EXPECT_FALSE(simplex.isEmpty());

  unsigned snapshot = simplex.getSnapshot();
  // (u - v) <= -1
  simplex.addInequality({-1, 1, -1});
  EXPECT_TRUE(simplex.isEmpty());
  simplex.rollback(snapshot);
  EXPECT_FALSE(simplex.isEmpty());
}

/// Check that the set gets marked as empty when we add contradictory
/// constraints.
TEST(SimplexTest, addEquality_separate) {
  Simplex simplex(1);
  simplex.addInequality({1, -1}); // x >= 1.
  ASSERT_FALSE(simplex.isEmpty());
  simplex.addEquality({1, 0}); // x == 0.
  EXPECT_TRUE(simplex.isEmpty());
}

void expectInequalityMakesSetEmpty(Simplex &simplex, ArrayRef<int64_t> coeffs,
                                   bool expect) {
  ASSERT_FALSE(simplex.isEmpty());
  unsigned snapshot = simplex.getSnapshot();
  simplex.addInequality(coeffs);
  EXPECT_EQ(simplex.isEmpty(), expect);
  simplex.rollback(snapshot);
}

TEST(SimplexTest, addInequality_rollback) {
  Simplex simplex(3);
  SmallVector<int64_t, 4> coeffs[]{{1, 0, 0, 0},   // u >= 0.
                                   {-1, 0, 0, 0},  // u <= 0.
                                   {1, -1, 1, 0},  // u - v + w >= 0.
                                   {1, 1, -1, 0}}; // u + v - w >= 0.
  // The above constraints force u = 0 and v = w.
  // The constraints below violate v = w.
  SmallVector<int64_t, 4> checkCoeffs[]{{0, 1, -1, -1},  // v - w >= 1.
                                        {0, -1, 1, -1}}; // v - w <= -1.

  for (int run = 0; run < 4; run++) {
    unsigned snapshot = simplex.getSnapshot();

    expectInequalityMakesSetEmpty(simplex, checkCoeffs[0], false);
    expectInequalityMakesSetEmpty(simplex, checkCoeffs[1], false);

    for (int i = 0; i < 4; i++)
      simplex.addInequality(coeffs[(run + i) % 4]);

    expectInequalityMakesSetEmpty(simplex, checkCoeffs[0], true);
    expectInequalityMakesSetEmpty(simplex, checkCoeffs[1], true);

    simplex.rollback(snapshot);
    EXPECT_EQ(simplex.numConstraints(), 0u);

    expectInequalityMakesSetEmpty(simplex, checkCoeffs[0], false);
    expectInequalityMakesSetEmpty(simplex, checkCoeffs[1], false);
  }
}

Simplex simplexFromConstraints(unsigned nDim,
                               SmallVector<SmallVector<int64_t, 8>, 8> ineqs,
                               SmallVector<SmallVector<int64_t, 8>, 8> eqs) {
  Simplex simplex(nDim);
  for (const auto &ineq : ineqs)
    simplex.addInequality(ineq);
  for (const auto &eq : eqs)
    simplex.addEquality(eq);
  return simplex;
}

TEST(SimplexTest, isUnbounded) {
  EXPECT_FALSE(simplexFromConstraints(
                   2, {{1, 1, 0}, {-1, -1, 0}, {1, -1, 5}, {-1, 1, -5}}, {})
                   .isUnbounded());

  EXPECT_TRUE(
      simplexFromConstraints(2, {{1, 1, 0}, {1, -1, 5}, {-1, 1, -5}}, {})
          .isUnbounded());

  EXPECT_TRUE(
      simplexFromConstraints(2, {{-1, -1, 0}, {1, -1, 5}, {-1, 1, -5}}, {})
          .isUnbounded());

  EXPECT_TRUE(simplexFromConstraints(2, {}, {}).isUnbounded());

  EXPECT_FALSE(simplexFromConstraints(3,
                                      {
                                          {2, 0, 0, -1},
                                          {-2, 0, 0, 1},
                                          {0, 2, 0, -1},
                                          {0, -2, 0, 1},
                                          {0, 0, 2, -1},
                                          {0, 0, -2, 1},
                                      },
                                      {})
                   .isUnbounded());

  EXPECT_TRUE(simplexFromConstraints(3,
                                     {
                                         {2, 0, 0, -1},
                                         {-2, 0, 0, 1},
                                         {0, 2, 0, -1},
                                         {0, -2, 0, 1},
                                         {0, 0, -2, 1},
                                     },
                                     {})
                  .isUnbounded());

  EXPECT_TRUE(simplexFromConstraints(3,
                                     {
                                         {2, 0, 0, -1},
                                         {-2, 0, 0, 1},
                                         {0, 2, 0, -1},
                                         {0, -2, 0, 1},
                                         {0, 0, 2, -1},
                                     },
                                     {})
                  .isUnbounded());

  // Bounded set with equalities.
  EXPECT_FALSE(simplexFromConstraints(2,
                                      {{1, 1, 1},    // x + y >= -1.
                                       {-1, -1, 1}}, // x + y <=  1.
                                      {{1, -1, 0}}   // x = y.
                                      )
                   .isUnbounded());

  // Unbounded set with equalities.
  EXPECT_TRUE(simplexFromConstraints(3,
                                     {{1, 1, 1, 1},     // x + y + z >= -1.
                                      {-1, -1, -1, 1}}, // x + y + z <=  1.
                                     {{1, -1, -1, 0}}   // x = y + z.
                                     )
                  .isUnbounded());

  // Rational empty set.
  EXPECT_FALSE(simplexFromConstraints(3,
                                      {
                                          {2, 0, 0, -1},
                                          {-2, 0, 0, 1},
                                          {0, 2, 2, -1},
                                          {0, -2, -2, 1},
                                          {3, 3, 3, -4},
                                      },
                                      {})
                   .isUnbounded());
}

TEST(SimplexTest, getSamplePointIfIntegral) {
  // Empty set.
  EXPECT_FALSE(simplexFromConstraints(3,
                                      {
                                          {2, 0, 0, -1},
                                          {-2, 0, 0, 1},
                                          {0, 2, 2, -1},
                                          {0, -2, -2, 1},
                                          {3, 3, 3, -4},
                                      },
                                      {})
                   .getSamplePointIfIntegral()
                   .hasValue());

  auto maybeSample = simplexFromConstraints(2,
                                            {// x = y - 2.
                                             {1, -1, 2},
                                             {-1, 1, -2},
                                             // x + y = 2.
                                             {1, 1, -2},
                                             {-1, -1, 2}},
                                            {})
                         .getSamplePointIfIntegral();

  EXPECT_TRUE(maybeSample.hasValue());
  EXPECT_THAT(*maybeSample, testing::ElementsAre(0, 2));

  auto maybeSample2 = simplexFromConstraints(2,
                                             {
                                                 {1, 0, 0},  // x >= 0.
                                                 {-1, 0, 0}, // x <= 0.
                                             },
                                             {
                                                 {0, 1, -2} // y = 2.
                                             })
                          .getSamplePointIfIntegral();
  EXPECT_TRUE(maybeSample2.hasValue());
  EXPECT_THAT(*maybeSample2, testing::ElementsAre(0, 2));

  EXPECT_FALSE(simplexFromConstraints(1,
                                      {// 2x = 1. (no integer solutions)
                                       {2, -1},
                                       {-2, +1}},
                                      {})
                   .getSamplePointIfIntegral()
                   .hasValue());
}

} // namespace mlir
