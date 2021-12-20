//===- AffineStructuresTest.cpp - Tests for AffineStructures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "./AffineStructuresParser.h"
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

/// Parses a FlatAffineConstraints from a StringRef. It is expected that the
/// string represents a valid IntegerSet, otherwise it will violate a gtest
/// assertion.
static FlatAffineConstraints parseFAC(StringRef str, MLIRContext *context) {
  FailureOr<FlatAffineConstraints> fac = parseIntegerSetToFAC(str, context);

  EXPECT_TRUE(succeeded(fac));

  return *fac;
}

TEST(FlatAffineConstraintsTest, FindSampleTest) {
  // Bounded sets with only inequalities.

  MLIRContext context;

  // 0 <= 7x <= 5
  checkSample(true, parseFAC("(x) : (7 * x >= 0, -7 * x + 5 >= 0)", &context));

  // 1 <= 5x and 5x <= 4 (no solution).
  checkSample(false,
              parseFAC("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)", &context));

  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  checkSample(true,
              parseFAC("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)", &context));

  // Bounded sets with equalities.
  // x >= 8 and 40 >= y and x = y.
  checkSample(true, parseFAC("(x,y) : (x - 8 >= 0, -y + 40 >= 0, x - y == 0)",
                             &context));

  // x <= 10 and y <= 10 and 10 <= z and x + 2y = 3z.
  // solution: x = y = z = 10.
  checkSample(true, parseFAC("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                             "z - 10 >= 0, x + 2 * y - 3 * z == 0)",
                             &context));

  // x <= 10 and y <= 10 and 11 <= z and x + 2y = 3z.
  // This implies x + 2y >= 33 and x + 2y <= 30, which has no solution.
  checkSample(false, parseFAC("(x,y,z) : (-x + 10 >= 0, -y + 10 >= 0, "
                              "z - 11 >= 0, x + 2 * y - 3 * z == 0)",
                              &context));

  // 0 <= r and r <= 3 and 4q + r = 7.
  // Solution: q = 1, r = 3.
  checkSample(
      true,
      parseFAC("(q,r) : (r >= 0, -r + 3 >= 0, 4 * q + r - 7 == 0)", &context));

  // 4q + r = 7 and r = 0.
  // Solution: q = 1, r = 3.
  checkSample(false,
              parseFAC("(q,r) : (4 * q + r - 7 == 0, r == 0)", &context));

  // The next two sets are large sets that should take a long time to sample
  // with a naive branch and bound algorithm but can be sampled efficiently with
  // the GBR algorithm.
  //
  // This is a triangle with vertices at (1/3, 0), (2/3, 0) and (10000, 10000).
  checkSample(true, parseFAC("(x,y) : (y >= 0, "
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
      parseFAC("(a,b,c,d,e) : (b + d - e >= 0, -b + c - d + e >= 0, "
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
      false, parseFAC("(x,y) : (x >= 0, -x + 100 >= 0, 3 * x - 3 * y + 1 == 0)",
                      &context));

  // A thin parallelogram. 0 <= x <= 100 and x + 1/3 <= y <= x + 2/3.
  checkSample(false,
              parseFAC("(x,y) : (x >= 0, -x + 100 >= 0, "
                       "3 * x - 3 * y + 2 >= 0, -3 * x + 3 * y - 1 >= 0)",
                       &context));

  checkSample(true, parseFAC("(x,y) : (2 * x >= 0, -2 * x + 99 >= 0, "
                             "2 * y >= 0, -2 * y + 99 >= 0)",
                             &context));

  // 2D cone with apex at (10000, 10000) and
  // edges passing through (1/3, 0) and (2/3, 0).
  checkSample(true, parseFAC("(x,y) : (300000 * x - 299999 * y - 100000 >= 0, "
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

  checkSample(true, parseFAC("(x, y, z) : (2 * x - 1 >= 0, x - y - 1 == 0, "
                             "y - z == 0)",
                             &context));

  // Regression tests for the computation of dual coefficients.
  checkSample(false, parseFAC("(x, y, z) : ("
                              "6*x - 4*y + 9*z + 2 >= 0,"
                              "x + 5*y + z + 5 >= 0,"
                              "-4*x + y + 2*z - 1 >= 0,"
                              "-3*x - 2*y - 7*z - 1 >= 0,"
                              "-7*x - 5*y - 9*z - 1 >= 0)",
                              &context));
  checkSample(true, parseFAC("(x, y, z) : ("
                             "3*x + 3*y + 3 >= 0,"
                             "-4*x - 8*y - z + 4 >= 0,"
                             "-7*x - 4*y + z + 1 >= 0,"
                             "2*x - 7*y - 8*z - 7 >= 0,"
                             "9*x + 8*y - 9*z - 7 >= 0)",
                             &context));
}

TEST(FlatAffineConstraintsTest, IsIntegerEmptyTest) {

  MLIRContext context;

  // 1 <= 5x and 5x <= 4 (no solution).
  EXPECT_TRUE(parseFAC("(x) : (5 * x - 1 >= 0, -5 * x + 4 >= 0)", &context)
                  .isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1).
  EXPECT_FALSE(parseFAC("(x) : (5 * x - 1 >= 0, -5 * x + 9 >= 0)", &context)
                   .isIntegerEmpty());

  // Unbounded sets.
  EXPECT_TRUE(parseFAC("(x,y,z) : (2 * y - 1 >= 0, -2 * y + 1 >= 0, "
                       "2 * z - 1 >= 0, 2 * x - 1 == 0)",
                       &context)
                  .isIntegerEmpty());

  EXPECT_FALSE(parseFAC("(x,y,z) : (2 * x - 1 >= 0, -3 * x + 3 >= 0, "
                        "5 * z - 6 >= 0, -7 * z + 17 >= 0, 3 * y - 2 >= 0)",
                        &context)
                   .isIntegerEmpty());

  EXPECT_FALSE(
      parseFAC("(x,y,z) : (2 * x - 1 >= 0, x - y - 1 == 0, y - z == 0)",
               &context)
          .isIntegerEmpty());

  // FlatAffineConstraints::isEmpty() does not detect the following sets to be
  // empty.

  // 3x + 7y = 1 and 0 <= x, y <= 10.
  // Since x and y are non-negative, 3x + 7y can never be 1.
  EXPECT_TRUE(parseFAC("(x,y) : (x >= 0, -x + 10 >= 0, y >= 0, -y + 10 >= 0, "
                       "3 * x + 7 * y - 1 == 0)",
                       &context)
                  .isIntegerEmpty());

  // 2x = 3y and y = x - 1 and x + y = 6z + 2 and 0 <= x, y <= 100.
  // Substituting y = x - 1 in 3y = 2x, we obtain x = 3 and hence y = 2.
  // Since x + y = 5 cannot be equal to 6z + 2 for any z, the set is empty.
  EXPECT_TRUE(
      parseFAC("(x,y,z) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
               "2 * x - 3 * y == 0, x - y - 1 == 0, x + y - 6 * z - 2 == 0)",
               &context)
          .isIntegerEmpty());

  // 2x = 3y and y = x - 1 + 6z and x + y = 6q + 2 and 0 <= x, y <= 100.
  // 2x = 3y implies x is a multiple of 3 and y is even.
  // Now y = x - 1 + 6z implies y = 2 mod 3. In fact, since y is even, we have
  // y = 2 mod 6. Then since x = y + 1 + 6z, we have x = 3 mod 6, implying
  // x + y = 5 mod 6, which contradicts x + y = 6q + 2, so the set is empty.
  EXPECT_TRUE(
      parseFAC(
          "(x,y,z,q) : (x >= 0, -x + 100 >= 0, y >= 0, -y + 100 >= 0, "
          "2 * x - 3 * y == 0, x - y + 6 * z - 1 == 0, x + y - 6 * q - 2 == 0)",
          &context)
          .isIntegerEmpty());

  // Set with symbols.
  EXPECT_FALSE(
      parseFAC("(x)[s] : (x + s >= 0, x - s == 0)", &context).isIntegerEmpty());
}

TEST(FlatAffineConstraintsTest, removeRedundantConstraintsTest) {
  MLIRContext context;

  FlatAffineConstraints fac =
      parseFAC("(x) : (x - 2 >= 0, -x + 2 >= 0, x - 2 == 0)", &context);
  fac.removeRedundantConstraints();

  // Both inequalities are redundant given the equality. Both have been removed.
  EXPECT_EQ(fac.getNumInequalities(), 0u);
  EXPECT_EQ(fac.getNumEqualities(), 1u);

  FlatAffineConstraints fac2 =
      parseFAC("(x,y) : (x - 3 >= 0, y - 2 >= 0, x - y == 0)", &context);
  fac2.removeRedundantConstraints();

  // The second inequality is redundant and should have been removed. The
  // remaining inequality should be the first one.
  EXPECT_EQ(fac2.getNumInequalities(), 1u);
  EXPECT_THAT(fac2.getInequality(0), ElementsAre(1, 0, -3));
  EXPECT_EQ(fac2.getNumEqualities(), 1u);

  FlatAffineConstraints fac3 =
      parseFAC("(x,y,z) : (x - y == 0, x - z == 0, y - z == 0)", &context);
  fac3.removeRedundantConstraints();

  // One of the three equalities can be removed.
  EXPECT_EQ(fac3.getNumInequalities(), 0u);
  EXPECT_EQ(fac3.getNumEqualities(), 2u);

  FlatAffineConstraints fac4 =
      parseFAC("(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) : ("
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

  FlatAffineConstraints fac5 = parseFAC(
      "(x,y) : (128 * x + 127 >= 0, -x + 7 >= 0, -128 * x + y >= 0, y >= 0)",
      &context);
  // 128x + 127 >= 0  implies that 128x >= 0, since x has to be an integer.
  // (This should be caught by GCDTightenInqualities().)
  // So -128x + y >= 0 and 128x + 127 >= 0 imply y >= 0 since we have
  // y >= 128x >= 0.
  fac5.removeRedundantConstraints();
  EXPECT_EQ(fac5.getNumInequalities(), 3u);
  SmallVector<int64_t, 8> redundantConstraint = {0, 1, 0};
  for (unsigned i = 0; i < 3; ++i) {
    // Ensure that the removed constraint was the redundant constraint [3].
    EXPECT_NE(fac5.getInequality(i), ArrayRef<int64_t>(redundantConstraint));
  }
}

TEST(FlatAffineConstraintsTest, addConstantUpperBound) {
  FlatAffineConstraints fac(2);
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
  FlatAffineConstraints fac(2);
  fac.addBound(FlatAffineConstraints::LB, 0, 1);
  EXPECT_EQ(fac.atIneq(0, 0), 1);
  EXPECT_EQ(fac.atIneq(0, 1), 0);
  EXPECT_EQ(fac.atIneq(0, 2), -1);

  fac.addBound(FlatAffineConstraints::LB, {1, 2, 3}, 1);
  EXPECT_EQ(fac.atIneq(1, 0), 1);
  EXPECT_EQ(fac.atIneq(1, 1), 2);
  EXPECT_EQ(fac.atIneq(1, 2), 2);
}

/// Check if the expected division representation of local variables matches the
/// computed representation. The expected division representation is given as
/// a vector of expressions set in `expectedDividends` and the corressponding
/// denominator in `expectedDenominators`. The `denominators` and `dividends`
/// obtained through `getLocalRepr` function is verified against the
/// `expectedDenominators` and `expectedDividends` respectively.
static void checkDivisionRepresentation(
    FlatAffineConstraints &fac,
    const std::vector<SmallVector<int64_t, 8>> &expectedDividends,
    const SmallVectorImpl<unsigned> &expectedDenominators) {

  std::vector<SmallVector<int64_t, 8>> dividends;
  SmallVector<unsigned, 4> denominators;

  fac.getLocalReprs(dividends, denominators);

  // Check that the `denominators` and `expectedDenominators` match.
  EXPECT_TRUE(expectedDenominators == denominators);

  // Check that the `dividends` and `expectedDividends` match. If the
  // denominator for a division is zero, we ignore its dividend.
  EXPECT_TRUE(dividends.size() == expectedDividends.size());
  for (unsigned i = 0, e = dividends.size(); i < e; ++i)
    if (denominators[i] != 0)
      EXPECT_TRUE(expectedDividends[i] == dividends[i]);
}

TEST(FlatAffineConstraintsTest, computeLocalReprSimple) {
  FlatAffineConstraints fac(1);

  fac.addLocalFloorDiv({1, 4}, 10);
  fac.addLocalFloorDiv({1, 0, 100}, 10);

  std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0, 4},
                                                    {1, 0, 0, 100}};
  SmallVector<unsigned, 8> denoms = {10, 10};

  // Check if floordivs can be computed when no other inequalities exist
  // and floor divs do not depend on each other.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, computeLocalReprConstantFloorDiv) {
  FlatAffineConstraints fac(4);

  fac.addInequality({1, 0, 3, 1, 2});
  fac.addInequality({1, 2, -8, 1, 10});
  fac.addEquality({1, 2, -4, 1, 10});

  fac.addLocalFloorDiv({0, 0, 0, 0, 10}, 30);
  fac.addLocalFloorDiv({0, 0, 0, 0, 0, 99}, 101);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0, 0, 0, 0, 10},
                                                    {0, 0, 0, 0, 0, 0, 99}};
  SmallVector<unsigned, 8> denoms = {30, 101};

  // Check if floordivs with constant numerator can be computed.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, computeLocalReprRecursive) {
  FlatAffineConstraints fac(4);
  fac.addInequality({1, 0, 3, 1, 2});
  fac.addInequality({1, 2, -8, 1, 10});
  fac.addEquality({1, 2, -4, 1, 10});

  fac.addLocalFloorDiv({0, -2, 7, 2, 10}, 3);
  fac.addLocalFloorDiv({3, 0, 9, 2, 2, 10}, 5);
  fac.addLocalFloorDiv({0, 1, -123, 2, 0, -4, 10}, 3);

  fac.addInequality({1, 2, -2, 1, -5, 0, 6, 100});
  fac.addInequality({1, 2, -8, 1, 3, 7, 0, -9});

  std::vector<SmallVector<int64_t, 8>> divisions = {
      {0, -2, 7, 2, 0, 0, 0, 10},
      {3, 0, 9, 2, 2, 0, 0, 10},
      {0, 1, -123, 2, 0, -4, 0, 10}};

  SmallVector<unsigned, 8> denoms = {3, 5, 3};

  // Check if floordivs which may depend on other floordivs can be computed.
  checkDivisionRepresentation(fac, divisions, denoms);
}

TEST(FlatAffineConstraintsTest, computeLocalReprTightUpperBound) {
  MLIRContext context;

  {
    FlatAffineConstraints fac = parseFAC("(i) : (i mod 3 - 1 >= 0)", &context);

    // The set formed by the fac is:
    //        3q - i + 2 >= 0             <-- Division lower bound
    //       -3q + i - 1 >= 0
    //       -3q + i     >= 0             <-- Division upper bound
    // We remove redundant constraints to get the set:
    //        3q - i + 2 >= 0             <-- Division lower bound
    //       -3q + i - 1 >= 0             <-- Tighter division upper bound
    // thus, making the upper bound tighter.
    fac.removeRedundantConstraints();

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 0, 0}};
    SmallVector<unsigned, 8> denoms = {3};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(fac, divisions, denoms);
  }

  {
    FlatAffineConstraints fac = parseFAC(
        "(i, j, q) : (4*q - i - j + 2 >= 0, -4*q + i + j >= 0)", &context);
    // Convert `q` to a local variable.
    fac.convertDimToLocal(2, 3);

    std::vector<SmallVector<int64_t, 8>> divisions = {{1, 1, 0, 1}};
    SmallVector<unsigned, 8> denoms = {4};

    // Check if the divisions can be computed even with a tighter upper bound.
    checkDivisionRepresentation(fac, divisions, denoms);
  }
}

TEST(FlatAffineConstraintsTest, computeLocalReprNoRepr) {
  MLIRContext context;
  FlatAffineConstraints fac =
      parseFAC("(x, q) : (x - 3 * q >= 0, -x + 3 * q + 3 >= 0)", &context);
  // Convert q to a local variable.
  fac.convertDimToLocal(1, 2);

  std::vector<SmallVector<int64_t, 8>> divisions = {{0, 0, 0}};
  SmallVector<unsigned, 8> denoms = {0};

  // Check that no division is computed.
  checkDivisionRepresentation(fac, divisions, denoms);
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

TEST(FlatAffineConstraintsTest, mergeDivisionsSimple) {
  {
    // (x) : (exists z, y  = [x / 2] : x = 3y and x + z + 1 >= 0).
    FlatAffineConstraints fac1(1, 0, 1);
    fac1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    fac1.addEquality({1, 0, -3, 0});     // x = 3y.
    fac1.addInequality({1, 1, 0, 1});    // x + z + 1 >= 0.

    // (x) : (exists y = [x / 2], z : x = 5y).
    FlatAffineConstraints fac2(1);
    fac2.addLocalFloorDiv({1, 0}, 2); // y = [x / 2].
    fac2.addEquality({1, -5, 0});     // x = 5y.
    fac2.appendLocalId();             // Add local id z.

    fac1.mergeLocalIds(fac2);

    // Local space should be same.
    EXPECT_EQ(fac1.getNumLocalIds(), fac2.getNumLocalIds());

    // 1 division should be matched + 2 unmatched local ids.
    EXPECT_EQ(fac1.getNumLocalIds(), 3u);
    EXPECT_EQ(fac2.getNumLocalIds(), 3u);
  }

  {
    // (x) : (exists z = [x / 5], y = [x / 2] : x = 3y).
    FlatAffineConstraints fac1(1);
    fac1.addLocalFloorDiv({1, 0}, 5);    // z = [x / 5].
    fac1.addLocalFloorDiv({1, 0, 0}, 2); // y = [x / 2].
    fac1.addEquality({1, 0, -3, 0});     // x = 3y.

    // (x) : (exists y = [x / 2], z = [x / 5]: x = 5z).
    FlatAffineConstraints fac2(1);
    fac2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    fac2.addLocalFloorDiv({1, 0, 0}, 5); // z = [x / 5].
    fac2.addEquality({1, 0, -5, 0});     // x = 5z.

    fac1.mergeLocalIds(fac2);

    // Local space should be same.
    EXPECT_EQ(fac1.getNumLocalIds(), fac2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(fac1.getNumLocalIds(), 2u);
    EXPECT_EQ(fac2.getNumLocalIds(), 2u);
  }
}

TEST(FlatAffineConstraintsTest, mergeDivisionsNestedDivsions) {
  {
    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z >= x).
    FlatAffineConstraints fac1(1);
    fac1.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    fac1.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    fac1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3]: y + z <= x).
    FlatAffineConstraints fac2(1);
    fac2.addLocalFloorDiv({1, 0}, 2);    // y = [x / 2].
    fac2.addLocalFloorDiv({1, 1, 0}, 3); // z = [x + y / 3].
    fac2.addInequality({1, -1, -1, 0});  // y + z <= x.

    fac1.mergeLocalIds(fac2);

    // Local space should be same.
    EXPECT_EQ(fac1.getNumLocalIds(), fac2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(fac1.getNumLocalIds(), 2u);
    EXPECT_EQ(fac2.getNumLocalIds(), 2u);
  }

  {
    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z >= x).
    FlatAffineConstraints fac1(1);
    fac1.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    fac1.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    fac1.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    fac1.addInequality({-1, 1, 1, 0, 0});   // y + z >= x.

    // (x) : (exists y = [x / 2], z = [x + y / 3], w = [z + 1 / 5]: y + z <= x).
    FlatAffineConstraints fac2(1);
    fac2.addLocalFloorDiv({1, 0}, 2);       // y = [x / 2].
    fac2.addLocalFloorDiv({1, 1, 0}, 3);    // z = [x + y / 3].
    fac2.addLocalFloorDiv({0, 0, 1, 1}, 5); // w = [z + 1 / 5].
    fac2.addInequality({1, -1, -1, 0, 0});  // y + z <= x.

    fac1.mergeLocalIds(fac2);

    // Local space should be same.
    EXPECT_EQ(fac1.getNumLocalIds(), fac2.getNumLocalIds());

    // 3 divisions should be matched.
    EXPECT_EQ(fac1.getNumLocalIds(), 3u);
    EXPECT_EQ(fac2.getNumLocalIds(), 3u);
  }
}

TEST(FlatAffineConstraintsTest, mergeDivisionsConstants) {
  {
    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z >= x).
    FlatAffineConstraints fac1(1);
    fac1.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    fac1.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    fac1.addInequality({-1, 1, 1, 0});   // y + z >= x.

    // (x) : (exists y = [x + 1 / 3], z = [x + 2 / 3]: y + z <= x).
    FlatAffineConstraints fac2(1);
    fac2.addLocalFloorDiv({1, 1}, 2);    // y = [x + 1 / 2].
    fac2.addLocalFloorDiv({1, 0, 2}, 3); // z = [x + 2 / 3].
    fac2.addInequality({1, -1, -1, 0});  // y + z <= x.

    fac1.mergeLocalIds(fac2);

    // Local space should be same.
    EXPECT_EQ(fac1.getNumLocalIds(), fac2.getNumLocalIds());

    // 2 divisions should be matched.
    EXPECT_EQ(fac1.getNumLocalIds(), 2u);
    EXPECT_EQ(fac2.getNumLocalIds(), 2u);
  }
}

} // namespace mlir
