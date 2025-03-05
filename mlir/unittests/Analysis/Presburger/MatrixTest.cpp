//===- MatrixTest.cpp - Tests for Matrix ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "./Utils.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(MatrixTest, ReadWrite) {
  IntMatrix mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));
}

TEST(MatrixTest, SwapColumns) {
  IntMatrix mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = col == 3 ? 1 : 0;
  mat.swapColumns(3, 1);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);

  // swap around all the other columns, swap (1, 3) twice for no effect.
  mat.swapColumns(3, 1);
  mat.swapColumns(2, 4);
  mat.swapColumns(1, 3);
  mat.swapColumns(0, 4);
  mat.swapColumns(2, 2);

  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);
}

TEST(MatrixTest, SwapRows) {
  IntMatrix mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = row == 2 ? 1 : 0;
  mat.swapRows(2, 0);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);

  // swap around all the other rows, swap (2, 0) twice for no effect.
  mat.swapRows(3, 4);
  mat.swapRows(1, 4);
  mat.swapRows(2, 0);
  mat.swapRows(1, 1);
  mat.swapRows(0, 2);

  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);
}

TEST(MatrixTest, resizeVertically) {
  IntMatrix mat(5, 5);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.resizeVertically(3);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 3u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));

  mat.resizeVertically(5);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row >= 3 ? 0 : int(10 * row + col));
}

TEST(MatrixTest, insertColumns) {
  IntMatrix mat(5, 5, 5, 10);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.insertColumns(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 105u);
  for (unsigned row = 0; row < 5; ++row) {
    for (unsigned col = 0; col < 105; ++col) {
      if (col < 3)
        EXPECT_EQ(mat(row, col), int(10 * row + col));
      else if (3 <= col && col <= 102)
        EXPECT_EQ(mat(row, col), 0);
      else
        EXPECT_EQ(mat(row, col), int(10 * row + col - 100));
    }
  }

  mat.removeColumns(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertColumns(0, 0);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertColumn(5);
  ASSERT_TRUE(mat.hasConsistentState());

  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 6u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 6; ++col)
      EXPECT_EQ(mat(row, col), col == 5 ? 0 : 10 * row + col);
}

TEST(MatrixTest, insertRows) {
  IntMatrix mat(5, 5, 5, 10);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.insertRows(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 105u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 105; ++row) {
    for (unsigned col = 0; col < 5; ++col) {
      if (row < 3)
        EXPECT_EQ(mat(row, col), int(10 * row + col));
      else if (3 <= row && row <= 102)
        EXPECT_EQ(mat(row, col), 0);
      else
        EXPECT_EQ(mat(row, col), int(10 * (row - 100) + col));
    }
  }

  mat.removeRows(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertRows(0, 0);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertRow(5);
  ASSERT_TRUE(mat.hasConsistentState());

  EXPECT_EQ(mat.getNumRows(), 6u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 6; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 5 ? 0 : 10 * row + col);
}

TEST(MatrixTest, resize) {
  IntMatrix mat(5, 5);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.resize(3, 3);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 3u);
  EXPECT_EQ(mat.getNumColumns(), 3u);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned col = 0; col < 3; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));

  mat.resize(7, 7);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 7u);
  EXPECT_EQ(mat.getNumColumns(), 7u);
  for (unsigned row = 0; row < 7; ++row)
    for (unsigned col = 0; col < 7; ++col)
      EXPECT_EQ(mat(row, col), row >= 3 || col >= 3 ? 0 : int(10 * row + col));
}

template <typename T>
static void checkMatEqual(const Matrix<T> m1, const Matrix<T> m2) {
  EXPECT_EQ(m1.getNumRows(), m2.getNumRows());
  EXPECT_EQ(m1.getNumColumns(), m2.getNumColumns());

  for (unsigned row = 0, rows = m1.getNumRows(); row < rows; ++row)
    for (unsigned col = 0, cols = m1.getNumColumns(); col < cols; ++col)
      EXPECT_EQ(m1(row, col), m2(row, col));
}

static void checkHermiteNormalForm(const IntMatrix &mat,
                                   const IntMatrix &hermiteForm) {
  auto [h, u] = mat.computeHermiteNormalForm();

  checkMatEqual(h, hermiteForm);
}

TEST(MatrixTest, computeHermiteNormalForm) {
  // TODO: Add a check to test the original statement of hermite normal form
  // instead of using a precomputed result.

  {
    // Hermite form of a unimodular matrix is the identity matrix.
    IntMatrix mat = makeIntMatrix(3, 3, {{2, 3, 6}, {3, 2, 3}, {17, 11, 16}});
    IntMatrix hermiteForm =
        makeIntMatrix(3, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    // Hermite form of a unimodular is the identity matrix.
    IntMatrix mat = makeIntMatrix(
        4, 4,
        {{-6, -1, -19, -20}, {0, 1, 0, 0}, {-5, 0, -15, -16}, {6, 0, 18, 19}});
    IntMatrix hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    IntMatrix mat = makeIntMatrix(
        4, 4, {{3, 3, 1, 4}, {0, 1, 0, 0}, {0, 0, 19, 16}, {0, 0, 0, 3}});
    IntMatrix hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 3, 0}, {18, 0, 54, 57}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    IntMatrix mat = makeIntMatrix(
        4, 4, {{3, 3, 1, 4}, {0, 1, 0, 0}, {0, 0, 19, 16}, {0, 0, 0, 3}});
    IntMatrix hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 3, 0}, {18, 0, 54, 57}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    IntMatrix mat = makeIntMatrix(
        3, 5, {{0, 2, 0, 7, 1}, {-1, 0, 0, -3, 0}, {0, 4, 1, 0, 8}});
    IntMatrix hermiteForm = makeIntMatrix(
        3, 5, {{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}});
    checkHermiteNormalForm(mat, hermiteForm);
  }
}

TEST(MatrixTest, inverse) {
  IntMatrix mat1 = makeIntMatrix(2, 2, {{2, 1}, {7, 0}});
  EXPECT_EQ(mat1.determinant(), -7);

  FracMatrix mat = makeFracMatrix(
      2, 2, {{Fraction(2), Fraction(1)}, {Fraction(7), Fraction(0)}});
  FracMatrix inverse = makeFracMatrix(
      2, 2, {{Fraction(0), Fraction(1, 7)}, {Fraction(1), Fraction(-2, 7)}});

  FracMatrix inv(2, 2);
  mat.determinant(&inv);

  EXPECT_EQ_FRAC_MATRIX(inv, inverse);

  mat = makeFracMatrix(
      2, 2, {{Fraction(0), Fraction(1)}, {Fraction(0), Fraction(2)}});
  Fraction det = mat.determinant(nullptr);

  EXPECT_EQ(det, Fraction(0));

  mat = makeFracMatrix(3, 3,
                       {{Fraction(1), Fraction(2), Fraction(3)},
                        {Fraction(4), Fraction(8), Fraction(6)},
                        {Fraction(7), Fraction(8), Fraction(6)}});
  inverse = makeFracMatrix(3, 3,
                           {{Fraction(0), Fraction(-1, 3), Fraction(1, 3)},
                            {Fraction(-1, 2), Fraction(5, 12), Fraction(-1, 6)},
                            {Fraction(2, 3), Fraction(-1, 6), Fraction(0)}});

  mat.determinant(&inv);
  EXPECT_EQ_FRAC_MATRIX(inv, inverse);

  mat = makeFracMatrix(0, 0, {});
  mat.determinant(&inv);
}

TEST(MatrixTest, intInverse) {
  IntMatrix mat = makeIntMatrix(2, 2, {{2, 1}, {7, 0}});
  IntMatrix inverse = makeIntMatrix(2, 2, {{0, -1}, {-7, 2}});

  IntMatrix inv(2, 2);
  mat.determinant(&inv);

  EXPECT_EQ_INT_MATRIX(inv, inverse);

  mat = makeIntMatrix(
      4, 4, {{4, 14, 11, 3}, {13, 5, 14, 12}, {13, 9, 7, 14}, {2, 3, 12, 7}});
  inverse = makeIntMatrix(4, 4,
                          {{155, 1636, -579, -1713},
                           {725, -743, 537, -111},
                           {210, 735, -855, 360},
                           {-715, -1409, 1401, 1482}});

  mat.determinant(&inv);

  EXPECT_EQ_INT_MATRIX(inv, inverse);

  mat = makeIntMatrix(2, 2, {{0, 0}, {1, 2}});

  DynamicAPInt det = mat.determinant(&inv);

  EXPECT_EQ(det, 0);
}

TEST(MatrixTest, gramSchmidt) {
  FracMatrix mat =
      makeFracMatrix(3, 5,
                     {{Fraction(3, 1), Fraction(4, 1), Fraction(5, 1),
                       Fraction(12, 1), Fraction(19, 1)},
                      {Fraction(4, 1), Fraction(5, 1), Fraction(6, 1),
                       Fraction(13, 1), Fraction(20, 1)},
                      {Fraction(7, 1), Fraction(8, 1), Fraction(9, 1),
                       Fraction(16, 1), Fraction(24, 1)}});

  FracMatrix gramSchmidt = makeFracMatrix(
      3, 5,
      {{Fraction(3, 1), Fraction(4, 1), Fraction(5, 1), Fraction(12, 1),
        Fraction(19, 1)},
       {Fraction(142, 185), Fraction(383, 555), Fraction(68, 111),
        Fraction(13, 185), Fraction(-262, 555)},
       {Fraction(53, 463), Fraction(27, 463), Fraction(1, 463),
        Fraction(-181, 463), Fraction(100, 463)}});

  FracMatrix gs = mat.gramSchmidt();

  EXPECT_EQ_FRAC_MATRIX(gs, gramSchmidt);
  for (unsigned i = 0; i < 3u; i++)
    for (unsigned j = i + 1; j < 3u; j++)
      EXPECT_EQ(dotProduct(gs.getRow(i), gs.getRow(j)), 0);

  mat = makeFracMatrix(3, 3,
                       {{Fraction(20, 1), Fraction(17, 1), Fraction(10, 1)},
                        {Fraction(20, 1), Fraction(18, 1), Fraction(6, 1)},
                        {Fraction(15, 1), Fraction(14, 1), Fraction(10, 1)}});

  gramSchmidt = makeFracMatrix(
      3, 3,
      {{Fraction(20, 1), Fraction(17, 1), Fraction(10, 1)},
       {Fraction(460, 789), Fraction(1180, 789), Fraction(-2926, 789)},
       {Fraction(-2925, 3221), Fraction(3000, 3221), Fraction(750, 3221)}});

  gs = mat.gramSchmidt();

  EXPECT_EQ_FRAC_MATRIX(gs, gramSchmidt);
  for (unsigned i = 0; i < 3u; i++)
    for (unsigned j = i + 1; j < 3u; j++)
      EXPECT_EQ(dotProduct(gs.getRow(i), gs.getRow(j)), 0);

  mat = makeFracMatrix(
      4, 4,
      {{Fraction(1, 26), Fraction(13, 12), Fraction(34, 13), Fraction(7, 10)},
       {Fraction(40, 23), Fraction(34, 1), Fraction(11, 19), Fraction(15, 1)},
       {Fraction(21, 22), Fraction(10, 9), Fraction(4, 11), Fraction(14, 11)},
       {Fraction(35, 22), Fraction(1, 15), Fraction(5, 8), Fraction(30, 1)}});

  gs = mat.gramSchmidt();

  // The integers involved are too big to construct the actual matrix.
  // but we can check that the result is linearly independent.
  ASSERT_FALSE(mat.determinant(nullptr) == 0);

  for (unsigned i = 0; i < 4u; i++)
    for (unsigned j = i + 1; j < 4u; j++)
      EXPECT_EQ(dotProduct(gs.getRow(i), gs.getRow(j)), 0);

  mat = FracMatrix::identity(/*dim=*/10);

  gs = mat.gramSchmidt();

  EXPECT_EQ_FRAC_MATRIX(gs, FracMatrix::identity(10));
}

void checkReducedBasis(FracMatrix mat, Fraction delta) {
  FracMatrix gsOrth = mat.gramSchmidt();

  // Size-reduced check.
  for (unsigned i = 0, e = mat.getNumRows(); i < e; i++) {
    for (unsigned j = 0; j < i; j++) {
      Fraction mu = dotProduct(mat.getRow(i), gsOrth.getRow(j)) /
                    dotProduct(gsOrth.getRow(j), gsOrth.getRow(j));
      EXPECT_TRUE(abs(mu) <= Fraction(1, 2));
    }
  }

  // Lovasz condition check.
  for (unsigned i = 1, e = mat.getNumRows(); i < e; i++) {
    Fraction mu = dotProduct(mat.getRow(i), gsOrth.getRow(i - 1)) /
                  dotProduct(gsOrth.getRow(i - 1), gsOrth.getRow(i - 1));
    EXPECT_TRUE(dotProduct(mat.getRow(i), mat.getRow(i)) >
                (delta - mu * mu) *
                    dotProduct(gsOrth.getRow(i - 1), gsOrth.getRow(i - 1)));
  }
}

TEST(MatrixTest, LLL) {
  FracMatrix mat =
      makeFracMatrix(3, 3,
                     {{Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)},
                      {Fraction(-1, 1), Fraction(0, 1), Fraction(2, 1)},
                      {Fraction(3, 1), Fraction(5, 1), Fraction(6, 1)}});
  mat.LLL(Fraction(3, 4));

  checkReducedBasis(mat, Fraction(3, 4));

  mat = makeFracMatrix(
      2, 2,
      {{Fraction(12, 1), Fraction(2, 1)}, {Fraction(13, 1), Fraction(4, 1)}});
  mat.LLL(Fraction(3, 4));

  checkReducedBasis(mat, Fraction(3, 4));

  mat = makeFracMatrix(3, 3,
                       {{Fraction(1, 1), Fraction(0, 1), Fraction(2, 1)},
                        {Fraction(0, 1), Fraction(1, 3), -Fraction(5, 3)},
                        {Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)}});
  mat.LLL(Fraction(3, 4));

  checkReducedBasis(mat, Fraction(3, 4));
}

TEST(MatrixTest, moveColumns) {
  IntMatrix mat =
      makeIntMatrix(3, 4, {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 4, 2}});

  {
    IntMatrix movedMat =
        makeIntMatrix(3, 4, {{0, 3, 1, 2}, {4, 7, 5, 6}, {8, 2, 9, 4}});

    movedMat.moveColumns(2, 2, 1);
    checkMatEqual(mat, movedMat);
  }

  {
    IntMatrix movedMat =
        makeIntMatrix(3, 4, {{0, 3, 1, 2}, {4, 7, 5, 6}, {8, 2, 9, 4}});

    movedMat.moveColumns(1, 1, 3);
    checkMatEqual(mat, movedMat);
  }

  {
    IntMatrix movedMat =
        makeIntMatrix(3, 4, {{1, 2, 0, 3}, {5, 6, 4, 7}, {9, 4, 8, 2}});

    movedMat.moveColumns(0, 2, 1);
    checkMatEqual(mat, movedMat);
  }

  {
    IntMatrix movedMat =
        makeIntMatrix(3, 4, {{1, 0, 2, 3}, {5, 4, 6, 7}, {9, 8, 4, 2}});

    movedMat.moveColumns(0, 1, 1);
    checkMatEqual(mat, movedMat);
  }
}
