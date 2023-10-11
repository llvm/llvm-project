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

static void checkHermiteNormalForm(const IntMatrix &mat,
                                   const IntMatrix &hermiteForm) {
  auto [h, u] = mat.computeHermiteNormalForm();

  for (unsigned row = 0; row < mat.getNumRows(); row++)
    for (unsigned col = 0; col < mat.getNumColumns(); col++)
      EXPECT_EQ(h(row, col), hermiteForm(row, col));
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
  FracMatrix mat = makeFracMatrix(
      2, 2,
      {{Fraction(2, 1), Fraction(1, 1)}, {Fraction(7, 1), Fraction(0, 1)}});
  FracMatrix inverse = makeFracMatrix(
      2, 2,
      {{Fraction(0, 1), Fraction(1, 7)}, {Fraction(1, 1), Fraction(-2, 7)}});

  FracMatrix inv = *(mat.inverse());

  for (unsigned row = 0; row < 2; row++)
    for (unsigned col = 0; col < 2; col++)
      EXPECT_EQ(inv(row, col), inverse(row, col));
  
  mat = makeFracMatrix(
      2, 2,
      {{Fraction(0, 1), Fraction(1, 1)}, {Fraction(0, 1), Fraction(2, 1)}});
  
  EXPECT_FALSE(mat.inverse());
}

TEST(MatrixTest, intInverse) {
  IntMatrix mat = makeIntMatrix(2, 2, {{2, 1}, {7, 0}});
  IntMatrix inverse = makeIntMatrix(2, 2, {{0, -1}, {-7, 2}});

  IntMatrix inv = *(mat.integerInverse());

  for (unsigned i = 0; i < 2u; i++)
    for (unsigned j = 0; j < 2u; j++)
      EXPECT_EQ(inv(i, j), inverse(i, j));

  mat = makeIntMatrix(
      4, 4, {{4, 14, 11, 3}, {13, 5, 14, 12}, {13, 9, 7, 14}, {2, 3, 12, 7}});
  inverse = makeIntMatrix(4, 4,
                          {{155, 1636, -579, -1713},
                           {725, -743, 537, -111},
                           {210, 735, -855, 360},
                           {-715, -1409, 1401, 1482}});

  inv = *(mat.integerInverse());

  for (unsigned i = 0; i < 2u; i++)
    for (unsigned j = 0; j < 2u; j++)
      EXPECT_EQ(inv(i, j), inverse(i, j));

  mat = makeIntMatrix(
      2, 2, {{0, 0}, {1, 2}});
  
  EXPECT_FALSE(mat.integerInverse());
}
