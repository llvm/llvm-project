//===- MatrixTest.cpp - Tests for Matrix ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(MatrixTest, ReadWrite) {
  Matrix<MPInt> mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));
}

TEST(MatrixTest, SwapColumns) {
  Matrix<MPInt> mat(5, 5);
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
  Matrix<MPInt> mat(5, 5);
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
  Matrix<MPInt> mat(5, 5);
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
  Matrix<MPInt> mat(5, 5, 5, 10);
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
  Matrix<MPInt> mat(5, 5, 5, 10);
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
  Matrix<MPInt> mat(5, 5);
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

static void checkHermiteNormalForm(const Matrix<MPInt> &mat,
                                   const Matrix<MPInt> &hermiteForm) {
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
    Matrix<MPInt> mat = makeMatrix<MPInt>(3, 3, {{MPInt(2), MPInt(3), MPInt(6)}, {MPInt(3), MPInt(2), MPInt(3)}, {MPInt(17), MPInt(11), MPInt(16)}});
    Matrix<MPInt> hermiteForm = makeMatrix<MPInt>(3, 3, {{MPInt(1), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(1), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(1)}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    // Hermite form of a unimodular is the identity matrix.
    Matrix<MPInt> mat = makeMatrix<MPInt>(
        4, 4,
        {{-MPInt(6), -MPInt(1), -MPInt(19), -MPInt(20)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {-MPInt(5), MPInt(0), -MPInt(15), -MPInt(16)}, {MPInt(6), MPInt(0), MPInt(18), MPInt(19)}});
    Matrix<MPInt> hermiteForm = makeMatrix<MPInt>(
        4, 4, {{MPInt(1), MPInt(0), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(1), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(0), MPInt(1)}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat = makeMatrix<MPInt>(
        4, 4, {{MPInt(3), MPInt(3), MPInt(1), MPInt(4)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(19), MPInt(16)}, {MPInt(0), MPInt(0), MPInt(0), MPInt(3)}});
    Matrix<MPInt> hermiteForm = makeMatrix<MPInt>(
        4, 4, {{MPInt(1), MPInt(0), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {MPInt(1), MPInt(0), MPInt(3), MPInt(0)}, {MPInt(18), MPInt(0), MPInt(54), MPInt(57)}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat = makeMatrix<MPInt>(
        4, 4, {{MPInt(3), MPInt(3), MPInt(1), MPInt(4)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(19), MPInt(16)}, {MPInt(0), MPInt(0), MPInt(0), MPInt(3)}});
    Matrix<MPInt> hermiteForm = makeMatrix<MPInt>(
        4, 4, {{MPInt(1), MPInt(0), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0)}, {MPInt(1), MPInt(0), MPInt(3), MPInt(0)}, {MPInt(18), MPInt(0), MPInt(54), MPInt(57)}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat =
        makeMatrix<MPInt>(3, 5, {{MPInt(0), MPInt(2), MPInt(0), MPInt(7), MPInt(1)}, {-MPInt(1), MPInt(0), MPInt(0), -MPInt(3), MPInt(0)}, {MPInt(0), MPInt(4), MPInt(1), MPInt(0), MPInt(8)}});
    Matrix<MPInt> hermiteForm =
        makeMatrix<MPInt>(3, 5, {{MPInt(1), MPInt(0), MPInt(0), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(1), MPInt(0), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(1), MPInt(0), MPInt(0)}});
    checkHermiteNormalForm(mat, hermiteForm);
  }
}

TEST (MatrixTest, inverse) {
    Matrix<Fraction> mat = makeMatrix<Fraction>(2, 2, {{Fraction(2, 1), Fraction(1, 1)}, {Fraction(7, 1), Fraction(0, 1)}});
    Matrix<Fraction> inverse = makeMatrix<Fraction>(2, 2, {{Fraction(0, 1), Fraction(1, 7)}, {Fraction(1, 1), Fraction(-2, 7)}});

    Matrix<Fraction> inv = mat.inverse();

    for (unsigned row = 0; row < 2; row++)
      for (unsigned col = 0; col < 2; col++)
        EXPECT_EQ(inv(row, col), inverse(row, col));
}

TEST(MatrixTest, gramSchmidt) {
    Matrix<Fraction> mat = makeMatrix<Fraction>(3, 5, {{Fraction(3, 1), Fraction(4, 1), Fraction(5, 1), Fraction(12, 1), Fraction(19, 1)},
                                     {Fraction(4, 1), Fraction(5, 1), Fraction(6, 1), Fraction(13, 1), Fraction(20, 1)},
                                     {Fraction(7, 1), Fraction(8, 1), Fraction(9, 1), Fraction(16, 1), Fraction(24, 1)}});

    Matrix<Fraction> gramSchmidt = makeMatrix<Fraction>(3, 5,
           {{Fraction(3, 1),     Fraction(4, 1),     Fraction(5, 1),    Fraction(12, 1),     Fraction(19, 1)},
            {Fraction(142, 185), Fraction(383, 555), Fraction(68, 111), Fraction(13, 185),   Fraction(-262, 555)},
            {Fraction(53, 463),  Fraction(27, 463),  Fraction(1, 463),  Fraction(-181, 463), Fraction(100, 463)}});

    Matrix<Fraction> gs = mat.gramSchmidt();

    for (unsigned row = 0; row < 3; row++)
      for (unsigned col = 0; col < 5; col++)
        EXPECT_EQ(gs(row, col), gramSchmidt(row, col));
}

TEST(MatrixTest, LLL) {
    Matrix<Fraction> mat = makeMatrix<Fraction>(3, 3, {{Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)},
                                     {Fraction(-1, 1), Fraction(0, 1), Fraction(2, 1)},
                                     {Fraction(3, 1), Fraction(5, 1), Fraction(6, 1)}});
    mat.LLL(Fraction(3, 4));
    
    Matrix<Fraction> LLL = makeMatrix<Fraction>(3, 3, {{Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)},
                                     {Fraction(1, 1), Fraction(0, 1), Fraction(1, 1)},
                                     {Fraction(-1, 1), Fraction(0, 1), Fraction(2, 1)}});

    for (unsigned row = 0; row < 3; row++)
      for (unsigned col = 0; col < 3; col++)
        EXPECT_EQ(mat(row, col), LLL(row, col));


    mat = makeMatrix<Fraction>(2, 2, {{Fraction(12, 1), Fraction(2, 1)}, {Fraction(13, 1), Fraction(4, 1)}});
    LLL = makeMatrix<Fraction>(2, 2, {{Fraction(1, 1),  Fraction(2, 1)}, {Fraction(9, 1),  Fraction(-4, 1)}});

    mat.LLL(Fraction(3, 4));

    for (unsigned row = 0; row < 2; row++)
      for (unsigned col = 0; col < 2; col++)
        EXPECT_EQ(mat(row, col), LLL(row, col));

}
