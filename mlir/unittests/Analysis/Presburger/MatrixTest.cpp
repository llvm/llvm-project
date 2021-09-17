//===- MatrixTest.cpp - Tests for Matrix ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(MatrixTest, ReadWrite) {
  Matrix mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));
}

TEST(MatrixTest, SwapColumns) {
  Matrix mat(5, 5);
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
  Matrix mat(5, 5);
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
  Matrix mat(5, 5);
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
  Matrix mat(5, 5, 5, 10);
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
  Matrix mat(5, 5, 5, 10);
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
  Matrix mat(5, 5);
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

} // namespace mlir
