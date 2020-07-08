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
  EXPECT_EQ(mat.getNumRows(), 3u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));

  mat.resizeVertically(5);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row >= 3 ? 0 : int(10 * row + col));
}

} // namespace mlir
