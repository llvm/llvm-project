//===- LinearTransformTest.cpp - Tests for LinearTransform ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/LinearTransform.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

void testColumnEchelonForm(const Matrix &m, unsigned expectedRank) {
  unsigned lastAllowedNonZeroCol = 0;
  std::pair<unsigned, LinearTransform> result =
      LinearTransform::makeTransformToColumnEchelon(m);
  unsigned rank = result.first;
  EXPECT_EQ(rank, expectedRank);
  LinearTransform transform = result.second;
  // In column echelon form, each row's last non-zero value can be at most one
  // column to the right of the last non-zero column among the previous rows.
  for (unsigned row = 0, nRows = m.getNumRows(); row < nRows; ++row) {
    SmallVector<MPInt, 8> rowVec = transform.preMultiplyWithRow(m.getRow(row));
    for (unsigned col = lastAllowedNonZeroCol + 1, nCols = m.getNumColumns();
         col < nCols; ++col) {
      EXPECT_EQ(rowVec[col], 0);
      if (rowVec[col] != 0) {
        llvm::errs() << "Failed at input matrix:\n";
        m.dump();
      }
    }
    if (rowVec[lastAllowedNonZeroCol] != 0)
      lastAllowedNonZeroCol++;
  }
  // The final value of lastAllowedNonZeroCol is the index of the first
  // all-zeros column, so it must be equal to the rank.
  EXPECT_EQ(lastAllowedNonZeroCol, rank);
}

TEST(LinearTransformTest, transformToColumnEchelonTest) {
  // m1, m2, m3 are rank 1 matrices -- the first and second rows are identical.
  Matrix m1(2, 2);
  m1(0, 0) = 4;
  m1(0, 1) = -7;
  m1(1, 0) = 4;
  m1(1, 1) = -7;
  testColumnEchelonForm(m1, 1u);

  Matrix m2(2, 2);
  m2(0, 0) = -4;
  m2(0, 1) = 7;
  m2(1, 0) = 4;
  m2(1, 1) = -7;
  testColumnEchelonForm(m2, 1u);

  Matrix m3(2, 2);
  m3(0, 0) = -4;
  m3(0, 1) = -7;
  m3(1, 0) = -4;
  m3(1, 1) = -7;
  testColumnEchelonForm(m3, 1u);

  // m4, m5, m6 are rank 2 matrices -- the first and second rows are different.
  Matrix m4(2, 2);
  m4(0, 0) = 4;
  m4(0, 1) = -7;
  m4(1, 0) = -4;
  m4(1, 1) = -7;
  testColumnEchelonForm(m4, 2u);

  Matrix m5(2, 2);
  m5(0, 0) = -4;
  m5(0, 1) = 7;
  m5(1, 0) = 4;
  m5(1, 1) = 7;
  testColumnEchelonForm(m5, 2u);

  Matrix m6(2, 2);
  m6(0, 0) = -4;
  m6(0, 1) = -7;
  m6(1, 0) = 4;
  m6(1, 1) = -7;
  testColumnEchelonForm(m5, 2u);
}
