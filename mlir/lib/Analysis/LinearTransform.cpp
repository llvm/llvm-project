//===- LinearTransform.cpp - MLIR LinearTransform Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LinearTransform.h"
#include "mlir/Analysis/AffineStructures.h"

namespace mlir {

LinearTransform::LinearTransform(Matrix &&oMatrix) : matrix(oMatrix) {}
LinearTransform::LinearTransform(const Matrix &oMatrix) : matrix(oMatrix) {}

// Set M(row, targetCol) to its remainder on division by M(row, sourceCol)
// by subtracting from column targetCol an appropriate integer multiple of
// sourceCol. This brings M(row, targetCol) to the range [0, M(row, sourceCol)).
// Apply the same column operation to otherMatrix, with the same integer
// multiple.
static void modEntryColumnOperation(Matrix &m, unsigned row, unsigned sourceCol,
                                    unsigned targetCol, Matrix &otherMatrix) {
  assert(m(row, sourceCol) != 0 && "Cannot divide by zero!");
  assert((m(row, sourceCol) > 0 && m(row, targetCol) > 0) &&
         "Operands must be positive!");
  int64_t ratio = m(row, targetCol) / m(row, sourceCol);
  m.addToColumn(sourceCol, targetCol, -ratio);
  otherMatrix.addToColumn(sourceCol, targetCol, -ratio);
}

std::pair<unsigned, LinearTransform>
LinearTransform::makeTransformToColumnEchelon(Matrix m) {
  // We start with an identity result matrix and perform operations on m
  // until m is in column echelon form. We apply the same sequence of operations
  // on resultMatrix to obtain a transform that takes m to column echelon
  // form.
  Matrix resultMatrix = Matrix::identity(m.getNumColumns());

  unsigned echelonCol = 0;
  // Invariant: in all rows above row, all columns from echelonCol onwards
  // are all zero elements. In an iteration, if the curent row has any non-zero
  // elements echelonCol onwards, we bring one to echelonCol and use it to
  // make all elements echelonCol + 1 onwards zero.
  for (unsigned row = 0; row < m.getNumRows(); ++row) {
    // Search row for a non-empty entry, starting at echelonCol.
    unsigned nonZeroCol = echelonCol;
    for (unsigned e = m.getNumColumns(); nonZeroCol < e; ++nonZeroCol) {
      if (m(row, nonZeroCol) == 0)
        continue;
      break;
    }

    // Continue to the next row with the same echelonCol if this row is all
    // zeros from echelonCol onwards.
    if (nonZeroCol == m.getNumColumns())
      continue;

    // Bring the non-zero column to echelonCol. This doesn't affect rows
    // above since they are all zero at these columns.
    if (nonZeroCol != echelonCol) {
      m.swapColumns(nonZeroCol, echelonCol);
      resultMatrix.swapColumns(nonZeroCol, echelonCol);
    }

    // Make m(row, echelonCol) non-negative.
    if (m(row, echelonCol) < 0) {
      m.negateColumn(echelonCol);
      resultMatrix.negateColumn(echelonCol);
    }

    // Make all the entries in row after echelonCol zero.
    for (unsigned i = echelonCol + 1, e = m.getNumColumns(); i < e; ++i) {
      // We make m(row, i) non-negative, and then apply the Euclidean GCD
      // algorithm to (row, i) and (row, echelonCol). At the end, one of them
      // has value equal to the gcd of the two entries, and the other is zero.

      if (m(row, i) < 0) {
        m.negateColumn(i);
        resultMatrix.negateColumn(i);
      }

      unsigned targetCol = i, sourceCol = echelonCol;
      // At every step, we set m(row, targetCol) %= m(row, sourceCol), and
      // swap the indices sourceCol and targetCol. (not the columns themselves)
      // This modulo is implemented as a subtraction
      // m(row, targetCol) -= quotient * m(row, sourceCol),
      // where quotient = floor(m(row, targetCol) / m(row, sourceCol)),
      // which brings m(row, targetCol) to the range [0, m(row, sourceCol)).
      //
      // We are only allowed column operations; we perform the above
      // for every row, i.e., the above subtraction is done as a column
      // operation. This does not affect any rows above us since they are
      // guaranteed to be zero at these columns.
      while (m(row, targetCol) != 0 && m(row, sourceCol) != 0) {
        modEntryColumnOperation(m, row, sourceCol, targetCol, resultMatrix);
        std::swap(targetCol, sourceCol);
      }

      // One of (row, echelonCol) and (row, i) is zero and the other is the gcd.
      // Make it so that (row, echelonCol) holds the non-zero value.
      if (m(row, echelonCol) == 0) {
        m.swapColumns(i, echelonCol);
        resultMatrix.swapColumns(i, echelonCol);
      }
    }

    ++echelonCol;
  }

  return {echelonCol, LinearTransform(std::move(resultMatrix))};
}

SmallVector<int64_t, 8>
LinearTransform::postMultiplyRow(ArrayRef<int64_t> rowVec) const {
  assert(rowVec.size() == matrix.getNumRows() &&
         "row vector dimension should match transform output dimension");

  SmallVector<int64_t, 8> result(matrix.getNumColumns(), 0);
  for (unsigned col = 0, e = matrix.getNumColumns(); col < e; ++col)
    for (unsigned i = 0, e = matrix.getNumRows(); i < e; ++i)
      result[col] += rowVec[i] * matrix(i, col);
  return result;
}

SmallVector<int64_t, 8>
LinearTransform::preMultiplyColumn(ArrayRef<int64_t> colVec) const {
  assert(matrix.getNumColumns() == colVec.size() &&
         "column vector dimension should match transform input dimension");

  SmallVector<int64_t, 8> result(matrix.getNumRows(), 0);
  for (unsigned row = 0, e = matrix.getNumRows(); row < e; row++)
    for (unsigned i = 0, e = matrix.getNumColumns(); i < e; i++)
      result[row] += matrix(row, i) * colVec[i];
  return result;
}

FlatAffineConstraints
LinearTransform::applyTo(const FlatAffineConstraints &fac) const {
  FlatAffineConstraints result(fac.getNumDimIds());

  for (unsigned i = 0, e = fac.getNumEqualities(); i < e; ++i) {
    ArrayRef<int64_t> eq = fac.getEquality(i);

    int64_t c = eq.back();

    SmallVector<int64_t, 8> newEq = postMultiplyRow(eq.drop_back());
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0, e = fac.getNumInequalities(); i < e; ++i) {
    ArrayRef<int64_t> ineq = fac.getInequality(i);

    int64_t c = ineq.back();

    SmallVector<int64_t, 8> newIneq = postMultiplyRow(ineq.drop_back());
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  return result;
}

} // namespace mlir
