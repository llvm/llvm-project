//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;

Matrix::Matrix(unsigned rows, unsigned columns, unsigned reservedRows,
               unsigned reservedColumns)
    : nRows(rows), nColumns(columns),
      nReservedColumns(std::max(nColumns, reservedColumns)),
      data(nRows * nReservedColumns) {
  data.reserve(std::max(nRows, reservedRows) * nReservedColumns);
}

Matrix Matrix::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

unsigned Matrix::getNumReservedRows() const {
  return data.capacity() / nReservedColumns;
}

void Matrix::reserveRows(unsigned rows) {
  data.reserve(rows * nReservedColumns);
}

unsigned Matrix::appendExtraRow() {
  resizeVertically(nRows + 1);
  return nRows - 1;
}

unsigned Matrix::appendExtraRow(ArrayRef<MPInt> elems) {
  assert(elems.size() == nColumns && "elems must match row length!");
  unsigned row = appendExtraRow();
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
  return row;
}

void Matrix::resizeHorizontally(unsigned newNColumns) {
  if (newNColumns < nColumns)
    removeColumns(newNColumns, nColumns - newNColumns);
  if (newNColumns > nColumns)
    insertColumns(nColumns, newNColumns - nColumns);
}

void Matrix::resize(unsigned newNRows, unsigned newNColumns) {
  resizeHorizontally(newNColumns);
  resizeVertically(newNRows);
}

void Matrix::resizeVertically(unsigned newNRows) {
  nRows = newNRows;
  data.resize(nRows * nReservedColumns);
}

void Matrix::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

void Matrix::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

MutableArrayRef<MPInt> Matrix::getRow(unsigned row) {
  return {&data[row * nReservedColumns], nColumns};
}

ArrayRef<MPInt> Matrix::getRow(unsigned row) const {
  return {&data[row * nReservedColumns], nColumns};
}

void Matrix::setRow(unsigned row, ArrayRef<MPInt> elems) {
  assert(elems.size() == getNumColumns() &&
         "elems size must match row length!");
  for (unsigned i = 0, e = getNumColumns(); i < e; ++i)
    at(row, i) = elems[i];
}

void Matrix::insertColumn(unsigned pos) { insertColumns(pos, 1); }
void Matrix::insertColumns(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos <= nColumns);
  unsigned oldNReservedColumns = nReservedColumns;
  if (nColumns + count > nReservedColumns) {
    nReservedColumns = llvm::NextPowerOf2(nColumns + count);
    data.resize(nRows * nReservedColumns);
  }
  nColumns += count;

  for (int ri = nRows - 1; ri >= 0; --ri) {
    for (int ci = nReservedColumns - 1; ci >= 0; --ci) {
      unsigned r = ri;
      unsigned c = ci;
      MPInt &dest = data[r * nReservedColumns + c];
      if (c >= nColumns) { // NOLINT
        // Out of bounds columns are zero-initialized. NOLINT because clang-tidy
        // complains about this branch being the same as the c >= pos one.
        //
        // TODO: this case can be skipped if the number of reserved columns
        // didn't change.
        dest = 0;
      } else if (c >= pos + count) {
        // Shift the data occuring after the inserted columns.
        dest = data[r * oldNReservedColumns + c - count];
      } else if (c >= pos) {
        // The inserted columns are also zero-initialized.
        dest = 0;
      } else {
        // The columns before the inserted columns stay at the same (row, col)
        // but this corresponds to a different location in the linearized array
        // if the number of reserved columns changed.
        if (nReservedColumns == oldNReservedColumns)
          break;
        dest = data[r * oldNReservedColumns + c];
      }
    }
  }
}

void Matrix::removeColumn(unsigned pos) { removeColumns(pos, 1); }
void Matrix::removeColumns(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos + count - 1 < nColumns);
  for (unsigned r = 0; r < nRows; ++r) {
    for (unsigned c = pos; c < nColumns - count; ++c)
      at(r, c) = at(r, c + count);
    for (unsigned c = nColumns - count; c < nColumns; ++c)
      at(r, c) = 0;
  }
  nColumns -= count;
}

void Matrix::insertRow(unsigned pos) { insertRows(pos, 1); }
void Matrix::insertRows(unsigned pos, unsigned count) {
  if (count == 0)
    return;

  assert(pos <= nRows);
  resizeVertically(nRows + count);
  for (int r = nRows - 1; r >= int(pos + count); --r)
    copyRow(r - count, r);
  for (int r = pos + count - 1; r >= int(pos); --r)
    for (unsigned c = 0; c < nColumns; ++c)
      at(r, c) = 0;
}

void Matrix::removeRow(unsigned pos) { removeRows(pos, 1); }
void Matrix::removeRows(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos + count - 1 <= nRows);
  for (unsigned r = pos; r + count < nRows; ++r)
    copyRow(r + count, r);
  resizeVertically(nRows - count);
}

void Matrix::copyRow(unsigned sourceRow, unsigned targetRow) {
  if (sourceRow == targetRow)
    return;
  for (unsigned c = 0; c < nColumns; ++c)
    at(targetRow, c) = at(sourceRow, c);
}

void Matrix::fillRow(unsigned row, const MPInt &value) {
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = value;
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow,
                      const MPInt &scale) {
  addToRow(targetRow, getRow(sourceRow), scale);
}

void Matrix::addToRow(unsigned row, ArrayRef<MPInt> rowVec,
                      const MPInt &scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) += scale * rowVec[col];
}

void Matrix::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         const MPInt &scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
}

void Matrix::negateColumn(unsigned column) {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) = -at(row, column);
}

void Matrix::negateRow(unsigned row) {
  for (unsigned column = 0, e = getNumColumns(); column < e; ++column)
    at(row, column) = -at(row, column);
}

MPInt Matrix::normalizeRow(unsigned row, unsigned cols) {
  return normalizeRange(getRow(row).slice(0, cols));
}

MPInt Matrix::normalizeRow(unsigned row) {
  return normalizeRow(row, getNumColumns());
}

SmallVector<MPInt, 8> Matrix::preMultiplyWithRow(ArrayRef<MPInt> rowVec) const {
  assert(rowVec.size() == getNumRows() && "Invalid row vector dimension!");

  SmallVector<MPInt, 8> result(getNumColumns(), MPInt(0));
  for (unsigned col = 0, e = getNumColumns(); col < e; ++col)
    for (unsigned i = 0, e = getNumRows(); i < e; ++i)
      result[col] += rowVec[i] * at(i, col);
  return result;
}

SmallVector<MPInt, 8>
Matrix::postMultiplyWithColumn(ArrayRef<MPInt> colVec) const {
  assert(getNumColumns() == colVec.size() &&
         "Invalid column vector dimension!");

  SmallVector<MPInt, 8> result(getNumRows(), MPInt(0));
  for (unsigned row = 0, e = getNumRows(); row < e; row++)
    for (unsigned i = 0, e = getNumColumns(); i < e; i++)
      result[row] += at(row, i) * colVec[i];
  return result;
}

/// Set M(row, targetCol) to its remainder on division by M(row, sourceCol)
/// by subtracting from column targetCol an appropriate integer multiple of
/// sourceCol. This brings M(row, targetCol) to the range [0, M(row,
/// sourceCol)). Apply the same column operation to otherMatrix, with the same
/// integer multiple.
static void modEntryColumnOperation(Matrix &m, unsigned row, unsigned sourceCol,
                                    unsigned targetCol, Matrix &otherMatrix) {
  assert(m(row, sourceCol) != 0 && "Cannot divide by zero!");
  assert(m(row, sourceCol) > 0 && "Source must be positive!");
  MPInt ratio = -floorDiv(m(row, targetCol), m(row, sourceCol));
  m.addToColumn(sourceCol, targetCol, ratio);
  otherMatrix.addToColumn(sourceCol, targetCol, ratio);
}

std::pair<Matrix, Matrix> Matrix::computeHermiteNormalForm() const {
  // We start with u as an identity matrix and perform operations on h until h
  // is in hermite normal form. We apply the same sequence of operations on u to
  // obtain a transform that takes h to hermite normal form.
  Matrix h = *this;
  Matrix u = Matrix::identity(h.getNumColumns());

  unsigned echelonCol = 0;
  // Invariant: in all rows above row, all columns from echelonCol onwards
  // are all zero elements. In an iteration, if the curent row has any non-zero
  // elements echelonCol onwards, we bring one to echelonCol and use it to
  // make all elements echelonCol + 1 onwards zero.
  for (unsigned row = 0; row < h.getNumRows(); ++row) {
    // Search row for a non-empty entry, starting at echelonCol.
    unsigned nonZeroCol = echelonCol;
    for (unsigned e = h.getNumColumns(); nonZeroCol < e; ++nonZeroCol) {
      if (h(row, nonZeroCol) == 0)
        continue;
      break;
    }

    // Continue to the next row with the same echelonCol if this row is all
    // zeros from echelonCol onwards.
    if (nonZeroCol == h.getNumColumns())
      continue;

    // Bring the non-zero column to echelonCol. This doesn't affect rows
    // above since they are all zero at these columns.
    if (nonZeroCol != echelonCol) {
      h.swapColumns(nonZeroCol, echelonCol);
      u.swapColumns(nonZeroCol, echelonCol);
    }

    // Make h(row, echelonCol) non-negative.
    if (h(row, echelonCol) < 0) {
      h.negateColumn(echelonCol);
      u.negateColumn(echelonCol);
    }

    // Make all the entries in row after echelonCol zero.
    for (unsigned i = echelonCol + 1, e = h.getNumColumns(); i < e; ++i) {
      // We make h(row, i) non-negative, and then apply the Euclidean GCD
      // algorithm to (row, i) and (row, echelonCol). At the end, one of them
      // has value equal to the gcd of the two entries, and the other is zero.

      if (h(row, i) < 0) {
        h.negateColumn(i);
        u.negateColumn(i);
      }

      unsigned targetCol = i, sourceCol = echelonCol;
      // At every step, we set h(row, targetCol) %= h(row, sourceCol), and
      // swap the indices sourceCol and targetCol. (not the columns themselves)
      // This modulo is implemented as a subtraction
      // h(row, targetCol) -= quotient * h(row, sourceCol),
      // where quotient = floor(h(row, targetCol) / h(row, sourceCol)),
      // which brings h(row, targetCol) to the range [0, h(row, sourceCol)).
      //
      // We are only allowed column operations; we perform the above
      // for every row, i.e., the above subtraction is done as a column
      // operation. This does not affect any rows above us since they are
      // guaranteed to be zero at these columns.
      while (h(row, targetCol) != 0 && h(row, sourceCol) != 0) {
        modEntryColumnOperation(h, row, sourceCol, targetCol, u);
        std::swap(targetCol, sourceCol);
      }

      // One of (row, echelonCol) and (row, i) is zero and the other is the gcd.
      // Make it so that (row, echelonCol) holds the non-zero value.
      if (h(row, echelonCol) == 0) {
        h.swapColumns(i, echelonCol);
        u.swapColumns(i, echelonCol);
      }
    }

    // Make all entries before echelonCol non-negative and strictly smaller
    // than the pivot entry.
    for (unsigned i = 0; i < echelonCol; ++i)
      modEntryColumnOperation(h, row, echelonCol, i, u);

    ++echelonCol;
  }

  return {h, u};
}

void Matrix::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << ' ';
    os << '\n';
  }
}

void Matrix::dump() const { print(llvm::errs()); }

bool Matrix::hasConsistentState() const {
  if (data.size() != nRows * nReservedColumns)
    return false;
  if (nColumns > nReservedColumns)
    return false;
#ifdef EXPENSIVE_CHECKS
  for (unsigned r = 0; r < nRows; ++r)
    for (unsigned c = nColumns; c < nReservedColumns; ++c)
      if (data[r * nReservedColumns + c] != 0)
        return false;
#endif
  return true;
}
