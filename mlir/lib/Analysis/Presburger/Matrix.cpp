//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

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

int64_t &Matrix::at(unsigned row, unsigned column) {
  assert(row < nRows && "Row outside of range");
  assert(column < nColumns && "Column outside of range");
  return data[row * nReservedColumns + column];
}

int64_t Matrix::at(unsigned row, unsigned column) const {
  assert(row < nRows && "Row outside of range");
  assert(column < nColumns && "Column outside of range");
  return data[row * nReservedColumns + column];
}

int64_t &Matrix::operator()(unsigned row, unsigned column) {
  return at(row, column);
}

int64_t Matrix::operator()(unsigned row, unsigned column) const {
  return at(row, column);
}

unsigned Matrix::getNumRows() const { return nRows; }

unsigned Matrix::getNumColumns() const { return nColumns; }

unsigned Matrix::getNumReservedColumns() const { return nReservedColumns; }

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

ArrayRef<int64_t> Matrix::getRow(unsigned row) const {
  return {&data[row * nReservedColumns], nColumns};
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
      int64_t &dest = data[r * nReservedColumns + c];
      if (c >= nColumns)
        dest = 0;
      else if (c >= pos + count)
        dest = data[r * oldNReservedColumns + c - count];
      else if (c >= pos)
        dest = 0;
      else
        dest = data[r * oldNReservedColumns + c];
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

void Matrix::fillRow(unsigned row, int64_t value) {
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = value;
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
}

void Matrix::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
}

void Matrix::negateColumn(unsigned column) {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) = -at(row, column);
}

SmallVector<int64_t, 8>
Matrix::preMultiplyWithRow(ArrayRef<int64_t> rowVec) const {
  assert(rowVec.size() == getNumRows() && "Invalid row vector dimension!");

  SmallVector<int64_t, 8> result(getNumColumns(), 0);
  for (unsigned col = 0, e = getNumColumns(); col < e; ++col)
    for (unsigned i = 0, e = getNumRows(); i < e; ++i)
      result[col] += rowVec[i] * at(i, col);
  return result;
}

SmallVector<int64_t, 8>
Matrix::postMultiplyWithColumn(ArrayRef<int64_t> colVec) const {
  assert(getNumColumns() == colVec.size() &&
         "Invalid column vector dimension!");

  SmallVector<int64_t, 8> result(getNumRows(), 0);
  for (unsigned row = 0, e = getNumRows(); row < e; row++)
    for (unsigned i = 0, e = getNumColumns(); i < e; i++)
      result[row] += at(row, i) * colVec[i];
  return result;
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
  for (unsigned r = 0; r < nRows; ++r)
    for (unsigned c = nColumns; c < nReservedColumns; ++c)
      if (data[r * nReservedColumns + c] != 0)
        return false;
  return true;
}

} // namespace mlir
