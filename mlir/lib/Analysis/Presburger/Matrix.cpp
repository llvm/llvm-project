//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {

Matrix::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), data(nRows * nColumns) {}

Matrix Matrix::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

int64_t &Matrix::at(unsigned row, unsigned column) {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t Matrix::at(unsigned row, unsigned column) const {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t &Matrix::operator()(unsigned row, unsigned column) {
  return at(row, column);
}

int64_t Matrix::operator()(unsigned row, unsigned column) const {
  return at(row, column);
}

unsigned Matrix::getNumRows() const { return nRows; }

unsigned Matrix::getNumColumns() const { return nColumns; }

void Matrix::resizeVertically(unsigned newNRows) {
  nRows = newNRows;
  data.resize(nRows * nColumns);
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
  return {&data[row * nColumns], nColumns};
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
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

void Matrix::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << ' ';
    os << '\n';
  }
}

void Matrix::dump() const { print(llvm::errs()); }

} // namespace mlir
