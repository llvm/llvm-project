//===- MatrixF.cpp - MLIR MatrixF Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/MatrixF.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;

MatrixF::MatrixF(unsigned rows, unsigned columns, unsigned reservedRows,
               unsigned reservedColumns)
    : nRows(rows), nColumns(columns),
      nReservedColumns(std::max(nColumns, reservedColumns)),
      data(nRows * nReservedColumns) {
  data.reserve(std::max(nRows, reservedRows) * nReservedColumns);
}

MatrixF MatrixF::identity(unsigned dimension) {
  MatrixF matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = Fraction(1, 1);
  return matrix;
}

unsigned MatrixF::getNumReservedRows() const {
  return data.capacity() / nReservedColumns;
}

void MatrixF::reserveRows(unsigned rows) {
  data.reserve(rows * nReservedColumns);
}

unsigned MatrixF::appendExtraRow() {
  resizeVertically(nRows + 1);
  return nRows - 1;
}

unsigned MatrixF::appendExtraRow(ArrayRef<Fraction> elems) {
  assert(elems.size() == nColumns && "elems must match row length!");
  unsigned row = appendExtraRow();
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
  return row;
}

void MatrixF::resizeHorizontally(unsigned newNColumns) {
  if (newNColumns < nColumns)
    removeColumns(newNColumns, nColumns - newNColumns);
  if (newNColumns > nColumns)
    insertColumns(nColumns, newNColumns - nColumns);
}

void MatrixF::resize(unsigned newNRows, unsigned newNColumns) {
  resizeHorizontally(newNColumns);
  resizeVertically(newNRows);
}

void MatrixF::resizeVertically(unsigned newNRows) {
  nRows = newNRows;
  data.resize(nRows * nReservedColumns);
}

void MatrixF::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

void MatrixF::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

MutableArrayRef<Fraction> MatrixF::getRow(unsigned row) {
  return {&data[row * nReservedColumns], nColumns};
}

ArrayRef<Fraction> MatrixF::getRow(unsigned row) const {
  return {&data[row * nReservedColumns], nColumns};
}

void MatrixF::setRow(unsigned row, ArrayRef<Fraction> elems) {
  assert(elems.size() == getNumColumns() &&
         "elems size must match row length!");
  for (unsigned i = 0, e = getNumColumns(); i < e; ++i)
    at(row, i) = elems[i];
}

void MatrixF::insertColumn(unsigned pos) { insertColumns(pos, 1); }
void MatrixF::insertColumns(unsigned pos, unsigned count) {
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
      Fraction &dest = data[r * nReservedColumns + c];
      if (c >= nColumns) { // NOLINT
        // Out of bounds columns are zero-initialized. NOLINT because clang-tidy
        // complains about this branch being the same as the c >= pos one.
        //
        // TODO: this case can be skipped if the number of reserved columns
        // didn't change.
        dest = Fraction(0, 1);
      } else if (c >= pos + count) {
        // Shift the data occuring after the inserted columns.
        dest = data[r * oldNReservedColumns + c - count];
      } else if (c >= pos) {
        // The inserted columns are also zero-initialized.
        dest = Fraction(0, 1);
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

void MatrixF::removeColumn(unsigned pos) { removeColumns(pos, 1); }
void MatrixF::removeColumns(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos + count - 1 < nColumns);
  for (unsigned r = 0; r < nRows; ++r) {
    for (unsigned c = pos; c < nColumns - count; ++c)
      at(r, c) = at(r, c + count);
    for (unsigned c = nColumns - count; c < nColumns; ++c)
      at(r, c) = Fraction(0, 1);
  }
  nColumns -= count;
}

void MatrixF::insertRow(unsigned pos) { insertRows(pos, 1); }
void MatrixF::insertRows(unsigned pos, unsigned count) {
  if (count == 0)
    return;

  assert(pos <= nRows);
  resizeVertically(nRows + count);
  for (int r = nRows - 1; r >= int(pos + count); --r)
    copyRow(r - count, r);
  for (int r = pos + count - 1; r >= int(pos); --r)
    for (unsigned c = 0; c < nColumns; ++c)
      at(r, c) = Fraction(0, 1);
}

void MatrixF::removeRow(unsigned pos) { removeRows(pos, 1); }
void MatrixF::removeRows(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos + count - 1 <= nRows);
  for (unsigned r = pos; r + count < nRows; ++r)
    copyRow(r + count, r);
  resizeVertically(nRows - count);
}

void MatrixF::copyRow(unsigned sourceRow, unsigned targetRow) {
  if (sourceRow == targetRow)
    return;
  for (unsigned c = 0; c < nColumns; ++c)
    at(targetRow, c) = at(sourceRow, c);
}

void MatrixF::fillRow(unsigned row, const Fraction &value) {
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = value;
}

void MatrixF::addToRow(unsigned sourceRow, unsigned targetRow,
                      const Fraction &scale) {
  addToRow(targetRow, getRow(sourceRow), scale);
}

void MatrixF::addToRow(unsigned row, ArrayRef<Fraction> rowVec,
                      const Fraction &scale) {
  if (scale == Fraction(0, 1))
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = at(row, col) + scale * rowVec[col];
}

void MatrixF::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         const Fraction &scale) {
  if (scale == Fraction(0, 1))
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) = at(row, targetColumn) + scale * at(row, sourceColumn);
}

void MatrixF::negateColumn(unsigned column) {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) = -at(row, column);
}

void MatrixF::negateRow(unsigned row) {
  for (unsigned column = 0, e = getNumColumns(); column < e; ++column)
    at(row, column) = -at(row, column);
}

SmallVector<Fraction, 8> MatrixF::preMultiplyWithRow(ArrayRef<Fraction> rowVec) const {
  assert(rowVec.size() == getNumRows() && "Invalid row vector dimension!");

  SmallVector<Fraction, 8> result(getNumColumns(), Fraction(0, 1));
  for (unsigned col = 0, e = getNumColumns(); col < e; ++col)
    for (unsigned i = 0, e = getNumRows(); i < e; ++i)
      result[col] = result[col] + rowVec[i] * at(i, col);
  return result;
}

SmallVector<Fraction, 8>
MatrixF::postMultiplyWithColumn(ArrayRef<Fraction> colVec) const {
  assert(getNumColumns() == colVec.size() &&
         "Invalid column vector dimension!");

  SmallVector<Fraction, 8> result(getNumRows(), Fraction(0, 1));
  for (unsigned row = 0, e = getNumRows(); row < e; row++)
    for (unsigned i = 0, e = getNumColumns(); i < e; i++)
      result[row] = result[row] + at(row, i) * colVec[i];
  return result;
}

MatrixF MatrixF::inverse()
{
    // We use Gaussian elimination on the rows of [M | I]
    // to find the integer inverse. We proceed left-to-right,
    // top-to-bottom. M is assumed to be a dim x dim matrix.

    unsigned dim = getNumRows();

    // Construct the augmented matrix [M | I]
    MatrixF augmented(dim, dim + dim);
    for (unsigned i = 0; i < dim; i++)
    {
        augmented.fillRow(i, 0);
        for (unsigned j = 0; j < dim; j++)
            augmented(i, j) = at(i, j);
        augmented(i, dim+i) = Fraction(1, 1);
    }

    Fraction a, b;
    for (unsigned i = 0; i < dim; i++)
    {
        b = augmented(i, i);
        for (unsigned j = 0; j < dim; j++)
        {
            if (i == j) continue;
            a = augmented(j, i);
            // Rj -> Rj - (b/a)Ri
            augmented.addToRow(j, augmented.getRow(i), - a / b);
            // Now (Rj)i = 0
        }
    }
    
    // Now only diagonal elements are nonzero, but they are
    // not necessarily 1.
    for (unsigned i = 0; i < dim; i++)
    {
        a = augmented(i, i);
        for (unsigned j = dim; j < dim + dim; j++)
            augmented(i, j) = augmented(i, j) / a;
    }

    // Copy the right half of the augmented matrix.
    MatrixF inverse(dim, dim);
    for (unsigned i = 0; i < dim; i++)
        for (unsigned j = 0; j < dim; j++)
            inverse(i, j) = augmented(i, j+dim);

    return inverse;
}

Fraction mlir::presburger::dotProduct(ArrayRef<Fraction> a, ArrayRef<Fraction> b)
{
    Fraction sum(0, 1);
    for (unsigned long i = 0; i < a.size(); i++)
    {
        sum = sum + a[i] * b[i];
    }
    return sum;
}

MatrixF MatrixF::gramSchmidt()
{
    Fraction projection;
    MatrixF copy(getNumRows(), getNumColumns());
    for (unsigned i = 0; i < getNumRows(); i++)
      copy.setRow(i, getRow(i));
    for (unsigned i = 1; i < getNumRows(); i++)
    {
        for (unsigned j = 0; j < i; j++)
        {
            projection = dotProduct(copy.getRow(i), copy.getRow(j)) /
                         dotProduct(copy.getRow(j), copy.getRow(j));
            copy.addToRow(i, copy.getRow(j), -projection);
        }
    }
    return copy;
}

void MatrixF::LLL(Fraction delta)
{
    MPInt nearest;
    Fraction mu;

   MatrixF bStar = gramSchmidt();

    unsigned k = 1;
    while (k < getNumRows())
    {
        for (unsigned j = k-1; j < k; j--)
        {
            mu = dotProduct(getRow(k), bStar.getRow(j)) / dotProduct(bStar.getRow(j), bStar.getRow(j));
            if (mu > Fraction(1, 2))
            {
                nearest = floor(mu + Fraction(1, 2));
                addToRow(k, getRow(j), -Fraction(nearest, 1));
                bStar = gramSchmidt();
            }
        }
        mu = dotProduct(getRow(k), bStar.getRow(k-1)) / dotProduct(bStar.getRow(k-1), bStar.getRow(k-1));
        if (dotProduct(bStar.getRow(k), bStar.getRow(k)) > (delta - mu*mu) * dotProduct(bStar.getRow(k-1), bStar.getRow(k-1)))
            k += 1;
        else
        {
            swapRows(k, k-1);
            bStar = gramSchmidt();
            k = k > 1 ? k-1 : 1;
        }
    }
    return;
}

void MatrixF::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << "(" << at(row, column).num << "/" << at(row, column).den << ")" << ' ';
    os << '\n';
  }
}

void MatrixF::dump() const { print(llvm::errs()); }

bool MatrixF::hasConsistentState() const {
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
