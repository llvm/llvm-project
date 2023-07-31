//===- MatrixF.h - MLIR MatrixF Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple 2D matrix class that supports reading, writing, resizing,
// swapping rows, and swapping columns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIXF_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIXF_H

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/MPInt.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir {
namespace presburger {

/// This is a class to represent a resizable matrix.
///
/// More columns and rows can be reserved than are currently used. The data is
/// stored as a single 1D array, viewed as a 2D matrix with nRows rows and
/// nReservedColumns columns, stored in row major form. Thus the element at
/// (i, j) is stored at data[i*nReservedColumns + j]. The reserved but unused
/// columns always have all zero values. The reserved rows are just reserved
/// space in the underlying SmallVector's capacity.
class MatrixF {
public:
  MatrixF() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// The number of reserved rows and columns will be at least the number
  /// specified, and will always be sufficient to accomodate the number of rows
  /// and columns specified.
  ///
  /// Initially, the entries are initialized to ero.
  MatrixF(unsigned rows, unsigned columns, unsigned reservedRows = 0,
         unsigned reservedColumns = 0);

  /// Return the identity matrix of the specified dimension.
  static MatrixF identity(unsigned dimension);

  /// Access the element at the specified row and column.
  Fraction &at(unsigned row, unsigned column) {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  Fraction at(unsigned row, unsigned column) const {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nReservedColumns + column];
  }

  Fraction &operator()(unsigned row, unsigned column) { return at(row, column); }

  Fraction operator()(unsigned row, unsigned column) const {
    return at(row, column);
  }

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const { return nRows; }

  unsigned getNumColumns() const { return nColumns; }

  /// Return the maximum number of rows/columns that can be added without
  /// incurring a reallocation.
  unsigned getNumReservedRows() const;
  unsigned getNumReservedColumns() const { return nReservedColumns; }

  /// Reserve enough space to resize to the specified number of rows without
  /// reallocations.
  void reserveRows(unsigned rows);

  /// Get a [Mutable]ArrayRef corresponding to the specified row.
  MutableArrayRef<Fraction> getRow(unsigned row);
  ArrayRef<Fraction> getRow(unsigned row) const;

  /// Set the specified row to `elems`.
  void setRow(unsigned row, ArrayRef<Fraction> elems);

  /// Insert columns having positions pos, pos + 1, ... pos + count - 1.
  /// Columns that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nColumns.
  void insertColumns(unsigned pos, unsigned count);
  void insertColumn(unsigned pos);

  /// Insert rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nRows.
  void insertRows(unsigned pos, unsigned count);
  void insertRow(unsigned pos);

  /// Remove the columns having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos + count - 1 or later will be pushed to
  /// the right. The columns to be deleted must be valid rows: pos + count - 1
  /// must be at most nColumns - 1.
  void removeColumns(unsigned pos, unsigned count);
  void removeColumn(unsigned pos);

  /// Remove the rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos + count - 1 or later will be pushed to the
  /// right. The rows to be deleted must be valid rows: pos + count - 1 must be
  /// at most nRows - 1.
  void removeRows(unsigned pos, unsigned count);
  void removeRow(unsigned pos);

  void copyRow(unsigned sourceRow, unsigned targetRow);

  void fillRow(unsigned row, const Fraction &value);
  void fillRow(unsigned row, int64_t value) { fillRow(row, Fraction(value, 1)); }

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, const Fraction &scale);
  void addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
    addToRow(sourceRow, targetRow, Fraction(scale, 1));
  }
  /// Add `scale` multiples of the rowVec row to the specified row.
  void addToRow(unsigned row, ArrayRef<Fraction> rowVec, const Fraction &scale);

  /// Add `scale` multiples of the source column to the target column.
  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   const Fraction &scale);
  void addToColumn(unsigned sourceColumn, unsigned targetColumn,
                   int64_t scale) {
    addToColumn(sourceColumn, targetColumn, Fraction(scale, 1));
  }

  /// Negate the specified column.
  void negateColumn(unsigned column);

  /// Negate the specified row.
  void negateRow(unsigned row);

  /// The given vector is interpreted as a row vector v. Post-multiply v with
  /// this matrix, say M, and return vM.
  SmallVector<Fraction, 8> preMultiplyWithRow(ArrayRef<Fraction> rowVec) const;

  /// The given vector is interpreted as a column vector v. Pre-multiply v with
  /// this matrix, say M, and return Mv.
  SmallVector<Fraction, 8> postMultiplyWithColumn(ArrayRef<Fraction> colVec) const;

  // Invert the matrix (it must be square), leaving the calling object unmodified.
  MatrixF inverse();

  // Run Gram-Schmidt orthogonalisation on the matrix, leaving the calling object
  // unmodified and returning the orthogonal (unnormalised) matrix.
  MatrixF gramSchmidt();

  // Run LLL basis reduction on the matrix, modifying it in-place.
  // The parameter is delta, taken to be 3/4 by default.
  void LLL(Fraction delta = Fraction(3, 4));

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are initialized
  /// to zero.
  ///
  /// Due to the representation of the matrix, resizing vertically (adding rows)
  /// is less expensive than increasing the number of columns beyond
  /// nReservedColumns.
  void resize(unsigned newNRows, unsigned newNColumns);
  void resizeHorizontally(unsigned newNColumns);
  void resizeVertically(unsigned newNRows);

  /// Add an extra row at the bottom of the matrix and return its position.
  unsigned appendExtraRow();
  /// Same as above, but copy the given elements into the row. The length of
  /// `elems` must be equal to the number of columns.
  unsigned appendExtraRow(ArrayRef<Fraction> elems);

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Return whether the MatrixF is in a consistent state with all its
  /// invariants satisfied.
  bool hasConsistentState() const;

private:
  /// The current number of rows, columns, and reserved columns. The underlying
  /// data vector is viewed as an nRows x nReservedColumns matrix, of which the
  /// first nColumns columns are currently in use, and the remaining are
  /// reserved columns filled with zeros.
  unsigned nRows, nColumns, nReservedColumns;

  /// Stores the data. data.size() is equal to nRows * nReservedColumns.
  /// data.capacity() / nReservedColumns is the number of reserved rows.
  SmallVector<Fraction, 16> data;
};

Fraction dotProduct(ArrayRef<Fraction>, ArrayRef<Fraction>);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
