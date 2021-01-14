//===- Matrix.h - MLIR Matrix Class -----------------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir {

/// This is a simple class to represent a resizable matrix.
///
/// The data is stored in the form of a vector of vectors.
class Matrix {
public:
  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// Initially, the values are default initialized.
  Matrix(unsigned rows, unsigned columns);

  /// Return the identity matrix of the specified dimension.
  static Matrix identity(unsigned dimension);

  /// Access the element at the specified row and column.
  int64_t &at(unsigned row, unsigned column);
  int64_t at(unsigned row, unsigned column) const;
  int64_t &operator()(unsigned row, unsigned column);
  int64_t operator()(unsigned row, unsigned column) const;

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const;

  unsigned getNumColumns() const;

  /// Get an ArrayRef corresponding to the specified row.
  ArrayRef<int64_t> getRow(unsigned row) const;

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale);

  /// Add `scale` multiples of the source column to the target column.
  void addToColumn(unsigned sourceColumn, unsigned targetColumn, int64_t scale);

  /// Negate the specified column.
  void negateColumn(unsigned column);

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resizeVertically(unsigned newNRows);

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

private:
  unsigned nRows, nColumns;

  /// Stores the data. data.size() is equal to nRows * nColumns.
  SmallVector<int64_t, 64> data;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
