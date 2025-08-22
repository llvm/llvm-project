//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <utility>

using namespace mlir;
using namespace presburger;

template <typename T>
Matrix<T>::Matrix(unsigned rows, unsigned columns, unsigned reservedRows,
                  unsigned reservedColumns)
    : nRows(rows), nColumns(columns),
      nReservedColumns(std::max(nColumns, reservedColumns)),
      data(nRows * nReservedColumns) {
  data.reserve(std::max(nRows, reservedRows) * nReservedColumns);
}

/// We cannot use the default implementation of operator== as it compares
/// fields like `reservedColumns` etc., which are not part of the data.
template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &m) const {
  if (nRows != m.getNumRows())
    return false;
  if (nColumns != m.getNumColumns())
    return false;

  for (unsigned i = 0; i < nRows; i++)
    if (getRow(i) != m.getRow(i))
      return false;

  return true;
}

template <typename T>
Matrix<T> Matrix<T>::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

template <typename T>
unsigned Matrix<T>::getNumReservedRows() const {
  return data.capacity() / nReservedColumns;
}

template <typename T>
void Matrix<T>::reserveRows(unsigned rows) {
  data.reserve(rows * nReservedColumns);
}

template <typename T>
unsigned Matrix<T>::appendExtraRow() {
  resizeVertically(nRows + 1);
  return nRows - 1;
}

template <typename T>
unsigned Matrix<T>::appendExtraRow(ArrayRef<T> elems) {
  assert(elems.size() == nColumns && "elems must match row length!");
  unsigned row = appendExtraRow();
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
  return row;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> transp(nColumns, nRows);
  for (unsigned row = 0; row < nRows; ++row)
    for (unsigned col = 0; col < nColumns; ++col)
      transp(col, row) = at(row, col);

  return transp;
}

template <typename T>
void Matrix<T>::resizeHorizontally(unsigned newNColumns) {
  if (newNColumns < nColumns)
    removeColumns(newNColumns, nColumns - newNColumns);
  if (newNColumns > nColumns)
    insertColumns(nColumns, newNColumns - nColumns);
}

template <typename T>
void Matrix<T>::resize(unsigned newNRows, unsigned newNColumns) {
  resizeHorizontally(newNColumns);
  resizeVertically(newNRows);
}

template <typename T>
void Matrix<T>::resizeVertically(unsigned newNRows) {
  nRows = newNRows;
  data.resize(nRows * nReservedColumns);
}

template <typename T>
void Matrix<T>::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

template <typename T>
void Matrix<T>::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

template <typename T>
MutableArrayRef<T> Matrix<T>::getRow(unsigned row) {
  return {&data[row * nReservedColumns], nColumns};
}

template <typename T>
ArrayRef<T> Matrix<T>::getRow(unsigned row) const {
  return {&data[row * nReservedColumns], nColumns};
}

template <typename T>
void Matrix<T>::setRow(unsigned row, ArrayRef<T> elems) {
  assert(elems.size() == getNumColumns() &&
         "elems size must match row length!");
  for (unsigned i = 0, e = getNumColumns(); i < e; ++i)
    at(row, i) = elems[i];
}

template <typename T>
void Matrix<T>::insertColumn(unsigned pos) {
  insertColumns(pos, 1);
}
template <typename T>
void Matrix<T>::insertColumns(unsigned pos, unsigned count) {
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
      T &dest = data[r * nReservedColumns + c];
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

template <typename T>
void Matrix<T>::removeColumn(unsigned pos) {
  removeColumns(pos, 1);
}
template <typename T>
void Matrix<T>::removeColumns(unsigned pos, unsigned count) {
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

template <typename T>
void Matrix<T>::insertRow(unsigned pos) {
  insertRows(pos, 1);
}
template <typename T>
void Matrix<T>::insertRows(unsigned pos, unsigned count) {
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

template <typename T>
void Matrix<T>::removeRow(unsigned pos) {
  removeRows(pos, 1);
}
template <typename T>
void Matrix<T>::removeRows(unsigned pos, unsigned count) {
  if (count == 0)
    return;
  assert(pos + count - 1 <= nRows);
  for (unsigned r = pos; r + count < nRows; ++r)
    copyRow(r + count, r);
  resizeVertically(nRows - count);
}

template <typename T>
void Matrix<T>::copyRow(unsigned sourceRow, unsigned targetRow) {
  if (sourceRow == targetRow)
    return;
  for (unsigned c = 0; c < nColumns; ++c)
    at(targetRow, c) = at(sourceRow, c);
}

template <typename T>
void Matrix<T>::fillRow(unsigned row, const T &value) {
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = value;
}

// moveColumns is implemented by moving the columns adjacent to the source range
// to their final position. When moving right (i.e. dstPos > srcPos), the range
// of the adjacent columns is [srcPos + num, dstPos + num). When moving left
// (i.e. dstPos < srcPos) the range of the adjacent columns is [dstPos, srcPos).
// First, zeroed out columns are inserted in the final positions of the adjacent
// columns. Then, the adjacent columns are moved to their final positions by
// swapping them with the zeroed columns. Finally, the now zeroed adjacent
// columns are deleted.
template <typename T>
void Matrix<T>::moveColumns(unsigned srcPos, unsigned num, unsigned dstPos) {
  if (num == 0)
    return;

  int offset = dstPos - srcPos;
  if (offset == 0)
    return;

  assert(srcPos + num <= getNumColumns() &&
         "move source range exceeds matrix columns");
  assert(dstPos + num <= getNumColumns() &&
         "move destination range exceeds matrix columns");

  unsigned insertCount = offset > 0 ? offset : -offset;
  unsigned finalAdjStart = offset > 0 ? srcPos : srcPos + num;
  unsigned curAdjStart = offset > 0 ? srcPos + num : dstPos;
  // TODO: This can be done using std::rotate.
  // Insert new zero columns in the positions where the adjacent columns are to
  // be moved.
  insertColumns(finalAdjStart, insertCount);
  // Update curAdjStart if insertion of new columns invalidates it.
  if (finalAdjStart < curAdjStart)
    curAdjStart += insertCount;

  // Swap the adjacent columns with inserted zero columns.
  for (unsigned i = 0; i < insertCount; ++i)
    swapColumns(finalAdjStart + i, curAdjStart + i);

  // Delete the now redundant zero columns.
  removeColumns(curAdjStart, insertCount);
}

template <typename T>
void Matrix<T>::addToRow(unsigned sourceRow, unsigned targetRow,
                         const T &scale) {
  addToRow(targetRow, getRow(sourceRow), scale);
}

template <typename T>
void Matrix<T>::addToRow(unsigned row, ArrayRef<T> rowVec, const T &scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) += scale * rowVec[col];
}

template <typename T>
void Matrix<T>::scaleRow(unsigned row, const T &scale) {
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) *= scale;
}

template <typename T>
void Matrix<T>::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                            const T &scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
}

template <typename T>
void Matrix<T>::negateColumn(unsigned column) {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) = -at(row, column);
}

template <typename T>
void Matrix<T>::negateRow(unsigned row) {
  for (unsigned column = 0, e = getNumColumns(); column < e; ++column)
    at(row, column) = -at(row, column);
}

template <typename T>
void Matrix<T>::negateMatrix() {
  for (unsigned row = 0; row < nRows; ++row)
    negateRow(row);
}

template <typename T>
SmallVector<T, 8> Matrix<T>::preMultiplyWithRow(ArrayRef<T> rowVec) const {
  assert(rowVec.size() == getNumRows() && "Invalid row vector dimension!");

  SmallVector<T, 8> result(getNumColumns(), T(0));
  for (unsigned col = 0, e = getNumColumns(); col < e; ++col)
    for (unsigned i = 0, e = getNumRows(); i < e; ++i)
      result[col] += rowVec[i] * at(i, col);
  return result;
}

template <typename T>
SmallVector<T, 8> Matrix<T>::postMultiplyWithColumn(ArrayRef<T> colVec) const {
  assert(getNumColumns() == colVec.size() &&
         "Invalid column vector dimension!");

  SmallVector<T, 8> result(getNumRows(), T(0));
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
static void modEntryColumnOperation(Matrix<DynamicAPInt> &m, unsigned row,
                                    unsigned sourceCol, unsigned targetCol,
                                    Matrix<DynamicAPInt> &otherMatrix) {
  assert(m(row, sourceCol) != 0 && "Cannot divide by zero!");
  assert(m(row, sourceCol) > 0 && "Source must be positive!");
  DynamicAPInt ratio = -floorDiv(m(row, targetCol), m(row, sourceCol));
  m.addToColumn(sourceCol, targetCol, ratio);
  otherMatrix.addToColumn(sourceCol, targetCol, ratio);
}

template <typename T>
Matrix<T> Matrix<T>::getSubMatrix(unsigned fromRow, unsigned toRow,
                                  unsigned fromColumn,
                                  unsigned toColumn) const {
  assert(fromRow <= toRow && "end of row range must be after beginning!");
  assert(toRow < nRows && "end of row range out of bounds!");
  assert(fromColumn <= toColumn &&
         "end of column range must be after beginning!");
  assert(toColumn < nColumns && "end of column range out of bounds!");
  Matrix<T> subMatrix(toRow - fromRow + 1, toColumn - fromColumn + 1);
  for (unsigned i = fromRow; i <= toRow; ++i)
    for (unsigned j = fromColumn; j <= toColumn; ++j)
      subMatrix(i - fromRow, j - fromColumn) = at(i, j);
  return subMatrix;
}

template <typename T>
void Matrix<T>::print(raw_ostream &os) const {
  PrintTableMetrics ptm = {0, 0, "-"};
  for (unsigned row = 0; row < nRows; ++row)
    for (unsigned column = 0; column < nColumns; ++column)
      updatePrintMetrics<T>(at(row, column), ptm);
  unsigned MIN_SPACING = 1;
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column) {
      printWithPrintMetrics<T>(os, at(row, column), MIN_SPACING, ptm);
    }
    os << "\n";
  }
}

/// We iterate over the `indicator` bitset, checking each bit. If a bit is 1,
/// we append it to one matrix, and if it is zero, we append it to the other.
template <typename T>
std::pair<Matrix<T>, Matrix<T>>
Matrix<T>::splitByBitset(ArrayRef<int> indicator) {
  Matrix<T> rowsForOne(0, nColumns), rowsForZero(0, nColumns);
  for (unsigned i = 0; i < nRows; i++) {
    if (indicator[i] == 1)
      rowsForOne.appendExtraRow(getRow(i));
    else
      rowsForZero.appendExtraRow(getRow(i));
  }
  return {rowsForOne, rowsForZero};
}

template <typename T>
void Matrix<T>::dump() const {
  print(llvm::errs());
}

template <typename T>
bool Matrix<T>::hasConsistentState() const {
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

namespace mlir {
namespace presburger {
template class Matrix<DynamicAPInt>;
template class Matrix<Fraction>;
} // namespace presburger
} // namespace mlir

IntMatrix IntMatrix::identity(unsigned dimension) {
  IntMatrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

std::pair<IntMatrix, IntMatrix> IntMatrix::computeHermiteNormalForm() const {
  // We start with u as an identity matrix and perform operations on h until h
  // is in hermite normal form. We apply the same sequence of operations on u to
  // obtain a transform that takes h to hermite normal form.
  IntMatrix h = *this;
  IntMatrix u = IntMatrix::identity(h.getNumColumns());

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

DynamicAPInt IntMatrix::normalizeRow(unsigned row, unsigned cols) {
  return normalizeRange(getRow(row).slice(0, cols));
}

DynamicAPInt IntMatrix::normalizeRow(unsigned row) {
  return normalizeRow(row, getNumColumns());
}

DynamicAPInt IntMatrix::determinant(IntMatrix *inverse) const {
  assert(nRows == nColumns &&
         "determinant can only be calculated for square matrices!");

  FracMatrix m(*this);

  FracMatrix fracInverse(nRows, nColumns);
  DynamicAPInt detM = m.determinant(&fracInverse).getAsInteger();

  if (detM == 0)
    return DynamicAPInt(0);

  if (!inverse)
    return detM;

  *inverse = IntMatrix(nRows, nColumns);
  for (unsigned i = 0; i < nRows; i++)
    for (unsigned j = 0; j < nColumns; j++)
      inverse->at(i, j) = (fracInverse.at(i, j) * detM).getAsInteger();

  return detM;
}

FracMatrix FracMatrix::identity(unsigned dimension) {
  return Matrix::identity(dimension);
}

FracMatrix::FracMatrix(IntMatrix m)
    : FracMatrix(m.getNumRows(), m.getNumColumns()) {
  for (unsigned i = 0, r = m.getNumRows(); i < r; i++)
    for (unsigned j = 0, c = m.getNumColumns(); j < c; j++)
      this->at(i, j) = m.at(i, j);
}

Fraction FracMatrix::determinant(FracMatrix *inverse) const {
  assert(nRows == nColumns &&
         "determinant can only be calculated for square matrices!");

  FracMatrix m(*this);
  FracMatrix tempInv(nRows, nColumns);
  if (inverse)
    tempInv = FracMatrix::identity(nRows);

  Fraction a, b;
  // Make the matrix into upper triangular form using
  // gaussian elimination with row operations.
  // If inverse is required, we apply more operations
  // to turn the matrix into diagonal form. We apply
  // the same operations to the inverse matrix,
  // which is initially identity.
  // Either way, the product of the diagonal elements
  // is then the determinant.
  for (unsigned i = 0; i < nRows; i++) {
    if (m(i, i) == 0)
      // First ensure that the diagonal
      // element is nonzero, by swapping
      // it with a nonzero row.
      for (unsigned j = i + 1; j < nRows; j++) {
        if (m(j, i) != 0) {
          m.swapRows(j, i);
          if (inverse)
            tempInv.swapRows(j, i);
          break;
        }
      }

    b = m.at(i, i);
    if (b == 0)
      return 0;

    // Set all elements above the
    // diagonal to zero.
    if (inverse) {
      for (unsigned j = 0; j < i; j++) {
        if (m.at(j, i) == 0)
          continue;
        a = m.at(j, i);
        // Set element (j, i) to zero
        // by subtracting the ith row,
        // appropriately scaled.
        m.addToRow(i, j, -a / b);
        tempInv.addToRow(i, j, -a / b);
      }
    }

    // Set all elements below the
    // diagonal to zero.
    for (unsigned j = i + 1; j < nRows; j++) {
      if (m.at(j, i) == 0)
        continue;
      a = m.at(j, i);
      // Set element (j, i) to zero
      // by subtracting the ith row,
      // appropriately scaled.
      m.addToRow(i, j, -a / b);
      if (inverse)
        tempInv.addToRow(i, j, -a / b);
    }
  }

  // Now only diagonal elements of m are nonzero, but they are
  // not necessarily 1. To get the true inverse, we should
  // normalize them and apply the same scale to the inverse matrix.
  // For efficiency we skip scaling m and just scale tempInv appropriately.
  if (inverse) {
    for (unsigned i = 0; i < nRows; i++)
      for (unsigned j = 0; j < nRows; j++)
        tempInv.at(i, j) = tempInv.at(i, j) / m(i, i);

    *inverse = std::move(tempInv);
  }

  Fraction determinant = 1;
  for (unsigned i = 0; i < nRows; i++)
    determinant *= m.at(i, i);

  return determinant;
}

FracMatrix FracMatrix::gramSchmidt() const {
  // Create a copy of the argument to store
  // the orthogonalised version.
  FracMatrix orth(*this);

  // For each vector (row) in the matrix, subtract its unit
  // projection along each of the previous vectors.
  // This ensures that it has no component in the direction
  // of any of the previous vectors.
  for (unsigned i = 1, e = getNumRows(); i < e; i++) {
    for (unsigned j = 0; j < i; j++) {
      Fraction jNormSquared = dotProduct(orth.getRow(j), orth.getRow(j));
      assert(jNormSquared != 0 && "some row became zero! Inputs to this "
                                  "function must be linearly independent.");
      Fraction projectionScale =
          dotProduct(orth.getRow(i), orth.getRow(j)) / jNormSquared;
      orth.addToRow(j, i, -projectionScale);
    }
  }
  return orth;
}

// Convert the matrix, interpreted (row-wise) as a basis
// to an LLL-reduced basis.
//
// This is an implementation of the algorithm described in
// "Factoring polynomials with rational coefficients" by
// A. K. Lenstra, H. W. Lenstra Jr., L. Lovasz.
//
// Let {b_1,  ..., b_n}  be the current basis and
//     {b_1*, ..., b_n*} be the Gram-Schmidt orthogonalised
//                          basis (unnormalized).
// Define the Gram-Schmidt coefficients μ_ij as
// (b_i • b_j*) / (b_j* • b_j*), where (•) represents the inner product.
//
// We iterate starting from the second row to the last row.
//
// For the kth row, we first check μ_kj for all rows j < k.
// We subtract b_j (scaled by the integer nearest to μ_kj)
// from b_k.
//
// Now, we update k.
// If b_k and b_{k-1} satisfy the Lovasz condition
//    |b_k|^2 ≥ (δ - μ_k{k-1}^2) |b_{k-1}|^2,
// we are done and we increment k.
// Otherwise, we swap b_k and b_{k-1} and decrement k.
//
// We repeat this until k = n and return.
void FracMatrix::LLL(const Fraction &delta) {
  DynamicAPInt nearest;
  Fraction mu;

  // `gsOrth` holds the Gram-Schmidt orthogonalisation
  // of the matrix at all times. It is recomputed every
  // time the matrix is modified during the algorithm.
  // This is naive and can be optimised.
  FracMatrix gsOrth = gramSchmidt();

  // We start from the second row.
  unsigned k = 1;
  while (k < getNumRows()) {
    for (unsigned j = k - 1; j < k; j--) {
      // Compute the Gram-Schmidt coefficient μ_jk.
      mu = dotProduct(getRow(k), gsOrth.getRow(j)) /
           dotProduct(gsOrth.getRow(j), gsOrth.getRow(j));
      nearest = round(mu);
      // Subtract b_j scaled by the integer nearest to μ_jk from b_k.
      addToRow(k, getRow(j), -Fraction(nearest, 1));
      gsOrth = gramSchmidt(); // Update orthogonalization.
    }
    mu = dotProduct(getRow(k), gsOrth.getRow(k - 1)) /
         dotProduct(gsOrth.getRow(k - 1), gsOrth.getRow(k - 1));
    // Check the Lovasz condition for b_k and b_{k-1}.
    if (dotProduct(gsOrth.getRow(k), gsOrth.getRow(k)) >
        (delta - mu * mu) *
            dotProduct(gsOrth.getRow(k - 1), gsOrth.getRow(k - 1))) {
      // If it is satisfied, proceed to the next k.
      k += 1;
    } else {
      // If it is not satisfied, decrement k (without
      // going beyond the second row).
      swapRows(k, k - 1);
      gsOrth = gramSchmidt(); // Update orthogonalization.
      k = k > 1 ? k - 1 : 1;
    }
  }
}

IntMatrix FracMatrix::normalizeRows() const {
  unsigned numRows = getNumRows();
  unsigned numColumns = getNumColumns();
  IntMatrix normalized(numRows, numColumns);

  DynamicAPInt lcmDenoms = DynamicAPInt(1);
  for (unsigned i = 0; i < numRows; i++) {
    // For a row, first compute the LCM of the denominators.
    for (unsigned j = 0; j < numColumns; j++)
      lcmDenoms = lcm(lcmDenoms, at(i, j).den);
    // Then, multiply by it throughout and convert to integers.
    for (unsigned j = 0; j < numColumns; j++)
      normalized(i, j) = (at(i, j) * lcmDenoms).getAsInteger();
  }
  return normalized;
}
