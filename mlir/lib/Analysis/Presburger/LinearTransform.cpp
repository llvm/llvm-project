//===- LinearTransform.cpp - MLIR LinearTransform Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"

using namespace mlir;
using namespace presburger;

template <typename T> LinearTransform<T>::LinearTransform(Matrix<T> &&oMatrix) : matrix(oMatrix) {}
template <typename T> LinearTransform<T>::LinearTransform(const Matrix<T> &oMatrix) : matrix(oMatrix) {}

template <typename T> std::pair<unsigned, LinearTransform<T>>
LinearTransform<T>::makeTransformToColumnEchelon(const Matrix<T> &m) {
  // Compute the hermite normal form of m. This, is by definition, is in column
  // echelon form.
  auto [h, u] = m.computeHermiteNormalForm();

  // Since the matrix is in column ecehlon form, a zero column means the rest of
  // the columns are zero. Thus, once we find a zero column, we can stop.
  unsigned col, e;
  for (col = 0, e = m.getNumColumns(); col < e; ++col) {
    bool zeroCol = true;
    for (unsigned row = 0, f = m.getNumRows(); row < f; ++row) {
      if (h(row, col) != 0) {
        zeroCol = false;
        break;
      }
    }

    if (zeroCol)
      break;
  }

  return {col, LinearTransform(std::move(u))};
}

template <typename T> IntegerRelation LinearTransform<T>::applyTo(const IntegerRelation &rel) const {
  IntegerRelation result(rel.getSpace());

  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i) {
    ArrayRef<T> eq = rel.getEquality(i);

    const T &c = eq.back();

    SmallVector<T, 8> newEq = preMultiplyWithRow(eq.drop_back());
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i) {
    ArrayRef<T> ineq = rel.getInequality(i);

    const T &c = ineq.back();

    SmallVector<T, 8> newIneq = preMultiplyWithRow(ineq.drop_back());
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  return result;
}

template <typename T> T LinearTransform<T>::determinant()
{
    // Convert to column echelon form. Now `colEchelon` is lower triangular.
    Matrix<T> m = this->matrix;
    LinearTransform<T> colEchelon = makeTransformToColumnEchelon(m).second;
    T determinant(1);
    for (unsigned i = 0; i < m.getNumColumns(); i++)
    {
        // Construct a one-hot vector for i.
        SmallVector<T, 8U> pickColumnVec(m.getNumRows(), T(0));
        pickColumnVec[i] = 1;
        // Select column i by post-multiplying.
        SmallVector<T, 8> iThColumn = colEchelon.postMultiplyWithColumn(ArrayRef(pickColumnVec));
        // Get ith element of column i.
        determinant *= iThColumn[i];
    }

    return determinant;
}
