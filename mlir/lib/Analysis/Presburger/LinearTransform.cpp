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

LinearTransform::LinearTransform(IntMatrix &&oMatrix) : matrix(oMatrix) {}
LinearTransform::LinearTransform(const IntMatrix &oMatrix) : matrix(oMatrix) {}

std::pair<unsigned, LinearTransform>
LinearTransform::makeTransformToColumnEchelon(const IntMatrix &m) {
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

IntegerRelation LinearTransform::applyTo(const IntegerRelation &rel) const {
  IntegerRelation result(rel.getSpace());

  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i) {
    ArrayRef<MPInt> eq = rel.getEquality(i);

    const MPInt &c = eq.back();

    SmallVector<MPInt, 8> newEq = preMultiplyWithRow(eq.drop_back());
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i) {
    ArrayRef<MPInt> ineq = rel.getInequality(i);

    const MPInt &c = ineq.back();

    SmallVector<MPInt, 8> newIneq = preMultiplyWithRow(ineq.drop_back());
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  return result;
}
