//===- LinearTransform.h - MLIR LinearTransform Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for linear transforms and applying them to FlatAffineConstraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LINEARTRANSFORM_H
#define MLIR_ANALYSIS_LINEARTRANSFORM_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class LinearTransform {
public:
  explicit LinearTransform(Matrix &&oMatrix);
  explicit LinearTransform(const Matrix &oMatrix);

  // Returns a linear transform T such that MT is M in column echelon form.
  // Also returns the number of non-zero columns in MT.
  //
  // Specifically, T is such that in every column the first non-zero row is
  // strictly below that of the previous column, and all columns which have only
  // zeros are at the end.
  static std::pair<unsigned, LinearTransform>
  makeTransformToColumnEchelon(Matrix m);

  // Returns a FlatAffineConstraints having a constraint vector vT for every
  // constraint vector v in fac, where T is this transform.
  FlatAffineConstraints applyTo(const FlatAffineConstraints &fac) const;

  // The given vector is interpreted as a row vector v. Post-multiply v with
  // this transform, say T, and return vT.
  SmallVector<int64_t, 8> postMultiplyRow(ArrayRef<int64_t> rowVec) const;

  // The given vector is interpreted as a column vector v. Pre-multiply v with
  // this transform, say T, and return Tv.
  SmallVector<int64_t, 8> preMultiplyColumn(ArrayRef<int64_t> colVec) const;

private:
  Matrix matrix;
};

} // namespace mlir
#endif // MLIR_ANALYSIS_LINEARTRANSFORM_H
