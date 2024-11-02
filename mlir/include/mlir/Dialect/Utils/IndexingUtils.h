//===- IndexingUtils.h - Helpers related to index computations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common canonicalization patterns for
// reshape operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_INDEXINGUTILS_H
#define MLIR_DIALECT_UTILS_INDEXINGUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ArrayAttr;

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension
/// space, returns the vector-space offsets in each dimension for a
/// de-linearized index.
SmallVector<int64_t, 4> delinearize(ArrayRef<int64_t> strides,
                                    int64_t linearIndex);

/// Apply the permutation defined by `permutation` to `inVec`.
/// Element `i` in `inVec` is mapped to location `j = permutation[i]`.
/// E.g.: for an input vector `inVec = ['a', 'b', 'c']` and a permutation vector
/// `permutation = [2, 0, 1]`, this function leaves `inVec = ['c', 'a', 'b']`.
template <typename T, unsigned N>
void applyPermutationToVector(SmallVector<T, N> &inVec,
                              ArrayRef<int64_t> permutation) {
  SmallVector<T, N> auxVec(inVec.size());
  for (const auto &en : enumerate(permutation))
    auxVec[en.index()] = inVec[en.value()];
  inVec = auxVec;
}

/// Helper that returns a subset of `arrayAttr` as a vector of int64_t.
SmallVector<int64_t, 4> getI64SubArray(ArrayAttr arrayAttr,
                                       unsigned dropFront = 0,
                                       unsigned dropBack = 0);

/// Computes and returns linearized affine expression w.r.t. `basis`.
mlir::AffineExpr getLinearAffineExpr(ArrayRef<int64_t> basis, mlir::Builder &b);

/// Given the strides in the dimension space, returns the affine expressions for
/// vector-space offsets in each dimension for a de-linearized index.
SmallVector<mlir::AffineExpr, 4>
getDelinearizedAffineExpr(ArrayRef<int64_t> strides, mlir::Builder &b);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_INDEXINGUTILS_H
