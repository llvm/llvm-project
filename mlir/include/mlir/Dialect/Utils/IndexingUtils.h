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
#include <optional>

namespace mlir {
class ArrayAttr;

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension
/// space, returns the vector-space offsets in each dimension for a
/// de-linearized index.
SmallVector<int64_t> delinearize(ArrayRef<int64_t> strides,
                                 int64_t linearIndex);

/// Given a set of sizes, compute and return the strides (i.e. the number of
/// linear incides to skip along the (k-1) most minor dimensions to get the next
/// k-slice). This is also the basis that one can use to linearize an n-D offset
/// confined to `[0 .. sizes]`.
SmallVector<int64_t> computeStrides(ArrayRef<int64_t> sizes);

/// Return a vector containing llvm::zip of v1 and v2 multiplied elementwise.
SmallVector<int64_t> computeElementwiseMul(ArrayRef<int64_t> v1,
                                           ArrayRef<int64_t> v2);

/// Compute and return the multi-dimensional integral ratio of `subShape` to
/// the trailing dimensions of `shape`. This represents how many times
/// `subShape` fits within `shape`.
/// If integral division is not possible, return std::nullopt.
/// The trailing `subShape.size()` entries of both shapes are assumed (and
/// enforced) to only contain noonnegative values.
///
/// Examples:
///   - shapeRatio({3, 5, 8}, {2, 5, 2}) returns {3, 2, 1}.
///   - shapeRatio({3, 8}, {2, 5, 2}) returns std::nullopt (subshape has higher
///     rank).
///   - shapeRatio({42, 2, 10, 32}, {2, 5, 2}) returns {42, 1, 2, 16} which is
///     derived as {42(leading shape dim), 2/2, 10/5, 32/2}.
///   - shapeRatio({42, 2, 11, 32}, {2, 5, 2}) returns std::nullopt  which is
///     derived as {42(leading shape dim), 2/2, 11/5(not divisible), 32/2}.
std::optional<SmallVector<int64_t>>
computeShapeRatio(ArrayRef<int64_t> shape, ArrayRef<int64_t> subShape);

/// Return the number of elements of basis (i.e. the max linear index).
/// Return `0` if `basis` is empty.
int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis);

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

/// Helper method to apply to inverse a permutation.
SmallVector<int64_t> invertPermutationVector(ArrayRef<int64_t> permutation);

/// Method to check if an interchange vector is a permutation.
bool isPermutationVector(ArrayRef<int64_t> interchange);

/// Helper that returns a subset of `arrayAttr` as a vector of int64_t.
SmallVector<int64_t> getI64SubArray(ArrayAttr arrayAttr, unsigned dropFront = 0,
                                    unsigned dropBack = 0);

/// Computes and returns linearized affine expression w.r.t. `basis`.
mlir::AffineExpr getLinearAffineExpr(ArrayRef<int64_t> basis, mlir::Builder &b);

/// Given the strides in the dimension space, returns the affine expressions for
/// vector-space offsets in each dimension for a de-linearized index.
SmallVector<mlir::AffineExpr>
getDelinearizedAffineExpr(ArrayRef<int64_t> strides, mlir::Builder &b);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_INDEXINGUTILS_H
