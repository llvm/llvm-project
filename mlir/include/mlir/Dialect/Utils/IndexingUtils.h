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

//===----------------------------------------------------------------------===//
// Utils that operate on static integer values.
//===----------------------------------------------------------------------===//

/// Given a set of sizes, return the suffix product.
///
/// When applied to slicing, this is the calculation needed to derive the
/// strides (i.e. the number of linear indices to skip along the (k-1) most
/// minor dimensions to get the next k-slice).
///
/// This is the basis to linearize an n-D offset confined to `[0 ... sizes]`.
///
/// Assuming `sizes` is `[s0, .. sn]`, return the vector<int64_t>
///   `[s1 * ... * sn, s2 * ... * sn, ..., sn, 1]`.
///
/// `sizes` elements are asserted to be non-negative.
///
/// Return an empty vector if `sizes` is empty.
SmallVector<int64_t> computeSuffixProduct(ArrayRef<int64_t> sizes);
inline SmallVector<int64_t> computeStrides(ArrayRef<int64_t> sizes) {
  return computeSuffixProduct(sizes);
}

/// Return a vector containing llvm::zip_equal(v1, v2) multiplied elementwise.
///
/// Return an empty vector if `v1` and `v2` are empty.
SmallVector<int64_t> computeElementwiseMul(ArrayRef<int64_t> v1,
                                           ArrayRef<int64_t> v2);

/// Self-explicit.
int64_t computeSum(ArrayRef<int64_t> basis);

/// Self-explicit.
int64_t computeProduct(ArrayRef<int64_t> basis);

/// Return the number of elements of basis (i.e. the max linear index).
/// Return `0` if `basis` is empty.
///
/// `basis` elements are asserted to be non-negative.
///
/// Return `0` if `basis` is empty.
inline int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  return computeProduct(basis);
}

/// Return the linearized index of 'offsets' w.r.t. 'basis'.
///
/// `basis` elements are asserted to be non-negative.
int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension space,
/// return the vector-space offsets in each dimension for a de-linearized index.
/// `strides` elements are asserted to be non-negative.
///
/// Let `li = linearIndex`, assuming `strides` are `[s0, .. sn]`, return the
/// vector of int64_t
///   `[li % s0, (li / s0) % s1, ..., (li / s0 / .. / sn-1) % sn]`
SmallVector<int64_t> delinearize(int64_t linearIndex,
                                 ArrayRef<int64_t> strides);

/// Return the multi-dimensional integral ratio of `subShape` to the trailing
/// dimensions of `shape`. This represents how many times `subShape` fits
/// within `shape`. If integral division is not possible, return std::nullopt.
/// The trailing `subShape.size()` entries of both shapes are assumed (and
/// enforced) to only contain non-negative values.
///
/// Examples:
///   - shapeRatio({3, 5, 8}, {2, 5, 2}) returns {3, 2, 1}.
///   - shapeRatio({3, 8}, {2, 5, 2}) returns std::nullopt (subshape has
///   higher
///     rank).
///   - shapeRatio({42, 2, 10, 32}, {2, 5, 2}) returns {42, 1, 2, 16} which is
///     derived as {42(leading shape dim), 2/2, 10/5, 32/2}.
///   - shapeRatio({42, 2, 11, 32}, {2, 5, 2}) returns std::nullopt  which is
///     derived as {42(leading shape dim), 2/2, 11/5(not divisible), 32/2}.
std::optional<SmallVector<int64_t>>
computeShapeRatio(ArrayRef<int64_t> shape, ArrayRef<int64_t> subShape);

//===----------------------------------------------------------------------===//
// Utils that operate on AffineExpr.
//===----------------------------------------------------------------------===//

/// Given a set of sizes, return the suffix product.
///
/// When applied to slicing, this is the calculation needed to derive the
/// strides (i.e. the number of linear indices to skip along the (k-1) most
/// minor dimensions to get the next k-slice).
///
/// This is the basis to linearize an n-D offset confined to `[0 ... sizes]`.
///
/// Assuming `sizes` is `[s0, .. sn]`, return the vector<AffineExpr>
///   `[s1 * ... * sn, s2 * ... * sn, ..., sn, 1]`.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// `sizes` elements are expected to bind to non-negative values.
///
/// Return an empty vector if `sizes` is empty.
SmallVector<AffineExpr> computeSuffixProduct(ArrayRef<AffineExpr> sizes);
inline SmallVector<AffineExpr> computeStrides(ArrayRef<AffineExpr> sizes) {
  return computeSuffixProduct(sizes);
}

/// Return a vector containing llvm::zip_equal(v1, v2) multiplied elementwise.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// Return an empty vector if `v1` and `v2` are empty.
SmallVector<AffineExpr> computeElementwiseMul(ArrayRef<AffineExpr> v1,
                                              ArrayRef<AffineExpr> v2);

/// Self-explicit.
AffineExpr computeSum(MLIRContext *ctx, ArrayRef<AffineExpr> basis);

/// Self-explicit.
AffineExpr computeProduct(MLIRContext *ctx, ArrayRef<AffineExpr> basis);

/// Return the number of elements of basis (i.e. the max linear index).
/// Return `0` if `basis` is empty.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that
/// result in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide
/// by an AffineDimExpr).
///
/// `basis` elements are expected to bind to non-negative values.
///
/// Return the `0` AffineConstantExpr if `basis` is empty.
inline AffineExpr computeMaxLinearIndex(MLIRContext *ctx,
                                        ArrayRef<AffineExpr> basis) {
  return computeProduct(ctx, basis);
}

/// Return the linearized index of 'offsets' w.r.t. 'basis'.
///
/// Assuming `offsets` is `[o0, .. on]` and `basis` is `[b0, .. bn]`, return the
/// AffineExpr `o0 * b0 + .. + on * bn`.
///
/// It is the caller's responsibility to pass proper AffineExpr kind that result
/// in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide by an
/// AffineDimExpr).
///
/// `basis` elements are expected to bind to non-negative values.
AffineExpr linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                     ArrayRef<AffineExpr> basis);
AffineExpr linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                     ArrayRef<int64_t> basis);

/// Given the strides together with a linear index in the dimension space,
/// return the vector-space offsets in each dimension for a de-linearized index.
///
/// Let `li = linearIndex`, assuming `strides` are `[s0, .. sn]`, return the
/// vector of AffineExpr
///   `[li % s0, (li / s0) % s1, ..., (li / s0 / .. / sn-1) % sn]`
///
/// It is the caller's responsibility to pass proper AffineExpr kind that result
/// in valid AffineExpr (i.e. cannot multiply 2 AffineDimExpr or divide by an
/// AffineDimExpr).
///
/// `strides` elements are expected to bind to non-negative values.
SmallVector<AffineExpr> delinearize(AffineExpr linearIndex,
                                    ArrayRef<AffineExpr> strides);
SmallVector<AffineExpr> delinearize(AffineExpr linearIndex,
                                    ArrayRef<int64_t> strides);

//===----------------------------------------------------------------------===//
// Permutation utils.
//===----------------------------------------------------------------------===//

/// Apply the permutation defined by `permutation` to `inVec`.
/// Element `i` in `inVec` is mapped to location `j = permutation[i]`.
/// E.g.: for an input vector `inVec = ['a', 'b', 'c']` and a permutation
/// vector `permutation = [2, 0, 1]`, this function leaves `inVec = ['c', 'a',
/// 'b']`.
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

/// Return a permutation vector of size permSize that would result in moving
/// positions into desiredPositions.
///
/// For example, permSize == 5, positions = {2, 4}, desiredPositions = {1, 0}
/// would result in a {4, 2, 0, 1, 3} permutation vector.
SmallVector<int64_t>
computePermutationVector(int64_t permSize, ArrayRef<int64_t> positions,
                         ArrayRef<int64_t> desiredPositions);

/// Helper to return a subset of `arrayAttr` as a vector of int64_t.
// TODO: Port everything relevant to DenseArrayAttr and drop this util.
SmallVector<int64_t> getI64SubArray(ArrayAttr arrayAttr, unsigned dropFront = 0,
                                    unsigned dropBack = 0);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_INDEXINGUTILS_H
