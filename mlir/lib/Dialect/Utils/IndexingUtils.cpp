//===- IndexingUtils.cpp - Helpers related to index computations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>
#include <optional>

using namespace mlir;

template <typename ExprType>
SmallVector<ExprType> computeSuffixProductImpl(ArrayRef<ExprType> sizes,
                                               ExprType unit) {
  if (sizes.empty())
    return {};
  SmallVector<ExprType> strides(sizes.size(), unit);
  for (int64_t r = strides.size() - 2; r >= 0; --r)
    strides[r] = strides[r + 1] * sizes[r + 1];
  return strides;
}

template <typename ExprType>
SmallVector<ExprType> computeElementwiseMulImpl(ArrayRef<ExprType> v1,
                                                ArrayRef<ExprType> v2) {
  // Early exit if both are empty, let zip_equal fail if only 1 is empty.
  if (v1.empty() && v2.empty())
    return {};
  SmallVector<ExprType> result;
  for (auto it : llvm::zip_equal(v1, v2))
    result.push_back(std::get<0>(it) * std::get<1>(it));
  return result;
}

template <typename ExprType>
ExprType linearizeImpl(ArrayRef<ExprType> offsets, ArrayRef<ExprType> basis,
                       ExprType zero) {
  assert(offsets.size() == basis.size());
  ExprType linearIndex = zero;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex = linearIndex + offsets[idx] * basis[idx];
  return linearIndex;
}

template <typename ExprType, typename DivOpTy>
SmallVector<ExprType> delinearizeImpl(ExprType linearIndex,
                                      ArrayRef<ExprType> strides,
                                      DivOpTy divOp) {
  int64_t rank = strides.size();
  SmallVector<ExprType> offsets(rank);
  for (int64_t r = 0; r < rank; ++r) {
    offsets[r] = divOp(linearIndex, strides[r]);
    linearIndex = linearIndex % strides[r];
  }
  return offsets;
}

//===----------------------------------------------------------------------===//
// Utils that operate on static integer values.
//===----------------------------------------------------------------------===//

SmallVector<int64_t> mlir::computeSuffixProduct(ArrayRef<int64_t> sizes) {
  assert(llvm::all_of(sizes, [](int64_t s) { return s >= 0; }) &&
         "sizes must be nonnegative");
  int64_t unit = 1;
  return ::computeSuffixProductImpl(sizes, unit);
}

SmallVector<int64_t> mlir::computeElementwiseMul(ArrayRef<int64_t> v1,
                                                 ArrayRef<int64_t> v2) {
  return computeElementwiseMulImpl(v1, v2);
}

int64_t mlir::computeSum(ArrayRef<int64_t> basis) {
  assert(llvm::all_of(basis, [](int64_t s) { return s > 0; }) &&
         "basis must be nonnegative");
  if (basis.empty())
    return 0;
  return std::accumulate(basis.begin(), basis.end(), 1, std::plus<int64_t>());
}

int64_t mlir::computeProduct(ArrayRef<int64_t> basis) {
  assert(llvm::all_of(basis, [](int64_t s) { return s > 0; }) &&
         "basis must be nonnegative");
  if (basis.empty())
    return 0;
  return std::accumulate(basis.begin(), basis.end(), 1,
                         std::multiplies<int64_t>());
}

int64_t mlir::linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(llvm::all_of(basis, [](int64_t s) { return s > 0; }) &&
         "basis must be nonnegative");
  int64_t zero = 0;
  return linearizeImpl(offsets, basis, zero);
}

SmallVector<int64_t> mlir::delinearize(int64_t linearIndex,
                                       ArrayRef<int64_t> strides) {
  assert(llvm::all_of(strides, [](int64_t s) { return s > 0; }) &&
         "strides must be nonnegative");
  return delinearizeImpl(linearIndex, strides,
                         [](int64_t e1, int64_t e2) { return e1 / e2; });
}

std::optional<SmallVector<int64_t>>
mlir::computeShapeRatio(ArrayRef<int64_t> shape, ArrayRef<int64_t> subShape) {
  if (shape.size() < subShape.size())
    return std::nullopt;
  assert(llvm::all_of(shape, [](int64_t s) { return s > 0; }) &&
         "shape must be nonnegative");
  assert(llvm::all_of(subShape, [](int64_t s) { return s > 0; }) &&
         "subShape must be nonnegative");

  // Starting from the end, compute the integer divisors.
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (auto [size, subSize] :
       llvm::zip(llvm::reverse(shape), llvm::reverse(subShape))) {
    // If integral division does not occur, return and let the caller decide.
    if (size % subSize != 0)
      return std::nullopt;
    result.push_back(size / subSize);
  }
  // At this point we computed the ratio (in reverse) for the common size.
  // Fill with the remaining entries from the shape (still in reverse).
  int commonSize = subShape.size();
  std::copy(shape.rbegin() + commonSize, shape.rend(),
            std::back_inserter(result));
  // Reverse again to get it back in the proper order and return.
  return SmallVector<int64_t>{result.rbegin(), result.rend()};
}

//===----------------------------------------------------------------------===//
// Utils that operate on AffineExpr.
//===----------------------------------------------------------------------===//

SmallVector<AffineExpr> mlir::computeSuffixProduct(ArrayRef<AffineExpr> sizes) {
  if (sizes.empty())
    return {};
  AffineExpr unit = getAffineConstantExpr(1, sizes.front().getContext());
  return ::computeSuffixProductImpl(sizes, unit);
}

SmallVector<AffineExpr> mlir::computeElementwiseMul(ArrayRef<AffineExpr> v1,
                                                    ArrayRef<AffineExpr> v2) {
  return computeElementwiseMulImpl(v1, v2);
}

AffineExpr mlir::computeSum(MLIRContext *ctx, ArrayRef<AffineExpr> basis) {
  if (basis.empty())
    return getAffineConstantExpr(0, ctx);
  return std::accumulate(basis.begin(), basis.end(),
                         getAffineConstantExpr(0, ctx),
                         std::plus<AffineExpr>());
}

AffineExpr mlir::computeProduct(MLIRContext *ctx, ArrayRef<AffineExpr> basis) {
  if (basis.empty())
    return getAffineConstantExpr(1, ctx);
  return std::accumulate(basis.begin(), basis.end(),
                         getAffineConstantExpr(1, ctx),
                         std::multiplies<AffineExpr>());
}

AffineExpr mlir::linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                           ArrayRef<AffineExpr> basis) {
  AffineExpr zero = getAffineConstantExpr(0, ctx);
  return linearizeImpl(offsets, basis, zero);
}

AffineExpr mlir::linearize(MLIRContext *ctx, ArrayRef<AffineExpr> offsets,
                           ArrayRef<int64_t> basis) {

  return linearize(ctx, offsets, getAffineConstantExprs(basis, ctx));
}

SmallVector<AffineExpr> mlir::delinearize(AffineExpr linearIndex,
                                          ArrayRef<AffineExpr> strides) {
  return delinearizeImpl(
      linearIndex, strides,
      [](AffineExpr e1, AffineExpr e2) { return e1.floorDiv(e2); });
}

SmallVector<AffineExpr> mlir::delinearize(AffineExpr linearIndex,
                                          ArrayRef<int64_t> strides) {
  MLIRContext *ctx = linearIndex.getContext();
  return delinearize(linearIndex, getAffineConstantExprs(strides, ctx));
}

//===----------------------------------------------------------------------===//
// Permutation utils.
//===----------------------------------------------------------------------===//

SmallVector<int64_t>
mlir::invertPermutationVector(ArrayRef<int64_t> permutation) {
  assert(llvm::all_of(permutation, [](int64_t s) { return s >= 0; }) &&
         "permutation must be non-negative");
  SmallVector<int64_t> inversion(permutation.size());
  for (const auto &pos : llvm::enumerate(permutation)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}

bool mlir::isIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (auto i : llvm::seq<int64_t>(0, permutation.size()))
    if (permutation[i] != i)
      return false;
  return true;
}

bool mlir::isPermutationVector(ArrayRef<int64_t> interchange) {
  assert(llvm::all_of(interchange, [](int64_t s) { return s >= 0; }) &&
         "permutation must be non-negative");
  llvm::SmallDenseSet<int64_t, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val))
      return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

SmallVector<int64_t>
mlir::computePermutationVector(int64_t permSize, ArrayRef<int64_t> positions,
                               ArrayRef<int64_t> desiredPositions) {
  SmallVector<int64_t> res(permSize, -1);
  DenseSet<int64_t> seen;
  for (auto [pos, desiredPos] : llvm::zip_equal(positions, desiredPositions)) {
    res[desiredPos] = pos;
    seen.insert(pos);
  }
  int64_t nextPos = 0;
  for (int64_t &entry : res) {
    if (entry != -1)
      continue;
    while (seen.contains(nextPos))
      ++nextPos;
    entry = nextPos;
    ++nextPos;
  }
  return res;
}

SmallVector<int64_t> mlir::getI64SubArray(ArrayAttr arrayAttr,
                                          unsigned dropFront,
                                          unsigned dropBack) {
  assert(arrayAttr.size() > dropFront + dropBack && "Out of bounds");
  auto range = arrayAttr.getAsRange<IntegerAttr>();
  SmallVector<int64_t> res;
  res.reserve(arrayAttr.size() - dropFront - dropBack);
  for (auto it = range.begin() + dropFront, eit = range.end() - dropBack;
       it != eit; ++it)
    res.push_back((*it).getValue().getSExtValue());
  return res;
}

// TODO: do we have any common utily for this?
static MLIRContext *getContext(OpFoldResult val) {
  assert(val && "Invalid value");
  if (auto attr = dyn_cast<Attribute>(val)) {
    return attr.getContext();
  }
  return cast<Value>(val).getContext();
}

std::pair<AffineExpr, SmallVector<OpFoldResult>>
mlir::computeLinearIndex(OpFoldResult sourceOffset,
                         ArrayRef<OpFoldResult> strides,
                         ArrayRef<OpFoldResult> indices) {
  assert(strides.size() == indices.size());
  auto sourceRank = static_cast<unsigned>(strides.size());

  // Hold the affine symbols and values for the computation of the offset.
  SmallVector<OpFoldResult> values(2 * sourceRank + 1);
  SmallVector<AffineExpr> symbols(2 * sourceRank + 1);

  bindSymbolsList(getContext(sourceOffset), MutableArrayRef{symbols});
  AffineExpr expr = symbols.front();
  values[0] = sourceOffset;

  for (unsigned i = 0; i < sourceRank; ++i) {
    // Compute the stride.
    OpFoldResult origStride = strides[i];

    // Build up the computation of the offset.
    unsigned baseIdxForDim = 1 + 2 * i;
    unsigned subOffsetForDim = baseIdxForDim;
    unsigned origStrideForDim = baseIdxForDim + 1;
    expr = expr + symbols[subOffsetForDim] * symbols[origStrideForDim];
    values[subOffsetForDim] = indices[i];
    values[origStrideForDim] = origStride;
  }

  return {expr, values};
}

std::pair<AffineExpr, SmallVector<OpFoldResult>>
mlir::computeLinearIndex(OpFoldResult sourceOffset, ArrayRef<int64_t> strides,
                         ArrayRef<Value> indices) {
  return computeLinearIndex(
      sourceOffset, getAsIndexOpFoldResult(sourceOffset.getContext(), strides),
      getAsOpFoldResult(ValueRange(indices)));
}

//===----------------------------------------------------------------------===//
// TileOffsetRange
//===----------------------------------------------------------------------===//

/// Apply left-padding by 1 to the tile shape if required.
static SmallVector<int64_t> padTileShapeToSize(ArrayRef<int64_t> tileShape,
                                               unsigned paddedSize) {
  assert(tileShape.size() <= paddedSize &&
         "expected tileShape to <= paddedSize");
  if (tileShape.size() == paddedSize)
    return to_vector(tileShape);
  SmallVector<int64_t> result(paddedSize - tileShape.size(), 1);
  llvm::append_range(result, tileShape);
  return result;
}

mlir::detail::TileOffsetRangeImpl::TileOffsetRangeImpl(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> tileShape,
    ArrayRef<int64_t> loopOrder)
    : tileShape(padTileShapeToSize(tileShape, shape.size())),
      inverseLoopOrder(invertPermutationVector(loopOrder)),
      sliceStrides(shape.size()) {
  // Divide the shape by the tile shape.
  std::optional<SmallVector<int64_t>> shapeRatio =
      mlir::computeShapeRatio(shape, tileShape);
  assert(shapeRatio && shapeRatio->size() == shape.size() &&
         "target shape does not evenly divide the original shape");
  assert(isPermutationVector(loopOrder) && loopOrder.size() == shape.size() &&
         "expected loop order to be a permutation of rank equal to outer "
         "shape");

  maxLinearIndex = mlir::computeMaxLinearIndex(*shapeRatio);
  mlir::applyPermutationToVector(*shapeRatio, loopOrder);
  sliceStrides = mlir::computeStrides(*shapeRatio);
}

SmallVector<int64_t> mlir::detail::TileOffsetRangeImpl::getStaticTileOffsets(
    int64_t linearIndex) const {
  SmallVector<int64_t> tileCoords = applyPermutation(
      delinearize(linearIndex, sliceStrides), inverseLoopOrder);
  return computeElementwiseMul(tileCoords, tileShape);
}

SmallVector<AffineExpr>
mlir::detail::TileOffsetRangeImpl::getDynamicTileOffsets(
    AffineExpr linearIndex) const {
  MLIRContext *ctx = linearIndex.getContext();
  SmallVector<AffineExpr> tileCoords = applyPermutation(
      delinearize(linearIndex, sliceStrides), inverseLoopOrder);
  return mlir::computeElementwiseMul(tileCoords,
                                     getAffineConstantExprs(tileShape, ctx));
}
