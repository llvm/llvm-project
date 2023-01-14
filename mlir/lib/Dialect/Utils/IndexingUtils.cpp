//===- IndexingUtils.cpp - Helpers related to index computations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/IndexingUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>
#include <optional>

using namespace mlir;

SmallVector<int64_t> mlir::computeStrides(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> strides(sizes.size(), 1);
  for (int64_t r = strides.size() - 2; r >= 0; --r)
    strides[r] = strides[r + 1] * sizes[r + 1];
  return strides;
}

SmallVector<int64_t> mlir::computeElementwiseMul(ArrayRef<int64_t> v1,
                                                 ArrayRef<int64_t> v2) {
  SmallVector<int64_t> result;
  for (auto it : llvm::zip(v1, v2))
    result.push_back(std::get<0>(it) * std::get<1>(it));
  return result;
}

Optional<SmallVector<int64_t>>
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

int64_t mlir::linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
}

llvm::SmallVector<int64_t> mlir::delinearize(ArrayRef<int64_t> sliceStrides,
                                             int64_t index) {
  int64_t rank = sliceStrides.size();
  SmallVector<int64_t> vectorOffsets(rank);
  for (int64_t r = 0; r < rank; ++r) {
    assert(sliceStrides[r] > 0);
    vectorOffsets[r] = index / sliceStrides[r];
    index %= sliceStrides[r];
  }
  return vectorOffsets;
}

int64_t mlir::computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  return std::accumulate(basis.begin(), basis.end(), 1,
                         std::multiplies<int64_t>());
}

llvm::SmallVector<int64_t>
mlir::invertPermutationVector(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inversion(permutation.size());
  for (const auto &pos : llvm::enumerate(permutation)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}

bool mlir::isPermutationVector(ArrayRef<int64_t> interchange) {
  llvm::SmallDenseSet<int64_t, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val))
      return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

llvm::SmallVector<int64_t> mlir::getI64SubArray(ArrayAttr arrayAttr,
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

mlir::AffineExpr mlir::getLinearAffineExpr(ArrayRef<int64_t> basis,
                                           mlir::Builder &b) {
  AffineExpr resultExpr = b.getAffineDimExpr(0);
  resultExpr = resultExpr * basis[0];
  for (unsigned i = 1; i < basis.size(); i++)
    resultExpr = resultExpr + b.getAffineDimExpr(i) * basis[i];
  return resultExpr;
}

llvm::SmallVector<mlir::AffineExpr>
mlir::getDelinearizedAffineExpr(mlir::ArrayRef<int64_t> strides, Builder &b) {
  AffineExpr resultExpr = b.getAffineDimExpr(0);
  int64_t rank = strides.size();
  SmallVector<AffineExpr> vectorOffsets(rank);
  vectorOffsets[0] = resultExpr.floorDiv(strides[0]);
  resultExpr = resultExpr % strides[0];
  for (unsigned i = 1; i < rank; i++) {
    vectorOffsets[i] = resultExpr.floorDiv(strides[i]);
    resultExpr = resultExpr % strides[i];
  }
  return vectorOffsets;
}
