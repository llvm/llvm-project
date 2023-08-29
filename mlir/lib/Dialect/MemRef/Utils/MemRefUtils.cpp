//===- MemRefUtils.cpp - Utilities to support the MemRef dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace memref {

bool isStaticShapeAndContiguousRowMajor(MemRefType type) {
  if (!type.hasStaticShape())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;

  // MemRef is contiguous if outer dimensions are size-1 and inner
  // dimensions have unit strides.
  int64_t runningStride = 1;
  int64_t curDim = strides.size() - 1;
  // Finds all inner dimensions with unit strides.
  while (curDim >= 0 && strides[curDim] == runningStride) {
    runningStride *= type.getDimSize(curDim);
    --curDim;
  }

  // Check if other dimensions are size-1.
  while (curDim >= 0 && type.getDimSize(curDim) == 1) {
    --curDim;
  }

  // All dims are unit-strided or size-1.
  return curDim < 0;
}

std::pair<LinearizedMemRefInfo, OpFoldResult> getLinearizedMemRefOffsetAndSize(
    OpBuilder &builder, Location loc, int srcBits, int dstBits,
    OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> indices) {
  unsigned sourceRank = sizes.size();
  assert(sizes.size() == strides.size() &&
         "expected as many sizes as strides for a memref");
  SmallVector<OpFoldResult> indicesVec = llvm::to_vector(indices);
  if (indices.empty())
    indicesVec.resize(sourceRank, builder.getIndexAttr(0));
  assert(indicesVec.size() == strides.size() &&
         "expected as many indices as rank of memref");

  // Create the affine symbols and values for linearization.
  SmallVector<AffineExpr> symbols(2 * sourceRank);
  bindSymbolsList(builder.getContext(), MutableArrayRef{symbols});
  AffineExpr addMulMap = builder.getAffineConstantExpr(0);
  AffineExpr mulMap = builder.getAffineConstantExpr(1);

  SmallVector<OpFoldResult> offsetValues(2 * sourceRank);
  SmallVector<OpFoldResult> sizeValues(sourceRank);

  for (unsigned i = 0; i < sourceRank; ++i) {
    unsigned offsetIdx = 2 * i;
    addMulMap = addMulMap + symbols[offsetIdx] * symbols[offsetIdx + 1];
    offsetValues[offsetIdx] = indicesVec[i];
    offsetValues[offsetIdx + 1] = strides[i];

    mulMap = mulMap * symbols[i];
  }

  // Adjust linearizedIndices, size and offset by the scale factor (dstBits /
  // srcBits).
  int64_t scaler = dstBits / srcBits;
  addMulMap = addMulMap.floorDiv(scaler);
  mulMap = mulMap.floorDiv(scaler);

  OpFoldResult linearizedIndices = affine::makeComposedFoldedAffineApply(
      builder, loc, addMulMap, offsetValues);
  OpFoldResult linearizedSize =
      affine::makeComposedFoldedAffineApply(builder, loc, mulMap, sizes);

  // Adjust baseOffset by the scale factor (dstBits / srcBits).
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  OpFoldResult adjustBaseOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(scaler), {offset});

  return {{adjustBaseOffset, linearizedSize}, linearizedIndices};
}

LinearizedMemRefInfo
getLinearizedMemRefOffsetAndSize(OpBuilder &builder, Location loc, int srcBits,
                                 int dstBits, OpFoldResult offset,
                                 ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> strides(sizes.size());
  if (!sizes.empty()) {
    strides.back() = builder.getIndexAttr(1);
    AffineExpr s0, s1;
    bindSymbols(builder.getContext(), s0, s1);
    for (int index = sizes.size() - 1; index > 0; --index) {
      strides[index - 1] = affine::makeComposedFoldedAffineApply(
          builder, loc, s0 * s1,
          ArrayRef<OpFoldResult>{strides[index], sizes[index]});
    }
  }

  LinearizedMemRefInfo linearizedMemRefInfo;
  std::tie(linearizedMemRefInfo, std::ignore) =
      getLinearizedMemRefOffsetAndSize(builder, loc, srcBits, dstBits, offset,
                                       sizes, strides);
  return linearizedMemRefInfo;
}

} // namespace memref
} // namespace mlir
