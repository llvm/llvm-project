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

std::pair<Value, Value>
getLinearizeMemRefAndOffset(Location loc, MemRefType sourceType, int srcBits,
                            int dstBits, SmallVector<Value> indices,
                            memref::ExtractStridedMetadataOp stridedMetadata,
                            OpBuilder &builder) {
  auto srcElementType = sourceType.getElementType();
  unsigned sourceRank = indices.size();

  Value baseBuffer = stridedMetadata.getBaseBuffer();
  SmallVector<Value> baseSizes = stridedMetadata.getSizes();
  SmallVector<Value> baseStrides = stridedMetadata.getStrides();
  Value baseOffset = stridedMetadata.getOffset();
  assert(indices.size() == baseStrides.size());

  // Create the affine symbols and values for linearization.
  SmallVector<AffineExpr> symbols(2 * sourceRank + 2);
  bindSymbolsList(builder.getContext(), MutableArrayRef{symbols});
  symbols[0] = builder.getAffineSymbolExpr(0);
  AffineExpr addMulMap = symbols.front();
  AffineExpr mulMap = symbols.front();

  SmallVector<OpFoldResult> offsetValues(2 * sourceRank + 2);
  offsetValues[0] = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> sizeValues(sourceRank + 1);
  sizeValues[0] = builder.getIndexAttr(1);

  for (unsigned i = 0; i < sourceRank; ++i) {
    unsigned offsetIdx = 2 * i + 1;
    addMulMap = addMulMap + symbols[offsetIdx] * symbols[offsetIdx + 1];
    offsetValues[offsetIdx] = indices[i];
    offsetValues[offsetIdx + 1] = baseStrides[i];

    unsigned sizeIdx = i + 1;
    mulMap = mulMap * symbols[sizeIdx];
    sizeValues[sizeIdx] = baseSizes[i];
  }

  // Adjust linearizedOffset by the scale factor (dstBits / srcBits).
  OpFoldResult scaler = builder.getIndexAttr(dstBits / srcBits);
  AffineExpr scaledAddMulMap = addMulMap.floorDiv(symbols.back());
  offsetValues.back() = scaler;

  OpFoldResult linearizedOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, scaledAddMulMap, offsetValues);
  OpFoldResult linearizedSize =
      affine::makeComposedFoldedAffineApply(builder, loc, mulMap, sizeValues);

  // Adjust baseOffset by the scale factor (dstBits / srcBits).
  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  OpFoldResult adjustBaseOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(s1), {baseOffset, scaler});

  // Flatten n-D MemRef to 1-D MemRef.
  std::optional<int64_t> stride =
      getConstantIntValue(stridedMetadata.getConstifiedMixedStrides().back());
  auto layoutAttr =
      StridedLayoutAttr::get(sourceType.getContext(), ShapedType::kDynamic,
                             {stride ? stride.value() : ShapedType::kDynamic});
  int64_t staticShape = sourceType.hasStaticShape()
                            ? sourceType.getNumElements()
                            : ShapedType::kDynamic;
  auto flattenMemrefType = MemRefType::get(
      staticShape, srcElementType, layoutAttr, sourceType.getMemorySpace());

  auto reinterpret = builder.create<memref::ReinterpretCastOp>(
      loc, flattenMemrefType, baseBuffer,
      getValueOrCreateConstantIndexOp(builder, loc, adjustBaseOffset),
      getValueOrCreateConstantIndexOp(builder, loc, linearizedSize),
      baseStrides.back());

  return std::make_pair(reinterpret, getValueOrCreateConstantIndexOp(
                                         builder, loc, linearizedOffset));
}

} // namespace memref
} // namespace mlir
