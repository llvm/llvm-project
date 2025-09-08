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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace memref {

bool isStaticShapeAndContiguousRowMajor(MemRefType type) {
  if (!type.hasStaticShape())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
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

  SmallVector<OpFoldResult> offsetValues(2 * sourceRank);

  for (unsigned i = 0; i < sourceRank; ++i) {
    unsigned offsetIdx = 2 * i;
    addMulMap = addMulMap + symbols[offsetIdx] * symbols[offsetIdx + 1];
    offsetValues[offsetIdx] = indicesVec[i];
    offsetValues[offsetIdx + 1] = strides[i];
  }
  // Adjust linearizedIndices and size by the scale factor (dstBits / srcBits).
  int64_t scaler = dstBits / srcBits;
  OpFoldResult linearizedIndices = affine::makeComposedFoldedAffineApply(
      builder, loc, addMulMap.floorDiv(scaler), offsetValues);

  size_t symbolIndex = 0;
  SmallVector<OpFoldResult> values;
  SmallVector<AffineExpr> productExpressions;
  for (unsigned i = 0; i < sourceRank; ++i) {
    AffineExpr strideExpr = symbols[symbolIndex++];
    values.push_back(strides[i]);
    AffineExpr sizeExpr = symbols[symbolIndex++];
    values.push_back(sizes[i]);

    productExpressions.push_back((strideExpr * sizeExpr).floorDiv(scaler));
  }
  AffineMap maxMap = AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/symbolIndex, productExpressions,
      builder.getContext());
  OpFoldResult linearizedSize =
      affine::makeComposedFoldedAffineMax(builder, loc, maxMap, values);

  // Adjust baseOffset by the scale factor (dstBits / srcBits).
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  OpFoldResult adjustBaseOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(scaler), {offset});

  OpFoldResult intraVectorOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, addMulMap % scaler, offsetValues);

  return {{adjustBaseOffset, linearizedSize, intraVectorOffset},
          linearizedIndices};
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

/// Returns true if all the uses of op are not read/load.
/// There can be SubviewOp users as long as all its users are also
/// StoreOp/transfer_write. If return true it also fills out the uses, if it
/// returns false uses is unchanged.
static bool resultIsNotRead(Operation *op, std::vector<Operation *> &uses) {
  std::vector<Operation *> opUses;
  for (OpOperand &use : op->getUses()) {
    Operation *useOp = use.getOwner();
    if (isa<memref::DeallocOp>(useOp) ||
        (useOp->getNumResults() == 0 && useOp->getNumRegions() == 0 &&
         !mlir::hasEffect<MemoryEffects::Read>(useOp)) ||
        (isa<memref::SubViewOp>(useOp) && resultIsNotRead(useOp, opUses))) {
      opUses.push_back(useOp);
      continue;
    }
    return false;
  }
  llvm::append_range(uses, opUses);
  return true;
}

void eraseDeadAllocAndStores(RewriterBase &rewriter, Operation *parentOp) {
  std::vector<Operation *> opToErase;
  parentOp->walk([&](Operation *op) {
    std::vector<Operation *> candidates;
    if (isa<memref::AllocOp, memref::AllocaOp>(op) &&
        resultIsNotRead(op, candidates)) {
      llvm::append_range(opToErase, candidates);
      opToErase.push_back(op);
    }
  });

  for (Operation *op : opToErase)
    rewriter.eraseOp(op);
}

static SmallVector<OpFoldResult>
computeSuffixProductIRBlockImpl(Location loc, OpBuilder &builder,
                                ArrayRef<OpFoldResult> sizes,
                                OpFoldResult unit) {
  SmallVector<OpFoldResult> strides(sizes.size(), unit);
  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);

  for (int64_t r = strides.size() - 1; r > 0; --r) {
    strides[r - 1] = affine::makeComposedFoldedAffineApply(
        builder, loc, s0 * s1, {strides[r], sizes[r]});
  }
  return strides;
}

SmallVector<OpFoldResult>
computeSuffixProductIRBlock(Location loc, OpBuilder &builder,
                            ArrayRef<OpFoldResult> sizes) {
  OpFoldResult unit = builder.getIndexAttr(1);
  return computeSuffixProductIRBlockImpl(loc, builder, sizes, unit);
}

MemrefValue skipFullyAliasingOperations(MemrefValue source) {
  while (auto op = source.getDefiningOp()) {
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(op);
        subViewOp && subViewOp.hasZeroOffset() && subViewOp.hasUnitStride()) {
      // A `memref.subview` with an all zero offset, and all unit strides, still
      // points to the same memory.
      source = cast<MemrefValue>(subViewOp.getSource());
    } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
      // A `memref.cast` still points to the same memory.
      source = castOp.getSource();
    } else {
      return source;
    }
  }
  return source;
}

MemrefValue skipViewLikeOps(MemrefValue source) {
  while (auto op = source.getDefiningOp()) {
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(op)) {
      if (source == viewLike.getViewDest()) {
        source = cast<MemrefValue>(viewLike.getViewSource());
        continue;
      }
    }
    return source;
  }
  return source;
}

LogicalResult resolveSourceIndicesExpandShape(
    Location loc, PatternRewriter &rewriter,
    memref::ExpandShapeOp expandShapeOp, ValueRange indices,
    SmallVectorImpl<Value> &sourceIndices, bool startsInbounds) {
  SmallVector<OpFoldResult> destShape = expandShapeOp.getMixedOutputShape();

  // Traverse all reassociation groups to determine the appropriate indices
  // corresponding to each one of them post op folding.
  for (ArrayRef<int64_t> group : expandShapeOp.getReassociationIndices()) {
    assert(!group.empty() && "association indices groups cannot be empty");
    int64_t groupSize = group.size();
    if (groupSize == 1) {
      sourceIndices.push_back(indices[group[0]]);
      continue;
    }
    SmallVector<OpFoldResult> groupBasis =
        llvm::map_to_vector(group, [&](int64_t d) { return destShape[d]; });
    SmallVector<Value> groupIndices =
        llvm::map_to_vector(group, [&](int64_t d) { return indices[d]; });
    Value collapsedIndex = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, groupIndices, groupBasis, /*disjoint=*/startsInbounds);
    sourceIndices.push_back(collapsedIndex);
  }
  return success();
}

LogicalResult
resolveSourceIndicesCollapseShape(Location loc, PatternRewriter &rewriter,
                                  memref::CollapseShapeOp collapseShapeOp,
                                  ValueRange indices,
                                  SmallVectorImpl<Value> &sourceIndices) {
  // Note: collapse_shape requires a strided memref, we can do this.
  auto metadata = memref::ExtractStridedMetadataOp::create(
      rewriter, loc, collapseShapeOp.getSrc());
  SmallVector<OpFoldResult> sourceSizes = metadata.getConstifiedMixedSizes();
  for (auto [index, group] :
       llvm::zip(indices, collapseShapeOp.getReassociationIndices())) {
    assert(!group.empty() && "association indices groups cannot be empty");
    int64_t groupSize = group.size();

    if (groupSize == 1) {
      sourceIndices.push_back(index);
      continue;
    }

    SmallVector<OpFoldResult> basis =
        llvm::map_to_vector(group, [&](int64_t d) { return sourceSizes[d]; });
    auto delinearize = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, index, basis, /*hasOuterBound=*/true);
    llvm::append_range(sourceIndices, delinearize.getResults());
  }
  if (collapseShapeOp.getReassociationIndices().empty()) {
    auto zeroAffineMap = rewriter.getConstantAffineMap(0);
    int64_t srcRank =
        cast<MemRefType>(collapseShapeOp.getViewSource().getType()).getRank();
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, zeroAffineMap, ArrayRef<OpFoldResult>{});
    for (int64_t i = 0; i < srcRank; i++) {
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
  }
  return success();
}

} // namespace memref
} // namespace mlir
