//===- ExtractAddressCmoputations.cpp - Extract address computations  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This transformation pass rewrites memory access operations with offsets into
/// accesses through a subview and without any offset on the access operation
/// itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for the `access base[off0...]`
//  => `access (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

/// Returns true if every index is zero.
static bool hasAllZeroIndices(ValueRange indices) {
  return llvm::all_of(getAsOpFoldResult(indices), isZeroInteger);
}

/// Get the remaining size in each dimension - that is, the size of the memref
/// dimension minus the index. Used to preserve in_bounds behavior for
/// transfer_read/write.
static SmallVector<OpFoldResult> getRemainingSizes(RewriterBase &rewriter,
                                                   Location loc,
                                                   Value srcMemRef,
                                                   ValueRange indices) {
  auto extractStridedMetadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, srcMemRef);
  SmallVector<OpFoldResult> srcSizes =
      extractStridedMetadataOp.getConstifiedMixedSizes();
  SmallVector<OpFoldResult> mixedIndices = getAsOpFoldResult(indices);
  SmallVector<OpFoldResult> finalSizes;

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);

  for (auto [srcSize, index] : llvm::zip_equal(srcSizes, mixedIndices)) {
    finalSizes.push_back(affine::makeComposedFoldedAffineApply(
        rewriter, loc, s0 - s1, {srcSize, index}));
  }
  return finalSizes;
}

/// Get the sizes needed to create a valid subview for an indexed access.
/// The trailing dimensions are sized using the accessed shape, taking
/// the minimum of that shape's size and what's available along the relevant
/// memref dimension as a dynamic value if the memref is dynamically shaped
/// (so as to avoid subviews that exceed the bounds of the relevant memref
/// dimension). If the operation accesses a dynamic number of elements along
/// the dimension, the size of the subview will always be the remaining element
/// count along the dimension.
static SmallVector<OpFoldResult>
getIndexedAccessViewSizes(RewriterBase &rewriter,
                          memref::IndexedAccessOpInterface op) {
  TypedValue<MemRefType> srcMemRef = op.getAccessedMemref();
  assert(srcMemRef && "expected indexed access with a memref");

  MemRefType srcType = srcMemRef.getType();
  int64_t srcRank = srcType.getRank();
  SmallVector<int64_t> accessedShape = op.getAccessedShape();
  int64_t accessedRank = static_cast<int64_t>(accessedShape.size());
  assert(accessedRank <= srcRank &&
         "can't access more dimensions than a memref has");

  SmallVector<OpFoldResult> indices = getAsOpFoldResult(op.getIndices());
  int64_t firstAccessedDim = srcRank - accessedRank;

  Location loc = op.getLoc();
  SmallVector<OpFoldResult> viewSizes(srcRank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> srcSizes;
  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  AffineExpr cst = rewriter.getAffineSymbolExpr(2);

  auto ensureSrcSizes = [&]() {
    if (srcSizes.empty()) {
      auto extractStridedMetadataOp =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, srcMemRef);
      srcSizes = extractStridedMetadataOp.getConstifiedMixedSizes();
    }
  };

  for (int64_t accessedDim : llvm::seq<int64_t>(0, accessedRank)) {
    int64_t accessedSize = accessedShape[accessedDim];
    int64_t dim = firstAccessedDim + accessedDim;
    if (!ShapedType::isDynamic(accessedSize)) {
      int64_t srcDimSize = srcType.getDimSize(dim);
      if (!ShapedType::isDynamic(srcDimSize) || accessedSize == 1) {
        viewSizes[dim] = rewriter.getIndexAttr(accessedSize);
        continue;
      }
      ensureSrcSizes();
      viewSizes[dim] = affine::makeComposedFoldedAffineMin(
          rewriter, loc,
          AffineMap::get(/*dimCount=*/0, /*symbolCount=*/3, {s0 - s1, cst},
                         rewriter.getContext()),
          {srcSizes[dim], indices[dim], rewriter.getIndexAttr(accessedSize)});
    } else {
      ensureSrcSizes();
      viewSizes[dim] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 - s1, {srcSizes[dim], indices[dim]});
    }
  }
  return viewSizes;
}

static memref::SubViewOp createSubviewForAccess(RewriterBase &rewriter,
                                                Location loc, Value srcMemRef,
                                                ValueRange indices,
                                                ArrayRef<OpFoldResult> sizes) {
  int64_t rank = cast<MemRefType>(srcMemRef.getType()).getRank();
  SmallVector<OpFoldResult> mixedIndices = getAsOpFoldResult(indices);
  SmallVector<OpFoldResult> ones(rank, rewriter.getIndexAttr(1));

  return memref::SubViewOp::create(rewriter, loc, /*source=*/srcMemRef,
                                   /*offsets=*/mixedIndices,
                                   /*sizes=*/sizes, /*strides=*/ones);
}

static SmallVector<Value> getZeroIndices(RewriterBase &rewriter, Location loc,
                                         int64_t rank) {
  if (rank == 0)
    return {};
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  return SmallVector<Value>(rank, zero);
}

/// Rewrite an indexed access op so that all its indices are zeros.
/// E.g., %res = indexed_access %base[%off0]...[%offN]
/// =>
/// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
/// %res = indexed_access %new_base[0,..,0] :
///    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
struct IndexedAccessOpRewriter final
    : OpInterfaceRewritePattern<memref::IndexedAccessOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedAccessOpInterface op,
                                PatternRewriter &rewriter) const override {
    TypedValue<MemRefType> srcMemRef = op.getAccessedMemref();
    if (!srcMemRef)
      return rewriter.notifyMatchFailure(op, "source is not a memref");

    int64_t rank = srcMemRef.getType().getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "0-D accesses don't need rewriting");

    if (static_cast<int64_t>(op.getAccessedShape().size()) > rank)
      return rewriter.notifyMatchFailure(
          op, "can't access more dimensions than a memref has");

    if (!op.hasInboundsIndices())
      return rewriter.notifyMatchFailure(op, "indices may be out of bounds");

    // If the access already has only zeros as indices there is nothing
    // to do.
    if (hasAllZeroIndices(op.getIndices()))
      return rewriter.notifyMatchFailure(
          op, "no computation to extract: offsets are 0s");

    SmallVector<OpFoldResult> subviewSizes =
        getIndexedAccessViewSizes(rewriter, op);

    Location loc = op.getLoc();
    auto subview = createSubviewForAccess(rewriter, loc, srcMemRef,
                                          op.getIndices(), subviewSizes);
    SmallVector<Value> zeros = getZeroIndices(rewriter, loc, rank);

    std::optional<SmallVector<Value>> newValues =
        op.updateMemrefAndIndices(rewriter, subview.getResult(), zeros);
    if (newValues)
      rewriter.replaceOp(op, *newValues);
    return success();
  }
};

/// Rewrite a vector transfer op so that all its indices are zeros.
struct TransferOpRewriter final
    : OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(VectorTransferOpInterface op,
                                PatternRewriter &rewriter) const override {
    Value srcMemRef = op.getBase();
    auto srcType = dyn_cast<MemRefType>(srcMemRef.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "source is not a memref");

    int64_t rank = srcType.getRank();

    if (rank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "0-D accesses don't need rewriting");

    if (hasAllZeroIndices(op.getIndices()))
      return rewriter.notifyMatchFailure(
          op, "no computation to extract: offsets are 0s");

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(op.getIndices());
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    // Approximate sizes needed so we can test the general case of the
    // replacement we're planning to do - this can be tightened up later when
    // this pattern is extended to reason about in_bounds, which dimensions are
    // accessed, etc.
    SmallVector<OpFoldResult> approximateSizes(
        rank, rewriter.getIndexAttr(ShapedType::kDynamic));
    MemRefType subviewType = memref::SubViewOp::inferResultType(
        srcType, offsets, approximateSizes, strides);
    if (!subviewType)
      return rewriter.notifyMatchFailure(op, "failed to infer subview type");

    AffineMap permutationMap = op.getPermutationMap();
    if (failed(op.mayUpdateStartingPosition(subviewType, permutationMap)))
      return rewriter.notifyMatchFailure(op,
                                         "failed op-specific preconditions");

    SmallVector<OpFoldResult> sizes =
        getRemainingSizes(rewriter, loc, srcMemRef, op.getIndices());
    auto subview = createSubviewForAccess(rewriter, loc, srcMemRef,
                                          op.getIndices(), sizes);
    SmallVector<Value> zeros = getZeroIndices(rewriter, loc, rank);

    op.updateStartingPosition(rewriter, subview.getResult(), zeros,
                              AffineMapAttr::get(permutationMap));
    return success();
  }
};
} // namespace

void memref::populateExtractAddressComputationsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<IndexedAccessOpRewriter, TransferOpRewriter>(
      patterns.getContext());
}
