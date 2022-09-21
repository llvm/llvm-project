//===- MergeConsecutiveInsertExtractSlicePatterns.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

/// Creates AffineExpr from `ofr`: if the OpFoldResult is a Value, creates a
/// AffineSymbolExpr and appends it to `symbols`; otherwise creates a
/// AffineConstantExpr.
static AffineExpr getAffineExpr(OpFoldResult ofr,
                                SmallVector<OpFoldResult> &symbols) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    return getAffineConstantExpr(attr.cast<IntegerAttr>().getInt(),
                                 attr.getContext());
  }
  Value v = ofr.get<Value>();
  AffineExpr expr = getAffineSymbolExpr(symbols.size(), v.getContext());
  symbols.push_back(v);
  return expr;
}

/// Builds the AffineExpr incrementally for arithmetic operations.
static AffineExpr add(AffineExpr expr, OpFoldResult ofr,
                      SmallVector<OpFoldResult> &symbols) {
  return expr + getAffineExpr(ofr, symbols);
}
static AffineExpr mul(OpFoldResult lhs, OpFoldResult rhs,
                      SmallVector<OpFoldResult> &symbols) {
  return getAffineExpr(lhs, symbols) * getAffineExpr(rhs, symbols);
}

/// Converts an AffineExpr to OpFoldResult by generating an `affine.apply`
/// op and fold it.
static OpFoldResult getOpFoldResult(OpBuilder &builder, Location loc,
                                    AffineExpr expr,
                                    SmallVector<OpFoldResult> &symbols) {
  AffineMap m = AffineMap::get(0, symbols.size(), expr);
  return makeComposedFoldedAffineApply(builder, loc, m, symbols);
}

LogicalResult tensor::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> producerOffsets,
    ArrayRef<OpFoldResult> producerSizes,
    ArrayRef<OpFoldResult> producerStrides,
    const llvm::SmallBitVector &droppedProducerDims,
    ArrayRef<OpFoldResult> consumerOffsets,
    ArrayRef<OpFoldResult> consumerSizes,
    ArrayRef<OpFoldResult> consumerStrides,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  combinedOffsets.resize(producerOffsets.size());
  combinedSizes.resize(producerOffsets.size());
  combinedStrides.resize(producerOffsets.size());
  unsigned consumerPos = 0;
  for (auto i : llvm::seq<unsigned>(0, producerOffsets.size())) {
    if (droppedProducerDims.test(i)) {
      // For dropped dims, get the values from the producer.
      combinedOffsets[i] = producerOffsets[i];
      combinedSizes[i] = producerSizes[i];
      combinedStrides[i] = producerStrides[i];
      continue;
    }
    SmallVector<OpFoldResult> offsetSymbols, strideSymbols;
    // The combined offset is computed as
    //    producer_offset + consumer_offset * producer_strides.
    combinedOffsets[i] =
        getOpFoldResult(builder, loc,
                        add(mul(consumerOffsets[consumerPos],
                                producerStrides[i], offsetSymbols),
                            producerOffsets[i], offsetSymbols),
                        offsetSymbols);
    combinedSizes[i] = consumerSizes[consumerPos];
    // The combined stride is computed as
    //    consumer_stride * producer_stride.
    combinedStrides[i] = getOpFoldResult(
        builder, loc,
        mul(consumerStrides[consumerPos], producerStrides[i], strideSymbols),
        strideSymbols);
    consumerPos++;
  }
  return success();
}

LogicalResult tensor::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    const llvm::SmallBitVector &droppedProducerDims,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  SmallVector<OpFoldResult> consumerOffsets = consumer.getMixedOffsets();
  SmallVector<OpFoldResult> consumerSizes = consumer.getMixedSizes();
  SmallVector<OpFoldResult> consumerStrides = consumer.getMixedStrides();
  SmallVector<OpFoldResult> producerOffsets = producer.getMixedOffsets();
  SmallVector<OpFoldResult> producerSizes = producer.getMixedSizes();
  SmallVector<OpFoldResult> producerStrides = producer.getMixedStrides();
  return tensor::mergeOffsetsSizesAndStrides(
      builder, loc, producerOffsets, producerSizes, producerStrides,
      droppedProducerDims, consumerOffsets, consumerSizes, consumerStrides,
      combinedOffsets, combinedSizes, combinedStrides);
}

namespace {
/// Merges consecutive tensor.extract_slice ops into one.
struct MergeConsecutiveExtractSlice : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().getDefiningOp<ExtractSliceOp>();
    if (!prevOp)
      return failure();

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(mergeOffsetsSizesAndStrides(rewriter, nextOp.getLoc(), prevOp,
                                           nextOp, prevOp.getDroppedDims(),
                                           newOffsets, newSizes, newStrides)))
      return failure();

    rewriter.replaceOpWithNewOp<ExtractSliceOp>(nextOp, nextOp.getType(),
                                                prevOp.getSource(), newOffsets,
                                                newSizes, newStrides);
    return success();
  }
};

/// Merges consecutive tensor.insert_slice ops into one.
struct MergeConsecutiveInsertSlice : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().getDefiningOp<InsertSliceOp>();
    if (!prevOp)
      return failure();

    if (!prevOp.hasUnitStride() || !nextOp.hasUnitStride())
      return failure();

    // The first insert_slice op should be rank reducing to make sure we cover
    // the full source tensor to be inserted in the second insert_slice op.
    SliceVerificationResult result =
        isRankReducedType(prevOp.getDestType(), prevOp.getSourceType());
    if (result != SliceVerificationResult::Success)
      return failure();

    // Dynamic dimensions can pass rank reducing check in the above, e.g,
    // inserting <?xf32> into <1x?x1xf32>. For such cases we cannot be certain
    // the dynamic size covers the full tensor.
    if (!prevOp.getSourceType().hasStaticShape() ||
        !prevOp.getDestType().hasStaticShape())
      return failure();

    rewriter.replaceOpWithNewOp<InsertSliceOp>(
        nextOp, prevOp.getSource(), nextOp.getDest(), nextOp.getMixedOffsets(),
        nextOp.getMixedSizes(), nextOp.getMixedStrides());
    return success();
  }
};
} // namespace

void mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns) {
  patterns.add<MergeConsecutiveExtractSlice, MergeConsecutiveInsertSlice>(
      patterns.getContext());
}
