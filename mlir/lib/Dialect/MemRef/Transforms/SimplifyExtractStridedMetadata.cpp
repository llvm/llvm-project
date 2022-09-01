//===- SimplifyExtractStridedMetadata.cpp - Simplify this operation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass simplifies extract_strided_metadata(other_op(memref) to
/// extract_strided_metadata(memref) when it is possible to express the effect
// of other_op using affine apply on the results of
// extract_strided_metadata(memref).
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_SIMPLIFYEXTRACTSTRIDEDMETADATA
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir
using namespace mlir;

namespace {
/// Replace `baseBuffer, offset, sizes, strides =
///              extract_strided_metadata(subview(memref, subOffset,
///                                               subSizes, subStrides))`
/// With
///
/// \verbatim
/// baseBuffer, baseOffset, baseSizes, baseStrides =
///     extract_strided_metadata(memref)
/// strides#i = baseStrides#i * subSizes#i
/// offset = baseOffset + sum(subOffset#i * strides#i)
/// sizes = subSizes
/// \endverbatim
///
/// In other words, get rid of the subview in that expression and canonicalize
/// on its effects on the offset, the sizes, and the strides using affine apply.
struct ExtractStridedMetadataOpSubviewFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
public:
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto subview = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    // Build a plain extract_strided_metadata(memref) from
    // extract_strided_metadata(subview(memref)).
    Location origLoc = op.getLoc();
    IndexType indexType = rewriter.getIndexType();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    unsigned sourceRank = sourceType.getRank();
    SmallVector<Type> sizeStrideTypes(sourceRank, indexType);

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(
            origLoc, op.getBaseBuffer().getType(), indexType, sizeStrideTypes,
            sizeStrideTypes, source);

    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;

    bool hasKnownStridesAndOffset =
        succeeded(getStridesAndOffset(sourceType, sourceStrides, sourceOffset));
    (void)hasKnownStridesAndOffset;
    assert(hasKnownStridesAndOffset &&
           "getStridesAndOffset must work on valid subviews");

    // Compute the new strides and offset from the base strides and offset:
    // newStride#i = baseStride#i * subStride#i
    // offset = baseOffset + sum(subOffsets#i * newStrides#i)
    SmallVector<OpFoldResult> strides;
    SmallVector<OpFoldResult> subStrides = subview.getMixedStrides();
    auto origStrides = newExtractStridedMetadata.getStrides();

    // Hold the affine symbols and values for the computation of the offset.
    SmallVector<OpFoldResult> values(3 * sourceRank + 1);
    SmallVector<AffineExpr> symbols(3 * sourceRank + 1);

    detail::bindSymbolsList(rewriter.getContext(), symbols);
    AffineExpr expr = symbols.front();
    values[0] = ShapedType::isDynamicStrideOrOffset(sourceOffset)
                    ? getAsOpFoldResult(newExtractStridedMetadata.getOffset())
                    : rewriter.getIndexAttr(sourceOffset);
    SmallVector<OpFoldResult> subOffsets = subview.getMixedOffsets();

    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    for (unsigned i = 0; i < sourceRank; ++i) {
      // Compute the stride.
      OpFoldResult origStride =
          ShapedType::isDynamicStrideOrOffset(sourceStrides[i])
              ? origStrides[i]
              : OpFoldResult(rewriter.getIndexAttr(sourceStrides[i]));
      strides.push_back(makeComposedFoldedAffineApply(
          rewriter, origLoc, s0 * s1, {subStrides[i], origStride}));

      // Build up the computation of the offset.
      unsigned baseIdxForDim = 1 + 3 * i;
      unsigned subOffsetForDim = baseIdxForDim;
      unsigned subStrideForDim = baseIdxForDim + 1;
      unsigned origStrideForDim = baseIdxForDim + 2;
      expr = expr + symbols[subOffsetForDim] * symbols[subStrideForDim] *
                        symbols[origStrideForDim];
      values[subOffsetForDim] = subOffsets[i];
      values[subStrideForDim] = subStrides[i];
      values[origStrideForDim] = origStride;
    }

    // Compute the offset.
    OpFoldResult finalOffset =
        makeComposedFoldedAffineApply(rewriter, origLoc, expr, values);

    SmallVector<Value> results;
    // The final result is  <baseBuffer, offset, sizes, strides>.
    // Thus we need 1 + 1 + subview.getRank() + subview.getRank(), to hold all
    // the values.
    auto subType = subview.getType().cast<MemRefType>();
    unsigned subRank = subType.getRank();
    // Properly size the array so that we can do random insertions
    // at the right indices.
    // We do that to populate the non-dropped sizes and strides in one go.
    results.resize_for_overwrite(subRank * 2 + 2);

    results[0] = newExtractStridedMetadata.getBaseBuffer();
    results[1] =
        getValueOrCreateConstantIndexOp(rewriter, origLoc, finalOffset);

    // The sizes of the final type are defined directly by the input sizes of
    // the subview.
    // Moreover subviews can drop some dimensions, some strides and sizes may
    // not end up in the final <base, offset, sizes, strides> value that we are
    // replacing.
    // Do the filtering here.
    SmallVector<OpFoldResult> subSizes = subview.getMixedSizes();
    const unsigned sizeStartIdx = 2;
    const unsigned strideStartIdx = sizeStartIdx + subRank;
    unsigned insertedDims = 0;
    llvm::SmallBitVector droppedDims = subview.getDroppedDims();
    for (unsigned i = 0; i < sourceRank; ++i) {
      if (droppedDims.test(i))
        continue;

      results[sizeStartIdx + insertedDims] =
          getValueOrCreateConstantIndexOp(rewriter, origLoc, subSizes[i]);
      results[strideStartIdx + insertedDims] =
          getValueOrCreateConstantIndexOp(rewriter, origLoc, strides[i]);
      ++insertedDims;
    }
    assert(insertedDims == subRank &&
           "Should have populated all the values at this point");

    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

void memref::populateSimplifyExtractStridedMetadataOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractStridedMetadataOpSubviewFolder>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct SimplifyExtractStridedMetadataPass final
    : public memref::impl::SimplifyExtractStridedMetadataBase<
          SimplifyExtractStridedMetadataPass> {
  void runOnOperation() override;
};

} // namespace

void SimplifyExtractStridedMetadataPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateSimplifyExtractStridedMetadataOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                     std::move(patterns));
}

std::unique_ptr<Pass> memref::createSimplifyExtractStridedMetadataPass() {
  return std::make_unique<SimplifyExtractStridedMetadataPass>();
}
