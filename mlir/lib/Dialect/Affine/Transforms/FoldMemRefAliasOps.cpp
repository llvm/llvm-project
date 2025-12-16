//===- FoldMemRefAliasOps.cpp - Fold memref alias ops for affine ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass contains affine-specif versions of the folding patterns for
// memref.expand_shape, memref.collapse_shape, and memref.subview, since
// those all need affine-specific handling that won't fit a general interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEFOLDMEMREFALIASOPS
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Given an AffineMap and a list of indices, apply the map to get the
/// underlying indices (expanding the affine map).
static void expandToUnderlyingIndices(AffineMap affineMap, ValueRange indices,
                                      Location loc, PatternRewriter &rewriter,
                                      SmallVectorImpl<Value> &result) {
  SmallVector<OpFoldResult> indicesOfr(
      llvm::map_to_vector(indices, [](Value v) -> OpFoldResult { return v; }));
  for (unsigned i = 0, e = affineMap.getNumResults(); i < e; ++i) {
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, affineMap.getSubMap({i}), indicesOfr);
    result.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {

class AffineLoadOpOfSubViewOpFolder final
    : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = loadOp.getMemref().getDefiningOp<memref::SubViewOp>();

    if (!subViewOp)
      return rewriter.notifyMatchFailure(loadOp, "not a subview producer");

    SmallVector<Value> indices;
    expandToUnderlyingIndices(loadOp.getAffineMap(), loadOp.getIndices(),
                              loadOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, loadOp.getLoc(), subViewOp.getMixedOffsets(),
        subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), indices,
        sourceIndices);

    rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, subViewOp.getSource(),
                                              sourceIndices);
    return success();
  }
};

class AffineLoadOpOfExpandShapeOpFolder final
    : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        loadOp.getMemref().getDefiningOp<memref::ExpandShapeOp>();

    if (!expandShapeOp)
      return failure();

    SmallVector<Value> indices;
    expandToUnderlyingIndices(loadOp.getAffineMap(), loadOp.getIndices(),
                              loadOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    // affine.load guarantees that indexes start inbounds, which impacts if our
    // linearization is `disjoint`.
    if (failed(memref::resolveSourceIndicesExpandShape(
            loadOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices,
            /*startsInbounds=*/true)))
      return failure();

    rewriter.replaceOpWithNewOp<AffineLoadOp>(
        loadOp, expandShapeOp.getViewSource(), sourceIndices);
    return success();
  }
};

class AffineLoadOpOfCollapseShapeOpFolder final
    : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp =
        loadOp.getMemref().getDefiningOp<memref::CollapseShapeOp>();

    if (!collapseShapeOp)
      return failure();

    SmallVector<Value> indices;
    expandToUnderlyingIndices(loadOp.getAffineMap(), loadOp.getIndices(),
                              loadOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    if (failed(memref::resolveSourceIndicesCollapseShape(
            loadOp.getLoc(), rewriter, collapseShapeOp, indices,
            sourceIndices)))
      return failure();

    rewriter.replaceOpWithNewOp<AffineLoadOp>(
        loadOp, collapseShapeOp.getViewSource(), sourceIndices);
    return success();
  }
};

class AffineStoreOpOfSubViewOpFolder final
    : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = storeOp.getMemref().getDefiningOp<memref::SubViewOp>();

    if (!subViewOp)
      return rewriter.notifyMatchFailure(storeOp, "not a subview producer");

    // For affine ops, we need to apply the map to get the "actual" indices.
    SmallVector<Value> indices;
    expandToUnderlyingIndices(storeOp.getAffineMap(), storeOp.getIndices(),
                              storeOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, storeOp.getLoc(), subViewOp.getMixedOffsets(),
        subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), indices,
        sourceIndices);

    rewriter.replaceOpWithNewOp<AffineStoreOp>(
        storeOp, storeOp.getValue(), subViewOp.getSource(), sourceIndices);
    return success();
  }
};

class AffineStoreOpOfExpandShapeOpFolder final
    : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        storeOp.getMemref().getDefiningOp<memref::ExpandShapeOp>();

    if (!expandShapeOp)
      return failure();

    SmallVector<Value> indices;
    expandToUnderlyingIndices(storeOp.getAffineMap(), storeOp.getIndices(),
                              storeOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    // affine.store guarantees that indexes start inbounds, which impacts if our
    // linearization is `disjoint`.
    if (failed(memref::resolveSourceIndicesExpandShape(
            storeOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices,
            /*startsInbounds=*/true)))
      return failure();

    rewriter.replaceOpWithNewOp<AffineStoreOp>(
        storeOp, storeOp.getValueToStore(), expandShapeOp.getViewSource(),
        sourceIndices);
    return success();
  }
};

class AffineStoreOpOfCollapseShapeOpFolder final
    : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp =
        storeOp.getMemref().getDefiningOp<memref::CollapseShapeOp>();

    if (!collapseShapeOp)
      return failure();

    // For affine ops, we need to apply the map to get the "actual" indices.
    SmallVector<Value> indices;
    expandToUnderlyingIndices(storeOp.getAffineMap(), storeOp.getIndices(),
                              storeOp.getLoc(), rewriter, indices);

    SmallVector<Value> sourceIndices;
    if (failed(memref::resolveSourceIndicesCollapseShape(
            storeOp.getLoc(), rewriter, collapseShapeOp, indices,
            sourceIndices)))
      return failure();

    rewriter.replaceOpWithNewOp<AffineStoreOp>(
        storeOp, storeOp.getValueToStore(), collapseShapeOp.getViewSource(),
        sourceIndices);
    return success();
  }
};

} // namespace

void affine::populateAffineFoldMemRefAliasOpPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<AffineLoadOpOfSubViewOpFolder, AffineLoadOpOfExpandShapeOpFolder,
           AffineLoadOpOfCollapseShapeOpFolder, AffineStoreOpOfSubViewOpFolder,
           AffineStoreOpOfExpandShapeOpFolder,
           AffineStoreOpOfCollapseShapeOpFolder>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct AffineFoldMemRefAliasOpsPass final
    : public affine::impl::AffineFoldMemRefAliasOpsBase<
          AffineFoldMemRefAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

void AffineFoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  affine::populateAffineFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
