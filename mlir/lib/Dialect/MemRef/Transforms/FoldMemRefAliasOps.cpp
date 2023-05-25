//===- FoldMemRefAliasOps.cpp - Fold memref alias ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass folds loading/storing from/to subview ops into
// loading/storing from/to the original memref.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fold-memref-alias-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FOLDMEMREFALIASOPS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Given the 'indices' of a load/store operation where the memref is a result
/// of a expand_shape op, returns the indices w.r.t to the source memref of the
/// expand_shape op. For example
///
/// %0 = ... : memref<12x42xf32>
/// %1 = memref.expand_shape %0 [[0, 1], [2]]
///    : memref<12x42xf32> into memref<2x6x42xf32>
/// %2 = load %1[%i1, %i2, %i3] : memref<2x6x42xf32
///
/// could be folded into
///
/// %2 = load %0[6 * i1 + i2, %i3] :
///          memref<12x42xf32>
static LogicalResult
resolveSourceIndicesExpandShape(Location loc, PatternRewriter &rewriter,
                                memref::ExpandShapeOp expandShapeOp,
                                ValueRange indices,
                                SmallVectorImpl<Value> &sourceIndices) {
  MLIRContext *ctx = rewriter.getContext();
  for (ArrayRef<int64_t> groups : expandShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    int64_t groupSize = groups.size();

    // Construct the expression for the index value w.r.t to expand shape op
    // source corresponding the indices wrt to expand shape op result.
    SmallVector<int64_t> sizes(groupSize);
    for (int64_t i = 0; i < groupSize; ++i)
      sizes[i] = expandShapeOp.getResultType().getDimSize(groups[i]);
    SmallVector<int64_t> suffixProduct = computeSuffixProduct(sizes);
    SmallVector<AffineExpr> dims(groupSize);
    bindDimsList(ctx, MutableArrayRef{dims});
    AffineExpr srcIndexExpr = linearize(ctx, dims, suffixProduct);

    /// Apply permutation and create AffineApplyOp.
    SmallVector<OpFoldResult> dynamicIndices(groupSize);
    for (int64_t i = 0; i < groupSize; i++)
      dynamicIndices[i] = indices[groups[i]];

    // Creating maximally folded and composd affine.apply composes better with
    // other transformations without interleaving canonicalization passes.
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc,
        AffineMap::get(/*numDims=*/groupSize,
                       /*numSymbols=*/0, srcIndexExpr),
        dynamicIndices);
    sourceIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
  return success();
}

/// Given the 'indices' of a load/store operation where the memref is a result
/// of a collapse_shape op, returns the indices w.r.t to the source memref of
/// the collapse_shape op. For example
///
/// %0 = ... : memref<2x6x42xf32>
/// %1 = memref.collapse_shape %0 [[0, 1], [2]]
///    : memref<2x6x42xf32> into memref<12x42xf32>
/// %2 = load %1[%i1, %i2] : memref<12x42xf32>
///
/// could be folded into
///
/// %2 = load %0[%i1 / 6, %i1 % 6, %i2] :
///          memref<2x6x42xf32>
static LogicalResult
resolveSourceIndicesCollapseShape(Location loc, PatternRewriter &rewriter,
                                  memref::CollapseShapeOp collapseShapeOp,
                                  ValueRange indices,
                                  SmallVectorImpl<Value> &sourceIndices) {
  int64_t cnt = 0;
  SmallVector<Value> tmp(indices.size());
  SmallVector<OpFoldResult> dynamicIndices;
  for (ArrayRef<int64_t> groups : collapseShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    dynamicIndices.push_back(indices[cnt++]);
    int64_t groupSize = groups.size();

    // Calculate suffix product for all collapse op source dimension sizes.
    SmallVector<int64_t> sizes(groupSize);
    for (int64_t i = 0; i < groupSize; ++i)
      sizes[i] = collapseShapeOp.getSrcType().getDimSize(groups[i]);
    SmallVector<int64_t> suffixProduct = computeSuffixProduct(sizes);

    // Derive the index values along all dimensions of the source corresponding
    // to the index wrt to collapsed shape op output.
    auto d0 = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, suffixProduct);

    // Construct the AffineApplyOp for each delinearizingExpr.
    for (int64_t i = 0; i < groupSize; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc,
          AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
                         delinearizingExprs[i]),
          dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
    dynamicIndices.clear();
  }
  if (collapseShapeOp.getReassociationIndices().empty()) {
    auto zeroAffineMap = rewriter.getConstantAffineMap(0);
    int64_t srcRank =
        cast<MemRefType>(collapseShapeOp.getViewSource().getType()).getRank();
    for (int64_t i = 0; i < srcRank; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, zeroAffineMap, dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
  }
  return success();
}

/// Helpers to access the memref operand for each op.
template <typename LoadOrStoreOpTy>
static Value getMemRefOperand(LoadOrStoreOpTy op) {
  return op.getMemref();
}

static Value getMemRefOperand(vector::TransferReadOp op) {
  return op.getSource();
}

static Value getMemRefOperand(nvgpu::LdMatrixOp op) {
  return op.getSrcMemref();
}

static Value getMemRefOperand(vector::LoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::TransferWriteOp op) {
  return op.getSource();
}

static Value getMemRefOperand(gpu::SubgroupMmaLoadMatrixOp op) {
  return op.getSrcMemref();
}

static Value getMemRefOperand(gpu::SubgroupMmaStoreMatrixOp op) {
  return op.getDstMemref();
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Merges subview operation with load/transferRead operation.
template <typename OpTy>
class LoadOpOfSubViewOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges expand_shape operation with load/transferRead operation.
template <typename OpTy>
class LoadOpOfExpandShapeOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges collapse_shape operation with load/transferRead operation.
template <typename OpTy>
class LoadOpOfCollapseShapeOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges subview operation with store/transferWriteOp operation.
template <typename OpTy>
class StoreOpOfSubViewOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges expand_shape operation with store/transferWriteOp operation.
template <typename OpTy>
class StoreOpOfExpandShapeOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges collapse_shape operation with store/transferWriteOp operation.
template <typename OpTy>
class StoreOpOfCollapseShapeOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;
};

/// Folds subview(subview(x)) to a single subview(x).
class SubViewOfSubViewFolder : public OpRewritePattern<memref::SubViewOp> {
public:
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subView,
                                PatternRewriter &rewriter) const override {
    auto srcSubView = subView.getSource().getDefiningOp<memref::SubViewOp>();
    if (!srcSubView)
      return failure();

    // TODO: relax unit stride assumption.
    if (!subView.hasUnitStride()) {
      return rewriter.notifyMatchFailure(subView, "requires unit strides");
    }
    if (!srcSubView.hasUnitStride()) {
      return rewriter.notifyMatchFailure(srcSubView, "requires unit strides");
    }

    // Resolve sizes according to dropped dims.
    SmallVector<OpFoldResult> resolvedSizes;
    llvm::SmallBitVector srcDroppedDims = srcSubView.getDroppedDims();
    affine::resolveSizesIntoOpWithSizes(srcSubView.getMixedSizes(),
                                        subView.getMixedSizes(), srcDroppedDims,
                                        resolvedSizes);

    // Resolve offsets according to source offsets and strides.
    SmallVector<Value> resolvedOffsets;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, subView.getLoc(), srcSubView.getMixedOffsets(),
        srcSubView.getMixedStrides(), srcDroppedDims, subView.getMixedOffsets(),
        resolvedOffsets);

    // Replace original op.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        subView, subView.getType(), srcSubView.getSource(),
        getAsOpFoldResult(resolvedOffsets), resolvedSizes,
        srcSubView.getMixedStrides());

    return success();
  }
};

/// Folds nvgpu.device_async_copy subviews into the copy itself. This pattern
/// is folds subview on src and dst memref of the copy.
class NvgpuAsyncCopyOpSubViewOpFolder final
    : public OpRewritePattern<nvgpu::DeviceAsyncCopyOp> {
public:
  using OpRewritePattern<nvgpu::DeviceAsyncCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nvgpu::DeviceAsyncCopyOp copyOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

static SmallVector<Value>
calculateExpandedAccessIndices(AffineMap affineMap,
                               const SmallVector<Value> &indices, Location loc,
                               PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> indicesOfr(llvm::to_vector(
      llvm::map_range(indices, [](Value v) -> OpFoldResult { return v; })));
  SmallVector<Value> expandedIndices;
  for (unsigned i = 0, e = affineMap.getNumResults(); i < e; i++) {
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, affineMap.getSubMap({i}), indicesOfr);
    expandedIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
  return expandedIndices;
}

template <typename XferOp>
static LogicalResult
preconditionsFoldSubViewOpImpl(RewriterBase &rewriter, XferOp xferOp,
                               memref::SubViewOp subviewOp) {
  static_assert(
      !llvm::is_one_of<vector::TransferReadOp, vector::TransferWriteOp>::value,
      "must be a vector transfer op");
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp, "out of bounds transfer dim");
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "masked transfer");
  if (!subviewOp.hasUnitStride()) {
    return rewriter.notifyMatchFailure(
        xferOp, "non-1 stride subview, need to track strides in folded memref");
  }
  return success();
}

static LogicalResult preconditionsFoldSubViewOp(RewriterBase &rewriter,
                                                Operation *op,
                                                memref::SubViewOp subviewOp) {
  return success();
}

static LogicalResult preconditionsFoldSubViewOp(RewriterBase &rewriter,
                                                vector::TransferReadOp readOp,
                                                memref::SubViewOp subviewOp) {
  return preconditionsFoldSubViewOpImpl(rewriter, readOp, subviewOp);
}

static LogicalResult preconditionsFoldSubViewOp(RewriterBase &rewriter,
                                                vector::TransferWriteOp writeOp,
                                                memref::SubViewOp subviewOp) {
  return preconditionsFoldSubViewOpImpl(rewriter, writeOp, subviewOp);
}

template <typename OpTy>
LogicalResult LoadOpOfSubViewOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto subViewOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::SubViewOp>();

  if (!subViewOp)
    return rewriter.notifyMatchFailure(loadOp, "not a subview producer");

  LogicalResult preconditionResult =
      preconditionsFoldSubViewOp(rewriter, loadOp, subViewOp);
  if (failed(preconditionResult))
    return preconditionResult;

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp =
          dyn_cast<affine::AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, loadOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), indices,
      sourceIndices);

  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](affine::AffineLoadOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
            loadOp, subViewOp.getSource(), sourceIndices);
      })
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, subViewOp.getSource(), sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            op, op.getType(), subViewOp.getSource(), sourceIndices);
      })
      .Case([&](vector::TransferReadOp op) {
        rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
            op, op.getVectorType(), subViewOp.getSource(), sourceIndices,
            AffineMapAttr::get(expandDimsToRank(
                op.getPermutationMap(), subViewOp.getSourceType().getRank(),
                subViewOp.getDroppedDims())),
            op.getPadding(), /*mask=*/Value(), op.getInBoundsAttr());
      })
      .Case([&](gpu::SubgroupMmaLoadMatrixOp op) {
        rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(
            op, op.getType(), subViewOp.getSource(), sourceIndices,
            op.getLeadDimension(), op.getTransposeAttr());
      })
      .Case([&](nvgpu::LdMatrixOp op) {
        rewriter.replaceOpWithNewOp<nvgpu::LdMatrixOp>(
            op, op.getType(), subViewOp.getSource(), sourceIndices,
            op.getTranspose(), op.getNumTiles());
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult LoadOpOfExpandShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto expandShapeOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::ExpandShapeOp>();

  if (!expandShapeOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp =
          dyn_cast<affine::AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesExpandShape(
          loadOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case<affine::AffineLoadOp, memref::LoadOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(
            loadOp, expandShapeOp.getViewSource(), sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult LoadOpOfCollapseShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto collapseShapeOp = getMemRefOperand(loadOp)
                             .template getDefiningOp<memref::CollapseShapeOp>();

  if (!collapseShapeOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp =
          dyn_cast<affine::AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesCollapseShape(
          loadOp.getLoc(), rewriter, collapseShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case<affine::AffineLoadOp, memref::LoadOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(
            loadOp, collapseShapeOp.getViewSource(), sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult StoreOpOfSubViewOpFolder<OpTy>::matchAndRewrite(
    OpTy storeOp, PatternRewriter &rewriter) const {
  auto subViewOp =
      getMemRefOperand(storeOp).template getDefiningOp<memref::SubViewOp>();

  if (!subViewOp)
    return rewriter.notifyMatchFailure(storeOp, "not a subview producer");

  LogicalResult preconditionResult =
      preconditionsFoldSubViewOp(rewriter, storeOp, subViewOp);
  if (failed(preconditionResult))
    return preconditionResult;

  SmallVector<Value> indices(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineStoreOp =
          dyn_cast<affine::AffineStoreOp>(storeOp.getOperation())) {
    AffineMap affineMap = affineStoreOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, storeOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, storeOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(), indices,
      sourceIndices);

  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case([&](affine::AffineStoreOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
            op, op.getValue(), subViewOp.getSource(), sourceIndices);
      })
      .Case([&](memref::StoreOp op) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            op, op.getValue(), subViewOp.getSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::TransferWriteOp op) {
        rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
            op, op.getValue(), subViewOp.getSource(), sourceIndices,
            AffineMapAttr::get(expandDimsToRank(
                op.getPermutationMap(), subViewOp.getSourceType().getRank(),
                subViewOp.getDroppedDims())),
            op.getInBoundsAttr());
      })
      .Case([&](gpu::SubgroupMmaStoreMatrixOp op) {
        rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
            op, op.getSrc(), subViewOp.getSource(), sourceIndices,
            op.getLeadDimension(), op.getTransposeAttr());
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult StoreOpOfExpandShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy storeOp, PatternRewriter &rewriter) const {
  auto expandShapeOp =
      getMemRefOperand(storeOp).template getDefiningOp<memref::ExpandShapeOp>();

  if (!expandShapeOp)
    return failure();

  SmallVector<Value> indices(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineStoreOp =
          dyn_cast<affine::AffineStoreOp>(storeOp.getOperation())) {
    AffineMap affineMap = affineStoreOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, storeOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesExpandShape(
          storeOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case<affine::AffineStoreOp, memref::StoreOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(storeOp, storeOp.getValue(),
                                                  expandShapeOp.getViewSource(),
                                                  sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult StoreOpOfCollapseShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy storeOp, PatternRewriter &rewriter) const {
  auto collapseShapeOp = getMemRefOperand(storeOp)
                             .template getDefiningOp<memref::CollapseShapeOp>();

  if (!collapseShapeOp)
    return failure();

  SmallVector<Value> indices(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineStoreOp =
          dyn_cast<affine::AffineStoreOp>(storeOp.getOperation())) {
    AffineMap affineMap = affineStoreOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, storeOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesCollapseShape(
          storeOp.getLoc(), rewriter, collapseShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case<affine::AffineStoreOp, memref::StoreOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(
            storeOp, storeOp.getValue(), collapseShapeOp.getViewSource(),
            sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

LogicalResult NvgpuAsyncCopyOpSubViewOpFolder::matchAndRewrite(
    nvgpu::DeviceAsyncCopyOp copyOp, PatternRewriter &rewriter) const {

  LLVM_DEBUG(DBGS() << "copyOp       : " << copyOp << "\n");

  auto srcSubViewOp =
      copyOp.getSrc().template getDefiningOp<memref::SubViewOp>();
  auto dstSubViewOp =
      copyOp.getDst().template getDefiningOp<memref::SubViewOp>();

  if (!(srcSubViewOp || dstSubViewOp))
    return rewriter.notifyMatchFailure(copyOp, "does not use subview ops for "
                                               "source or destination");

  // If the source is a subview, we need to resolve the indices.
  SmallVector<Value> srcindices(copyOp.getSrcIndices().begin(),
                                copyOp.getSrcIndices().end());
  SmallVector<Value> foldedSrcIndices(srcindices);

  if (srcSubViewOp) {
    LLVM_DEBUG(DBGS() << "srcSubViewOp : " << srcSubViewOp << "\n");
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, copyOp.getLoc(), srcSubViewOp.getMixedOffsets(),
        srcSubViewOp.getMixedStrides(), srcSubViewOp.getDroppedDims(),
        srcindices, foldedSrcIndices);
  }

  // If the destination is a subview, we need to resolve the indices.
  SmallVector<Value> dstindices(copyOp.getDstIndices().begin(),
                                copyOp.getDstIndices().end());
  SmallVector<Value> foldedDstIndices(dstindices);

  if (dstSubViewOp) {
    LLVM_DEBUG(DBGS() << "dstSubViewOp : " << dstSubViewOp << "\n");
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, copyOp.getLoc(), dstSubViewOp.getMixedOffsets(),
        dstSubViewOp.getMixedStrides(), dstSubViewOp.getDroppedDims(),
        dstindices, foldedDstIndices);
  }

  // Replace the copy op with a new copy op that uses the source and destination
  // of the subview.
  rewriter.replaceOpWithNewOp<nvgpu::DeviceAsyncCopyOp>(
      copyOp, nvgpu::DeviceAsyncTokenType::get(copyOp.getContext()),
      (dstSubViewOp ? dstSubViewOp.getSource() : copyOp.getDst()),
      foldedDstIndices,
      (srcSubViewOp ? srcSubViewOp.getSource() : copyOp.getSrc()),
      foldedSrcIndices, copyOp.getDstElements(), copyOp.getSrcElements(),
      copyOp.getBypassL1Attr());

  return success();
}

void memref::populateFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadOpOfSubViewOpFolder<affine::AffineLoadOp>,
               LoadOpOfSubViewOpFolder<memref::LoadOp>,
               LoadOpOfSubViewOpFolder<nvgpu::LdMatrixOp>,
               LoadOpOfSubViewOpFolder<vector::LoadOp>,
               LoadOpOfSubViewOpFolder<vector::TransferReadOp>,
               LoadOpOfSubViewOpFolder<gpu::SubgroupMmaLoadMatrixOp>,
               StoreOpOfSubViewOpFolder<affine::AffineStoreOp>,
               StoreOpOfSubViewOpFolder<memref::StoreOp>,
               StoreOpOfSubViewOpFolder<vector::TransferWriteOp>,
               StoreOpOfSubViewOpFolder<gpu::SubgroupMmaStoreMatrixOp>,
               LoadOpOfExpandShapeOpFolder<affine::AffineLoadOp>,
               LoadOpOfExpandShapeOpFolder<memref::LoadOp>,
               StoreOpOfExpandShapeOpFolder<affine::AffineStoreOp>,
               StoreOpOfExpandShapeOpFolder<memref::StoreOp>,
               LoadOpOfCollapseShapeOpFolder<affine::AffineLoadOp>,
               LoadOpOfCollapseShapeOpFolder<memref::LoadOp>,
               StoreOpOfCollapseShapeOpFolder<affine::AffineStoreOp>,
               StoreOpOfCollapseShapeOpFolder<memref::StoreOp>,
               SubViewOfSubViewFolder, NvgpuAsyncCopyOpSubViewOpFolder>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct FoldMemRefAliasOpsPass final
    : public memref::impl::FoldMemRefAliasOpsBase<FoldMemRefAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

void FoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> memref::createFoldMemRefAliasOpsPass() {
  return std::make_unique<FoldMemRefAliasOpsPass>();
}
