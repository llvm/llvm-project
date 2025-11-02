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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
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
#define GEN_PASS_DEF_FOLDMEMREFALIASOPSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Helpers to access the memref operand for each op.
template <typename LoadOrStoreOpTy>
static Value getMemRefOperand(LoadOrStoreOpTy op) {
  return op.getMemref();
}

static Value getMemRefOperand(vector::TransferReadOp op) {
  return op.getBase();
}

static Value getMemRefOperand(nvgpu::LdMatrixOp op) {
  return op.getSrcMemref();
}

static Value getMemRefOperand(vector::LoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::StoreOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedLoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedStoreOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::TransferWriteOp op) {
  return op.getBase();
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
class NVGPUAsyncCopyOpSubViewOpFolder final
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
      .Case([&](vector::MaskedLoadOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
            op, op.getType(), subViewOp.getSource(), sourceIndices,
            op.getMask(), op.getPassThru());
      })
      .Case([&](vector::TransferReadOp op) {
        rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
            op, op.getVectorType(), subViewOp.getSource(), sourceIndices,
            AffineMapAttr::get(expandDimsToRank(
                op.getPermutationMap(), subViewOp.getSourceType().getRank(),
                subViewOp.getDroppedDims())),
            op.getPadding(), op.getMask(), op.getInBoundsAttr());
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
      .DefaultUnreachable("unexpected operation");
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
  // memref.load and affine.load guarantee that indexes start inbounds
  // while the vector operations don't. This impacts if our linearization
  // is `disjoint`
  if (failed(resolveSourceIndicesExpandShape(
          loadOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices,
          isa<affine::AffineLoadOp, memref::LoadOp>(loadOp.getOperation()))))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](affine::AffineLoadOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
            loadOp, expandShapeOp.getViewSource(), sourceIndices);
      })
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::MaskedLoadOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getMask(), op.getPassThru());
      })
      .DefaultUnreachable("unexpected operation");
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
      .Case([&](affine::AffineLoadOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
            loadOp, collapseShapeOp.getViewSource(), sourceIndices);
      })
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, collapseShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            op, op.getType(), collapseShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
      })
      .Case([&](vector::MaskedLoadOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
            op, op.getType(), collapseShapeOp.getViewSource(), sourceIndices,
            op.getMask(), op.getPassThru());
      })
      .DefaultUnreachable("unexpected operation");
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
            op.getMask(), op.getInBoundsAttr());
      })
      .Case([&](vector::StoreOp op) {
        rewriter.replaceOpWithNewOp<vector::StoreOp>(
            op, op.getValueToStore(), subViewOp.getSource(), sourceIndices);
      })
      .Case([&](vector::MaskedStoreOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
            op, subViewOp.getSource(), sourceIndices, op.getMask(),
            op.getValueToStore());
      })
      .Case([&](gpu::SubgroupMmaStoreMatrixOp op) {
        rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
            op, op.getSrc(), subViewOp.getSource(), sourceIndices,
            op.getLeadDimension(), op.getTransposeAttr());
      })
      .DefaultUnreachable("unexpected operation");
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
  // memref.store and affine.store guarantee that indexes start inbounds
  // while the vector operations don't. This impacts if our linearization
  // is `disjoint`
  if (failed(resolveSourceIndicesExpandShape(
          storeOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices,
          isa<affine::AffineStoreOp, memref::StoreOp>(storeOp.getOperation()))))
    return failure();
  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case([&](affine::AffineStoreOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
            storeOp, op.getValueToStore(), expandShapeOp.getViewSource(),
            sourceIndices);
      })
      .Case([&](memref::StoreOp op) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            storeOp, op.getValueToStore(), expandShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::StoreOp op) {
        rewriter.replaceOpWithNewOp<vector::StoreOp>(
            op, op.getValueToStore(), expandShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::MaskedStoreOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
            op, expandShapeOp.getViewSource(), sourceIndices, op.getMask(),
            op.getValueToStore());
      })
      .DefaultUnreachable("unexpected operation");
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
      .Case([&](affine::AffineStoreOp op) {
        rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
            storeOp, op.getValueToStore(), collapseShapeOp.getViewSource(),
            sourceIndices);
      })
      .Case([&](memref::StoreOp op) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            storeOp, op.getValueToStore(), collapseShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::StoreOp op) {
        rewriter.replaceOpWithNewOp<vector::StoreOp>(
            op, op.getValueToStore(), collapseShapeOp.getViewSource(),
            sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::MaskedStoreOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
            op, collapseShapeOp.getViewSource(), sourceIndices, op.getMask(),
            op.getValueToStore());
      })
      .DefaultUnreachable("unexpected operation");
  return success();
}

LogicalResult NVGPUAsyncCopyOpSubViewOpFolder::matchAndRewrite(
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
               LoadOpOfSubViewOpFolder<vector::MaskedLoadOp>,
               LoadOpOfSubViewOpFolder<vector::TransferReadOp>,
               LoadOpOfSubViewOpFolder<gpu::SubgroupMmaLoadMatrixOp>,
               StoreOpOfSubViewOpFolder<affine::AffineStoreOp>,
               StoreOpOfSubViewOpFolder<memref::StoreOp>,
               StoreOpOfSubViewOpFolder<vector::TransferWriteOp>,
               StoreOpOfSubViewOpFolder<vector::StoreOp>,
               StoreOpOfSubViewOpFolder<vector::MaskedStoreOp>,
               StoreOpOfSubViewOpFolder<gpu::SubgroupMmaStoreMatrixOp>,
               LoadOpOfExpandShapeOpFolder<affine::AffineLoadOp>,
               LoadOpOfExpandShapeOpFolder<memref::LoadOp>,
               LoadOpOfExpandShapeOpFolder<vector::LoadOp>,
               LoadOpOfExpandShapeOpFolder<vector::MaskedLoadOp>,
               StoreOpOfExpandShapeOpFolder<affine::AffineStoreOp>,
               StoreOpOfExpandShapeOpFolder<memref::StoreOp>,
               StoreOpOfExpandShapeOpFolder<vector::StoreOp>,
               StoreOpOfExpandShapeOpFolder<vector::MaskedStoreOp>,
               LoadOpOfCollapseShapeOpFolder<affine::AffineLoadOp>,
               LoadOpOfCollapseShapeOpFolder<memref::LoadOp>,
               LoadOpOfCollapseShapeOpFolder<vector::LoadOp>,
               LoadOpOfCollapseShapeOpFolder<vector::MaskedLoadOp>,
               StoreOpOfCollapseShapeOpFolder<affine::AffineStoreOp>,
               StoreOpOfCollapseShapeOpFolder<memref::StoreOp>,
               StoreOpOfCollapseShapeOpFolder<vector::StoreOp>,
               StoreOpOfCollapseShapeOpFolder<vector::MaskedStoreOp>,
               SubViewOfSubViewFolder, NVGPUAsyncCopyOpSubViewOpFolder>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct FoldMemRefAliasOpsPass final
    : public memref::impl::FoldMemRefAliasOpsPassBase<FoldMemRefAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

void FoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
