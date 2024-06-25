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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
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
  // Record the rewriter context for constructing ops later.
  MLIRContext *ctx = rewriter.getContext();

  // Capture expand_shape's input dimensions as `SmallVector<OpFoldResult>`.
  // This is done for the purpose of inferring the output shape via
  // `inferExpandOutputShape` which will in turn be used for suffix product
  // calculation later.
  SmallVector<OpFoldResult> srcShape;
  MemRefType srcType = expandShapeOp.getSrcType();

  for (int64_t i = 0, e = srcType.getRank(); i < e; ++i) {
    if (srcType.isDynamicDim(i)) {
      srcShape.push_back(
          rewriter.create<memref::DimOp>(loc, expandShapeOp.getSrc(), i)
              .getResult());
    } else {
      srcShape.push_back(rewriter.getIndexAttr(srcType.getShape()[i]));
    }
  }

  auto outputShape = inferExpandShapeOutputShape(
      rewriter, loc, expandShapeOp.getResultType(),
      expandShapeOp.getReassociationIndices(), srcShape);
  if (!outputShape.has_value())
    return failure();

  // Traverse all reassociation groups to determine the appropriate indices
  // corresponding to each one of them post op folding.
  for (ArrayRef<int64_t> groups : expandShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    // Flag to indicate the presence of dynamic dimensions in current
    // reassociation group.
    int64_t groupSize = groups.size();

    // Group output dimensions utilized in this reassociation group for suffix
    // product calculation.
    SmallVector<OpFoldResult> sizesVal(groupSize);
    for (int64_t i = 0; i < groupSize; ++i) {
      sizesVal[i] = (*outputShape)[groups[i]];
    }

    // Calculate suffix product of relevant output dimension sizes.
    SmallVector<OpFoldResult> suffixProduct =
        memref::computeSuffixProductIRBlock(loc, rewriter, sizesVal);

    // Create affine expression variables for dimensions and symbols in the
    // newly constructed affine map.
    SmallVector<AffineExpr> dims(groupSize), symbols(groupSize);
    bindDimsList<AffineExpr>(ctx, dims);
    bindSymbolsList<AffineExpr>(ctx, symbols);

    // Linearize binded dimensions and symbols to construct the resultant
    // affine expression for this indice.
    AffineExpr srcIndexExpr = linearize(ctx, dims, symbols);

    // Record the load index corresponding to each dimension in the
    // reassociation group. These are later supplied as operands to the affine
    // map used for calulating relevant index post op folding.
    SmallVector<OpFoldResult> dynamicIndices(groupSize);
    for (int64_t i = 0; i < groupSize; i++)
      dynamicIndices[i] = indices[groups[i]];

    // Supply suffix product results followed by load op indices as operands
    // to the map.
    SmallVector<OpFoldResult> mapOperands;
    llvm::append_range(mapOperands, suffixProduct);
    llvm::append_range(mapOperands, dynamicIndices);

    // Creating maximally folded and composed affine.apply composes better
    // with other transformations without interleaving canonicalization
    // passes.
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc,
        AffineMap::get(/*numDims=*/groupSize,
                       /*numSymbols=*/groupSize, /*expression=*/srcIndexExpr),
        mapOperands);

    // Push index value in the op post folding corresponding to this
    // reassociation group.
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

    // Calculate suffix product for all collapse op source dimension sizes
    // except the most major one of each group.
    // We allow the most major source dimension to be dynamic but enforce all
    // others to be known statically.
    SmallVector<int64_t> sizes(groupSize, 1);
    for (int64_t i = 1; i < groupSize; ++i) {
      sizes[i] = collapseShapeOp.getSrcType().getDimSize(groups[i]);
      if (sizes[i] == ShapedType::kDynamic)
        return failure();
    }
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

static Value getMemRefOperand(vector::StoreOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedLoadOp op) { return op.getBase(); }

static Value getMemRefOperand(vector::MaskedStoreOp op) { return op.getBase(); }

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
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
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
