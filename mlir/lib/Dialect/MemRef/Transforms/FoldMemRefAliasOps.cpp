//===- FoldMemRefAliasOps.cpp - Fold memref alias ops ---------------------===//
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

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

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

/// Deterimine if the last N indices of `reassocitaion` are trivial - that is,
/// check if they all contain exactly one dimension to collape/expand into.
static bool
hasTrivialReassociationSuffix(ArrayRef<ReassociationIndices> reassocs,
                              int64_t n) {
  if (n <= 0)
    return true;
  return llvm::all_of(
      reassocs.take_back(n),
      [&](const ReassociationIndices &indices) { return indices.size() == 1; });
}

static bool hasTrailingUnitStrides(memref::SubViewOp subview, int64_t n) {
  if (n <= 0)
    return true;
  return llvm::all_of(subview.getStaticStrides().take_back(n),
                      [](int64_t s) { return s == 1; });
}

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

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, subView.getLoc(), srcSubView, subView,
            srcSubView.getDroppedDims(), newOffsets, newSizes, newStrides)))
      return failure();

    // Replace original op.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        subView, subView.getType(), srcSubView.getSource(), newOffsets,
        newSizes, newStrides);
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

/// Merges subview operations with load/store like operations unless such a
/// merger would cause the strides between dimensions accessed by that operaton
/// to change.
struct AccessOpOfSubViewOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedAccessOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedAccessOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merge a memref.expand_shape operation with an operation that accesses a
/// memref by index unless that operation accesss more than one dimension of
/// memory and any dimension other than the outermost dimension accessed this
/// way would be merged. This prevents issuses from arising with, say, a
/// vector.load of a 4x2 vector having the two trailing dimensions of the access
/// get merged.
struct AccessOpOfExpandShapeOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedAccessOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedAccessOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges an operation that accesses a memref by index with a
/// memref.collapse_shape, unless this would break apart a dimension other than
/// the outermost one that an operation accesses. This prevents, for example,
/// transforming a load of a 3x8 vector from a 6x8 memref into a load
/// from a 3x4x2 memref (as this would require special handling and could lead
/// to invalid IR if that higher-dimensional memref comes from a subview) but
/// does permit turning a load of a length-8 vector from a 3x8 memref into a
/// load from a 3x2x8 one.
struct AccessOpOfCollapseShapeOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedAccessOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedAccessOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges memref.subview operations present on the source or destination
/// operands of indexed memory copy operations (DMA operations) into those
/// operations. This is perfromed unconditionally, since folding in a subview
/// cannot change the starting position of the copy, which is what the
/// memref/index pair represent in DMA operations.
struct IndexedMemCopyOpOfSubViewOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedMemCopyOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedMemCopyOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges memref.expand_shape operations that are present on the source or
/// destination of an indexed memory copy/DMA into the memref/index arguments of
/// that DMA. As with subviews, this can be done unconditionally.
struct IndexedMemCopyOpOfExpandShapeOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedMemCopyOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedMemCopyOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges memref.collapse_shape operations that are present on the source or
/// destination of an indexed memory copy/DMA into the memref/index arguments of
/// that DMA. As with subviews, this can be done unconditionally.
struct IndexedMemCopyOpOfCollapseShapeOpFolder final
    : OpInterfaceRewritePattern<memref::IndexedMemCopyOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedMemCopyOpInterface op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

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

  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, loadOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(),
      loadOp.getIndices(), sourceIndices);

  llvm::TypeSwitch<Operation *, void>(loadOp)
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

  // For vector::TransferReadOp, validate preconditions before creating any IR.
  // resolveSourceIndicesExpandShape creates new ops, so all checks that can
  // fail must happen before that call to avoid "pattern returned failure but
  // IR did change" errors (caught by MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS).
  SmallVector<AffineExpr> transferReadNewResults;
  if (auto transferOp =
          dyn_cast<vector::TransferReadOp>(loadOp.getOperation())) {
    const int64_t vectorRank = transferOp.getVectorType().getRank();
    const int64_t sourceRank =
        cast<MemRefType>(expandShapeOp.getViewSource().getType()).getRank();
    if (sourceRank < vectorRank)
      return failure();

    // We can only fold if the permutation map uses only the least significant
    // dimension from each expanded reassociation group.
    for (AffineExpr result : transferOp.getPermutationMap().getResults()) {
      bool foundExpr = false;
      for (auto reassocationIndices :
           llvm::enumerate(expandShapeOp.getReassociationIndices())) {
        auto reassociation = reassocationIndices.value();
        AffineExpr dim = getAffineDimExpr(
            reassociation[reassociation.size() - 1], rewriter.getContext());
        if (dim == result) {
          transferReadNewResults.push_back(getAffineDimExpr(
              reassocationIndices.index(), rewriter.getContext()));
          foundExpr = true;
          break;
        }
      }
      if (!foundExpr)
        return failure();
    }
  }

  SmallVector<Value> sourceIndices;
  // memref.load guarantees that indexes start inbounds while the vector
  // operations don't. This impacts if our linearization is `disjoint`
  resolveSourceIndicesExpandShape(loadOp.getLoc(), rewriter, expandShapeOp,
                                  loadOp.getIndices(), sourceIndices,
                                  isa<memref::LoadOp>(loadOp.getOperation()));

  return llvm::TypeSwitch<Operation *, LogicalResult>(loadOp)
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
        return success();
      })
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getNontemporal());
        return success();
      })
      .Case([&](vector::MaskedLoadOp op) {
        rewriter.replaceOpWithNewOp<vector::MaskedLoadOp>(
            op, op.getType(), expandShapeOp.getViewSource(), sourceIndices,
            op.getMask(), op.getPassThru());
        return success();
      })
      .Case([&](vector::TransferReadOp op) {
        const int64_t sourceRank = sourceIndices.size();
        auto newMap = AffineMap::get(sourceRank, 0, transferReadNewResults,
                                     op.getContext());
        rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
            op, op.getVectorType(), expandShapeOp.getViewSource(),
            sourceIndices, newMap, op.getPadding(), op.getMask(),
            op.getInBounds());
        return success();
      })
      .DefaultUnreachable("unexpected operation");
}

template <typename OpTy>
LogicalResult LoadOpOfCollapseShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto collapseShapeOp = getMemRefOperand(loadOp)
                             .template getDefiningOp<memref::CollapseShapeOp>();

  if (!collapseShapeOp)
    return failure();

  SmallVector<Value> sourceIndices;
  resolveSourceIndicesCollapseShape(loadOp.getLoc(), rewriter, collapseShapeOp,
                                    loadOp.getIndices(), sourceIndices);
  llvm::TypeSwitch<Operation *, void>(loadOp)
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

  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, storeOp.getLoc(), subViewOp.getMixedOffsets(),
      subViewOp.getMixedStrides(), subViewOp.getDroppedDims(),
      storeOp.getIndices(), sourceIndices);

  llvm::TypeSwitch<Operation *, void>(storeOp)
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

  SmallVector<Value> sourceIndices;
  // memref.store guarantees that indexes start inbounds while the vector
  // operations don't. This impacts if our linearization is `disjoint`
  resolveSourceIndicesExpandShape(storeOp.getLoc(), rewriter, expandShapeOp,
                                  storeOp.getIndices(), sourceIndices,
                                  isa<memref::StoreOp>(storeOp.getOperation()));
  llvm::TypeSwitch<Operation *, void>(storeOp)
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

  SmallVector<Value> sourceIndices;
  resolveSourceIndicesCollapseShape(storeOp.getLoc(), rewriter, collapseShapeOp,
                                    storeOp.getIndices(), sourceIndices);
  llvm::TypeSwitch<Operation *, void>(storeOp)
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

LogicalResult
AccessOpOfSubViewOpFolder::matchAndRewrite(memref::IndexedAccessOpInterface op,
                                           PatternRewriter &rewriter) const {
  auto subview = op.getAccessedMemref().getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return rewriter.notifyMatchFailure(op, "not accessing a subview");

  SmallVector<int64_t> accessedShape = op.getAccessedShape();
  // Note the subtle difference between accesedShape = {1} and accessedShape =
  // {} here. The former prevents us from fdolding in a subview that doesn't
  // have a unit stride on the final dimension, while the latter does not (since
  // it indices scalar accesss).
  int64_t accessedDims = accessedShape.size();
  if (!hasTrailingUnitStrides(subview, accessedDims))
    return rewriter.notifyMatchFailure(
        op, "non-unit stride on accessed dimensions");

  llvm::SmallBitVector droppedDims = subview.getDroppedDims();
  int64_t sourceRank = subview.getSourceType().getRank();

  // Ignore outermost access dimension - we only care about dropped dimensions
  // between the accessed op's results, as those could break the accessing op's
  // sematics.
  int64_t secondAccessedDim = sourceRank - (accessedDims - 1);
  if (secondAccessedDim < sourceRank) {
    for (int64_t d : llvm::seq(secondAccessedDim, sourceRank)) {
      if (droppedDims.test(d))
        return rewriter.notifyMatchFailure(
            op, "reintroducing dropped dimension " + Twine(d) +
                    " would break access op semantics");
    }
  }

  SmallVector<Value> sourceIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, op.getLoc(), subview.getMixedOffsets(),
      subview.getMixedStrides(), droppedDims, op.getIndices(), sourceIndices);

  std::optional<SmallVector<Value>> newValues =
      op.updateMemrefAndIndices(rewriter, subview.getSource(), sourceIndices);
  if (newValues)
    rewriter.replaceOp(op, *newValues);
  return success();
}

LogicalResult AccessOpOfExpandShapeOpFolder::matchAndRewrite(
    memref::IndexedAccessOpInterface op, PatternRewriter &rewriter) const {
  auto expand = op.getAccessedMemref().getDefiningOp<memref::ExpandShapeOp>();
  if (!expand)
    return rewriter.notifyMatchFailure(op, "not accessing an expand_shape");

  SmallVector<int64_t> rawAccessedShape = op.getAccessedShape();
  ArrayRef<int64_t> accessedShape = rawAccessedShape;
  // Cut off the leading dimension, since we don't care about monifying its
  // strides.
  if (!accessedShape.empty())
    accessedShape = accessedShape.drop_front();

  SmallVector<ReassociationIndices, 4> reassocs =
      expand.getReassociationIndices();
  if (!hasTrivialReassociationSuffix(reassocs, accessedShape.size()))
    return rewriter.notifyMatchFailure(
        op,
        "expand_shape folding would merge semanvtically important dimensions");

  SmallVector<Value> sourceIndices;
  memref::resolveSourceIndicesExpandShape(op.getLoc(), rewriter, expand,
                                          op.getIndices(), sourceIndices,
                                          op.hasInboundsIndices());

  std::optional<SmallVector<Value>> newValues = op.updateMemrefAndIndices(
      rewriter, expand.getViewSource(), sourceIndices);
  if (newValues)
    rewriter.replaceOp(op, *newValues);
  return success();
}

LogicalResult AccessOpOfCollapseShapeOpFolder::matchAndRewrite(
    memref::IndexedAccessOpInterface op, PatternRewriter &rewriter) const {
  auto collapse =
      op.getAccessedMemref().getDefiningOp<memref::CollapseShapeOp>();
  if (!collapse)
    return rewriter.notifyMatchFailure(op, "not accessing a collapse_shape");

  SmallVector<int64_t> rawAccessedShape = op.getAccessedShape();
  ArrayRef<int64_t> accessedShape = rawAccessedShape;
  // Cut off the leading dimension, since we don't care about its strides being
  // modified and we know that the dimensions within its reassociation group, if
  // it's non-trivial, must be contiguous.
  if (!accessedShape.empty())
    accessedShape = accessedShape.drop_front();

  SmallVector<ReassociationIndices, 4> reassocs =
      collapse.getReassociationIndices();
  if (!hasTrivialReassociationSuffix(reassocs, accessedShape.size()))
    return rewriter.notifyMatchFailure(op,
                                       "collapse_shape folding would merge "
                                       "semanvtically important dimensions");

  SmallVector<Value> sourceIndices;
  memref::resolveSourceIndicesCollapseShape(op.getLoc(), rewriter, collapse,
                                            op.getIndices(), sourceIndices);

  std::optional<SmallVector<Value>> newValues = op.updateMemrefAndIndices(
      rewriter, collapse.getViewSource(), sourceIndices);
  if (newValues)
    rewriter.replaceOp(op, *newValues);
  return success();
}

LogicalResult IndexedMemCopyOpOfSubViewOpFolder::matchAndRewrite(
    memref::IndexedMemCopyOpInterface op, PatternRewriter &rewriter) const {
  auto srcSubview = op.getSrc().getDefiningOp<memref::SubViewOp>();
  auto dstSubview = op.getDst().getDefiningOp<memref::SubViewOp>();
  if (!srcSubview && !dstSubview)
    return rewriter.notifyMatchFailure(
        op, "no subviews found on indexed copy inputs");

  Value newSrc = op.getSrc();
  SmallVector<Value> newSrcIndices = llvm::to_vector(op.getSrcIndices());
  Value newDst = op.getDst();
  SmallVector<Value> newDstIndices = llvm::to_vector(op.getDstIndices());
  if (srcSubview) {
    newSrc = srcSubview.getSource();
    newSrcIndices.clear();
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, op.getLoc(), srcSubview.getMixedOffsets(),
        srcSubview.getMixedStrides(), srcSubview.getDroppedDims(),
        op.getSrcIndices(), newSrcIndices);
  }
  if (dstSubview) {
    newDst = dstSubview.getSource();
    newDstIndices.clear();
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, op.getLoc(), dstSubview.getMixedOffsets(),
        dstSubview.getMixedStrides(), dstSubview.getDroppedDims(),
        op.getDstIndices(), newDstIndices);
  }
  op.setMemrefsAndIndices(rewriter, newSrc, newSrcIndices, newDst,
                          newDstIndices);
  return success();
}

LogicalResult IndexedMemCopyOpOfExpandShapeOpFolder::matchAndRewrite(
    memref::IndexedMemCopyOpInterface op, PatternRewriter &rewriter) const {
  auto srcExpand = op.getSrc().getDefiningOp<memref::ExpandShapeOp>();
  auto dstExpand = op.getDst().getDefiningOp<memref::ExpandShapeOp>();
  if (!srcExpand && !dstExpand)
    return rewriter.notifyMatchFailure(
        op, "no expand_shapes found on indexed copy inputs");

  Value newSrc = op.getSrc();
  SmallVector<Value> newSrcIndices = llvm::to_vector(op.getSrcIndices());
  Value newDst = op.getDst();
  SmallVector<Value> newDstIndices = llvm::to_vector(op.getDstIndices());
  if (srcExpand) {
    newSrc = srcExpand.getViewSource();
    newSrcIndices.clear();
    memref::resolveSourceIndicesExpandShape(op.getLoc(), rewriter, srcExpand,
                                            op.getSrcIndices(), newSrcIndices,
                                            /*startsInbounds=*/true);
  }
  if (dstExpand) {
    newDst = dstExpand.getViewSource();
    newDstIndices.clear();
    memref::resolveSourceIndicesExpandShape(op.getLoc(), rewriter, dstExpand,
                                            op.getDstIndices(), newDstIndices,
                                            /*startsInbounds=*/true);
  }
  op.setMemrefsAndIndices(rewriter, newSrc, newSrcIndices, newDst,
                          newDstIndices);
  return success();
}

LogicalResult IndexedMemCopyOpOfCollapseShapeOpFolder::matchAndRewrite(
    memref::IndexedMemCopyOpInterface op, PatternRewriter &rewriter) const {
  auto srcCollapse = op.getSrc().getDefiningOp<memref::CollapseShapeOp>();
  auto dstCollapse = op.getDst().getDefiningOp<memref::CollapseShapeOp>();
  if (!srcCollapse && !dstCollapse)
    return rewriter.notifyMatchFailure(
        op, "no collapse_shapes found on indexed copy inputs");

  Value newSrc = op.getSrc();
  SmallVector<Value> newSrcIndices = llvm::to_vector(op.getSrcIndices());
  Value newDst = op.getDst();
  SmallVector<Value> newDstIndices = llvm::to_vector(op.getDstIndices());
  if (srcCollapse) {
    newSrc = srcCollapse.getViewSource();
    newSrcIndices.clear();
    memref::resolveSourceIndicesCollapseShape(
        op.getLoc(), rewriter, srcCollapse, op.getSrcIndices(), newSrcIndices);
  }
  if (dstCollapse) {
    newDst = dstCollapse.getViewSource();
    newDstIndices.clear();
    memref::resolveSourceIndicesCollapseShape(
        op.getLoc(), rewriter, dstCollapse, op.getDstIndices(), newDstIndices);
  }
  op.setMemrefsAndIndices(rewriter, newSrc, newSrcIndices, newDst,
                          newDstIndices);
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
  SmallVector<Value> foldedSrcIndices(copyOp.getSrcIndices().begin(),
                                      copyOp.getSrcIndices().end());

  if (srcSubViewOp) {
    LLVM_DEBUG(DBGS() << "srcSubViewOp : " << srcSubViewOp << "\n");
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, copyOp.getLoc(), srcSubViewOp.getMixedOffsets(),
        srcSubViewOp.getMixedStrides(), srcSubViewOp.getDroppedDims(),
        copyOp.getSrcIndices(), foldedSrcIndices);
  }

  // If the destination is a subview, we need to resolve the indices.
  SmallVector<Value> foldedDstIndices(copyOp.getDstIndices().begin(),
                                      copyOp.getDstIndices().end());

  if (dstSubViewOp) {
    LLVM_DEBUG(DBGS() << "dstSubViewOp : " << dstSubViewOp << "\n");
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, copyOp.getLoc(), dstSubViewOp.getMixedOffsets(),
        dstSubViewOp.getMixedStrides(), dstSubViewOp.getDroppedDims(),
        copyOp.getDstIndices(), foldedDstIndices);
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
  patterns.add<
      // Interface-based patterns to which we will be migrating.
      AccessOpOfSubViewOpFolder, AccessOpOfExpandShapeOpFolder,
      AccessOpOfCollapseShapeOpFolder, IndexedMemCopyOpOfSubViewOpFolder,
      IndexedMemCopyOpOfExpandShapeOpFolder,
      IndexedMemCopyOpOfCollapseShapeOpFolder,
      // The old way of doing things. Don't add more of these.
      LoadOpOfSubViewOpFolder<nvgpu::LdMatrixOp>,
      LoadOpOfSubViewOpFolder<vector::LoadOp>,
      LoadOpOfSubViewOpFolder<vector::MaskedLoadOp>,
      LoadOpOfSubViewOpFolder<vector::TransferReadOp>,
      StoreOpOfSubViewOpFolder<vector::TransferWriteOp>,
      StoreOpOfSubViewOpFolder<vector::StoreOp>,
      StoreOpOfSubViewOpFolder<vector::MaskedStoreOp>,
      LoadOpOfExpandShapeOpFolder<vector::LoadOp>,
      LoadOpOfExpandShapeOpFolder<vector::MaskedLoadOp>,
      LoadOpOfExpandShapeOpFolder<vector::TransferReadOp>,
      StoreOpOfExpandShapeOpFolder<vector::StoreOp>,
      StoreOpOfExpandShapeOpFolder<vector::MaskedStoreOp>,
      LoadOpOfCollapseShapeOpFolder<vector::LoadOp>,
      LoadOpOfCollapseShapeOpFolder<vector::MaskedLoadOp>,
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
