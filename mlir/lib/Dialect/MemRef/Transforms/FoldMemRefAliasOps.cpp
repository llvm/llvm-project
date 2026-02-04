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

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
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

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
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
/// vector.load of a 4x2 vector having the two traliing dimensions of the access
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
/// transforming a load of a vector 3x8 vector from a 6x8 memref into a load
/// from a 3x4x2 memref (as this would require special handling and could lead
/// to invalid IR if that higher-dimensional memref comes from a subview) but
/// does permit turning a load of a length-8 vector from a 3x8 memref into a
/// load from a 6x4x2 one.
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

/// Merges memref.subview ops on the base argument to vector transfer operations
/// into the base and indices of that transfer if:
/// - The subview has unit strides on transfer dimensions
/// - All the transfer dimensions are in-bounds
/// This will correctly update said permutation map to account for dropped
/// dimensions in rank-reducing subviews.
struct TransferOpOfSubViewOpFolder final
    : OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(VectorTransferOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges memref.expand_shape ops that create the base of a vector transfer
/// operation into the base and indices of that transfer. Does not act when the
/// a dimension is potentially out of bounds, if one of the transfer dimensions
/// would need to be strided because of the collapse, or if it would merge two
/// dimensions that are both transfer dimensions.
/// TODO: become more sophisticated about length-1 dimensions that are the
/// result of an expansion becoming broadcasts.
struct TransferOpOfExpandShapeOpFolder final
    : OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(VectorTransferOpInterface op,
                                PatternRewriter &rewriter) const override;
};

/// Merges memref.collapse_shape ops that create the base of a vector transfer
/// operation into the base and indices of that transfer. Does not act when the
/// permutation map is not trivial, a dimension could be performing out of
/// bounds reads, or if it would break apart a transfer dimension.
struct TransferOpOfCollapseShapeOpFolder final
    : OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(VectorTransferOpInterface op,
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
} // namespace

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

  auto reassocs = expand.getReassociationIndices();
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

  auto reassocs = collapse.getReassociationIndices();
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

LogicalResult
TransferOpOfSubViewOpFolder::matchAndRewrite(VectorTransferOpInterface op,
                                             PatternRewriter &rewriter) const {
  auto subview = op.getBase().getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return rewriter.notifyMatchFailure(op, "not accessing a subview");

  AffineMap perm = op.getPermutationMap();
  // Note: no identity permutation check here, since subview foldin can handle
  // complex permutations because it doesn't merge or split any individual
  // dimension.
  if (op.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(op, "out of bounds dimension");
  VectorType vecTy = op.getVectorType();
  // Because we know the permutation map is a minor identity, we know that the
  // last N dimensions must have unit stride, where N is the vector rank.

  if (!hasTrailingUnitStrides(subview, vecTy.getRank()))
    return rewriter.notifyMatchFailure(subview, "non-unit stride within last " +
                                                    Twine(vecTy.getRank()) +
                                                    " dimensions");

  AffineMap newPerm = expandDimsToRank(perm, subview.getSourceType().getRank(),
                                       subview.getDroppedDims());

  if (failed(op.mayUpdateStartingPosition(subview.getSourceType(), newPerm)))
    return rewriter.notifyMatchFailure(subview,
                                       "failed op-specific preconditions");

  SmallVector<Value> newIndices;
  affine::resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, op.getLoc(), subview.getMixedOffsets(),
      subview.getMixedStrides(), subview.getDroppedDims(), op.getIndices(),
      newIndices);
  op.updateStartingPosition(rewriter, subview.getSource(), newIndices,
                            AffineMapAttr::get(newPerm));
  return success();
}

LogicalResult TransferOpOfExpandShapeOpFolder::matchAndRewrite(
    VectorTransferOpInterface op, PatternRewriter &rewriter) const {
  auto expand = op.getBase().getDefiningOp<memref::ExpandShapeOp>();
  if (!expand)
    return rewriter.notifyMatchFailure(op, "not accessing an expand_shape");

  if (op.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(op, "out of bounds dimension");

  int64_t srcRank = expand.getSrc().getType().getRank();
  int64_t vecRank = op.getVectorType().getRank();
  if (srcRank < vecRank)
    return rewriter.notifyMatchFailure(op,
                                       "source rank is less than vector rank");

  llvm::SmallDenseMap<int64_t, int64_t, 8> unstridedResDimToSrcDim;
  for (auto [srcIdx, reassoc] :
       llvm::enumerate(expand.getReassociationIndices())) {
    unstridedResDimToSrcDim.insert({reassoc.back(), srcIdx});
  }
  // If every dimension of the expanded shape that appears in the permutation
  // map is also present in the final entry of the expansions (meaning that
  // collapsing in more values won't cause us to need to stride the index), we
  // can fold in the expansion. (This doesn't currently account for expanding
  // length X to X by 1, but it could in the future).
  AffineMap permMap = op.getPermutationMap();
  SmallVector<AffineExpr> newPermMapResults;
  newPermMapResults.reserve(permMap.getNumResults());
  for (AffineExpr permRes : permMap.getResults()) {
    auto resDim = dyn_cast<AffineDimExpr>(permRes);
    if (!resDim)
      return rewriter.notifyMatchFailure(
          op, "has non-dim entry in permutation map");
    auto dimInSrc = unstridedResDimToSrcDim.find(resDim.getPosition());
    if (dimInSrc == unstridedResDimToSrcDim.end())
      return rewriter.notifyMatchFailure(op,
                                         "permutation map result would be made "
                                         "strided by expand_shape folding");
    newPermMapResults.push_back(rewriter.getAffineDimExpr(dimInSrc->second));
  }

  auto newPerm = AffineMap::get(srcRank, 0, newPermMapResults, op.getContext());

  if (failed(op.mayUpdateStartingPosition(expand.getSrc().getType(), newPerm)))
    return rewriter.notifyMatchFailure(op, "failed op-specific preconditions");

  SmallVector<Value> newIndices;
  // We can use a disjoint linearization if we aren't masking, because then all
  // indicators show that the start position will be in bounds.
  memref::resolveSourceIndicesExpandShape(op.getLoc(), rewriter, expand,
                                          op.getIndices(), newIndices,
                                          /*startsInbounds=*/!op.getMask());

  op.updateStartingPosition(rewriter, expand.getViewSource(), newIndices,
                            AffineMapAttr::get(newPerm));
  return success();
}

LogicalResult TransferOpOfCollapseShapeOpFolder::matchAndRewrite(
    VectorTransferOpInterface op, PatternRewriter &rewriter) const {
  auto collapse = op.getBase().getDefiningOp<memref::CollapseShapeOp>();
  if (!collapse)
    return rewriter.notifyMatchFailure(op, "not accessing a collapse_shape");

  if (!op.getPermutationMap().isMinorIdentity())
    return rewriter.notifyMatchFailure(op,
                                       "non-minor identity permutation map");

  if (op.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(op, "out of bounds dimension");

  int64_t srcRank = collapse.getSrc().getType().getRank();
  int64_t vecRank = op.getVectorType().getRank();
  if (srcRank < vecRank)
    return rewriter.notifyMatchFailure(op,
                                       "source rank is less than vector rank");

  // Note: no -  1 on the rank here. While we could treat the collapse of [1, 1,
  // N] into N as a specila case, that is left as future work for those who need
  // such a pattern.
  SmallVector<ReassociationIndices> reassocs =
      collapse.getReassociationIndices();
  if (!hasTrivialReassociationSuffix(reassocs, vecRank))
    return rewriter.notifyMatchFailure(
        op, "collapse_shape folding would split a transfer dimension");

  AffineMap newPerm =
      AffineMap::getMinorIdentityMap(srcRank, vecRank, op.getContext());
  if (failed(
          op.mayUpdateStartingPosition(collapse.getSrc().getType(), newPerm)))
    return rewriter.notifyMatchFailure(op, "failed op-specific preconditions");

  SmallVector<Value> newIndices;
  memref::resolveSourceIndicesCollapseShape(op.getLoc(), rewriter, collapse,
                                            op.getIndices(), newIndices);

  op.updateStartingPosition(rewriter, collapse.getViewSource(), newIndices,
                            AffineMapAttr::get(newPerm));
  return success();
}

void memref::populateFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
  patterns
      .add<AccessOpOfSubViewOpFolder, AccessOpOfExpandShapeOpFolder,
           AccessOpOfCollapseShapeOpFolder, IndexedMemCopyOpOfSubViewOpFolder,
           IndexedMemCopyOpOfExpandShapeOpFolder,
           IndexedMemCopyOpOfCollapseShapeOpFolder, TransferOpOfSubViewOpFolder,
           TransferOpOfExpandShapeOpFolder, TransferOpOfCollapseShapeOpFolder,
           SubViewOfSubViewFolder>(patterns.getContext());
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
