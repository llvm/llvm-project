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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

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
    OpFoldResult ofr = makeComposedFoldedAffineApply(
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
      OpFoldResult ofr = makeComposedFoldedAffineApply(
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
        collapseShapeOp.getViewSource().getType().cast<MemRefType>().getRank();
    for (int64_t i = 0; i < srcRank; i++) {
      OpFoldResult ofr = makeComposedFoldedAffineApply(
          rewriter, loc, zeroAffineMap, dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
  }
  return success();
}

/// Given the 'indices' of an load/store operation where the memref is a result
/// of a subview op, returns the indices w.r.t to the source memref of the
/// subview op. For example
///
/// %0 = ... : memref<12x42xf32>
/// %1 = subview %0[%arg0, %arg1][][%stride1, %stride2] : memref<12x42xf32> to
///          memref<4x4xf32, offset=?, strides=[?, ?]>
/// %2 = load %1[%i1, %i2] : memref<4x4xf32, offset=?, strides=[?, ?]>
///
/// could be folded into
///
/// %2 = load %0[%arg0 + %i1 * %stride1][%arg1 + %i2 * %stride2] :
///          memref<12x42xf32>
static LogicalResult
resolveSourceIndicesSubView(Location loc, PatternRewriter &rewriter,
                            memref::SubViewOp subViewOp, ValueRange indices,
                            SmallVectorImpl<Value> &sourceIndices) {
  SmallVector<OpFoldResult> mixedOffsets = subViewOp.getMixedOffsets();
  SmallVector<OpFoldResult> mixedSizes = subViewOp.getMixedSizes();
  SmallVector<OpFoldResult> mixedStrides = subViewOp.getMixedStrides();

  SmallVector<Value> useIndices;
  // Check if this is rank-reducing case. Then for every unit-dim size add a
  // zero to the indices.
  int64_t resultDim = 0;
  llvm::SmallBitVector unusedDims = subViewOp.getDroppedDims();
  for (auto dim : llvm::seq<int64_t>(0, subViewOp.getSourceType().getRank())) {
    if (unusedDims.test(dim))
      useIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    else
      useIndices.push_back(indices[resultDim++]);
  }
  if (useIndices.size() != mixedOffsets.size())
    return failure();
  sourceIndices.resize(useIndices.size());
  for (auto index : llvm::seq<size_t>(0, mixedOffsets.size())) {
    SmallVector<OpFoldResult> dynamicOperands;
    AffineExpr expr = rewriter.getAffineDimExpr(0);
    int64_t numSymbols = 0;
    dynamicOperands.push_back(useIndices[index]);

    // Multiply the stride;
    if (auto attr = mixedStrides[index].dyn_cast<Attribute>()) {
      expr = expr * attr.cast<IntegerAttr>().getInt();
    } else {
      dynamicOperands.push_back(mixedStrides[index].get<Value>());
      expr = expr * rewriter.getAffineSymbolExpr(numSymbols++);
    }

    // Add the offset.
    if (auto attr = mixedOffsets[index].dyn_cast<Attribute>()) {
      expr = expr + attr.cast<IntegerAttr>().getInt();
    } else {
      dynamicOperands.push_back(mixedOffsets[index].get<Value>());
      expr = expr + rewriter.getAffineSymbolExpr(numSymbols++);
    }
    Location loc = subViewOp.getLoc();
    OpFoldResult ofr = makeComposedFoldedAffineApply(
        rewriter, loc, AffineMap::get(1, numSymbols, expr), dynamicOperands);
    sourceIndices[index] = getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
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

static Value getMemRefOperand(vector::TransferWriteOp op) {
  return op.getSource();
}

static Value getMemRefOperand(gpu::SubgroupMmaLoadMatrixOp op) {
  return op.getSrcMemref();
}

static Value getMemRefOperand(gpu::SubgroupMmaStoreMatrixOp op) {
  return op.getDstMemref();
}

/// Given the permutation map of the original
/// `vector.transfer_read`/`vector.transfer_write` operations compute the
/// permutation map to use after the subview is folded with it.
static AffineMapAttr getPermutationMapAttr(MLIRContext *context,
                                           memref::SubViewOp subViewOp,
                                           AffineMap currPermutationMap) {
  llvm::SmallBitVector unusedDims = subViewOp.getDroppedDims();
  SmallVector<AffineExpr> exprs;
  int64_t sourceRank = subViewOp.getSourceType().getRank();
  for (auto dim : llvm::seq<int64_t>(0, sourceRank)) {
    if (unusedDims.test(dim))
      continue;
    exprs.push_back(getAffineDimExpr(dim, context));
  }
  auto resultDimToSourceDimMap = AffineMap::get(sourceRank, 0, exprs, context);
  return AffineMapAttr::get(
      currPermutationMap.compose(resultDimToSourceDimMap));
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
    Location loc = subView.getLoc();
    auto srcSubView = subView.getSource().getDefiningOp<memref::SubViewOp>();
    if (!srcSubView)
      return failure();
    int64_t srcRank = srcSubView.getSourceType().getRank();

    // TODO: Only stride 1 is supported.
    for (auto s : {subView.getMixedStrides(), srcSubView.getMixedStrides()})
      if (!llvm::all_of(
              s, [](OpFoldResult ofr) { return isConstantIntValue(ofr, 1); }))
        return failure();

    // Get original offsets and sizes.
    SmallVector<OpFoldResult> offsets = subView.getMixedOffsets();
    SmallVector<OpFoldResult> srcOffsets = srcSubView.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subView.getMixedSizes();
    SmallVector<OpFoldResult> srcSizes = srcSubView.getMixedSizes();

    // Compute new offsets and sizes.
    llvm::SmallBitVector srcReducedDims = srcSubView.getDroppedDims();
    SmallVector<OpFoldResult> newOffsets, newSizes;
    int64_t dim = 0;
    for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
      if (srcReducedDims[srcDim]) {
        // Dim is reduced in srcSubView.
        assert(isConstantIntValue(srcSizes[srcDim], 1) && "expected size 1");
        newOffsets.push_back(srcOffsets[srcDim]);
        newSizes.push_back(srcSizes[srcDim]);
        continue;
      }
      AffineExpr sym0, sym1;
      bindSymbols(subView.getContext(), sym0, sym1);
      newOffsets.push_back(makeComposedFoldedAffineApply(
          rewriter, loc, sym0 + sym1, {srcOffsets[srcDim], offsets[dim]}));
      newSizes.push_back(sizes[dim]);
      ++dim;
    }

    // Replace original op.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        subView, subView.getType(), srcSubView.getSource(), newOffsets,
        newSizes, srcSubView.getMixedStrides());
    return success();
  }
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
    OpFoldResult ofr = makeComposedFoldedAffineApply(
        rewriter, loc, affineMap.getSubMap({i}), indicesOfr);
    expandedIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
  return expandedIndices;
}

template <typename OpTy>
LogicalResult LoadOpOfSubViewOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto subViewOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::SubViewOp>();

  if (!subViewOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp = dyn_cast<AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesSubView(loadOp.getLoc(), rewriter, subViewOp,
                                         indices, sourceIndices)))
    return failure();

  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](AffineLoadOp op) {
        rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, subViewOp.getSource(),
                                                  sourceIndices);
      })
      .Case([&](memref::LoadOp op) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, subViewOp.getSource(), sourceIndices, op.getNontemporal());
      })
      .Case([&](vector::TransferReadOp transferReadOp) {
        rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
            transferReadOp, transferReadOp.getVectorType(),
            subViewOp.getSource(), sourceIndices,
            getPermutationMapAttr(rewriter.getContext(), subViewOp,
                                  transferReadOp.getPermutationMap()),
            transferReadOp.getPadding(),
            /*mask=*/Value(), transferReadOp.getInBoundsAttr());
      })
      .Case([&](gpu::SubgroupMmaLoadMatrixOp op) {
        rewriter.replaceOpWithNewOp<gpu::SubgroupMmaLoadMatrixOp>(
            op, op.getType(), subViewOp.getSource(), sourceIndices,
            op.getLeadDimension(), op.getTransposeAttr());
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
  if (auto affineLoadOp = dyn_cast<AffineLoadOp>(loadOp.getOperation())) {
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
      .Case<AffineLoadOp, memref::LoadOp>([&](auto op) {
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
  if (auto affineLoadOp = dyn_cast<AffineLoadOp>(loadOp.getOperation())) {
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
      .Case<AffineLoadOp, memref::LoadOp>([&](auto op) {
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
    return failure();

  SmallVector<Value> indices(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineStoreOp = dyn_cast<AffineStoreOp>(storeOp.getOperation())) {
    AffineMap affineMap = affineStoreOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, storeOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesSubView(storeOp.getLoc(), rewriter, subViewOp,
                                         indices, sourceIndices)))
    return failure();

  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case([&](AffineStoreOp op) {
        rewriter.replaceOpWithNewOp<AffineStoreOp>(
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
            getPermutationMapAttr(rewriter.getContext(), subViewOp,
                                  op.getPermutationMap()),
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
  if (auto affineStoreOp = dyn_cast<AffineStoreOp>(storeOp.getOperation())) {
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
      .Case<AffineStoreOp, memref::StoreOp>([&](auto op) {
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
  if (auto affineStoreOp = dyn_cast<AffineStoreOp>(storeOp.getOperation())) {
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
      .Case<AffineStoreOp, memref::StoreOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(
            storeOp, storeOp.getValue(), collapseShapeOp.getViewSource(),
            sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

void memref::populateFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadOpOfSubViewOpFolder<AffineLoadOp>,
               LoadOpOfSubViewOpFolder<memref::LoadOp>,
               LoadOpOfSubViewOpFolder<vector::TransferReadOp>,
               LoadOpOfSubViewOpFolder<gpu::SubgroupMmaLoadMatrixOp>,
               StoreOpOfSubViewOpFolder<AffineStoreOp>,
               StoreOpOfSubViewOpFolder<memref::StoreOp>,
               StoreOpOfSubViewOpFolder<vector::TransferWriteOp>,
               StoreOpOfSubViewOpFolder<gpu::SubgroupMmaStoreMatrixOp>,
               LoadOpOfExpandShapeOpFolder<AffineLoadOp>,
               LoadOpOfExpandShapeOpFolder<memref::LoadOp>,
               StoreOpOfExpandShapeOpFolder<AffineStoreOp>,
               StoreOpOfExpandShapeOpFolder<memref::StoreOp>,
               LoadOpOfCollapseShapeOpFolder<AffineLoadOp>,
               LoadOpOfCollapseShapeOpFolder<memref::LoadOp>,
               StoreOpOfCollapseShapeOpFolder<AffineStoreOp>,
               StoreOpOfCollapseShapeOpFolder<memref::StoreOp>,
               SubViewOfSubViewFolder>(patterns.getContext());
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
