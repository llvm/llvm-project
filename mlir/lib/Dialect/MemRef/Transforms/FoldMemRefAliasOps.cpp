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

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
  for (SmallVector<int64_t, 2> groups :
       expandShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    unsigned groupSize = groups.size();
    SmallVector<int64_t> suffixProduct(groupSize);
    // Calculate suffix product of dimension sizes for all dimensions of expand
    // shape op result.
    suffixProduct[groupSize - 1] = 1;
    for (unsigned i = groupSize - 1; i > 0; i--)
      suffixProduct[i - 1] =
          suffixProduct[i] *
          expandShapeOp.getType().cast<MemRefType>().getDimSize(groups[i]);
    SmallVector<Value> dynamicIndices(groupSize);
    for (unsigned i = 0; i < groupSize; i++)
      dynamicIndices[i] = indices[groups[i]];
    // Construct the expression for the index value w.r.t to expand shape op
    // source corresponding the indices wrt to expand shape op result.
    AffineExpr srcIndexExpr = getLinearAffineExpr(suffixProduct, rewriter);
    sourceIndices.push_back(rewriter.create<AffineApplyOp>(
        loc,
        AffineMap::get(/*numDims=*/groupSize, /*numSymbols=*/0, srcIndexExpr),
        dynamicIndices));
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
  unsigned cnt = 0;
  SmallVector<Value> tmp(indices.size());
  SmallVector<Value> dynamicIndices;
  for (SmallVector<int64_t, 2> groups :
       collapseShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    dynamicIndices.push_back(indices[cnt++]);
    unsigned groupSize = groups.size();
    SmallVector<int64_t> suffixProduct(groupSize);
    // Calculate suffix product for all collapse op source dimension sizes.
    suffixProduct[groupSize - 1] = 1;
    for (unsigned i = groupSize - 1; i > 0; i--)
      suffixProduct[i - 1] =
          suffixProduct[i] * collapseShapeOp.getSrcType().getDimSize(groups[i]);
    // Derive the index values along all dimensions of the source corresponding
    // to the index wrt to collapsed shape op output.
    SmallVector<AffineExpr, 4> srcIndexExpr =
        getDelinearizedAffineExpr(suffixProduct, rewriter);
    for (unsigned i = 0; i < groupSize; i++)
      sourceIndices.push_back(rewriter.create<AffineApplyOp>(
          loc, AffineMap::get(/*numDims=*/1, /*numSymbols=*/0, srcIndexExpr[i]),
          dynamicIndices));
    dynamicIndices.clear();
  }
  if (collapseShapeOp.getReassociationIndices().empty()) {
    auto zeroAffineMap = rewriter.getConstantAffineMap(0);
    unsigned srcRank =
        collapseShapeOp.getViewSource().getType().cast<MemRefType>().getRank();
    for (unsigned i = 0; i < srcRank; i++)
      sourceIndices.push_back(
          rewriter.create<AffineApplyOp>(loc, zeroAffineMap, dynamicIndices));
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
  unsigned resultDim = 0;
  llvm::SmallBitVector unusedDims = subViewOp.getDroppedDims();
  for (auto dim : llvm::seq<unsigned>(0, subViewOp.getSourceType().getRank())) {
    if (unusedDims.test(dim))
      useIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    else
      useIndices.push_back(indices[resultDim++]);
  }
  if (useIndices.size() != mixedOffsets.size())
    return failure();
  sourceIndices.resize(useIndices.size());
  for (auto index : llvm::seq<size_t>(0, mixedOffsets.size())) {
    SmallVector<Value> dynamicOperands;
    AffineExpr expr = rewriter.getAffineDimExpr(0);
    unsigned numSymbols = 0;
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
    sourceIndices[index] = rewriter.create<AffineApplyOp>(
        loc, AffineMap::get(1, numSymbols, expr), dynamicOperands);
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

} // namespace

static SmallVector<Value>
calculateExpandedAccessIndices(AffineMap affineMap,
                               const SmallVector<Value> &indices, Location loc,
                               PatternRewriter &rewriter) {
  SmallVector<Value> expandedIndices;
  for (unsigned i = 0, e = affineMap.getNumResults(); i < e; i++)
    expandedIndices.push_back(
        rewriter.create<AffineApplyOp>(loc, affineMap.getSubMap({i}), indices));
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
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndicesSubView(loadOp.getLoc(), rewriter, subViewOp,
                                         indices, sourceIndices)))
    return failure();

  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case<AffineLoadOp, memref::LoadOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(loadOp, subViewOp.getSource(),
                                                  sourceIndices);
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
  SmallVector<Value, 4> sourceIndices;
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
  SmallVector<Value, 4> sourceIndices;
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
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndicesSubView(storeOp.getLoc(), rewriter, subViewOp,
                                         indices, sourceIndices)))
    return failure();

  llvm::TypeSwitch<Operation *, void>(storeOp)
      .Case<AffineStoreOp, memref::StoreOp>([&](auto op) {
        rewriter.replaceOpWithNewOp<decltype(op)>(
            storeOp, storeOp.getValue(), subViewOp.getSource(), sourceIndices);
      })
      .Case([&](vector::TransferWriteOp op) {
        rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
            op, op.getValue(), subViewOp.getSource(), sourceIndices,
            getPermutationMapAttr(rewriter.getContext(), subViewOp,
                                  op.getPermutationMap()),
            op.getInBoundsAttr());
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
  SmallVector<Value, 4> sourceIndices;
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
  SmallVector<Value, 4> sourceIndices;
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
               StoreOpOfSubViewOpFolder<AffineStoreOp>,
               StoreOpOfSubViewOpFolder<memref::StoreOp>,
               StoreOpOfSubViewOpFolder<vector::TransferWriteOp>,
               LoadOpOfExpandShapeOpFolder<AffineLoadOp>,
               LoadOpOfExpandShapeOpFolder<memref::LoadOp>,
               StoreOpOfExpandShapeOpFolder<AffineStoreOp>,
               StoreOpOfExpandShapeOpFolder<memref::StoreOp>,
               LoadOpOfCollapseShapeOpFolder<AffineLoadOp>,
               LoadOpOfCollapseShapeOpFolder<memref::LoadOp>,
               StoreOpOfCollapseShapeOpFolder<AffineStoreOp>,
               StoreOpOfCollapseShapeOpFolder<memref::StoreOp>>(
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
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                     std::move(patterns));
}

std::unique_ptr<Pass> memref::createFoldMemRefAliasOpsPass() {
  return std::make_unique<FoldMemRefAliasOpsPass>();
}
