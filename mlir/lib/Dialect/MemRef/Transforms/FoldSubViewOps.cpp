//===- FoldSubViewOps.cpp - Fold memref.subview ops -----------------------===//
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

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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
resolveSourceIndices(Location loc, PatternRewriter &rewriter,
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
  return op.memref();
}

static Value getMemRefOperand(vector::TransferReadOp op) { return op.source(); }

static Value getMemRefOperand(vector::TransferWriteOp op) {
  return op.source();
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
class LoadOpOfSubViewFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;

private:
  void replaceOp(OpTy loadOp, memref::SubViewOp subViewOp,
                 ArrayRef<Value> sourceIndices,
                 PatternRewriter &rewriter) const;
};

/// Merges subview operation with store/transferWriteOp operation.
template <typename OpTy>
class StoreOpOfSubViewFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;

private:
  void replaceOp(OpTy storeOp, memref::SubViewOp subViewOp,
                 ArrayRef<Value> sourceIndices,
                 PatternRewriter &rewriter) const;
};

template <typename LoadOpTy>
void LoadOpOfSubViewFolder<LoadOpTy>::replaceOp(
    LoadOpTy loadOp, memref::SubViewOp subViewOp, ArrayRef<Value> sourceIndices,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LoadOpTy>(loadOp, subViewOp.source(),
                                        sourceIndices);
}

template <>
void LoadOpOfSubViewFolder<vector::TransferReadOp>::replaceOp(
    vector::TransferReadOp transferReadOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  // TODO: support 0-d corner case.
  if (transferReadOp.getTransferRank() == 0)
    return;
  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
      transferReadOp, transferReadOp.getVectorType(), subViewOp.source(),
      sourceIndices,
      getPermutationMapAttr(rewriter.getContext(), subViewOp,
                            transferReadOp.permutation_map()),
      transferReadOp.padding(),
      /*mask=*/Value(), transferReadOp.in_boundsAttr());
}

template <typename StoreOpTy>
void StoreOpOfSubViewFolder<StoreOpTy>::replaceOp(
    StoreOpTy storeOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<StoreOpTy>(storeOp, storeOp.value(),
                                         subViewOp.source(), sourceIndices);
}

template <>
void StoreOpOfSubViewFolder<vector::TransferWriteOp>::replaceOp(
    vector::TransferWriteOp transferWriteOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  // TODO: support 0-d corner case.
  if (transferWriteOp.getTransferRank() == 0)
    return;
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      transferWriteOp, transferWriteOp.vector(), subViewOp.source(),
      sourceIndices,
      getPermutationMapAttr(rewriter.getContext(), subViewOp,
                            transferWriteOp.permutation_map()),
      transferWriteOp.in_boundsAttr());
}
} // namespace

template <typename OpTy>
LogicalResult
LoadOpOfSubViewFolder<OpTy>::matchAndRewrite(OpTy loadOp,
                                             PatternRewriter &rewriter) const {
  auto subViewOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::SubViewOp>();
  if (!subViewOp)
    return failure();

  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                  loadOp.indices(), sourceIndices)))
    return failure();

  replaceOp(loadOp, subViewOp, sourceIndices, rewriter);
  return success();
}

template <typename OpTy>
LogicalResult
StoreOpOfSubViewFolder<OpTy>::matchAndRewrite(OpTy storeOp,
                                              PatternRewriter &rewriter) const {
  auto subViewOp =
      getMemRefOperand(storeOp).template getDefiningOp<memref::SubViewOp>();
  if (!subViewOp)
    return failure();

  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(storeOp.getLoc(), rewriter, subViewOp,
                                  storeOp.indices(), sourceIndices)))
    return failure();

  replaceOp(storeOp, subViewOp, sourceIndices, rewriter);
  return success();
}

void memref::populateFoldSubViewOpPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadOpOfSubViewFolder<AffineLoadOp>,
               LoadOpOfSubViewFolder<memref::LoadOp>,
               LoadOpOfSubViewFolder<vector::TransferReadOp>,
               StoreOpOfSubViewFolder<AffineStoreOp>,
               StoreOpOfSubViewFolder<memref::StoreOp>,
               StoreOpOfSubViewFolder<vector::TransferWriteOp>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct FoldSubViewOpsPass final
    : public FoldSubViewOpsBase<FoldSubViewOpsPass> {
  void runOnOperation() override;
};

} // namespace

void FoldSubViewOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldSubViewOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> memref::createFoldSubViewOpsPass() {
  return std::make_unique<FoldSubViewOpsPass>();
}
