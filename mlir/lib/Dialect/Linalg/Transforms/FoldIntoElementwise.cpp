//===- FoldIntoElementwise.cpp - Fold Ops into elementwise if possible ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements folding ops such as transpose and broadcast into the
// affine maps of elementwise consumers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGFOLDINTOELEMENTWISEPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-fold-into-elementwise"

namespace {
template <typename ProducerOpTy>
struct ElementwiseOpFolder {
  // Helper function to fold broadcast etc. into a consumer operand.
  static bool fold(OpOperand *elwiseOperand, AffineMap elwiseMap,
                   SmallVector<Value> &newIns,
                   SmallVector<AffineMap> &newMaps) {
    auto producerOp = elwiseOperand->get().getDefiningOp<ProducerOpTy>();
    if (!producerOp || !elwiseMap.isProjectedPermutation())
      return false;
    newIns.push_back(producerOp.getInput());
    // push in the new composed affine map
    newMaps.push_back(
        producerOp.getMatchingIndexingMap(producerOp.getDpsInputOperand(0))
            .compose(elwiseMap));
    return true;
  }
};

template <typename ConsumerOpTy, typename... ProducerOps>
static bool foldInputOperands(ConsumerOpTy op, SmallVector<Value> &newIns,
                              SmallVector<AffineMap> &newMaps) {
  bool changed = false;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    AffineMap consumerMap = op.getMatchingIndexingMap(operand);
    const bool folded = (ElementwiseOpFolder<ProducerOps>::fold(
                             operand, consumerMap, newIns, newMaps) ||
                         ...);
    if (folded) {
      changed = true;
    } else {
      newIns.push_back(operand->get());
      newMaps.push_back(consumerMap);
    }
  }
  return changed;
}

template <typename... ProducerOps>
struct FoldIntoElementwisePattern : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern<ElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newIns;
    SmallVector<AffineMap> newMaps;
    if (!foldInputOperands<ElementwiseOp, ProducerOps...>(op, newIns, newMaps))
      return failure();
    newMaps.push_back(op.getIndexingMapsArray().back());

    rewriter.replaceOpWithNewOp<ElementwiseOp>(
        op, newIns, op.getDpsInits()[0], op.getKindAttr(),
        rewriter.getAffineMapArrayAttr(newMaps));
    return success();
  }
};

template <typename... ProducerOps>
struct FoldIntoGenericPattern : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Restrict this pattern to elementwise-like generic ops.
    // It may be safe to do so reduction dimensions in some cases. But we try
    // to focus on simple cases here.
    if (!op.isAllParallelLoops())
      return failure();

    SmallVector<Value> newIns;
    SmallVector<AffineMap> newMaps;
    if (!foldInputOperands<GenericOp, ProducerOps...>(op, newIns, newMaps))
      return failure();

    // Keep all output operands and their maps unchanged. The body is cloned
    // so that the block arguments continue to correspond to the new operand
    // list.
    SmallVector<AffineMap> allMaps = op.getIndexingMapsArray();
    newMaps.append(allMaps.begin() + op.getNumDpsInputs(), allMaps.end());
    // The maps of the rewritten op must still determine bounds for every loop
    // dimension. Folding a broadcast can otherwise drop the only map result
    // that covers a dimension.
    // See `generic_broadcast_not_folded_non_invertible` in
    // mlir/test/Dialect/Linalg/elementwise/fold.mlir for an example.
    if (!inversePermutation(concatAffineMaps(newMaps, op.getContext())))
      return failure();
    auto newOp =
        GenericOp::create(rewriter, op.getLoc(), op.getResultTypes(), newIns,
                          op.getDpsInits(), newMaps, op.getIteratorTypesArray(),
                          /*bodyBuild=*/nullptr, getPrunedAttributeList(op));
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct LinalgFoldIntoElementwisePass
    : public impl::LinalgFoldIntoElementwisePassBase<
          LinalgFoldIntoElementwisePass> {
  using impl::LinalgFoldIntoElementwisePassBase<
      LinalgFoldIntoElementwisePass>::LinalgFoldIntoElementwisePassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateLinalgFoldIntoElementwisePatterns(patterns);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void mlir::linalg::populateLinalgFoldIntoElementwisePatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldIntoElementwisePattern<TransposeOp, BroadcastOp>,
               FoldIntoGenericPattern<TransposeOp, BroadcastOp>>(
      patterns.getContext());
}
