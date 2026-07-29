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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
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

template <typename... ProducerOps>
struct FoldIntoElementwisePattern : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<GenericOp, ElementwiseOp>(op.getOperation()) || !isElementwise(op))
      return failure();

    bool changed = false;
    SmallVector<Value> newIns;
    SmallVector<AffineMap> newMaps;
    for (OpOperand *operand : op.getDpsInputOperands()) {
      AffineMap consumerMap = op.getMatchingIndexingMap(operand);
      const bool folded = (ElementwiseOpFolder<ProducerOps>::fold(
                               operand, consumerMap, newIns, newMaps) ||
                           ...);
      if (folded) {
        changed = true;
      } else {
        // push in original operand and its map.
        newIns.push_back(operand->get());
        newMaps.push_back(consumerMap);
      }
    }
    if (!changed)
      return failure();

    // Keep all output operands and their maps unchanged.
    SmallVector<AffineMap> originalMaps = op.getIndexingMapsArray();
    newMaps.append(originalMaps.begin() + op.getNumDpsInputs(),
                   originalMaps.end());
    // The maps of the rewritten op must still determine bounds for every loop
    // dimension. Folding a broadcast can otherwise drop the only map result
    // that covers a dimension.
    // See `generic_broadcast_not_folded_non_invertible` in
    // mlir/test/Dialect/Linalg/elementwise/fold.mlir for an example.
    if (!inversePermutation(concatAffineMaps(newMaps, op.getContext())))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (auto [index, operand] : llvm::enumerate(op.getDpsInputOperands()))
        op->setOperand(operand->getOperandNumber(), newIns[index]);
      op->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(newMaps));
    });
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
  patterns.add<FoldIntoElementwisePattern<TransposeOp, BroadcastOp>>(
      patterns.getContext());
}
