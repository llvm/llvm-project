//===- FoldIntoElementwise.cpp - Fold Ops into elementwise if possible ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements folding ops such as transpose and broadcast into the
// affine maps of the elementwise op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGFOLDINTOELEMENTWISEPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-fold-into-elementwise"

namespace {
struct FoldTransposePattern : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern<ElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<Value> newIns;
    SmallVector<AffineMap> newMaps;
    for (OpOperand *operand : op.getDpsInputOperands()) {
      AffineMap map = op.getMatchingIndexingMap(operand);
      auto transposeOp = operand->get().getDefiningOp<TransposeOp>();

      if (!map.isIdentity() || !transposeOp) {
        // push in original operand and its map.
        newIns.push_back(operand->get());
        newMaps.push_back(map);
        continue;
      }
      newIns.push_back(transposeOp.getInput());
      // push in transposeOp's inverse permutation map.
      newMaps.push_back(transposeOp.getMatchingIndexingMap(
          transposeOp.getDpsInputOperand(0)));
      changed = true;
    }
    if (!changed)
      return failure();
    newMaps.push_back(op.getIndexingMapsArray().back());

    rewriter.replaceOpWithNewOp<ElementwiseOp>(
        op, newIns, op.getDpsInits()[0], op.getKindAttr(),
        rewriter.getAffineMapArrayAttr(newMaps));
    return success();
  }
};

struct LinalgFoldIntoElementwisePass
    : public impl::LinalgFoldIntoElementwisePassBase<
          LinalgFoldIntoElementwisePass> {
  using impl::LinalgFoldIntoElementwisePassBase<
      LinalgFoldIntoElementwisePass>::LinalgFoldIntoElementwisePassBase;

  void runOnOperation() override {
    llvm::outs() << "Hellow from fold into elemenwise \n";
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
  patterns.add<FoldTransposePattern>(patterns.getContext());
}