//===- DecomposeNamedLinalgOps.cpp - Patterns to break up complex ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGDECOMPOSEAGGREGATENAMEDOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-decompose-named-ops"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct DecomposeSoftmaxPattern : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    // Decompose softmax(x) into tmp = exp(x - max(x)); tmp / sum(tmp)
    FailureOr<DecompositionResult> results = op.decomposeOperation(rewriter);
    if (failed(results))
      return rewriter.notifyMatchFailure(op, "Failed to decompose SoftmaxOp");
    rewriter.replaceOp(op, results->replacementValues);
    return success();
  }
};

} // namespace

struct LinalgDecomposeAggregateNamedOpsPass
    : public impl::LinalgDecomposeAggregateNamedOpsPassBase<
          LinalgDecomposeAggregateNamedOpsPass> {
  using impl::LinalgDecomposeAggregateNamedOpsPassBase<
      LinalgDecomposeAggregateNamedOpsPass>::
      LinalgDecomposeAggregateNamedOpsPassBase;

  void runOnOperation() override;
};

void LinalgDecomposeAggregateNamedOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateDecomposeAggregateNamedOpsPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void mlir::linalg::populateDecomposeAggregateNamedOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<DecomposeSoftmaxPattern>(patterns.getContext());
}
