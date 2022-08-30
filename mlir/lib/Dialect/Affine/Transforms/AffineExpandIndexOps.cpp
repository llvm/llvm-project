//===- AffineExpandIndexOps.cpp - Affine expand index ops pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to expand affine index ops into one or more more
// fundamental operations.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_AFFINEEXPANDINDEXOPSPASS
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Lowers `affine.delinearize_index` into a sequence of division and remainder
/// operations.
struct LowerDelinearizeIndexOps
    : public OpRewritePattern<AffineDelinearizeIndexOp> {
  using OpRewritePattern<AffineDelinearizeIndexOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineDelinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> multiIndex =
        delinearizeIndex(rewriter, op->getLoc(), op.getLinearIndex(),
                         llvm::to_vector(op.getBasis()));
    if (failed(multiIndex))
      return failure();
    rewriter.replaceOp(op, *multiIndex);
    return success();
  }
};

class ExpandAffineIndexOpsPass
    : public impl::AffineExpandIndexOpsPassBase<ExpandAffineIndexOpsPass> {
public:
  ExpandAffineIndexOpsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateAffineExpandIndexOpsPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::populateAffineExpandIndexOpsPatterns(RewritePatternSet &patterns) {
  patterns.insert<LowerDelinearizeIndexOps>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createAffineExpandIndexOpsPass() {
  return std::make_unique<ExpandAffineIndexOpsPass>();
}
