//===- AffineExpandIndexOpsAsAffine.cpp - Expand index ops to apply pass --===//
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
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEEXPANDINDEXOPSASAFFINE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

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
                         op.getEffectiveBasis(), /*hasOuterBound=*/false);
    if (failed(multiIndex))
      return failure();
    rewriter.replaceOp(op, *multiIndex);
    return success();
  }
};

/// Lowers `affine.linearize_index` into a sequence of multiplications and
/// additions.
struct LowerLinearizeIndexOps final : OpRewritePattern<AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    // Should be folded away, included here for safety.
    if (op.getMultiIndex().empty()) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      return success();
    }

    SmallVector<OpFoldResult> multiIndex =
        getAsOpFoldResult(op.getMultiIndex());
    OpFoldResult linearIndex =
        linearizeIndex(rewriter, op.getLoc(), multiIndex, op.getMixedBasis());
    Value linearIndexValue =
        getValueOrCreateConstantIntOp(rewriter, op.getLoc(), linearIndex);
    rewriter.replaceOp(op, linearIndexValue);
    return success();
  }
};

class ExpandAffineIndexOpsAsAffinePass
    : public affine::impl::AffineExpandIndexOpsAsAffineBase<
          ExpandAffineIndexOpsAsAffinePass> {
public:
  ExpandAffineIndexOpsAsAffinePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateAffineExpandIndexOpsAsAffinePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::affine::populateAffineExpandIndexOpsAsAffinePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<LowerDelinearizeIndexOps, LowerLinearizeIndexOps>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::affine::createAffineExpandIndexOpsAsAffinePass() {
  return std::make_unique<ExpandAffineIndexOpsAsAffinePass>();
}
