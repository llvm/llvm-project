//===- CanonicalizeF32Promotion.cpp - Remove redundant extf/truncf pairs -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements removing redundant extf/truncf pairs inserted from
// LegalizeToF32.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::math {
#define GEN_PASS_DEF_MATHCANONICALIZEF32PROMOTION
#include "mlir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace mlir::math

using namespace mlir;

namespace {

struct CanonicalizeF32PromotionRewritePattern final
    : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const final {
    if (auto innertruncop = op.getOperand().getDefiningOp<arith::TruncFOp>()) {
      if (auto truncinput = innertruncop.getOperand()) {
        auto outter_type = op.getType();
        auto intermediate_type = innertruncop.getType();
        auto inner_type = truncinput.getType();
        if (outter_type.isa<ShapedType>()) {
          outter_type = op.getType().cast<ShapedType>().getElementType();
          intermediate_type =
              innertruncop.getType().cast<ShapedType>().getElementType();
          inner_type = truncinput.getType().cast<ShapedType>().getElementType();
        }
        if (outter_type.isF32() &&
            (intermediate_type.isF16() || intermediate_type.isBF16()) &&
            inner_type.isF32()) {
          rewriter.replaceOp(op, {truncinput});
        }
      } else
        return failure();
    } else
      return failure();
    return success();
  }
};

struct MathCanonicalizeF32Promotion final
    : math::impl::MathCanonicalizeF32PromotionBase<
          MathCanonicalizeF32Promotion> {
  using MathCanonicalizeF32PromotionBase::MathCanonicalizeF32PromotionBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<CanonicalizeF32PromotionRewritePattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
