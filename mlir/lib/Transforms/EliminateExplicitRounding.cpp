//===- EliminateExplicitRounding.cpp - Remove redundant extf/truncf pairs -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements removing redundant extf/truncf pairs inserted from
// LegalizeToF32 and EmulateUnsupportedFloats.
//
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ELIMINATEEXPLICITROUNDING
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct EliminateExplicitRoundingRewritePattern final
    : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp extfop,
                                PatternRewriter &rewriter) const final {
    // check whether the extfop is eliminatable
    auto extfAttr = extfop->getAttrOfType<BoolAttr>("eliminatable");
    if (!extfAttr || (extfAttr && !extfAttr.getValue())) return failure();

    // check whether match `eliminatable truncf->extf` pair
    auto truncfop = extfop.getOperand().getDefiningOp<arith::TruncFOp>();
    if (!truncfop) return failure();
    auto truncfAttr = truncfop->getAttrOfType<BoolAttr>("eliminatable");
    if (!truncfAttr || (truncfAttr && !truncfAttr.getValue())) return failure();

    // check whether the the rounding pair's input and output data type are the
    // same Currently only consider to eliminate rounding pairs for (bf16 / f16
    // <-> f32)
    if (auto input = truncfop.getOperand()) {
        auto inTy = input.getType();
        auto outTy = extfop.getType();
        auto shortTy = getElementTypeOrSelf(truncfop.getType());
        if (inTy == outTy && getElementTypeOrSelf(inTy).isF32() &&
            (shortTy.isF16() || shortTy.isBF16())) {
          rewriter.replaceOp(extfop, {input});
        }
    }
    return success();
  }
};

struct EliminateExplicitRounding final
    : impl::EliminateExplicitRoundingBase<
          EliminateExplicitRounding> {
  using EliminateExplicitRoundingBase::EliminateExplicitRoundingBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<EliminateExplicitRoundingRewritePattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (isa<arith::ExtFOp>(op))
        ops.push_back(op);
    });
    if (failed(applyOpPatternsAndFold(ops, patternSet)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createEliminateExplicitRoundingPass() {
  return std::make_unique<EliminateExplicitRounding>();
}
