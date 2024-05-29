//===- EliminateExplicitRounding.cpp - Remove intermediate extf/truncf pairs
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements removing intermediate extf/truncf pairs inserted from
// type conversion.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ELIMINATEEXPLICITROUNDING
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;

namespace {

struct EliminateExplicitRoundingRewritePattern final
    : OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  using FilterFunction = std::function<bool(Operation *)>;

  EliminateExplicitRoundingRewritePattern(MLIRContext *context,
                                          FilterFunction filterFunc = nullptr)
      : OpRewritePattern(context), filterFunc(filterFunc) {}

  LogicalResult matchAndRewrite(arith::ExtFOp extFOp,
                                PatternRewriter &rewriter) const final {
    // Check whether match `truncF->extF` pair.
    auto truncFOp = extFOp.getOperand().getDefiningOp<arith::TruncFOp>();
    if (!truncFOp)
      return failure();

    // Check whether need to filter out.
    if (filterFunc && filterFunc(extFOp))
      return failure();

    // Check whether the rounding pair's input and output data type are the
    // same. Currently only consider to eliminate rounding pairs for (bf16 / f16
    // <-> f32).
    auto input = truncFOp.getOperand();
    auto inTy = input.getType();
    auto outTy = extFOp.getType();
    auto shortTy = getElementTypeOrSelf(truncFOp.getType());
    if (inTy == outTy && getElementTypeOrSelf(inTy).isF32() &&
        (shortTy.isF16() || shortTy.isBF16())) {
      rewriter.replaceOp(extFOp, {input});
      return success();
    }

    return failure();
  }

private:
  FilterFunction filterFunc;
};

struct EliminateExplicitRounding final
    : arith::impl::EliminateExplicitRoundingBase<EliminateExplicitRounding> {
  using EliminateExplicitRoundingBase::EliminateExplicitRoundingBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<EliminateExplicitRoundingRewritePattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    SmallVector<Operation *> ops;
    getOperation()->walk([&](arith::ExtFOp op) { ops.push_back(op); });
    if (failed(applyOpPatternsAndFold(ops, patternSet)))
      signalPassFailure();
  }
};

} // namespace
