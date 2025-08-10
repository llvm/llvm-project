//===- SCFToAffine.cpp - SCF to Affine conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise scf.for, scf.if and loop.terminator
// ops into affine ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_RAISESCFTOAFFINEPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct SCFToAffinePass
    : public impl::RaiseSCFToAffinePassBase<SCFToAffinePass> {
  void runOnOperation() override;
};

bool canRaiseToAffine(scf::ForOp op) {
  return affine::isValidDim(op.getLowerBound()) &&
         affine::isValidDim(op.getUpperBound()) &&
         affine::isValidSymbol(op.getStep());
}

struct ForOpRewrite : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  std::pair<affine::AffineForOp, Value>
  createAffineFor(scf::ForOp op, PatternRewriter &rewriter) const {
    if (auto constantStep = op.getStep().getDefiningOp<arith::ConstantOp>()) {
      int64_t step = cast<IntegerAttr>(constantStep.getValue()).getInt();
      if (step > 0)
        return positiveConstantStep(op, step, rewriter);
    }
    return genericBounds(op, rewriter);
  }

  std::pair<affine::AffineForOp, Value>
  positiveConstantStep(scf::ForOp op, int64_t step,
                       PatternRewriter &rewriter) const {
    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), ValueRange(op.getLowerBound()),
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0)),
        ValueRange(op.getUpperBound()),
        AffineMap::get(1, 0, rewriter.getAffineDimExpr(0)), step,
        op.getInits());
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }

  std::pair<affine::AffineForOp, Value>
  genericBounds(scf::ForOp op, PatternRewriter &rewriter) const {
    Value lower = op.getLowerBound();
    Value upper = op.getUpperBound();
    Value step = op.getStep();
    AffineExpr lowerExpr = rewriter.getAffineDimExpr(0);
    AffineExpr upperExpr = rewriter.getAffineDimExpr(1);
    AffineExpr stepExpr = rewriter.getAffineSymbolExpr(0);
    auto affineFor = affine::AffineForOp::create(
        rewriter, op.getLoc(), ValueRange(), rewriter.getConstantAffineMap(0),
        ValueRange({lower, upper, step}),
        AffineMap::get(
            2, 1, (upperExpr - lowerExpr + stepExpr - 1).floorDiv(stepExpr)),
        1, op.getInits());

    rewriter.setInsertionPointToStart(affineFor.getBody());
    auto actualIndexMap = AffineMap::get(
        2, 1, lowerExpr + rewriter.getAffineDimExpr(1) * stepExpr);
    auto actualIndex = affine::AffineApplyOp::create(
        rewriter, op.getLoc(), actualIndexMap,
        ValueRange({lower, affineFor.getInductionVar(), step}));
    return std::make_pair(affineFor, actualIndex.getResult());
  }

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!canRaiseToAffine(op))
      return failure();

    auto [affineFor, actualIndex] = createAffineFor(op, rewriter);
    Block *affineBody = affineFor.getBody();

    if (affineBody->mightHaveTerminator())
      rewriter.eraseOp(affineBody->getTerminator());

    SmallVector<Value> argValues;
    argValues.push_back(actualIndex);
    llvm::append_range(argValues, affineFor.getRegionIterArgs());
    rewriter.inlineBlockBefore(op.getBody(), affineBody, affineBody->end(),
                               argValues);

    auto scfYieldOp = cast<scf::YieldOp>(affineBody->getTerminator());
    rewriter.setInsertionPointToEnd(affineBody);
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
        scfYieldOp, scfYieldOp->getOperands());

    rewriter.replaceOp(op, affineFor);
    return success();
  }
};

} // namespace

void mlir::populateSCFToAffineConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForOpRewrite>(patterns.getContext());
}

void SCFToAffinePass::runOnOperation() {
  MLIRContext &ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateSCFToAffineConversionPatterns(patterns);

  // Configure conversion to raise SCF operations.
  ConversionTarget target(ctx);
  target.addDynamicallyLegalOp<scf::ForOp>(
      [](scf::ForOp op) { return !canRaiseToAffine(op); });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
