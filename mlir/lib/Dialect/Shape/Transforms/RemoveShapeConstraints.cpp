//===-- RemoveShapeConstraints.cpp - Remove Shape Cstr and Assuming Ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
/// Removal patterns.
class RemoveCstrBroadcastableOp
    : public OpRewritePattern<shape::CstrBroadcastableOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op.getOperation(), true);
    return success();
  }
};

class RemoveCstrEqOp : public OpRewritePattern<shape::CstrEqOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrEqOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op.getOperation(), true);
    return success();
  }
};

/// Removal pass.
class RemoveShapeConstraintsPass
    : public RemoveShapeConstraintsBase<RemoveShapeConstraintsPass> {

  void runOnFunction() override {
    MLIRContext &ctx = getContext();

    OwningRewritePatternList patterns;
    populateRemoveShapeConstraintsPatterns(patterns, &ctx);

    applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

} // namespace

void mlir::populateRemoveShapeConstraintsPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<RemoveCstrBroadcastableOp, RemoveCstrEqOp>(ctx);
}

std::unique_ptr<FunctionPass> mlir::createRemoveShapeConstraintsPass() {
  return std::make_unique<RemoveShapeConstraintsPass>();
}
