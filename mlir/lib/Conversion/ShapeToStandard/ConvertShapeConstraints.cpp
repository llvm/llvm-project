//===- ConvertShapeConstraints.cpp - Conversion of shape constraints ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {
#include "ShapeToStandard.cpp.inc"
} // namespace

namespace {
class ConvertCstrRequireOp : public OpRewritePattern<shape::CstrRequireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(shape::CstrRequireOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<AssertOp>(op.getLoc(), op.pred(), op.msgAttr());
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

void mlir::populateConvertShapeConstraintsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<CstrBroadcastableToRequire>(ctx);
  patterns.insert<CstrEqToRequire>(ctx);
  patterns.insert<ConvertCstrRequireOp>(ctx);
}

namespace {
// This pass eliminates shape constraints from the program, converting them to
// eager (side-effecting) error handling code. After eager error handling code
// is emitted, witnesses are satisfied, so they are replace with
// `shape.const_witness true`.
class ConvertShapeConstraints
    : public ConvertShapeConstraintsBase<ConvertShapeConstraints> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    populateConvertShapeConstraintsConversionPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertShapeConstraintsPass() {
  return std::make_unique<ConvertShapeConstraints>();
}
