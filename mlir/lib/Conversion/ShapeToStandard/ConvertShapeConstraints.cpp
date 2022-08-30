//===- ConvertShapeConstraints.cpp - Conversion of shape constraints ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSHAPECONSTRAINTSPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

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
    rewriter.create<cf::AssertOp>(op.getLoc(), op.getPred(), op.getMsgAttr());
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
    return success();
  }
};
} // namespace

void mlir::populateConvertShapeConstraintsConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CstrBroadcastableToRequire>(patterns.getContext());
  patterns.add<CstrEqToRequire>(patterns.getContext());
  patterns.add<ConvertCstrRequireOp>(patterns.getContext());
}

namespace {
// This pass eliminates shape constraints from the program, converting them to
// eager (side-effecting) error handling code. After eager error handling code
// is emitted, witnesses are satisfied, so they are replace with
// `shape.const_witness true`.
class ConvertShapeConstraintsPass
    : public impl::ConvertShapeConstraintsPassBase<
          ConvertShapeConstraintsPass> {
  using ConvertShapeConstraintsPassBase::ConvertShapeConstraintsPassBase;

  void runOnOperation() override {
    auto *func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    populateConvertShapeConstraintsConversionPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
