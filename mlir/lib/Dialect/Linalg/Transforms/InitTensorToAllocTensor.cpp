//===- InitTensorToAllocTensor.cpp - Lower init_tensor to alloc_tensor ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGINITTENSORTOALLOCTENSOR
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

namespace {
struct InitTensorLoweringPattern : public OpRewritePattern<InitTensorOp> {
  using OpRewritePattern<InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InitTensorOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(op, op.getType(),
                                                              op.getSizes());
    return success();
  }
};

struct LinalgInitTensorToAllocTensor
    : public impl::LinalgInitTensorToAllocTensorBase<
          LinalgInitTensorToAllocTensor> {
  LinalgInitTensorToAllocTensor() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, bufferization::BufferizationDialect>();
  }
};
} // namespace

void LinalgInitTensorToAllocTensor::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  patterns.insert<InitTensorLoweringPattern>(op->getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLinalgInitTensorToAllocTensorPass() {
  return std::make_unique<LinalgInitTensorToAllocTensor>();
}
