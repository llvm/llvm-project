//===- InitTensorToAllocTensor.cpp - Lower init_tensor to alloc_tensor ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_EMPTYTENSORTOALLOCTENSOR
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace {
struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};

struct EmptyTensorToAllocTensor
    : public impl::EmptyTensorToAllocTensorBase<EmptyTensorToAllocTensor> {
  EmptyTensorToAllocTensor() = default;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tensor::TensorDialect, bufferization::BufferizationDialect>();
  }
};
} // namespace

void EmptyTensorToAllocTensor::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  patterns.insert<EmptyTensorLoweringPattern>(op->getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createEmptyTensorToAllocTensorPass() {
  return std::make_unique<EmptyTensorToAllocTensor>();
}
