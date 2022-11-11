//===- TosaToLinalgPass.cpp - Lowering Tosa to Linalg Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOLINALGNAMED
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TosaToLinalgNamed
    : public impl::TosaToLinalgNamedBase<TosaToLinalgNamed> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tosa::TosaDialect,
                           tensor::TensorDialect, scf::SCFDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addIllegalOp<tosa::Conv2DOp>();
    target.addIllegalOp<tosa::DepthwiseConv2DOp>();
    target.addIllegalOp<tosa::MaxPool2dOp>();
    target.addIllegalOp<tosa::AvgPool2dOp>();
    target.addIllegalOp<tosa::MatMulOp>();
    target.addIllegalOp<tosa::FullyConnectedOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FunctionOpInterface func = getOperation();
    mlir::tosa::populateTosaToLinalgNamedConversionPatterns(&patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToLinalgNamed() {
  return std::make_unique<TosaToLinalgNamed>();
}
