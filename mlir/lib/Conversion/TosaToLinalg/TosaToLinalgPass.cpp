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
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOLINALG
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TosaToLinalg : public impl::TosaToLinalgBase<TosaToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           scf::SCFDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addLegalOp<tosa::ApplyScaleOp>();
    target.addLegalOp<tosa::IfOp>();
    target.addLegalOp<tosa::ConstOp>();
    target.addLegalOp<tosa::WhileOp>();
    target.addLegalOp<tosa::SliceOp>();
    target.addLegalOp<tosa::PadOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FunctionOpInterface func = getOperation();
    mlir::tosa::populateTosaToLinalgConversionPatterns(&patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToLinalg() {
  return std::make_unique<TosaToLinalg>();
}

void mlir::tosa::addTosaToLinalgPasses(OpPassManager &pm,
                                       bool disableTosaDecompositions) {
  // Optional decompositions are designed to benefit linalg.
  if (!disableTosaDecompositions)
    pm.addNestedPass<func::FuncOp>(tosa::createTosaOptionalDecompositions());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // TODO: Remove pass that operates on const tensor and enable optionality
  pm.addNestedPass<func::FuncOp>(tosa::createTosaLayerwiseConstantFoldPass());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaValidationPass());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
}
