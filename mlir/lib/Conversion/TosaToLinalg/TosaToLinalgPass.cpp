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
#include "mlir/Dialect/Index/IR/IndexDialect.h"
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
                index::IndexDialect, tensor::TensorDialect, scf::SCFDialect>();
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
    target.addLegalOp<tosa::ConstShapeOp>();
    target.addLegalOp<tosa::WhileOp>();
    target.addLegalOp<tosa::ConcatOp>();
    target.addLegalOp<tosa::SliceOp>();
    target.addLegalOp<tosa::ReshapeOp>();
    target.addLegalOp<tosa::PadOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    TypeConverter converter;
    tosa::populateTosaTypeConversion(converter);

    FunctionOpInterface func = getOperation();
    mlir::tosa::populateTosaToLinalgConversionPatterns(converter, &patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToLinalg() {
  return std::make_unique<TosaToLinalg>();
}

void mlir::tosa::addTosaToLinalgPasses(
    OpPassManager &pm, const TosaToLinalgOptions &options,
    const TosaToLinalgNamedOptions &tosaToLinalgNamedOptions,
    std::optional<tosa::TosaValidationOptions> validationOptions) {
  // Optional decompositions are designed to benefit linalg.
  if (!options.disableTosaDecompositions)
    pm.addNestedPass<func::FuncOp>(tosa::createTosaOptionalDecompositions());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(tosa::createTosaInferShapesPass());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
  pm.addNestedPass<func::FuncOp>(
      tosa::createTosaToLinalgNamed(tosaToLinalgNamedOptions));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // TODO: Remove pass that operates on const tensor and enable optionality
  pm.addNestedPass<func::FuncOp>(tosa::createTosaLayerwiseConstantFoldPass(
      {options.aggressiveReduceConstant}));
  pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
  if (validationOptions)
    pm.addPass(tosa::createTosaValidation(*validationOptions));
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::tosa::registerTosaToLinalgPipelines() {
  PassPipelineRegistration<>(
      "tosa-to-linalg-pipeline",
      "The default pipeline for converting TOSA operators to the equivalent "
      "operations using the tensor operations in LinAlg as well as LinAlg "
      "named operations.",
      [](OpPassManager &pm) {
        TosaToLinalgOptions tosaToLinalgOptions;
        TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
        TosaValidationOptions validationOptions;
        validationOptions.profile = {"none"};
        validationOptions.extension = {"none"};
        validationOptions.strictOpSpecAlignment = false;
        validationOptions.level = tosa::TosaLevelEnum::EightK;
        tosa::addTosaToLinalgPasses(pm, tosaToLinalgOptions,
                                    tosaToLinalgNamedOptions,
                                    validationOptions);
      });
}
