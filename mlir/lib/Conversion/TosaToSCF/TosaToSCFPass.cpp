//===- TosaToSCFPass.cpp - Lowering Tosa to SCF Dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOSCFPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToSCF : public impl::TosaToSCFPassBase<TosaToSCF> {
public:
  TosaToSCF(bool scatterHardening)
      : impl::TosaToSCFPassBase<TosaToSCF>(),
        scatterHardening(scatterHardening) {};

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<tosa::IfOp, tosa::ScatterOp, tosa::WhileOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    auto *op = getOperation();
    mlir::tosa::populateTosaToSCFConversionPatterns(&patterns,
                                                    scatterHardening);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

private:
  bool scatterHardening = true;
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToSCFPass(bool scatterHardening) {
  return std::make_unique<TosaToSCF>(scatterHardening);
}

void mlir::tosa::addTosaToSCFPasses(OpPassManager &pm,
                                    const TosaToSCFPassOptions &options) {
  pm.addNestedPass<func::FuncOp>(createTosaToSCFPass(options.scatterHardening));
}
