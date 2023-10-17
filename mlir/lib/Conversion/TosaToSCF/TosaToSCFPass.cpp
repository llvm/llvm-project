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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOSCF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToSCF : public impl::TosaToSCFBase<TosaToSCF> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<tosa::IfOp, tosa::ScatterOp, tosa::WhileOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    auto *op = getOperation();
    mlir::tosa::populateTosaToSCFConversionPatterns(&patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToSCF() {
  return std::make_unique<TosaToSCF>();
}

void mlir::tosa::addTosaToSCFPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createTosaToSCF());
}
