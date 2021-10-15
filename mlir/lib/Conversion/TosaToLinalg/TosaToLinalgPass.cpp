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

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TosaToLinalg : public TosaToLinalgBase<TosaToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, linalg::LinalgDialect,
                    math::MathDialect, StandardOpsDialect,
                    tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnFunction() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addLegalOp<tosa::ApplyScaleOp>();
    target.addLegalOp<tosa::IfOp>();
    target.addLegalOp<tosa::ConstOp>();
    target.addLegalOp<tosa::WhileOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FuncOp func = getFunction();
    mlir::tosa::populateTosaToLinalgConversionPatterns(&patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToLinalg() {
  return std::make_unique<TosaToLinalg>();
}

void mlir::tosa::addTosaToLinalgPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTosaMakeBroadcastablePass());
  pm.addNestedPass<FuncOp>(createTosaToLinalg());
}
