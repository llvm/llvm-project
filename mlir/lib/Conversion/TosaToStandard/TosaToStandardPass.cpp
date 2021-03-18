//===- TosaToStandardPass.cpp - Lowering Tosa to Linalg Dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Standard dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToStandard : public TosaToStandardBase<TosaToStandard> {
public:
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ConstOp>();
    target.addIllegalOp<tosa::SliceOp>();
    target.addLegalDialect<StandardOpsDialect>();

    auto *op = getOperation();
    mlir::tosa::populateTosaToStandardConversionPatterns(op->getContext(),
                                                         &patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToStandard() {
  return std::make_unique<TosaToStandard>();
}

void mlir::tosa::addTosaToStandardPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTosaToStandard());
}
