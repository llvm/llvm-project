//===- TosaToMLProgramPass.cpp - Lowering Tosa to MLProgram Dialect--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes the TOSA dialect to the MLProgram dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOMLPROGRAM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace tosa;

namespace {
struct TosaToMLProgram : public impl::TosaToMLProgramBase<TosaToMLProgram> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addIllegalOp<tosa::VariableOp, tosa::VariableReadOp,
                        tosa::VariableWriteOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    mlir::tosa::populateTosaToMLProgramConversionPatterns(&patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
