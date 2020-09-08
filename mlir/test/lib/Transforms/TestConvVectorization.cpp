//===- TestConvVectorization.cpp - Linalg to Vector dialect conversion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// A pass converting MLIR Linalg ops into Vector ops.
class TestConvVectorization
    : public PassWrapper<TestConvVectorization, OperationPass<ModuleOp>> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<StandardOpsDialect>();
  }
};
} // namespace

void TestConvVectorization::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, StandardOpsDialect,
                         vector::VectorDialect>();
  target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ReturnOp>();
  target.addLegalOp<linalg::FillOp, linalg::YieldOp>();

  OwningRewritePatternList patterns;
  linalg::populateConvVectorizationPatterns(context, patterns);

  if (failed(applyPartialConversion(module, target, patterns)))
    return signalPassFailure();
}

namespace mlir {
void registerTestConvVectorization() {
  PassRegistration<TestConvVectorization> testTransformPatternsPass(
      "test-conv-vectorization", "Test vectorization of convolutions");
}
} // namespace mlir
