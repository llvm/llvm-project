//===- TestDataLayoutPropagation.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestDataLayoutPropagationPass
    : public PassWrapper<TestDataLayoutPropagationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataLayoutPropagationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  StringRef getArgument() const final {
    return "test-linalg-data-layout-propagation";
  }
  StringRef getDescription() const final {
    return "Test data layout propagation";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateDataLayoutPropagationPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestDataLayoutPropagation() {
  PassRegistration<TestDataLayoutPropagationPass>();
}
} // namespace test
} // namespace mlir
