//===- TestLinalgDecomposeOps.cpp - Test Linalg decomposition  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing decomposition of Linalg ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestLinalgDecomposeOps
    : public PassWrapper<TestLinalgDecomposeOps, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgDecomposeOps)

  TestLinalgDecomposeOps() = default;
  TestLinalgDecomposeOps(const TestLinalgDecomposeOps &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-decompose-ops"; }
  StringRef getDescription() const final {
    return "Test Linalg decomposition patterns";
  }

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    RewritePatternSet decompositionPatterns(context);
    linalg::populateDecomposeLinalgOpsPattern(decompositionPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(decompositionPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgDecomposeOps() {
  PassRegistration<TestLinalgDecomposeOps>();
}
} // namespace test
} // namespace mlir
