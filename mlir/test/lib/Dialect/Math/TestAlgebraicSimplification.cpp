//===- TestAlgebraicSimplification.cpp - Test algebraic simplification ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for algebraic simplification patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestMathAlgebraicSimplificationPass
    : public PassWrapper<TestMathAlgebraicSimplificationPass, FunctionPass> {
  void runOnFunction() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, math::MathDialect>();
  }
  StringRef getArgument() const final {
    return "test-math-algebraic-simplification";
  }
  StringRef getDescription() const final {
    return "Test math algebraic simplification";
  }
};
} // end anonymous namespace

void TestMathAlgebraicSimplificationPass::runOnFunction() {
  RewritePatternSet patterns(&getContext());
  populateMathAlgebraicSimplificationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestMathAlgebraicSimplificationPass() {
  PassRegistration<TestMathAlgebraicSimplificationPass>();
}
} // namespace test
} // namespace mlir
