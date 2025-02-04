//===- TestPolynomialApproximation.cpp - Test math ops approximations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for expanding math operations into
// polynomial approximations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestMathPolynomialApproximationPass
    : public PassWrapper<TestMathPolynomialApproximationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestMathPolynomialApproximationPass)

  TestMathPolynomialApproximationPass() = default;
  TestMathPolynomialApproximationPass(
      const TestMathPolynomialApproximationPass &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect>();
    if (enableAvx2)
      registry.insert<x86vector::X86VectorDialect>();
  }
  StringRef getArgument() const final {
    return "test-math-polynomial-approximation";
  }
  StringRef getDescription() const final {
    return "Test math polynomial approximations";
  }

  Option<bool> enableAvx2{
      *this, "enable-avx2",
      llvm::cl::desc("Enable approximations that emit AVX2 intrinsics via the "
                     "X86Vector dialect"),
      llvm::cl::init(false)};
};
} // namespace

void TestMathPolynomialApproximationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  MathPolynomialApproximationOptions approxOptions;
  approxOptions.enableAvx2 = enableAvx2;
  populateMathPolynomialApproximationPatterns(patterns, approxOptions);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestMathPolynomialApproximationPass() {
  PassRegistration<TestMathPolynomialApproximationPass>();
}
} // namespace test
} // namespace mlir
