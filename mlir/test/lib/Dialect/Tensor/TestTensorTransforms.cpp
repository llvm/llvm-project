//===- TestTensorTransforms.cpp - Test Tensor transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Tensor transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestTensorTransforms
    : public PassWrapper<TestTensorTransforms, OperationPass<FuncOp>> {
  TestTensorTransforms() = default;
  TestTensorTransforms(const TestTensorTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, scf::SCFDialect>();
  }

  StringRef getArgument() const final {
    return "test-tensor-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Tensor transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testSplitPaddingPatterns{
      *this, "test-split-padding-patterns",
      llvm::cl::desc("Test patterns to split tensor.pad ops"),
      llvm::cl::init(false)};
};
} // namespace

static void applySplitPaddingPatterns(FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  tensor::populateSplitPaddingPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

void TestTensorTransforms::runOnOperation() {
  FuncOp func = getOperation();
  if (testSplitPaddingPatterns)
    applySplitPaddingPatterns(func);
}

namespace mlir {
namespace test {
void registerTestTensorTransforms() {
  PassRegistration<TestTensorTransforms>();
}
} // namespace test
} // namespace mlir
