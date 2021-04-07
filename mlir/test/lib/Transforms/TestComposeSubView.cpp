//===- TestComposeSubView.cpp - Test composed subviews --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the composed subview patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/Transforms/ComposeSubView.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestComposeSubViewPass
    : public PassWrapper<TestComposeSubViewPass, FunctionPass> {
  void runOnFunction() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};

void TestComposeSubViewPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<AffineDialect>();
}

void TestComposeSubViewPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  populateComposeSubViewPatterns(patterns, &getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
} // namespace

namespace mlir {
namespace test {
void registerTestComposeSubView() {
  PassRegistration<TestComposeSubViewPass> pass(
      "test-compose-subview", "Test combining composed subviews");
}
} // namespace test
} // namespace mlir
