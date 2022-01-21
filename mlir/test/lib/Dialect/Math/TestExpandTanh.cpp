//===- TestExpandTanh.cpp - Test expand tanh op into exp form -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for expanding tanh.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestExpandTanhPass
    : public PassWrapper<TestExpandTanhPass, OperationPass<FuncOp>> {
  void runOnOperation() override;
  StringRef getArgument() const final { return "test-expand-tanh"; }
  StringRef getDescription() const final { return "Test expanding tanh"; }
};
} // namespace

void TestExpandTanhPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateExpandTanhPattern(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestExpandTanhPass() { PassRegistration<TestExpandTanhPass>(); }
} // namespace test
} // namespace mlir
