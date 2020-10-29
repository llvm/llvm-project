//===- TestExpandMemRefReshape.cpp - Test expansion of memref_reshape -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for expanding memref reshape.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestExpandMemRefReshapePass
    : public PassWrapper<TestExpandMemRefReshapePass, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestExpandMemRefReshapePass::runOnFunction() {
  OwningRewritePatternList patterns;
  populateExpandMemRefReshapePattern(patterns, &getContext());
  applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
void registerTestExpandMemRefReshapePass() {
  PassRegistration<TestExpandMemRefReshapePass> pass(
      "test-expand-memref-reshape", "Test expanding memref reshape");
}
} // namespace mlir
