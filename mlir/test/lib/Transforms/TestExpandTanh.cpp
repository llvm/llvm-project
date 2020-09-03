//===- TestExpandTanh.cpp - Test expand tanh op into exp form ------===//
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

#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestExpandTanhPass
    : public PassWrapper<TestExpandTanhPass, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestExpandTanhPass::runOnFunction() {
  OwningRewritePatternList patterns;
  populateExpandTanhPattern(patterns, &getContext());
  applyPatternsAndFoldGreedily(getOperation(), patterns);
}

namespace mlir {
void registerTestExpandTanhPass() {
  PassRegistration<TestExpandTanhPass> pass("test-expand-tanh",
                                            "Test expanding tanh");
}
} // namespace mlir
