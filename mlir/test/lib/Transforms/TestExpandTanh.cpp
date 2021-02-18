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
    : public PassWrapper<TestExpandTanhPass, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestExpandTanhPass::runOnFunction() {
  OwningRewritePatternList patterns;
  populateExpandTanhPattern(patterns, &getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestExpandTanhPass() {
  PassRegistration<TestExpandTanhPass> pass("test-expand-tanh",
                                            "Test expanding tanh");
}
} // namespace test
} // namespace mlir
