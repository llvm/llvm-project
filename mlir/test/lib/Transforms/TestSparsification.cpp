//===- TestSparsification.cpp - Test sparsification of tensors ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestSparsification
    : public PassWrapper<TestSparsification, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnFunction() override {
    auto *ctx = &getContext();
    OwningRewritePatternList patterns;
    linalg::populateSparsificationPatterns(ctx, patterns);
    applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {

void registerTestSparsification() {
  PassRegistration<TestSparsification> sparsificationPass(
      "test-sparsification",
      "Test automatic geneneration of sparse tensor code");
}

} // namespace test
} // namespace mlir
