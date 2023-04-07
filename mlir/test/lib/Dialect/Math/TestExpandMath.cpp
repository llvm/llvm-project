//===- TestExpandMath.cpp - Test expand math op into exp form -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for expanding math operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestExpandMathPass
    : public PassWrapper<TestExpandMathPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestExpandMathPass)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-expand-math"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  StringRef getDescription() const final { return "Test expanding math"; }
};
} // namespace

void TestExpandMathPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateExpandCtlzPattern(patterns);
  populateExpandTanPattern(patterns);
  populateExpandTanhPattern(patterns);
  populateExpandFmaFPattern(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestExpandMathPass() { PassRegistration<TestExpandMathPass>(); }
} // namespace test
} // namespace mlir
