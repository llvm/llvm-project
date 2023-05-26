//===- TestDecomposeAffineOps.cpp - Test affine ops decomposition utility -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test affine data copy utility functions and
// options.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "test-decompose-affine-ops"

using namespace mlir;
using namespace mlir::affine;

namespace {

struct TestDecomposeAffineOps
    : public PassWrapper<TestDecomposeAffineOps, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDecomposeAffineOps)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine ops decomposition utility functions.";
  }
  TestDecomposeAffineOps() = default;
  TestDecomposeAffineOps(const TestDecomposeAffineOps &pass)
      : PassWrapper(pass){};

  void runOnOperation() override;
};

} // namespace

void TestDecomposeAffineOps::runOnOperation() {
  IRRewriter rewriter(&getContext());
  this->getOperation().walk([&](AffineApplyOp op) {
    rewriter.setInsertionPoint(op);
    reorderOperandsByHoistability(rewriter, op);
    (void)decompose(rewriter, op);
  });
}

namespace mlir {
void registerTestDecomposeAffineOpPass() {
  PassRegistration<TestDecomposeAffineOps>();
}
} // namespace mlir
