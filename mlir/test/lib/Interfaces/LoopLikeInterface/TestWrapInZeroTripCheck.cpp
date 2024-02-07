//===- TestWrapInZeroTripCheck.cpp -- Pass to test wrapInZeroTripCheck ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the passes to test wrapInZeroTripCheck of loop ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestWrapInZeroTripCheck
    : public PassWrapper<TestWrapInZeroTripCheck, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWrapInZeroTripCheck)

  StringRef getArgument() const final { return "test-wrap-in-zero-trip-check"; }
  StringRef getDescription() const final {
    return "test wrapInZeroTripCheck of loop ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    func.walk([&](LoopLikeOpInterface op) {
      auto result = op.wrapInZeroTripCheck(rewriter);
      if (failed(result)) {
        // Ignore not implemented failure in tests. The expected output should
        // catch problems (e.g. transformation doesn't happen).
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestWrapInZeroTripCheckPass() {
  PassRegistration<TestWrapInZeroTripCheck>();
}
} // namespace test
} // namespace mlir
