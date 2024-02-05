//===- TestLoopZeroTripCheck.cpp.cpp -- Pass to test replaceWithZeroTripC--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the passes to test replaceWithZeroTripCheck of loop ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestLoopZeroTripCheck
    : public PassWrapper<TestLoopZeroTripCheck, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopZeroTripCheck)

  StringRef getArgument() const final { return "test-loop-zero-trip-check"; }
  StringRef getDescription() const final {
    return "test replaceWithZeroTripCheck of loop ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    func.walk([&](LoopLikeOpInterface op) {
      auto result = op.replaceWithZeroTripCheck(rewriter);
      if (failed(result)) {
        // Ignore failures (e.g. not implemented) in tests.
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestLoopZeroTripCheckPass() {
  PassRegistration<TestLoopZeroTripCheck>();
}
} // namespace test
} // namespace mlir
