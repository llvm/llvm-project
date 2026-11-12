//===- TestSCFWrapInZeroTripCheck.cpp -- Pass to test SCF zero-trip-check -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the passes to test wrap-in-zero-trip-check transforms on
// SCF loop ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TestWrapWhileLoopInZeroTripCheckPass
    : public PassWrapper<TestWrapWhileLoopInZeroTripCheckPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestWrapWhileLoopInZeroTripCheckPass)

  StringRef getArgument() const final {
    return "test-wrap-scf-while-loop-in-zero-trip-check";
  }

  StringRef getDescription() const final {
    return "test scf::wrapWhileLoopInZeroTripCheck";
  }

  TestWrapWhileLoopInZeroTripCheckPass() = default;
  TestWrapWhileLoopInZeroTripCheckPass(
      const TestWrapWhileLoopInZeroTripCheckPass &) {}
  explicit TestWrapWhileLoopInZeroTripCheckPass(bool forceCreateCheckParam) {
    forceCreateCheck = forceCreateCheckParam;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (forceCreateCheck) {
      func.walk([&](scf::WhileOp op) {
        FailureOr<scf::WhileOp> result =
            scf::wrapWhileLoopInZeroTripCheck(op, rewriter, forceCreateCheck);
        // Ignore not implemented failure in tests. The expected output should
        // catch problems (e.g. transformation doesn't happen).
        (void)result;
      });
    } else {
      RewritePatternSet patterns(context);
      scf::populateSCFRotateWhileLoopPatterns(patterns);
      (void)applyPatternsGreedily(func, std::move(patterns));
    }
  }

  Option<bool> forceCreateCheck{
      *this, "force-create-check",
      llvm::cl::desc("Force to create zero-trip-check."),
      llvm::cl::init(false)};
};

} // namespace

namespace mlir {
namespace test {
void registerTestSCFWrapInZeroTripCheckPasses() {
  PassRegistration<TestWrapWhileLoopInZeroTripCheckPass>();
}
} // namespace test
} // namespace mlir
