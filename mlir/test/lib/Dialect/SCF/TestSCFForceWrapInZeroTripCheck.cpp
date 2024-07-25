//===- TestSCFForceWrapInZeroTripCheck.cpp -- Pass to test SCF zero-trip-check//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass to test wrap-in-zero-trip-check transforms on
// SCF loop ops. This pass only tests the case in which transformation is
// forced, i.e., when `forceCreateCheck = true`, as the other case is covered by
// the `-scf-rotate-while` pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestForceWrapWhileLoopInZeroTripCheckPass
    : PassWrapper<TestForceWrapWhileLoopInZeroTripCheckPass,
                  OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestForceWrapWhileLoopInZeroTripCheckPass)

  using PassWrapper<TestForceWrapWhileLoopInZeroTripCheckPass,
                    OperationPass<func::FuncOp>>::PassWrapper;

  StringRef getArgument() const final {
    return "test-force-wrap-scf-while-loop-in-zero-trip-check";
  }

  StringRef getDescription() const final {
    return "test scf::wrapWhileLoopInZeroTripCheck whith forceCreateCheck=true";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    func.walk([&](scf::WhileOp op) {
      // `forceCreateCheck=false` case is already tested by the
      // `-scf-rotate-while` pass using this function in its pattern.
      constexpr bool forceCreateCheck = true;
      FailureOr<scf::WhileOp> result =
          scf::wrapWhileLoopInZeroTripCheck(op, rewriter, forceCreateCheck);
      // Ignore not implemented failure in tests. The expected output should
      // catch problems (e.g. transformation doesn't happen).
      (void)result;
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestSCFForceWrapInZeroTripCheckPasses() {
  PassRegistration<TestForceWrapWhileLoopInZeroTripCheckPass>();
}
} // namespace test
} // namespace mlir
