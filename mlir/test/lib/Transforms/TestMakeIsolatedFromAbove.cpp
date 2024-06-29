//===- TestMakeIsolatedFromAbove.cpp - Test makeIsolatedFromAbove method -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

/// Helper function to call the `makeRegionIsolatedFromAbove` to convert
/// `test.one_region_op` to `test.isolated_one_region_op`.
static LogicalResult
makeIsolatedFromAboveImpl(RewriterBase &rewriter,
                          test::OneRegionWithOperandsOp regionOp,
                          llvm::function_ref<bool(Operation *)> callBack) {
  Region &region = regionOp.getRegion();
  SmallVector<Value> capturedValues =
      makeRegionIsolatedFromAbove(rewriter, region, callBack);
  SmallVector<Value> operands = regionOp.getOperands();
  operands.append(capturedValues);
  auto isolatedRegionOp =
      rewriter.create<test::IsolatedOneRegionOp>(regionOp.getLoc(), operands);
  rewriter.inlineRegionBefore(region, isolatedRegionOp.getRegion(),
                              isolatedRegionOp.getRegion().begin());
  rewriter.eraseOp(regionOp);
  return success();
}

namespace {

/// Simple test for making region isolated from above without cloning any
/// operations.
struct SimpleMakeIsolatedFromAbove
    : OpRewritePattern<test::OneRegionWithOperandsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(test::OneRegionWithOperandsOp regionOp,
                                PatternRewriter &rewriter) const override {
    return makeIsolatedFromAboveImpl(rewriter, regionOp,
                                     [](Operation *) { return false; });
  }
};

/// Test for making region isolated from above while clong operations
/// with no operands.
struct MakeIsolatedFromAboveAndCloneOpsWithNoOperands
    : OpRewritePattern<test::OneRegionWithOperandsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(test::OneRegionWithOperandsOp regionOp,
                                PatternRewriter &rewriter) const override {
    return makeIsolatedFromAboveImpl(rewriter, regionOp, [](Operation *op) {
      return op->getNumOperands() == 0;
    });
  }
};

/// Test for making region isolated from above while clong operations
/// with no operands.
struct MakeIsolatedFromAboveAndCloneOpsWithOperands
    : OpRewritePattern<test::OneRegionWithOperandsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(test::OneRegionWithOperandsOp regionOp,
                                PatternRewriter &rewriter) const override {
    return makeIsolatedFromAboveImpl(rewriter, regionOp,
                                     [](Operation *op) { return true; });
  }
};

/// Test pass for testing the `makeIsolatedFromAbove` function.
struct TestMakeIsolatedFromAbovePass
    : public PassWrapper<TestMakeIsolatedFromAbovePass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMakeIsolatedFromAbovePass)

  TestMakeIsolatedFromAbovePass() = default;
  TestMakeIsolatedFromAbovePass(const TestMakeIsolatedFromAbovePass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-make-isolated-from-above";
  }

  StringRef getDescription() const final {
    return "Test making a region isolated from above";
  }

  Option<bool> simple{
      *this, "simple",
      llvm::cl::desc("Test simple case with no cloning of operations"),
      llvm::cl::init(false)};

  Option<bool> cloneOpsWithNoOperands{
      *this, "clone-ops-with-no-operands",
      llvm::cl::desc("Test case with cloning of operations with no operands"),
      llvm::cl::init(false)};

  Option<bool> cloneOpsWithOperands{
      *this, "clone-ops-with-operands",
      llvm::cl::desc("Test case with cloning of operations with no operands"),
      llvm::cl::init(false)};

  void runOnOperation() override;
};

} // namespace

void TestMakeIsolatedFromAbovePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();

  if (simple) {
    RewritePatternSet patterns(context);
    patterns.insert<SimpleMakeIsolatedFromAbove>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
    return;
  }

  if (cloneOpsWithNoOperands) {
    RewritePatternSet patterns(context);
    patterns.insert<MakeIsolatedFromAboveAndCloneOpsWithNoOperands>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
    return;
  }

  if (cloneOpsWithOperands) {
    RewritePatternSet patterns(context);
    patterns.insert<MakeIsolatedFromAboveAndCloneOpsWithOperands>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
    return;
  }
}

namespace mlir {
namespace test {
void registerTestMakeIsolatedFromAbovePass() {
  PassRegistration<TestMakeIsolatedFromAbovePass>();
}
} // namespace test
} // namespace mlir
