//===- TestTensorTransforms.cpp - Test Tensor transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Tensor transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestTensorTransforms
    : public PassWrapper<TestTensorTransforms, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTensorTransforms)

  TestTensorTransforms() = default;
  TestTensorTransforms(const TestTensorTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, scf::SCFDialect>();
  }

  StringRef getArgument() const final {
    return "test-tensor-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Tensor transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testSplitPaddingPatterns{
      *this, "test-split-padding-patterns",
      llvm::cl::desc("Test patterns to split tensor.pad ops"),
      llvm::cl::init(false)};

  Option<bool> testFoldConstantExtractSlice{
      *this, "test-fold-constant-extract-slice",
      llvm::cl::desc("Test folding arith.constant and tensor.extract_slice"),
      llvm::cl::init(false)};
};
} // namespace

static void applySplitPaddingPatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateSplitPaddingPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void applyFoldConstantExtractSlicePatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::ControlConstantExtractSliceFusionFn controlFn =
      [](tensor::ExtractSliceOp op) {
        if (!op.source().hasOneUse())
          return false;

        auto resultType = op.result().getType().cast<ShapedType>();
        constexpr int64_t kConstantFoldingMaxNumElements = 1024;
        return resultType.getNumElements() <= kConstantFoldingMaxNumElements;
      };

  tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

void TestTensorTransforms::runOnOperation() {
  Operation *rootOp = getOperation();
  if (testSplitPaddingPatterns)
    applySplitPaddingPatterns(rootOp);
  if (testFoldConstantExtractSlice)
    applyFoldConstantExtractSlicePatterns(rootOp);
}

namespace mlir {
namespace test {
void registerTestTensorTransforms() {
  PassRegistration<TestTensorTransforms>();
}
} // namespace test
} // namespace mlir
