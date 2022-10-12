//===- TestLinalgTransforms.cpp - Test Linalg transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgTransforms
    : public PassWrapper<TestLinalgTransforms, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgTransforms)

  TestLinalgTransforms() = default;
  TestLinalgTransforms(const TestLinalgTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    bufferization::BufferizationDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    linalg::LinalgDialect,
                    vector::VectorDialect,
                    gpu::GPUDialect>();
    // clang-format on
  }
  StringRef getArgument() const final {
    return "test-linalg-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Linalg transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testPatterns{*this, "test-patterns",
                            llvm::cl::desc("Test a mixed set of patterns"),
                            llvm::cl::init(false)};
  Option<bool> testVectorTransferForwardingPatterns{
      *this, "test-vector-transfer-forwarding-patterns",
      llvm::cl::desc(
          "Test a fused pass that forwards memref.copy to vector.transfer"),
      llvm::cl::init(false)};
  Option<bool> testGenericToVectorPattern{
      *this, "test-linalg-to-vector-patterns",
      llvm::cl::desc("Test a set of patterns that rewrite a linalg contraction "
                     "in vector.contract form"),
      llvm::cl::init(false)};
  Option<bool> testTransformPadTensor{
      *this, "test-transform-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testGeneralizePadTensor{
      *this, "test-generalize-pad-tensor",
      llvm::cl::desc("Test transform pad tensor by copying with generic ops"),
      llvm::cl::init(false)};
  Option<bool> testSwapSubTensorPadTensor{
      *this, "test-swap-subtensor-padtensor",
      llvm::cl::desc("Test rewrite of subtensor(tensor.pad) into "
                     "tensor.pad(subtensor)"),
      llvm::cl::init(false)};
  Option<bool> testSplitReduction{
      *this, "test-split-reduction",
      llvm::cl::desc("Test split reduction transformation"),
      llvm::cl::init(false)};
  Option<bool> testSplitReductionInnerParallel{
      *this, "test-split-reduction-inner-parallel",
      llvm::cl::desc("Test split reduction with inner parallel transformation"),
      llvm::cl::init(false)};
  ListOption<int64_t> peeledLoops{
      *this, "peeled-loops",
      llvm::cl::desc("Loops to be peeled when test-tile-pattern")};
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Linalg tile sizes for test-tile-pattern")};
  Option<bool> skipPartial{
      *this, "skip-partial",
      llvm::cl::desc("Skip loops inside partial iterations during peeling"),
      llvm::cl::init(false)};
  Option<std::string> loopType{
      *this, "loop-type",
      llvm::cl::desc("Specify the type of loops to generate: for, parallel or "
                     "tiled_loop"),
      llvm::cl::init("for")};
  Option<bool> testBubbleUpExtractSliceOpPattern{
      *this, "test-bubble-up-extract-slice-op-pattern",
      llvm::cl::desc("Test rewrite of linalgOp + extract_slice into "
                     "extract_slice + linalgOp"),
      llvm::cl::init(false)};
  Option<bool> testSwapExtractSliceWithFill{
      *this, "test-swap-extract-slice-with-fill-pattern",
      llvm::cl::desc(
          "Test patterns to swap tensor.extract_slice(linalg.fill())"),
      llvm::cl::init(false)};
};
} // namespace

static void applyPatterns(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  RewritePatternSet patterns(ctx);

  //===--------------------------------------------------------------------===//
  // Linalg distribution patterns.
  //===--------------------------------------------------------------------===//
  LinalgLoopDistributionOptions distributionOptions;

  //===--------------------------------------------------------------------===//
  // Linalg to vector contraction patterns.
  //===--------------------------------------------------------------------===//
  patterns.add<CopyVectorizationPattern>(ctx);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // Drop the marker.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

static void applyVectorTransferForwardingPatterns(func::FuncOp funcOp) {
  RewritePatternSet forwardPattern(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTRForwardingPattern>(funcOp.getContext());
  forwardPattern.add<LinalgCopyVTWForwardingPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(forwardPattern));
}

static void applyLinalgToVectorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  auto *ctx = funcOp.getContext();
  patterns.add<CopyVectorizationPattern>(ctx);
  populatePadOpVectorizationPatterns(patterns);
  populateConvolutionVectorizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyPadTensorToGenericPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<PadOpTransformationPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyGeneralizePadTensorPatterns(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<GeneralizePadOpPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyExtractSliceOfPadTensorSwapPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<ExtractSliceOfPadTensorSwapPattern>(funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applySplitReduction(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  linalg::populateSplitReductionPattern(
      patterns,
      [](LinalgOp op) {
        unsigned insertDimIndex = op.getNumLoops() - 1;
        return SplitReductionOptions{4, insertDimIndex, false};
      },
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{},
          StringAttr::get(funcOp.getContext(), "SPLIT")));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applySplitReductionInnerParallel(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  linalg::populateSplitReductionPattern(
      patterns,
      [](LinalgOp op) {
        unsigned insertDimIndex = op.getNumLoops() - 1;
        return SplitReductionOptions{4, insertDimIndex, true};
      },
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{},
          StringAttr::get(funcOp.getContext(), "SPLIT")));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applyBubbleUpExtractSliceOpPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateBubbleUpExtractSliceOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

static void applySwapExtractSliceWithFillPattern(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateSwapExtractSliceWithFillPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

/// Apply transformations specified as patterns.
void TestLinalgTransforms::runOnOperation() {
  auto lambda = [&](void *) {
    getOperation().walk([](LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  };
  std::unique_ptr<void, decltype(lambda)> cleanupGuard{(void *)1, lambda};

  if (testPatterns)
    return applyPatterns(getOperation());
  if (testVectorTransferForwardingPatterns)
    return applyVectorTransferForwardingPatterns(getOperation());
  if (testGenericToVectorPattern)
    return applyLinalgToVectorPatterns(getOperation());
  if (testTransformPadTensor)
    return applyPadTensorToGenericPatterns(getOperation());
  if (testGeneralizePadTensor)
    return applyGeneralizePadTensorPatterns(getOperation());
  if (testSwapSubTensorPadTensor)
    return applyExtractSliceOfPadTensorSwapPattern(getOperation());
  if (testSplitReduction)
    return applySplitReduction(getOperation());
  if (testSplitReductionInnerParallel)
    return applySplitReductionInnerParallel(getOperation());
  if (testBubbleUpExtractSliceOpPattern)
    return applyBubbleUpExtractSliceOpPattern(getOperation());
  if (testSwapExtractSliceWithFill)
    return applySwapExtractSliceWithFillPattern(getOperation());
}

namespace mlir {
namespace test {
void registerTestLinalgTransforms() {
  PassRegistration<TestLinalgTransforms>();
}
} // namespace test
} // namespace mlir
