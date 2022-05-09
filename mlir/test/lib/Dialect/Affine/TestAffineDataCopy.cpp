//===- TestAffineDataCopy.cpp - Test affine data copy utility -------------===//
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

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "test-affine-data-copy"

using namespace mlir;

namespace {

struct TestAffineDataCopy
    : public PassWrapper<TestAffineDataCopy, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAffineDataCopy)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine data copy utility functions.";
  }
  TestAffineDataCopy() = default;
  TestAffineDataCopy(const TestAffineDataCopy &pass) : PassWrapper(pass){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override;

private:
  Option<bool> clMemRefFilter{
      *this, "memref-filter",
      llvm::cl::desc(
          "Enable memref filter testing in affine data copy optimization"),
      llvm::cl::init(false)};
  Option<bool> clTestGenerateCopyForMemRegion{
      *this, "for-memref-region",
      llvm::cl::desc("Test copy generation for a single memref region"),
      llvm::cl::init(false)};
};

} // namespace

void TestAffineDataCopy::runOnOperation() {
  // Gather all AffineForOps by loop depth.
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  gatherLoops(getOperation(), depthToLoops);
  assert(!depthToLoops.empty() && "Loop nest not found");

  // Only support tests with a single loop nest and a single innermost loop
  // for now.
  unsigned innermostLoopIdx = depthToLoops.size() - 1;
  if (depthToLoops[0].size() != 1 || depthToLoops[innermostLoopIdx].size() != 1)
    return;

  auto loopNest = depthToLoops[0][0];
  auto innermostLoop = depthToLoops[innermostLoopIdx][0];
  AffineLoadOp load;
  if (clMemRefFilter || clTestGenerateCopyForMemRegion) {
    // Gather MemRef filter. For simplicity, we use the first loaded memref
    // found in the innermost loop.
    for (auto &op : *innermostLoop.getBody()) {
      if (auto ld = dyn_cast<AffineLoadOp>(op)) {
        load = ld;
        break;
      }
    }
  }
  if (!load)
    return;

  AffineCopyOptions copyOptions = {/*generateDma=*/false,
                                   /*slowMemorySpace=*/0,
                                   /*fastMemorySpace=*/0,
                                   /*tagMemorySpace=*/0,
                                   /*fastMemCapacityBytes=*/32 * 1024 * 1024UL};
  DenseSet<Operation *> copyNests;
  if (clMemRefFilter) {
    if (failed(affineDataCopyGenerate(loopNest, copyOptions, load.getMemRef(),
                                      copyNests)))
      return;
  } else if (clTestGenerateCopyForMemRegion) {
    CopyGenerateResult result;
    MemRefRegion region(loopNest.getLoc());
    if (failed(region.compute(load, /*loopDepth=*/0)))
      return;
    if (failed(generateCopyForMemRegion(region, loopNest, copyOptions, result)))
      return;
  }

  // Promote any single iteration loops in the copy nests and simplify
  // load/stores.
  SmallVector<Operation *, 4> copyOps;
  for (Operation *nest : copyNests) {
    // With a post order walk, the erasure of loops does not affect
    // continuation of the walk or the collection of load/store ops.
    nest->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        (void)promoteIfSingleIteration(forOp);
      else if (auto loadOp = dyn_cast<AffineLoadOp>(op))
        copyOps.push_back(loadOp);
      else if (auto storeOp = dyn_cast<AffineStoreOp>(op))
        copyOps.push_back(storeOp);
    });
  }

  // Promoting single iteration loops could lead to simplification of
  // generated load's/store's, and the latter could anyway also be
  // canonicalized.
  RewritePatternSet patterns(&getContext());
  for (Operation *op : copyOps) {
    patterns.clear();
    if (isa<AffineLoadOp>(op)) {
      AffineLoadOp::getCanonicalizationPatterns(patterns, &getContext());
    } else {
      assert(isa<AffineStoreOp>(op) && "expected affine store op");
      AffineStoreOp::getCanonicalizationPatterns(patterns, &getContext());
    }
  }
  (void)applyOpPatternsAndFold(copyOps, std::move(patterns), /*strict=*/true);
}

namespace mlir {
void registerTestAffineDataCopyPass() {
  PassRegistration<TestAffineDataCopy>();
}
} // namespace mlir
