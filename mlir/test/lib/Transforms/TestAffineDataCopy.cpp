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

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "test-affine-data-copy"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(PASS_NAME " options");

namespace {

struct TestAffineDataCopy : public FunctionPass<TestAffineDataCopy> {
  TestAffineDataCopy() = default;
  TestAffineDataCopy(const TestAffineDataCopy &pass){};

  void runOnFunction() override;

private:
  Option<bool> clMemRefFilter{
      *this, "memref-filter",
      llvm::cl::desc(
          "Enable memref filter testing in affine data copy optimization"),
      llvm::cl::init(false)};
};

} // end anonymous namespace

void TestAffineDataCopy::runOnFunction() {
  // Gather all AffineForOps by loop depth.
  DenseMap<unsigned, SmallVector<AffineForOp, 2>> depthToLoops;
  gatherLoops(getFunction(), depthToLoops);
  assert(depthToLoops.size() && "Loop nest not found");

  // Only support tests with a single loop nest and a single innermost loop
  // for now.
  unsigned innermostLoopIdx = depthToLoops.size() - 2;
  if (depthToLoops[0].size() != 1 || depthToLoops[innermostLoopIdx].size() != 1)
    return;

  auto loopNest = depthToLoops[0][0];
  auto innermostLoop = depthToLoops[innermostLoopIdx][0];
  Optional<Value> memrefFilter;
  if (clMemRefFilter) {
    // Gather MemRef filter. For simplicity, we use the first loaded memref
    // found in the innermost loop.
    for (auto &op : *innermostLoop.getBody()) {
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        memrefFilter = load.getMemRef();
        break;
      }
    }
  }

  AffineCopyOptions copyOptions = {/*generateDma=*/false,
                                   /*slowMemorySpace=*/0,
                                   /*fastMemorySpace=*/0,
                                   /*tagMemorySpace=*/0,
                                   /*fastMemCapacityBytes=*/32 * 1024 * 1024UL};
  DenseSet<Operation *> copyNests;
  affineDataCopyGenerate(loopNest.getBody()->begin(),
                         std::prev(loopNest.getBody()->end()), copyOptions,
                         memrefFilter, copyNests);
}

namespace mlir {
void registerTestAffineDataCopyPass() {
  PassRegistration<TestAffineDataCopy>(
      PASS_NAME, "Tests affine data copy utility functions.");
}
} // namespace mlir
