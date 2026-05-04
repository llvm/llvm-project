//===- TestAccessAnalysis.cpp - Test affine access analysis utility -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test affine access analysis utilities.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "test-affine-access-analysis"

using namespace mlir;
using namespace mlir::affine;

namespace {

struct TestAccessAnalysis
    : public PassWrapper<TestAccessAnalysis, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAccessAnalysis)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine memory access analysis utility";
  }

  void runOnOperation() override;
};

} // namespace

/// Gathers all affine load/store ops in loop nest rooted at 'forOp' into
/// 'loadAndStoreOps'.
static void
gatherLoadsAndStores(AffineForOp forOp,
                     SmallVectorImpl<Operation *> &loadAndStoreOps) {
  forOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });
}

void TestAccessAnalysis::runOnOperation() {
  SmallVector<Operation *> loadStores;
  SmallVector<AffineForOp> enclosingOps;
  // Go over all top-level affine.for ops and test each contained affine
  // access's contiguity along every surrounding loop IV.
  for (auto forOp : getOperation().getOps<AffineForOp>()) {
    loadStores.clear();
    gatherLoadsAndStores(forOp, loadStores);
    for (Operation *memOp : loadStores) {
      enclosingOps.clear();
      getAffineForIVs(*memOp, &enclosingOps);
      for (unsigned d = 0, e = enclosingOps.size(); d < e; d++) {
        AffineForOp loop = enclosingOps[d];
        int memRefDim;
        bool isContiguous, isInvariant;
        if (auto read = dyn_cast<AffineReadOpInterface>(memOp)) {
          isContiguous =
              isContiguousAccess(loop.getInductionVar(), read, &memRefDim);
          isInvariant = isInvariantAccess(read, loop);
        } else {
          auto write = cast<AffineWriteOpInterface>(memOp);
          isContiguous =
              isContiguousAccess(loop.getInductionVar(), write, &memRefDim);
          isInvariant = isInvariantAccess(write, loop);
        }
        // Check for contiguity for the innermost memref dimension to avoid
        // emitting too many diagnostics.
        if (isContiguous && memRefDim == 0)
          memOp->emitRemark("contiguous along loop ") << d << '\n';
        if (isInvariant)
          memOp->emitRemark("invariant along loop ") << d << '\n';
      }
    }
  }
}

namespace mlir {
void registerTestAffineAccessAnalysisPass() {
  PassRegistration<TestAccessAnalysis>();
}
} // namespace mlir
