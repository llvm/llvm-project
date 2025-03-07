//===- AffineParallelize.cpp - Affineparallelize Pass---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parallelizer for affine loop nests that is able to
// perform inner or outer loop parallelization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include <deque>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEPARALLELIZE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-parallel"

using namespace mlir;
using namespace mlir::affine;

namespace {
/// Convert all parallel affine.for op into 1-D affine.parallel op.
struct AffineParallelize
    : public affine::impl::AffineParallelizeBase<AffineParallelize> {
  using AffineParallelizeBase<AffineParallelize>::AffineParallelizeBase;

  void runOnOperation() override;
};

/// Descriptor of a potentially parallelizable loop.
struct ParallelizationCandidate {
  ParallelizationCandidate(AffineForOp l, SmallVector<LoopReduction> &&r)
      : loop(l), reductions(std::move(r)) {}

  /// The potentially parallelizable loop.
  AffineForOp loop;
  /// Desciprtors of reductions that can be parallelized in the loop.
  SmallVector<LoopReduction> reductions;
};
} // namespace

void AffineParallelize::runOnOperation() {
  func::FuncOp f = getOperation();

  // The walker proceeds in pre-order to process the outer loops first
  // and control the number of outer parallel loops.
  std::vector<ParallelizationCandidate> parallelizableLoops;
  f.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
    SmallVector<LoopReduction> reductions;
    if (isLoopParallel(loop, parallelReductions ? &reductions : nullptr))
      parallelizableLoops.emplace_back(loop, std::move(reductions));
  });

  for (const ParallelizationCandidate &candidate : parallelizableLoops) {
    unsigned numParentParallelOps = 0;
    AffineForOp loop = candidate.loop;
    for (Operation *op = loop->getParentOp();
         op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
         op = op->getParentOp()) {
      if (isa<AffineParallelOp>(op))
        ++numParentParallelOps;
    }

    if (numParentParallelOps < maxNested) {
      if (failed(affineParallelize(loop, candidate.reductions))) {
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] failed to parallelize\n"
                                << loop);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] too many nested loops\n"
                              << loop);
    }
  }
}
