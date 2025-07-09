//===- ParallelForToNestedFors.cpp - scf.parallel to nested scf.for ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ParallelOp to nested scf.for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELFORTONESTEDFORS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

LogicalResult mlir::scf::parallelForToNestedFors(RewriterBase &rewriter,
                                                 scf::ParallelOp parallelOp,
                                                 scf::ForOp *result) {

  if (!parallelOp.getResults().empty()) {
    parallelOp->emitError("Currently ScfParallel to ScfFor conversion "
                          "doesn't support ScfParallel with results.");
    return failure();
  }

  rewriter.setInsertionPoint(parallelOp);

  Location loc = parallelOp.getLoc();
  auto lowerBounds = parallelOp.getLowerBound();
  auto upperBounds = parallelOp.getUpperBound();
  auto steps = parallelOp.getStep();

  assert(lowerBounds.size() == upperBounds.size() &&
         lowerBounds.size() == steps.size() &&
         "Mismatched parallel loop bounds");

  SmallVector<Value> ivs;
  auto loopNest =
      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps);

  auto oldInductionVars = parallelOp.getInductionVars();
  auto newInductionVars = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp forOp) { return forOp.getInductionVar(); });
  assert(oldInductionVars.size() == newInductionVars.size() &&
         "Mismatched induction variables");
  for (auto [oldIV, newIV] : llvm::zip(oldInductionVars, newInductionVars))
    oldIV.replaceAllUsesWith(newIV);

  auto *linearizedBody = loopNest.loops.back().getBody();
  Block &parallelBody = *parallelOp.getBody();
  for (Operation &op : llvm::make_early_inc_range(parallelBody)) {
    // Skip the terminator of the parallelOp body.
    if (&op == parallelBody.getTerminator())
      continue;
    op.moveBefore(linearizedBody->getTerminator());
  }
  rewriter.eraseOp(parallelOp);
  if (result)
    *result = loopNest.loops.front();
  return success();
}

namespace {
struct ParallelForToNestedFors final
    : public impl::SCFParallelForToNestedForsBase<ParallelForToNestedFors> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](scf::ParallelOp parallelOp) {
      if (failed(scf::parallelForToNestedFors(rewriter, parallelOp))) {
        return signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelForToNestedForsPass() {
  return std::make_unique<ParallelForToNestedFors>();
}
