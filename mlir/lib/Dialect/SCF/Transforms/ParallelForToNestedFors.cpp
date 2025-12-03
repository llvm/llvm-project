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

#define DEBUG_TYPE "parallel-for-to-nested-fors"
using namespace mlir;

FailureOr<scf::LoopNest>
mlir::scf::parallelForToNestedFors(RewriterBase &rewriter,
                                   scf::ParallelOp parallelOp) {

  if (!parallelOp.getResults().empty())
    return rewriter.notifyMatchFailure(
        parallelOp, "Currently scf.parallel to scf.for conversion doesn't "
                    "support scf.parallel with results.");

  rewriter.setInsertionPoint(parallelOp);

  Location loc = parallelOp.getLoc();
  SmallVector<Value> lowerBounds = parallelOp.getLowerBound();
  SmallVector<Value> upperBounds = parallelOp.getUpperBound();
  SmallVector<Value> steps = parallelOp.getStep();

  assert(lowerBounds.size() == upperBounds.size() &&
         lowerBounds.size() == steps.size() &&
         "Mismatched parallel loop bounds");

  scf::LoopNest loopNest =
      scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps);

  SmallVector<Value> newInductionVars = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp forOp) { return forOp.getInductionVar(); });
  Block *linearizedBody = loopNest.loops.back().getBody();
  Block *parallelBody = parallelOp.getBody();
  rewriter.eraseOp(parallelBody->getTerminator());
  rewriter.inlineBlockBefore(parallelBody, linearizedBody->getTerminator(),
                             newInductionVars);
  rewriter.eraseOp(parallelOp);
  return loopNest;
}

namespace {
struct ParallelForToNestedFors final
    : public impl::SCFParallelForToNestedForsBase<ParallelForToNestedFors> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk(
        [&](scf::ParallelOp parallelOp) {
          if (failed(scf::parallelForToNestedFors(rewriter, parallelOp))) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "Failed to convert scf.parallel to nested scf.for ops for:\n"
                << parallelOp << "\n");
            return WalkResult::advance();
          }
          return WalkResult::advance();
        });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelForToNestedForsPass() {
  return std::make_unique<ParallelForToNestedFors>();
}
