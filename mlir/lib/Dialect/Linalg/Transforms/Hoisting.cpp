//===- Hoisting.cpp - Linalg hoisting transformations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with hoisting invariant operations
// in the context of Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

/// Replace `loop` with a new loop that has a different init operand at
/// position `index`. The body of this loop is moved over to the new loop.
///
/// `newInitOperands` specifies the replacement "init" operands.
/// `newYieldValue` is the replacement yield value of the loop at position
/// `index`.
static scf::ForOp replaceWithDifferentYield(RewriterBase &rewriter,
                                            scf::ForOp loop,
                                            Value newInitOperand,
                                            unsigned index,
                                            Value newYieldValue) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop.getOperation());
  auto inits = llvm::to_vector(loop.getInits());

  // Replace the init value with the new operand.
  assert(index < inits.size());
  inits[index] = newInitOperand;

  scf::ForOp newLoop = scf::ForOp::create(
      rewriter, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
      loop.getStep(), inits, [](OpBuilder &, Location, Value, ValueRange) {},
      loop.getUnsignedCmp());

  // Generate the new yield with the replaced operand.
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  yieldOp.setOperand(index, newYieldValue);

  // Move the loop body to the new op.
  rewriter.mergeBlocks(loop.getBody(), newLoop.getBody(),
                       newLoop.getBody()->getArguments());

  // Replace the old loop.
  rewriter.replaceOp(loop.getOperation(), newLoop->getResults());
  return newLoop;
}

// Hoist out a pair of corresponding vector.extract+vector.broadcast
// operations. This function transforms a loop like this:
//  %res = scf.for _ = _ to _ step _ iter_args(%iarg = %v) -> (t1) {
//   %e = vector.extract %iarg : t1 to t2
//   %u = "some_use"(%e) : (t2) -> t2
//   %b = vector.broadcast %u : t2 to t1
//   scf.yield %b : t1
//  }
// into the following:
//  %e = vector.extract %v: t1 to t2
//  %res' = scf.for _ = _ to _ step _ iter_args(%iarg = %e) -> (t2) {
//   %u' = "some_use"(%iarg) : (t2) -> t2
//   scf.yield %u' : t2
//  }
//  %res = vector.broadcast %res' : t2 to t1
void mlir::linalg::hoistRedundantVectorBroadcasts(RewriterBase &rewriter,
                                                  Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interrupting the function walk.
    root->walk(
        [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

    root->walk([&](vector::ExtractOp extractOp) {
      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *extractOp.getOperation() << "\n");

      auto loop = dyn_cast<scf::ForOp>(extractOp->getParentOp());
      if (!loop)
        return WalkResult::advance();

      // Check that the vector to extract from is a BlockArgument.
      auto blockArg = dyn_cast<BlockArgument>(extractOp.getSource());
      if (!blockArg)
        return WalkResult::advance();

      // Check that the blockArg is an iter_arg of the loop.
      OpOperand *initArg = loop.getTiedLoopInit(blockArg);
      if (!initArg)
        return WalkResult::advance();

      // If the iter_arg does not have only one use, it won't be possible to
      // hoist the extractOp out.
      if (!blockArg.hasOneUse())
        return WalkResult::advance();

      unsigned index = blockArg.getArgNumber() - loop.getNumInductionVars();

      // Check that the loop yields a broadcast that has just one use.
      Operation *yieldedVal =
          loop.getTiedLoopYieldedValue(blockArg)->get().getDefiningOp();
      auto broadcast = dyn_cast<vector::BroadcastOp>(yieldedVal);
      if (!broadcast || !broadcast.getResult().hasOneUse())
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate broadcast: " << broadcast << "\n");

      Type broadcastInputType = broadcast.getSourceType();
      if (broadcastInputType != extractOp.getType())
        return WalkResult::advance();

      // The position of the extract must be defined outside of the loop if
      // it is dynamic.
      for (auto operand : extractOp.getDynamicPosition())
        if (!loop.isDefinedOutsideOfLoop(operand))
          return WalkResult::advance();

      rewriter.modifyOpInPlace(broadcast, [&] {
        extractOp.getSourceMutable().assign(initArg->get());
      });
      loop.moveOutOfLoop(extractOp);
      rewriter.moveOpAfter(broadcast, loop);

      scf::ForOp newLoop = replaceWithDifferentYield(
          rewriter, loop, extractOp.getResult(), index, broadcast.getSource());

      LLVM_DEBUG(DBGS() << "New loop: " << newLoop << "\n");

      rewriter.replaceAllUsesWith(newLoop.getResult(index), broadcast);
      rewriter.modifyOpInPlace(
          broadcast, [&] { broadcast.setOperand(newLoop.getResult(index)); });

      changed = true;
      return WalkResult::interrupt();
    });
  }
}

void mlir::linalg::hoistRedundantVectorTransfers(Operation *root,
                                                 bool verifyNonZeroTrip) {
  // Run LICM first to expose loop-invariant operands (e.g. subview ops, padding
  // values) that would otherwise block transfer-pair hoisting.
  root->walk([](LoopLikeOpInterface loop) { moveLoopInvariantCode(loop); });
  mlir::vector::hoistRedundantVectorTransfers(root, verifyNonZeroTrip);
}
