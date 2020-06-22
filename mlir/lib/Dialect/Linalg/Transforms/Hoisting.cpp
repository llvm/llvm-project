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
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

void mlir::linalg::hoistViewAllocOps(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&changed](Operation *op) {
      if (!isa<AllocOp>(op) && !isa<AllocaOp>(op) && !isa<DeallocOp>(op))
        return;

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: " << *op << "\n");
      auto loop = dyn_cast<scf::ForOp>(op->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *op->getParentOp() << "\n");

      // Only hoist out of immediately enclosing scf::ForOp.
      if (!loop)
        return;

      // If any operand is defined inside the loop don't hoist.
      if (llvm::any_of(op->getOperands(), [&](Value v) {
            return !loop.isDefinedOutsideOfLoop(v);
          }))
        return;

      LLVM_DEBUG(DBGS() << "All operands defined outside \n");

      // If alloc has other uses than ViewLikeOp and DeallocOp don't hoist.
      Value v;
      if (op->getNumResults() > 0) {
        assert(op->getNumResults() == 1 && "Unexpected multi-result alloc");
        v = op->getResult(0);
      }
      if (v && !llvm::all_of(v.getUses(), [&](OpOperand &operand) {
            return isa<ViewLikeOpInterface>(operand.getOwner()) ||
                   isa<DeallocOp>(operand.getOwner());
          })) {
        LLVM_DEBUG(DBGS() << "Found non view-like or dealloc use: bail\n");
        return;
      }

      // Move AllocOp before the loop.
      if (isa<AllocOp>(op) || isa<AllocaOp>(op))
        loop.moveOutOfLoop({op});
      else // Move DeallocOp outside of the loop.
        op->moveAfter(loop);
      changed = true;
    });
  }
}

void mlir::linalg::hoistRedundantVectorTransfers(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;

    func.walk([&](vector::TransferReadOp transferRead) {
      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<scf::ForOp>(transferRead.getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead.getParentOp()
                        << "\n");
      if (!loop)
        return WalkResult::advance();

      if (failed(moveLoopInvariantCode(
              cast<LoopLikeOpInterface>(loop.getOperation()))))
        llvm_unreachable(
            "Unexpected failure to move invariant code out of loop");

      LLVM_DEBUG(DBGS() << "Candidate read: " << *transferRead.getOperation()
                        << "\n");

      llvm::SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead, &forwardSlice);

      // Look for the last TransferWriteOp in the forwardSlice of
      // `transferRead` that operates on the same memref.
      vector::TransferWriteOp transferWrite;
      for (auto *sliceOp : llvm::reverse(forwardSlice)) {
        auto candidateWrite = dyn_cast<vector::TransferWriteOp>(sliceOp);
        if (!candidateWrite || candidateWrite.memref() != transferRead.memref())
          continue;
        transferWrite = candidateWrite;
      }

      // All operands of the TransferRead must be defined outside of the loop.
      for (auto operand : transferRead.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand))
          return WalkResult::advance();

      // Only hoist transfer_read / transfer_write pairs for now.
      if (!transferWrite)
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate: " << *transferWrite.getOperation()
                        << "\n");

      // Approximate aliasing by checking that:
      //   1. indices are the same,
      //   2. no other use either dominates the transfer_read or is dominated
      //   by the transfer_write (i.e. aliasing between the write and the read
      //   across the loop).
      if (transferRead.indices() != transferWrite.indices())
        return WalkResult::advance();

      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.memref().getUses())
        if (dom.properlyDominates(use.getOwner(),
                                  transferRead.getOperation()) ||
            dom.properlyDominates(transferWrite, use.getOwner()))
          return WalkResult::advance();

      // Hoist read before.
      if (failed(loop.moveOutOfLoop({transferRead})))
        llvm_unreachable(
            "Unexpected failure to move transfer read out of loop");

      // Hoist write after.
      transferWrite.getOperation()->moveAfter(loop);

      // Rewrite `loop` with new yields by cloning and erase the original loop.
      OpBuilder b(transferRead);
      auto newForOp = cloneWithNewYields(b, loop, transferRead.vector(),
                                         transferWrite.vector());

      // Transfer write has been hoisted, need to update the written value to
      // the value yielded by the newForOp.
      transferWrite.vector().replaceAllUsesWith(
          newForOp.getResults().take_back()[0]);

      changed = true;
      loop.erase();
      // Need to interrupt and restart because erasing the loop messes up the
      // walk.
      return WalkResult::interrupt();
    });
  }
}
