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
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
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
      if (!isa<AllocOp, AllocaOp, DeallocOp>(op))
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
            return isa<ViewLikeOpInterface, DeallocOp>(operand.getOwner());
          })) {
        LLVM_DEBUG(DBGS() << "Found non view-like or dealloc use: bail\n");
        return;
      }

      // Move AllocOp before the loop.
      if (isa<AllocOp, AllocaOp>(op))
        loop.moveOutOfLoop({op});
      else // Move DeallocOp outside of the loop.
        op->moveAfter(loop);
      changed = true;
    });
  }
}

/// Look for a transfer_read, in the given tensor uses, accessing the same
/// offset as the transfer_write.
static vector::TransferReadOp
findMatchingTransferRead(vector::TransferWriteOp write, Value srcTensor) {
  for (Operation *user : srcTensor.getUsers()) {
    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (read && read.indices() == write.indices() &&
        read.getVectorType() == write.getVectorType()) {
      return read;
    }
  }
  return nullptr;
}

/// Check if the chunk of data inserted by the transfer_write in the given
/// tensor are read by any other op than the read candidate.
static bool tensorChunkAccessedByUnknownOp(vector::TransferWriteOp write,
                                           vector::TransferReadOp candidateRead,
                                           Value srcTensor) {
  // Make sure none of the other uses read the part of the tensor modified
  // by the transfer_write.
  llvm::SmallVector<Value::use_range, 1> uses;
  uses.push_back(srcTensor.getUses());
  while (!uses.empty()) {
    for (OpOperand &use : uses.pop_back_val()) {
      Operation *user = use.getOwner();
      // Skip the candidate use, only inspect the "other" uses.
      if (user == candidateRead.getOperation() || user == write.getOperation())
        continue;
      // Consider all transitive uses through a vector.transfer_write.
      if (auto writeUser = dyn_cast<vector::TransferWriteOp>(user)) {
        uses.push_back(writeUser->getResult(0).getUses());
        continue;
      }
      // Consider all nested uses through an scf::ForOp. We may have
      // pass-through tensor arguments left from previous level of
      // hoisting.
      if (auto forUser = dyn_cast<scf::ForOp>(user)) {
        Value arg = forUser.getLoopBody().getArgument(
            use.getOperandNumber() - forUser.getNumControlOperands() +
            /*iv value*/ 1);
        uses.push_back(arg.getUses());
        continue;
      }
      // Follow the use yield as long as it doesn't escape the original
      // region.
      scf::YieldOp yieldUser = dyn_cast<scf::YieldOp>(user);
      if (yieldUser &&
          write->getParentOp()->isAncestor(yieldUser->getParentOp())) {
        Value ret = yieldUser->getParentOp()->getResult(use.getOperandNumber());
        uses.push_back(ret.getUses());
        continue;
      }
      auto read = dyn_cast<vector::TransferReadOp>(user);
      if (!read || !isDisjointTransferIndices(
                       cast<VectorTransferOpInterface>(read.getOperation()),
                       cast<VectorTransferOpInterface>(write.getOperation()))) {
        return true;
      }
    }
  }
  return false;
}

// To hoist transfer op on tensor the logic can be significantly simplified
// compared to the case on buffer. The transformation follows this logic:
// 1. Look for transfer_write with a single use from ForOp yield
// 2. Check the uses of the matching block argument and look for a transfer_read
// with the same indices.
// 3. Check that all the other uses of the tensor argument are either disjoint
// tensor_read or transfer_write. For transfer_write uses recurse to make sure
// the new tensor has the same restrictions on its uses.
// 4. Hoist the tensor_read/tensor_write and update the tensor SSA links.
// After this transformation the scf.forOp may have unused arguments that can be
// remove by the canonicalization pass.
void mlir::linalg::hoistRedundantVectorTransfersOnTensor(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](scf::ForOp forOp) {
      Operation *yield = forOp.getBody()->getTerminator();
      for (auto it : llvm::enumerate(forOp.getRegionIterArgs())) {
        Value ret = yield->getOperand(it.index());
        auto write = ret.getDefiningOp<vector::TransferWriteOp>();
        if (!write || !write->hasOneUse())
          continue;
        LLVM_DEBUG(DBGS() << "Candidate write for hoisting: "
                          << *write.getOperation() << "\n");
        if (llvm::any_of(write.indices(), [&forOp](Value index) {
              return !forOp.isDefinedOutsideOfLoop(index);
            }))
          continue;
        // Find a read with the same type and indices.
        vector::TransferReadOp matchingRead =
            findMatchingTransferRead(write, it.value());
        // Make sure none of the other uses read the part of the tensor modified
        // by the transfer_write.
        if (!matchingRead ||
            tensorChunkAccessedByUnknownOp(write, matchingRead, it.value()))
          continue;

        // Hoist read before.
        if (failed(forOp.moveOutOfLoop({matchingRead})))
          llvm_unreachable(
              "Unexpected failure to move transfer read out of loop");
        // Update the source tensor.
        matchingRead.sourceMutable().assign(forOp.initArgs()[it.index()]);

        // Hoist write after.
        write->moveAfter(forOp);
        yield->setOperand(it.index(), write.source());

        // Rewrite `loop` with new yields by cloning and erase the original
        // loop.
        OpBuilder b(matchingRead);
        auto newForOp =
            cloneWithNewYields(b, forOp, matchingRead.vector(), write.vector());

        // Transfer write has been hoisted, need to update the vector and tensor
        // source. Replace the result of the loop to use the new tensor created
        // outside the loop.
        newForOp.getResult(it.index()).replaceAllUsesWith(write.getResult(0));
        write.vectorMutable().assign(newForOp.getResults().back());
        write.sourceMutable().assign(newForOp.getResult(it.index()));

        changed = true;
        forOp.erase();
        // Need to interrupt and restart because erasing the loop messes up the
        // walk.
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
}

void mlir::linalg::hoistRedundantVectorTransfers(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;

    func.walk([&](vector::TransferReadOp transferRead) {
      if (!transferRead.getShapedType().isa<MemRefType>())
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<scf::ForOp>(transferRead->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead->getParentOp()
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
        if (!candidateWrite || candidateWrite.source() != transferRead.source())
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
      //   2. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.
      if (transferRead.indices() != transferWrite.indices() &&
          transferRead.getVectorType() == transferWrite.getVectorType())
        return WalkResult::advance();

      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.source().getUses()) {
        if (!dom.properlyDominates(loop, use.getOwner()))
          continue;
        if (use.getOwner() == transferRead.getOperation() ||
            use.getOwner() == transferWrite.getOperation())
          continue;
        if (auto transferWriteUse =
                dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
          if (!isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(transferWrite.getOperation()),
                  cast<VectorTransferOpInterface>(
                      transferWriteUse.getOperation())))
            return WalkResult::advance();
        } else if (auto transferReadUse =
                       dyn_cast<vector::TransferReadOp>(use.getOwner())) {
          if (!isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(transferWrite.getOperation()),
                  cast<VectorTransferOpInterface>(
                      transferReadUse.getOperation())))
            return WalkResult::advance();
        } else {
          // Unknown use, we cannot prove that it doesn't alias with the
          // transferRead/transferWrite operations.
          return WalkResult::advance();
        }
      }

      // Hoist read before.
      if (failed(loop.moveOutOfLoop({transferRead})))
        llvm_unreachable(
            "Unexpected failure to move transfer read out of loop");

      // Hoist write after.
      transferWrite->moveAfter(loop);

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
