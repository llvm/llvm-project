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
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

void mlir::linalg::hoistRedundantVectorTransfersOnTensor(func::FuncOp func) {
  IRRewriter rewriter(func->getContext());
  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  func.walk([&](scf::ForOp forOp) {
    hoistRedundantSubsetExtractInsert(rewriter, forOp);
  });
}

static bool noAliasingUseInLoop(vector::TransferReadOp transferRead,
                                LoopLikeOpInterface loop) {
  Value source = transferRead.getSource();

  // Skip view-like Ops and retrive the actual soruce Operation
  while (auto srcOp =
             dyn_cast_or_null<ViewLikeOpInterface>(source.getDefiningOp()))
    source = srcOp.getViewSource();

  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      users.append(viewLike->getUsers().begin(), viewLike->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user) || isa<vector::TransferReadOp>(user))
      continue;
    if (!loop->isAncestor(user))
      continue;
    return false;
  }
  return true;
}

void mlir::linalg::hoistRedundantVectorTransfers(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interrupting the function walk.
    func.walk(
        [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

    func.walk([&](vector::TransferReadOp transferRead) {
      if (!isa<MemRefType>(transferRead.getShapedType()))
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<LoopLikeOpInterface>(transferRead->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead->getParentOp()
                        << "\n");
      if (!isa_and_nonnull<scf::ForOp, affine::AffineForOp>(loop))
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate read: " << *transferRead.getOperation()
                        << "\n");

      SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead.getOperation(), &forwardSlice);

      // Look for the last TransferWriteOp in the forwardSlice of
      // `transferRead` that operates on the same memref.
      vector::TransferWriteOp transferWrite;
      for (auto *sliceOp : llvm::reverse(forwardSlice)) {
        auto candidateWrite = dyn_cast<vector::TransferWriteOp>(sliceOp);
        if (!candidateWrite ||
            candidateWrite.getSource() != transferRead.getSource())
          continue;
        transferWrite = candidateWrite;
      }

      // All operands of the TransferRead must be defined outside of the loop.
      for (auto operand : transferRead.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand))
          return WalkResult::advance();

      // Only hoist transfer_read / transfer_write pairs and singleton
      // transfer_reads for now.
      if (!transferWrite) {
        // Make sure there are no other accesses to the memref before
        // hoisting transfer_read.
        if (noAliasingUseInLoop(transferRead, loop))
          loop.moveOutOfLoop(transferRead);
        return WalkResult::advance();
      }

      LLVM_DEBUG(DBGS() << "Candidate: " << *transferWrite.getOperation()
                        << "\n");

      // Approximate aliasing by checking that:
      //   1. indices, vector type and permutation map are the same (i.e., the
      //      transfer_read/transfer_write ops are matching),
      //   2. source operands for transfer.{read|write} do not originate from
      //      Ops implementing ViewLikeOpInterface.
      //   3. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.
      if (transferRead.getIndices() != transferWrite.getIndices() ||
          transferRead.getVectorType() != transferWrite.getVectorType() ||
          transferRead.getPermutationMap() != transferWrite.getPermutationMap())
        return WalkResult::advance();

      auto *source = transferRead.getSource().getDefiningOp();
      if (source && isa_and_nonnull<ViewLikeOpInterface>(source))
        return WalkResult::advance();

      source = transferWrite.getSource().getDefiningOp();
      if (source && isa_and_nonnull<ViewLikeOpInterface>(source))
        return WalkResult::advance();

      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.getSource().getUses()) {
        if (!loop->isAncestor(use.getOwner()))
          continue;
        if (use.getOwner() == transferRead.getOperation() ||
            use.getOwner() == transferWrite.getOperation())
          continue;
        if (auto transferWriteUse =
                dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(transferWrite.getOperation()),
                  cast<VectorTransferOpInterface>(
                      transferWriteUse.getOperation())))
            return WalkResult::advance();
        } else if (auto transferReadUse =
                       dyn_cast<vector::TransferReadOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
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
      loop.moveOutOfLoop(transferRead);

      // Hoist write after.
      transferWrite->moveAfter(loop);

      // Rewrite `loop` with new yields by cloning and erase the original loop.
      IRRewriter rewriter(transferRead.getContext());
      NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                     ArrayRef<BlockArgument> newBBArgs) {
        return SmallVector<Value>{transferWrite.getVector()};
      };

      auto maybeNewLoop = loop.replaceWithAdditionalYields(
          rewriter, transferRead.getVector(),
          /*replaceInitOperandUsesInLoop=*/true, yieldFn);
      if (failed(maybeNewLoop))
        return WalkResult::interrupt();

      transferWrite.getVectorMutable().assign(
          maybeNewLoop->getOperation()->getResults().back());
      changed = true;
      // Need to interrupt and restart because erasing the loop messes up
      // the walk.
      return WalkResult::interrupt();
    });
  }
}
