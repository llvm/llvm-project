//===- VectorHoisting.cpp - Hoist redundant vector transfer operations ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hoisting of redundant vector.transfer_read /
// vector.transfer_write pairs out of loops.  The transformation detects pairs
// with loop-invariant indices that act as accumulator loads/stores and rewrites
// the loop with iter_args so the accumulator stays in a register across
// iterations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-hoisting"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::vector;

/// Return true if there are no aliasing uses of `transferRead`'s memref in
/// `loop` other than the read itself and disjoint transfer reads.  The function
/// walks up through a chain of view-like ops to reach the underlying memref
/// before collecting users.
static bool noAliasingUseInLoop(vector::TransferReadOp transferRead,
                                LoopLikeOpInterface loop) {
  Value source = transferRead.getBase();

  // Walk up through view-like ops to find the real source memref.
  while (auto viewLike = source.getDefiningOp<ViewLikeOpInterface>()) {
    if (viewLike.getViewDest() != source)
      break;
    source = viewLike.getViewSource();
  }

  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    if (!processed.insert(user).second)
      continue;
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      Value viewDest = viewLike.getViewDest();
      users.append(viewDest.getUsers().begin(), viewDest.getUsers().end());
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

void mlir::vector::hoistRedundantVectorTransfers(Operation *root,
                                                 bool verifyNonZeroTrip) {
  bool changed = true;
  while (changed) {
    changed = false;

    // Collect loops whose trip count is provably non-zero when requested.
    // Hoisting from a loop with a potentially zero trip count would execute a
    // transfer unconditionally when it should be skipped.
    llvm::DenseSet<LoopLikeOpInterface> definiteNonZeroTripCountLoops;
    if (verifyNonZeroTrip) {
      root->walk([&](LoopLikeOpInterface loopLike) {
        std::optional<SmallVector<OpFoldResult>> lbs =
            loopLike.getLoopLowerBounds();
        std::optional<SmallVector<OpFoldResult>> ubs =
            loopLike.getLoopUpperBounds();
        if (!lbs || !ubs)
          return;
        for (auto [lb, ub] : llvm::zip_equal(lbs.value(), ubs.value())) {
          FailureOr<int64_t> maxLb =
              ValueBoundsConstraintSet::computeConstantBound(
                  presburger::BoundType::UB, lb,
                  /*stopCondition=*/nullptr,
                  ValueBoundsOptions{/*closedUB=*/true});
          if (failed(maxLb))
            return;
          FailureOr<int64_t> minUb =
              ValueBoundsConstraintSet::computeConstantBound(
                  presburger::BoundType::LB, ub);
          if (failed(minUb))
            return;
          if (minUb.value() <= maxLb.value())
            return;
          definiteNonZeroTripCountLoops.insert(loopLike);
        }
      });
    }

    root->walk([&](vector::TransferReadOp transferRead) {
      if (!isa<MemRefType>(transferRead.getShapedType()))
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<LoopLikeOpInterface>(transferRead->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead->getParentOp()
                        << "\n");
      if (!isa_and_nonnull<scf::ForOp, affine::AffineForOp>(loop))
        return WalkResult::advance();

      if (verifyNonZeroTrip && !definiteNonZeroTripCountLoops.contains(loop)) {
        LLVM_DEBUG(DBGS() << "Loop may have zero trip count: " << *loop
                          << "\n");
        return WalkResult::advance();
      }

      LLVM_DEBUG(DBGS() << "Candidate read: " << *transferRead.getOperation()
                        << "\n");

      SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead.getOperation(), &forwardSlice);

      // Look for the last TransferWriteOp in the forward slice of
      // `transferRead` that operates on the same memref.
      vector::TransferWriteOp transferWrite;
      for (auto *sliceOp : llvm::reverse(forwardSlice)) {
        auto candidateWrite = dyn_cast<vector::TransferWriteOp>(sliceOp);
        if (!candidateWrite ||
            candidateWrite.getBase() != transferRead.getBase())
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
      //      nor have users that are Ops implementing ViewLikeOpInterface
      //      (with a relaxation for view-like ops whose parent memref has no
      //      other accesses inside the loop),
      //   3. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.

      // Check 1.
      if (transferRead.getIndices() != transferWrite.getIndices() ||
          transferRead.getVectorType() != transferWrite.getVectorType() ||
          transferRead.getPermutationMap() != transferWrite.getPermutationMap())
        return WalkResult::advance();

      // Check 2. Note, since both xfer ops share the base, we only need to
      // look at one of them.
      auto base = transferRead.getBase();
      auto *source = base.getDefiningOp();
      if (source) {
        // Special-case memref.assume_alignment: it is safe to look through it
        // when (a) it has exactly two in-loop uses (the xfer pair), and (b)
        // the underlying memref has a single use (the assume_alignment itself).
        if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(source)) {
          Value memPreAlignment = assume.getMemref();
          auto numInLoopUses =
              llvm::count_if(base.getUses(), [&loop](OpOperand &use) {
                return loop->isAncestor(use.getOwner());
              });
          if (numInLoopUses && memPreAlignment.hasOneUse())
            source = memPreAlignment.getDefiningOp();
        }

        // For view-like ops (e.g., memref.subview), rather than bailing
        // unconditionally, allow hoisting when:
        //   (a) no other view-like op derives from the same source (aliasing),
        //   (b) the source memref has no non-view direct uses inside the loop.
        // Only one level of view indirection is handled; chained views bail
        // conservatively.
        if (auto viewLike = dyn_cast_if_present<ViewLikeOpInterface>(source)) {
          Value parent = viewLike.getViewSource();
          // Chained view: bail conservatively.
          if (parent.getDefiningOp<ViewLikeOpInterface>())
            return WalkResult::advance();
          for (auto &use : parent.getUses()) {
            Operation *user = use.getOwner();
            if (user == source)
              continue;
            // Another view from the same parent can alias.
            if (isa<ViewLikeOpInterface>(user))
              return WalkResult::advance();
            // A direct use of the parent inside the loop is a conflict.
            if (loop->isAncestor(user))
              return WalkResult::advance();
          }
        }
      }

      if (llvm::any_of(base.getUsers(), llvm::IsaPred<ViewLikeOpInterface>))
        return WalkResult::advance();

      // Check 3.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.getBase().getUses()) {
        if (!loop->isAncestor(use.getOwner()))
          continue;
        if (use.getOwner() == transferRead.getOperation() ||
            use.getOwner() == transferWrite.getOperation())
          continue;
        if (auto transferWriteUse =
                dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(*transferWrite),
                  cast<VectorTransferOpInterface>(*transferWriteUse),
                  /*testDynamicValueUsingBounds=*/true))
            return WalkResult::advance();
        } else if (auto transferReadUse =
                       dyn_cast<vector::TransferReadOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(*transferWrite),
                  cast<VectorTransferOpInterface>(*transferReadUse),
                  /*testDynamicValueUsingBounds=*/true))
            return WalkResult::advance();
        } else {
          return WalkResult::advance();
        }
      }

      // Hoist read before the loop.
      loop.moveOutOfLoop(transferRead);

      // Hoist write after the loop.
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

      transferWrite.getValueToStoreMutable().assign(
          maybeNewLoop->getOperation()->getResults().back());
      changed = true;
      return WalkResult::interrupt();
    });
  }
}
