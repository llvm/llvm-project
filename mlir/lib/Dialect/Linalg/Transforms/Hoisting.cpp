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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

static bool noAliasingUseInLoop(vector::TransferReadOp transferRead,
                                LoopLikeOpInterface loop) {
  Value source = transferRead.getBase();

  // Skip view-like Ops and retrive the actual soruce Operation
  while (auto viewLike = source.getDefiningOp<ViewLikeOpInterface>()) {
    if (viewLike.getViewDest() != source) {
      break;
    }
    source = viewLike.getViewSource();
  }

  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
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

namespace {
/// Static footprint of a memref value resolved through a chain of
/// rank-preserving, unit-stride `memref.subview` ops: the underlying root
/// memref and, per root dimension, the `[offset, offset + size)` interval it
/// covers.
struct SubviewFootprint {
  Value root;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
};
} // namespace

/// Resolve `base` through a `memref.subview` chain to its static footprint, or
/// `std::nullopt` if it cannot be modelled statically.
static std::optional<SubviewFootprint> resolveSubviewFootprint(Value base) {
  SmallVector<memref::SubViewOp> chain;
  Value source = base;
  while (auto sv = source.getDefiningOp<memref::SubViewOp>()) {
    chain.push_back(sv);
    source = sv.getSource();
  }
  // The resolved root must be a genuine buffer, not another kind of view.
  if (isa_and_nonnull<ViewLikeOpInterface>(source.getDefiningOp()))
    return std::nullopt;

  auto rootTy = dyn_cast<MemRefType>(source.getType());
  if (!rootTy || !rootTy.hasStaticShape())
    return std::nullopt;
  SubviewFootprint fp;
  fp.root = source;
  // Seed with the whole-root footprint (offset 0, full extent per dimension):
  // this is the result for a bare buffer, and each subview in the chain narrows
  // it below.
  fp.offsets.assign(rootTy.getRank(), 0);
  fp.sizes.assign(rootTy.getShape().begin(), rootTy.getShape().end());
  // Compose from the outermost subview (closest to the root) inward.
  for (memref::SubViewOp sv : llvm::reverse(chain)) {
    // Rank-reducing subviews would misalign the per-dimension bookkeeping.
    if (sv.getSourceType().getRank() != sv.getType().getRank())
      return std::nullopt;
    ArrayRef<int64_t> offs = sv.getStaticOffsets();
    ArrayRef<int64_t> szs = sv.getStaticSizes();
    ArrayRef<int64_t> strides = sv.getStaticStrides();
    if (llvm::any_of(offs, ShapedType::isDynamic) ||
        llvm::any_of(szs, ShapedType::isDynamic) ||
        llvm::any_of(strides, [](int64_t s) { return s != 1; }))
      return std::nullopt;
    for (unsigned d = 0, e = offs.size(); d < e; ++d) {
      fp.offsets[d] += offs[d];
      fp.sizes[d] = szs[d];
    }
  }
  return fp;
}

/// True if two footprints provably cover disjoint memory: they must share the
/// same root and be separated along at least one dimension.
static bool footprintsDisjoint(const SubviewFootprint &a,
                               const SubviewFootprint &b) {
  if (a.root != b.root)
    return false;
  for (unsigned d = 0, e = a.offsets.size(); d < e; ++d) {
    int64_t aHi = a.offsets[d] + a.sizes[d];
    int64_t bHi = b.offsets[d] + b.sizes[d];
    if (aHi <= b.offsets[d] || bHi <= a.offsets[d])
      return true;
  }
  return false;
}

/// Subview-aware safety check for hoisting `read`/`write` out of `loop`.
static bool footprintAliasingIsSafe(LoopLikeOpInterface loop,
                                    vector::TransferReadOp read,
                                    vector::TransferWriteOp write) {
  std::optional<SubviewFootprint> ourFp =
      resolveSubviewFootprint(read.getBase());
  if (!ourFp)
    return false;
  // `resolveSubviewFootprint` guarantees the root is a genuine buffer.
  Value ourBuffer = ourFp->root;
  auto reachesOurBuffer = [&](Value operand) {
    auto memrefOperand = dyn_cast<MemrefValue>(operand);
    return memrefOperand && memref::skipViewLikeOps(memrefOperand) == ourBuffer;
  };
  WalkResult res = loop->walk([&](Operation *op) {
    if (op == read.getOperation() || op == write.getOperation())
      return WalkResult::advance();
    if (auto xfer = dyn_cast<VectorTransferOpInterface>(op)) {
      Value otherBase = xfer.getBase();
      // A tensor transfer or a different underlying buffer cannot alias ours.
      if (!reachesOurBuffer(otherBase))
        return WalkResult::advance();
      // Same buffer: only safe if we can prove the accessed slices are
      // statically disjoint.
      std::optional<SubviewFootprint> otherFp =
          resolveSubviewFootprint(otherBase);
      if (otherFp && footprintsDisjoint(*ourFp, *otherFp))
        return WalkResult::advance();
      return WalkResult::interrupt();
    }
    // Any other op that may touch memory aliases ours if one of its memref
    // operands reaches our buffer.
    if (!isMemoryEffectFree(op) &&
        llvm::any_of(op->getOperands(), reachesOurBuffer))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !res.wasInterrupted();
}

void mlir::linalg::hoistRedundantVectorTransfers(Operation *root,
                                                 bool verifyNonZeroTrip) {
  bool changed = true;
  while (changed) {
    changed = false;
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interrupting the function walk.
    root->walk(
        [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

    // Find all loops that are certain to have non zero trip count. Any loops
    // that are not part of this set cannot be hoisted from, since hoisting from
    // a potentially zero trip count loop may cause a vector transfer to be
    // executed when it shouldn't be.
    llvm::DenseSet<LoopLikeOpInterface> definiteNonZeroTripCountLoops;
    if (verifyNonZeroTrip) {
      root->walk([&](LoopLikeOpInterface loopLike) {
        std::optional<SmallVector<OpFoldResult>> lbs =
            loopLike.getLoopLowerBounds();
        std::optional<SmallVector<OpFoldResult>> ubs =
            loopLike.getLoopUpperBounds();
        // If loop bounds cannot be found, assume possibly zero trip count.
        if (!lbs || !ubs)
          return;

        // Otherwise, use ValueBounds to find the maximum lower bound and
        // minimum upper bound. If the bounds are found, and maxLb is less
        // than the minUb, then the loop will not have zero trip count.
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

      // Look for the last TransferWriteOp in the forwardSlice of
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
      //      nor have users that are Ops implementing ViewLikeOpInterface.
      //   3. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.

      // Check 1.
      if (transferRead.getIndices() != transferWrite.getIndices() ||
          transferRead.getVectorType() != transferWrite.getVectorType() ||
          transferRead.getPermutationMap() != transferWrite.getPermutationMap())
        return WalkResult::advance();

      // Check 2. Note, since both xfer Ops share the source, we only need to
      // look at one of them.
      auto base = transferRead.getBase();
      // Whether hoisting is safe despite a view base. Computed lazily and
      // cached since it is only consulted when a view is present.
      std::optional<bool> viewAliasingIsSafeCache;
      auto viewAliasingIsSafe = [&]() {
        if (!viewAliasingIsSafeCache)
          viewAliasingIsSafeCache =
              footprintAliasingIsSafe(loop, transferRead, transferWrite);
        return *viewAliasingIsSafeCache;
      };
      auto *source = base.getDefiningOp();
      if (source) {
        // NOTE: We treat `memref.assume_alignment` as a special case.
        //
        // The idea is that it is safe to look past AssumeAlignmemtOp (i.e.
        // MemRef _before_ alignment) iff:
        //  1. It has exactly two uses (these have to be the xfer Ops
        //     being looked at).
        //  2. The original MemRef has only one use (i.e.
        //     AssumeAlignmentOp).
        //
        // Relaxing these conditions will most likely require proper alias
        // analysis.
        if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(source)) {
          Value memPreAlignment = assume.getMemref();
          auto numInLoopUses =
              llvm::count_if(base.getUses(), [&loop](OpOperand &use) {
                return loop->isAncestor(use.getOwner());
              });

          if (numInLoopUses && memPreAlignment.hasOneUse())
            source = memPreAlignment.getDefiningOp();
        }
        if (isa_and_nonnull<ViewLikeOpInterface>(source) &&
            !viewAliasingIsSafe())
          return WalkResult::advance();
      }

      if (llvm::any_of(base.getUsers(), llvm::IsaPred<ViewLikeOpInterface>) &&
          !viewAliasingIsSafe())
        return WalkResult::advance();

      // Check 3.
      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
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
          // Unknown use, we cannot prove that it doesn't alias with the
          // transferRead/transferWrite operations.
          return WalkResult::advance();
        }
      }

      // Hoist read before.
      loop.moveOutOfLoop(transferRead);

      // Hoist write after.
      transferWrite->moveAfter(loop);

      // Rewrite `loop` with new yields by cloning and erase the original
      // loop.
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
      // Need to interrupt and restart because erasing the loop messes up
      // the walk.
      return WalkResult::interrupt();
    });
  }
}
