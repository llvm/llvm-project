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

/// Ensure prerequisites that guarantee pad op hoisting can occur.
/// Return failure in the cases when we cannot perform hoisting; i.e. if either:
///   1. There exists a use of `simplePadOp` that is not a linalg input operand.
///   2. There isn't an enclosing `outermostEnclosingForOp` loop.
///   3. There exists an op with a region that is dominated by
///   `outermostEnclosingForOp` and that isn't a LoopLikeInterface or a
///    LinalgOp.
///   3. There exists an op with side effects that is dominated by
///    `outermostEnclosingForOp` and that isn't a LoopLikeInterface.
///
/// While ensuring prerequisites:
///   1. Fill the `backwardSlice` to contain the topologically sorted ops
///   dominated by `outermostEnclosingForOp`.
///   2. Fill the `packingLoops` to contain only the enclosing loops of
///   `backwardSlice` whose IV is actually used in computing padding. Loops that
///   remain in `backwardSlice` but that are not in `packingLoops` are
///   dimensions of reuse.
static LogicalResult
hoistPaddingOnTensorsPrerequisites(linalg::SimplePadOp simplePadOp, int nLevels,
                                   llvm::SetVector<Operation *> &backwardSlice,
                                   llvm::SetVector<Operation *> &packingLoops) {
  // Bail on any use that isn't an input of a Linalg op.
  // Hoisting of inplace updates happens after vectorization.
  for (OpOperand &use : simplePadOp.result().getUses()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgUser || !linalgUser.isInputTensor(&use))
      return failure();
  }

  // Get at most nLevels of enclosing loops.
  SmallVector<LoopLikeOpInterface> reverseEnclosingLoops;
  Operation *outermostEnclosingForOp = nullptr,
            *nextEnclosingForOp =
                simplePadOp->getParentOfType<LoopLikeOpInterface>();
  while (nLevels-- > 0 && nextEnclosingForOp) {
    outermostEnclosingForOp = nextEnclosingForOp;
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingForOp =
        nextEnclosingForOp->getParentOfType<LoopLikeOpInterface>();
  }
  if (!outermostEnclosingForOp)
    return failure();

  // Get the backwards slice from `simplePadOp` that is dominated by the
  // outermost enclosing loop.
  DominanceInfo domInfo(outermostEnclosingForOp);
  getBackwardSlice(simplePadOp, &backwardSlice, [&](Operation *op) {
    return domInfo.dominates(outermostEnclosingForOp, op);
  });

  #if 0

  // Bail on any op with a region that is not a LoopLikeInterface or a LinalgOp.
  // Bail on any op with side effects that is not a LoopLikeInterface.
  if (llvm::any_of(backwardSlice, [](Operation *op) {
        if (isa<LoopLikeOpInterface>(op))
          return false;
        if (!MemoryEffectOpInterface::hasNoEffect(op))
          return true;
        return op->getNumRegions() > 0 && !isa<LinalgOp>(op);
      }))
    return failure();

  #else

  // Bail on any op with a region that is not a LoopLikeInterface or a LinalgOp.
  if (llvm::any_of(backwardSlice, [](Operation *op) {
        return op->getNumRegions() > 0 && !isa<LoopLikeOpInterface>(op) &&
               !isa<LinalgOp>(op);
      }))
    return failure();

  #endif

  // Filter out the loops whose induction variable is not used to compute the
  // padded result. As a first approximation, just look for IVs that have no use
  // in the backwardSlice.
  // These are the dimensions of reuse that we can exploit to reduce the amount
  // of work / memory.
  // TODO: would this optimization compose better as a canonicalization?
  for (LoopLikeOpInterface loop : reverseEnclosingLoops) {
    auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
    if (!forOp)
      continue;
    for (Operation *user : forOp.getInductionVar().getUsers()) {
      if (backwardSlice.contains(user)) {
        packingLoops.insert(forOp);
        break;
      }
    }
  }

  // Backward slice is a topologically sorted list of ops starting at
  // `outermostEnclosingForOp`.
  assert(outermostEnclosingForOp == backwardSlice.front());

  return success();
}

static Value buildLoopTripCount(OpBuilder &b, Operation *op) {
  MLIRContext *ctx = op->getContext();
  AffineExpr lb, ub, step = getAffineSymbolExpr(0, ctx);
  bindDims(ctx, lb, ub);
  scf::ForOp forOp = cast<scf::ForOp>(op);
  return b.create<AffineApplyOp>(
      op->getLoc(), AffineMap::get(2, 1, {(ub - lb).ceilDiv(step)}, ctx),
      ValueRange{forOp.lowerBound(), forOp.upperBound(), forOp.step()});
}

/// Mechanically hoist padding operations on tensors by at most `nLoops` into a
/// new, generally larger tensor. This achieves packing of multiple padding ops
/// into a larger tensor. On success, `simplePadOp` is replaced by the cloned
/// version in the packing loop so the caller can continue reasoning about the
/// padding operation.
///
/// Example in pseudo-mlir:
/// =======================
///
/// If hoistPaddingOnTensors is called with `nLoops` = 2 on the following IR.
/// ```
///    scf.for (%i, %j, %k)
///      %st0 = subtensor f(%i, %k) : ... to tensor<?x?xf32>
///      %0 = linalg.simple_pad %st0 pad %pad :
///             tensor<?x?xf32> to tensor<4x8xf32>
///      compute(%0)
/// ```
///
/// IR resembling the following is produced:
///
/// ```
///    scf.for (%i) {
///      %packed_init = linalg.init_tensor range(%j) : tensor<?x4x8xf32>
///      %packed = scf.for (%k) iter_args(%p : %packed_init)
///        %st0 = subtensor f(%i, %k) : ... to tensor<?x?xf32>
///        %0 = linalg.simple_pad %st0 pad %pad :
///               tensor<?x?xf32> to tensor<4x8xf32>
///        scf.yield %1: tensor<?x4x8xf32>
///      } -> tensor<?x4x8xf32>
///      scf.for (%j, %k) {
///        %st0 = subtensor %packed [%k, 0, 0][1, 4, 8][1, 1, 1] :
///                 tensor<?x4x8xf32> to tensor<4x8xf32>
///        compute(%st0)
///      }
///    }
/// ```
LogicalResult mlir::linalg::hoistPaddingOnTensors(SimplePadOp &simplePadOp,
                                                  unsigned nLoops) {
  llvm::SetVector<Operation *> backwardSlice, packingLoops;
  if (failed(hoistPaddingOnTensorsPrerequisites(simplePadOp, nLoops,
                                                backwardSlice, packingLoops)))
    return failure();

  // Update actual number of loops, which may be smaller.
  nLoops = packingLoops.size();

  Location loc = simplePadOp->getLoc();
  RankedTensorType paddedTensorType = simplePadOp.getResultType();
  unsigned paddedRank = paddedTensorType.getRank();

  // Backward slice is a topologically sorted list of ops starting at
  // `outermostEnclosingForOp`.
  Operation *outermostEnclosingForOp = backwardSlice.front();
  // IP just before the outermost loop considered that we hoist above.
  OpBuilder b(outermostEnclosingForOp);

  // Create the packed tensor<?x?x..?xpadded_shape> into which we amortize
  // padding.
  SmallVector<int64_t> packedShape(nLoops, ShapedType::kDynamicSize);
  // TODO: go grab dims when necessary, for now SimplePadOp returns a static
  // tensor.
  llvm::append_range(packedShape, paddedTensorType.getShape());
  auto packedTensorType =
      RankedTensorType::get(packedShape, paddedTensorType.getElementType());
  auto dynamicSizes = llvm::to_vector<4>(llvm::map_range(
      packingLoops, [&](Operation *op) { return buildLoopTripCount(b, op); }));
  Value packedTensor = b.create<linalg::InitTensorOp>(
      loc, dynamicSizes, packedTensorType.getShape(),
      packedTensorType.getElementType());

  // Clone the operations involved in the backward slice, iteratively stepping
  // into the loops that we encounter.
  // The implementation proceeds in a stack-like fashion:
  //   1. Iteratively clone and step into the loops, pushing the `packedTensor`
  //      deeper in the stack.
  //   2. Create a SubTensorInsert at the top of the stack.
  //   3. Iteratively pop and yield the result of the SubTensorInsertOp across
  //     the cloned loops.
  SmallVector<Value> clonedLoopIvs;
  clonedLoopIvs.reserve(nLoops);
  BlockAndValueMapping bvm;
  // Stack step 1. iteratively clone loops and push `packedTensor`.
  // Insert `simplePadOp` into the backwardSlice so we clone it too.
  backwardSlice.insert(simplePadOp);
  for (Operation *op : backwardSlice) {
    if (op->getNumRegions() == 0) {
      b.clone(*op, bvm);
      continue;
    }
    // TODO: support more cases as they appear.
    auto forOp = dyn_cast<scf::ForOp>(op);
    assert(forOp && "Expected scf::ForOp when hoisting pad ops");
    // Unused loop, just skip it.
    if (!packingLoops.contains(forOp))
      continue;
    auto clonedForOp =
        b.create<scf::ForOp>(loc, forOp.lowerBound(), forOp.upperBound(),
                             forOp.step(), packedTensor);
    assert(clonedForOp->getNumRegions() == 1);
    clonedLoopIvs.push_back(clonedForOp.getInductionVar());
    b.setInsertionPointToStart(&clonedForOp->getRegion(0).front());
    bvm.map(forOp.getInductionVar(), clonedLoopIvs.back());
    packedTensor = clonedForOp.getRegionIterArgs().front();
  }

  // Stack step 2. create SubTensorInsertOp at the top of the stack.
  // offsets = [clonedLoopIvs, 0 .. 0].
  SmallVector<OpFoldResult> offsets(clonedLoopIvs.begin(), clonedLoopIvs.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape].
  SmallVector<OpFoldResult> sizes(nLoops, b.getIndexAttr(1));
  for (int64_t sz : paddedTensorType.getShape()) {
    // TODO: go grab dims when necessary, for now SimplePadOp returns a static
    // tensor.
    assert(!ShapedType::isDynamic(sz) && "padded tensor needs static sizes");
    sizes.push_back(b.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  SmallVector<OpFoldResult> strides(nLoops + paddedRank, b.getIndexAttr(1));

  Value inserted =
      b.create<SubTensorInsertOp>(loc, bvm.lookup(simplePadOp.result()),
                                  packedTensor, offsets, sizes, strides);

  // Stack step 3. iteratively pop the stack and propagate the yield.
  Value valueToYield = inserted;
  for (Value iv : llvm::reverse(clonedLoopIvs)) {
    auto forOp = scf::getForInductionVarOwner(iv);
    b.setInsertionPointToEnd(&forOp.getRegion().front());
    b.create<scf::YieldOp>(loc, valueToYield);
    valueToYield = forOp.getResult(0);
  }

  // Now the packed tensor is ready, replace the original padding op by a
  // 1x..x1 SubTensor [originalLoopIvs, 0 .. 0][1 .. 1, paddedShape][1 .. 1].
  b.setInsertionPoint(simplePadOp);
  SmallVector<Value> originalLoopIvs =
      llvm::to_vector<4>(llvm::map_range(packingLoops, [](Operation *loop) {
        return cast<scf::ForOp>(loop).getInductionVar();
      }));
  // offsets = [originalLoopIvs, 0 .. 0].
  offsets.assign(originalLoopIvs.begin(), originalLoopIvs.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape] (definedabove).
  // strides = [1 .. 1] (defined above)
  packedTensor =
      scf::getForInductionVarOwner(clonedLoopIvs.front())->getResult(0);
  simplePadOp.replaceAllUsesWith(
      b.create<SubTensorOp>(loc, simplePadOp.getResultType(), packedTensor,
                            offsets, sizes, strides)
          ->getResult(0));

  Operation *toErase = simplePadOp;

  // Make the newly cloned `simplePadOp` available to the caller.
  simplePadOp =
      cast<SimplePadOp>(bvm.lookup(simplePadOp.result()).getDefiningOp());

  toErase->erase();

  return success();
}
