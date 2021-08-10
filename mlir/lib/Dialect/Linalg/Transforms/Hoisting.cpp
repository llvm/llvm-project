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
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/Analysis/ConstraintsSet.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Represents a unit of hoistable TransferWriteOp. This may comprise other
/// instructions that need to be hoisted too.
struct HoistableWrite {
  vector::TransferWriteOp transferWriteOp;
  tensor::InsertSliceOp insertSliceOp;
};
/// Represents a unit of hoistable TransferReadOp. This may comprise other
/// instructions that need to be hoisted too.
struct HoistableRead {
  vector::TransferReadOp transferReadOp;
  tensor::ExtractSliceOp extractSliceOp;
};
} // namespace

/// Return true if op1 and op2 are the same constant or the same SSA value.
static bool isEqualOffsetSizeOrStride(OpFoldResult op1, OpFoldResult op2) {
  auto getConstantIntValue = [](OpFoldResult ofr) -> llvm::Optional<int64_t> {
    Attribute attr = ofr.dyn_cast<Attribute>();
    // Note: isa+cast-like pattern allows writing the condition below as 1 line.
    if (!attr && ofr.get<Value>().getDefiningOp<ConstantOp>())
      attr = ofr.get<Value>().getDefiningOp<ConstantOp>().getValue();
    if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>())
      return intAttr.getValue().getSExtValue();
    return llvm::None;
  };
  auto cst1 = getConstantIntValue(op1), cst2 = getConstantIntValue(op2);
  if (cst1 && cst2 && *cst1 == *cst2)
    return true;
  auto v1 = op1.dyn_cast<Value>(), v2 = op2.dyn_cast<Value>();
  return v1 && v2 && v1 == v2;
}

/// Return true is all offsets, sizes and strides are equal.
static bool sameOffsetsSizesAndStrides(tensor::ExtractSliceOp s,
                                       tensor::InsertSliceOp si) {
  if (s.static_offsets().size() != si.static_offsets().size())
    return false;
  if (s.static_sizes().size() != si.static_sizes().size())
    return false;
  if (s.static_strides().size() != si.static_strides().size())
    return false;
  for (auto it : llvm::zip(s.getMixedOffsets(), si.getMixedOffsets()))
    if (!isEqualOffsetSizeOrStride(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(s.getMixedSizes(), si.getMixedSizes()))
    if (!isEqualOffsetSizeOrStride(std::get<0>(it), std::get<1>(it)))
      return false;
  for (auto it : llvm::zip(s.getMixedStrides(), si.getMixedStrides()))
    if (!isEqualOffsetSizeOrStride(std::get<0>(it), std::get<1>(it)))
      return false;
  return true;
}

/// Look for a HoistableRead, in the given tensor uses, accessing the same
/// offset as the HoistableWrite.
static HoistableRead findMatchingTransferRead(HoistableWrite write,
                                              Value srcTensor) {
  assert(write.transferWriteOp &&
         "expected hoistable write to have a .transfer_write");

  LLVM_DEBUG(DBGS() << "findMatchingTransferRead for: "
                    << *write.transferWriteOp.getOperation() << "\n");
  if (write.insertSliceOp)
    LLVM_DEBUG(DBGS() << "findMatchingTransferRead inserSliceOp: "
                      << *write.insertSliceOp.getOperation() << "\n");

  for (Operation *user : srcTensor.getUsers()) {
    LLVM_DEBUG(DBGS() << "findMatchingTransferRead inspect user: " << *user
                      << "\n");

    // If HoistableWrite involves a InsertSliceOp, we need to find a
    // matching ExtractSliceOp.
    tensor::ExtractSliceOp sliceOp;
    Operation *maybeTransferReadUser = user;
    if (write.insertSliceOp) {
      sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!sliceOp || sliceOp.getResult().getType() !=
                          write.insertSliceOp.source().getType())
        continue;

      LLVM_DEBUG(DBGS() << "check whether sameOffsetsSizesAndStrides: "
                        << *sliceOp << " vs " << *write.insertSliceOp << "\n");
      if (!sameOffsetsSizesAndStrides(sliceOp, write.insertSliceOp))
        continue;

      LLVM_DEBUG(DBGS() << "sameOffsetsSizesAndStrides: SUCCESS\n");
      // If we got here, sliceOp is hoistable iff it has exactly 2 uses:
      //   1. the transfer_write we want to hoist.
      //   2. a matching transfer_read.
      // Anything else, we skip.
      bool skip = false;
      Operation *otherUser = nullptr;
      for (Operation *u : sliceOp->getUsers()) {
        if (u == write.transferWriteOp)
          continue;
        if (otherUser) {
          skip = true;
          break;
        }
        otherUser = u;
      }
      if (skip || !otherUser)
        continue;
      maybeTransferReadUser = otherUser;
    }

    LLVM_DEBUG(DBGS() << "maybeTransferReadUser: " << *maybeTransferReadUser
                      << "\n");
    auto read = dyn_cast<vector::TransferReadOp>(maybeTransferReadUser);
    if (read && read.indices() == write.transferWriteOp.indices() &&
        read.getVectorType() == write.transferWriteOp.getVectorType())
      return HoistableRead{read, sliceOp};
  }
  return HoistableRead();
}

/// Check if the chunk of data inserted by the HoistableWrite are read by any
/// other op than the HoistableRead candidate.
static bool tensorChunkAccessedByUnknownOp(HoistableWrite write,
                                           HoistableRead candidateRead,
                                           BlockArgument tensorArg) {
  // Make sure none of the other uses read the part of the tensor modified
  // by the transfer_write.
  llvm::SmallVector<Value::use_range, 1> uses;
  uses.push_back(tensorArg.getUses());
  while (!uses.empty()) {
    for (OpOperand &use : uses.pop_back_val()) {
      Operation *user = use.getOwner();
      // Skip the candidate use, only inspect the "other" uses.
      if (user == candidateRead.transferReadOp ||
          user == candidateRead.extractSliceOp ||
          user == write.transferWriteOp || user == write.insertSliceOp)
        continue;
      // Consider all transitive uses through a extract_slice / insert_slice.
      // TODO: atm we just bail because a stronger analysis is needed for these
      // cases.
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user))
        return true;
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
      if (yieldUser && write.transferWriteOp->getParentOp()->isAncestor(
                           yieldUser->getParentOp())) {
        Value ret = yieldUser->getParentOp()->getResult(use.getOperandNumber());
        uses.push_back(ret.getUses());
        continue;
      }
      auto read = dyn_cast<vector::TransferReadOp>(user);
      if (!read || !isDisjointTransferIndices(
                       cast<VectorTransferOpInterface>(read.getOperation()),
                       cast<VectorTransferOpInterface>(
                           write.transferWriteOp.getOperation()))) {
        return true;
      }
    }
  }
  return false;
}

/// Return the `forOp`-invariant HoistableWrite that produces `yieldOperand`.
/// Return the null HoistableWrite() if it is not comprised of a
/// vector.transfer_write + optional insert_slice or if any of the indexings
/// is `forOp`-dependent.
static HoistableWrite
getLoopInvariantTransferWriteOpDefining(scf::ForOp forOp,
                                        OpOperand &yieldOperand) {
  Value v = yieldOperand.get();
  if (auto write = v.getDefiningOp<vector::TransferWriteOp>()) {
    // Indexing must not depend on `forOp`.
    for (Value operand : write.indices())
      if (!forOp.isDefinedOutsideOfLoop(operand))
        return HoistableWrite();

    return HoistableWrite{write, nullptr};
  }

  if (auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>()) {
    // Inserted slice must come from vector.transfer_write.
    auto write =
        insertSliceOp.source().getDefiningOp<vector::TransferWriteOp>();
    if (!write)
      return HoistableWrite();

    // Tensor inserted into must be a BBArg at position matching yieldOperand's.
    auto bbArg = insertSliceOp.dest().dyn_cast<BlockArgument>();
    if (!bbArg || bbArg.getOwner()->getParentOp() != forOp ||
        bbArg.getArgNumber() != /*num iv=*/1 + yieldOperand.getOperandNumber())
      return HoistableWrite();

    // Indexing inserted into must not depend on `forOp`.
    for (Value operand : insertSliceOp->getOperands().drop_front(
             tensor::InsertSliceOp::getOffsetSizeAndStrideStartOperandIndex()))
      if (!forOp.isDefinedOutsideOfLoop(operand))
        return HoistableWrite();

    return HoistableWrite{write, insertSliceOp};
  }

  return HoistableWrite();
}

/// Mechanical hoisting of a matching HoistableRead / HoistableWrite pair.
static void hoistReadWrite(HoistableRead read, HoistableWrite write,
                           BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());
  assert(read.transferReadOp && write.transferWriteOp &&
         "expected transfer_read and transfer_write ops to be set");
  assert(((read.extractSliceOp && write.insertSliceOp) ||
          (!read.extractSliceOp && !write.insertSliceOp)) &&
         "expected matching extract_slice / insert_slice");
  LLVM_DEBUG(DBGS() << "In forOp:\n"
                    << *forOp.getOperation()
                    << "\nHoist: " << *read.transferReadOp.getOperation()
                    << "\nHoist: " << *write.transferWriteOp.getOperation()
                    << "\nInvolving: " << tensorBBArg << "\n");

  // If a read slice is present, hoist it.
  if (read.extractSliceOp && failed(forOp.moveOutOfLoop({read.extractSliceOp})))
    llvm_unreachable("Unexpected failure moving extract_slice out of loop");

  // Hoist the transfer_read op.
  if (failed(forOp.moveOutOfLoop({read.transferReadOp})))
    llvm_unreachable("Unexpected failure moving transfer read out of loop");

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  unsigned initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // Update the source tensor.
  if (read.extractSliceOp)
    read.extractSliceOp.sourceMutable().assign(forOp.initArgs()[initArgNumber]);
  else
    read.transferReadOp.sourceMutable().assign(forOp.initArgs()[initArgNumber]);

  // Hoist write after.
  if (write.insertSliceOp)
    write.insertSliceOp->moveAfter(forOp);
  write.transferWriteOp->moveAfter(forOp);

  // Update the yield.
  auto yieldOp = cast<scf::YieldOp>(forOp.region().front().getTerminator());
  if (write.insertSliceOp)
    yieldOp->setOperand(initArgNumber, write.insertSliceOp.dest());
  else
    yieldOp->setOperand(initArgNumber, write.transferWriteOp.source());

  // Rewrite `loop` with additional new yields.
  OpBuilder b(read.transferReadOp);
  auto newForOp = cloneWithNewYields(b, forOp, read.transferReadOp.vector(),
                                     write.transferWriteOp.vector());
  // Transfer write has been hoisted, need to update the vector and tensor
  // source. Replace the result of the loop to use the new tensor created
  // outside the loop.
  // Depending on whether a insert_slice is present or not, it carries the
  // update on the tensor operands.
  if (write.insertSliceOp) {
    newForOp.getResult(initArgNumber)
        .replaceAllUsesWith(write.insertSliceOp.getResult());
    write.transferWriteOp.sourceMutable().assign(read.extractSliceOp.result());
    write.insertSliceOp.destMutable().assign(read.extractSliceOp.source());
  } else {
    newForOp.getResult(initArgNumber)
        .replaceAllUsesWith(write.transferWriteOp.getResult(0));
    write.transferWriteOp.sourceMutable().assign(
        newForOp.getResult(initArgNumber));
  }

  // Always update with the newly yield tensor and vector.
  write.transferWriteOp.vectorMutable().assign(newForOp.getResults().back());
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
        OpOperand &ret = yield->getOpOperand(it.index());
        HoistableWrite write =
            getLoopInvariantTransferWriteOpDefining(forOp, ret);
        if (!write.transferWriteOp || !write.transferWriteOp->hasOneUse())
          continue;
        LLVM_DEBUG(dbgs() << "\n";
                   DBGS() << "Candidate write for hoisting: "
                          << *write.transferWriteOp.getOperation() << "\n");
        if (write.insertSliceOp)
          LLVM_DEBUG(DBGS() << "Candidate insert_slice for hoisting: "
                            << *write.insertSliceOp.getOperation() << "\n");
        if (llvm::any_of(write.transferWriteOp.indices(),
                         [&forOp](Value index) {
                           return !forOp.isDefinedOutsideOfLoop(index);
                         }))
          continue;
        // Find a read with the same type and indices.
        HoistableRead matchingRead =
            findMatchingTransferRead(write, it.value());
        // Make sure none of the other uses read the part of the tensor modified
        // by the transfer_write.
        if (!matchingRead.transferReadOp ||
            tensorChunkAccessedByUnknownOp(write, matchingRead, it.value()))
          continue;

        LLVM_DEBUG(DBGS() << "Start hoisting\n");
        hoistReadWrite(matchingRead, write, it.value());
        changed = true;
        forOp.erase();

        // Need to interrupt and restart: erasing the loop messes up the walk.
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // Apply canonicalization so the newForOp + yield folds immediately, thus
    // cleaning up the IR and potentially enabling more hoisting.
    if (changed) {
      RewritePatternSet patterns(func->getContext());
      scf::ForOp::getCanonicalizationPatterns(patterns, func->getContext());
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
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

      SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead.getOperation(), &forwardSlice);

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

/// Return success if `v` is a value that is only transitively defined by ops of
/// type in `OpTypeList`.
template <typename... OpTypeList>
static bool backwardsSliceOnlyHasOpsOfType(scf::ForOp outerLimit, Value v) {
  // Compute a backward slice up to, but not including, `outerLimit`.
  SetVector<Operation *> backwardSlice;
  getBackwardSlice(v, &backwardSlice, [&](Operation *op) {
    return outerLimit->isProperAncestor(op);
  });
  // Traverse the backward slice and ensure we can perform the computation to
  // hoist.
  for (Operation *op : backwardSlice) {
    if (isa<OpTypeList...>(op))
      continue;
    LLVM_DEBUG(DBGS() << "Abort: unadmissible op in slice " << *op << "\n");
    return false;
  }
  return true;
}

bool isDefinedOutsideOrConstant(scf::ForOp outer, Value v) {
  return outer.isDefinedOutsideOfLoop(v) || v.getDefiningOp<ConstantOp>();
}

/// Return the current iteration number in the loop (iv - lb).ceilDiv(step).
/// The returned Value is guaranteed not to depend on any loop comprised in
/// [`outer`, `forOp`].
/// Return null if such a loop-independent quantity cannot be computed.
static Value buildLoopIterationCount(OpBuilder &b, scf::ForOp outer,
                                     scf::ForOp forOp) {
  MLIRContext *ctx = forOp->getContext();
  AffineExpr iv, lb, step;
  bindDims(ctx, iv, lb);
  bindSymbols(ctx, step);
  if (!isDefinedOutsideOrConstant(outer, forOp.lowerBound()) ||
      !isDefinedOutsideOrConstant(outer, forOp.step()))
    return Value();
  Value ivVal = forOp.getInductionVar(), lbVal = forOp.lowerBound(),
        stepVal = forOp.step();
  auto loc = forOp->getLoc();
  return b.createOrFold<AffineApplyOp>(loc, (iv - lb).ceilDiv(step),
                                       ValueRange{ivVal, lbVal, stepVal});
}

/// Given a set of loops, assumed to be scf::ForOp, create a constraint set
/// containing the inequalities `iv - lb >= 0` and `-iv + ub - 1 >= 0` for each
/// loop.
static ConstraintsSet initLoopIvsAndBounds(ArrayRef<Operation *> loops) {
  ConstraintsSet constraints;
  for (Operation *op : loops)
    constraints.addDimId(constraints.getNumDimIds(),
                         cast<scf::ForOp>(op).getInductionVar());
  for (Operation *op : loops)
    constraints.addDimId(constraints.getNumDimIds(),
                         cast<scf::ForOp>(op).lowerBound());
  for (Operation *op : loops)
    constraints.addDimId(constraints.getNumDimIds(),
                         cast<scf::ForOp>(op).upperBound());
  unsigned numLoops = loops.size();
  for (unsigned ivIdx = 0, e = numLoops; ivIdx < e; ++ivIdx) {
    // iv - lb >= 0
    SmallVector<int64_t, 8> ineqLb(constraints.getNumCols(), 0);
    ineqLb[ivIdx] = 1;
    ineqLb[ivIdx + numLoops] = -1;
    // -iv + ub >= 0
    SmallVector<int64_t, 8> ineqUb(constraints.getNumCols(), 0);
    ineqUb[ivIdx] = -1;
    ineqUb[ivIdx + 2 * numLoops] = 1;
    ineqUb[constraints.getNumCols() - 1] = -1;
    constraints.addInequality(ineqLb);
    constraints.addInequality(ineqUb);
  }
  return constraints;
}

/// For each loop in `loops`, determine the ops involved in the construction of
/// its upper bound---up to the outerLimit loop--- and fold them as new
/// inequalities in the constraint set.
/// This is achieved by computing the backwardSlice of the loop's upper bound
/// and iteratively folding each op in reverse topological order to guarantee
/// use-def ordering.
/// As operations are folded in, their result is projected out of the
/// constraints set.
/// The following operations are supported:
///   - scf::ForOp are simply skipped.
///   - AffineApplyOp are composed to replace the result by an equality.
///   - AffineMinOp are composed by adding each entry as an upper bound.
/// If any other operation is met, return failure.
// TODO: extend on a per-need basis.
static LogicalResult
foldUpperBoundsIntoConstraintsSet(ConstraintsSet &constraints,
                                  scf::ForOp outerLimit,
                                  ArrayRef<Operation *> loops) {
  SetVector<Value> toProjectOut;
  for (Operation *loop : loops) {
    auto ub = cast<scf::ForOp>(loop).upperBound();
    if (isDefinedOutsideOrConstant(outerLimit, ub))
      continue;

    // Compute a backward slice up to, but not including, `outerLimit`.
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(ub, &backwardSlice, [&](Operation *op) {
      return outerLimit->isProperAncestor(op);
    });
    backwardSlice.insert(ub.getDefiningOp());

    // Iterate over all ops in the slice and compose them in the constraints.
    for (Operation *op : llvm::reverse(backwardSlice)) {
      if (!isa<scf::ForOp, AffineApplyOp, AffineMinOp>(op))
        return failure();
      if (isa<scf::ForOp>(op))
        continue;
      // Ensure there is a
      auto ensureIdFailed = [&](Value v) {
        return failed(constraints.ensureIdOfType(v, /*asDim=*/true));
      };

      // Ensure all ids exist and add results for later projection.
      if (llvm::any_of(op->getResults(), ensureIdFailed) ||
          llvm::any_of(op->getOperands(), ensureIdFailed))
        return failure();

      // All supported ops have 1 result.
      // TODO: extend when needed.
      toProjectOut.insert(op->getResult(0));

      // Compose supported ops.
      if (auto affineApplyOp = dyn_cast<AffineApplyOp>(op)) {
        if (failed(constraints.composeAffineApply(affineApplyOp.getResult(),
                                                  affineApplyOp.getAffineMap(),
                                                  affineApplyOp.getOperands())))
          return failure();
        continue;
      }
      auto affineMinOp = cast<AffineMinOp>(op);
      if (failed(constraints.composeMin(affineMinOp.getResult(),
                                        affineMinOp.getAffineMap(),
                                        affineMinOp.operands())))
        return failure();
    }
  }
  for (Value v : toProjectOut)
    constraints.projectOut(v);
  return success();
}

/// Compute dynamic tensor sizes, independent of any value defined inside
/// `outer` and such that every n-D iteration of the packingLoops has its own
/// space (so that each packed buffer has a storage location). This is achieved
/// by computing the extent for each of the packing loops.
static LogicalResult computeBounds(scf::ForOp outer,
                                   ArrayRef<Operation *> packingLoops,
                                   SmallVector<AffineMap> &lbs,
                                   SmallVector<AffineMap> &ubs) {
  // Packing loop IVs are introduced as the first positions.
  ConstraintsSet constraints = initLoopIvsAndBounds(packingLoops);
  if (failed(
          foldUpperBoundsIntoConstraintsSet(constraints, outer, packingLoops)))
    return failure();
  // Compute the bounds of the first positions, assuming the others are fixed.
  constraints.getSliceBounds(/*pos=*/0, /*num=*/packingLoops.size(),
                             outer->getContext(), &lbs, &ubs);
  return success();
}

/// Ensure prerequisites that guarantee pad op hoisting can occur.
/// Return failure in the cases when we cannot perform hoisting; i.e. if either:
///   1. There exists a use of `padTensorOp` that is not a linalg input operand.
///   2. There isn't an enclosing `outermostEnclosingForOp` loop.
///   3. There exists an op with a region that is dominated by
///   `outermostEnclosingForOp` and that isn't a LoopLikeInterface or a
///    LinalgOp.
///   4. There exists an op with side effects that is dominated by
///   `outermostEnclosingForOp` and that isn't a LoopLikeInterface.
///   5. The lower bound, upper bound and step of all the loops involved in the
///   hoisting can be
///
/// While ensuring prerequisites:
///   1. Fill the `backwardSlice` to contain the topologically sorted ops
///   dominated by `outermostEnclosingForOp`.
///   2. Fill the `packingLoops` to contain only the enclosing loops of
///   `backwardSlice` whose IV is actually used in computing padding. Loops that
///   remain in `backwardSlice` but that are not in `packingLoops` are
///   dimensions of reuse.
static LogicalResult
hoistPaddingOnTensorsPrerequisites(linalg::PadTensorOp padTensorOp, int nLevels,
                                   SetVector<Operation *> &backwardSlice,
                                   SetVector<Operation *> &packingLoops,
                                   SmallVector<Value> &dynamicTensorSizes) {
  // Bail on any use that isn't an input of a Linalg op.
  // Hoisting of inplace updates happens after vectorization.
  for (OpOperand &use : padTensorOp.result().getUses()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgUser || !linalgUser.isInputTensor(&use))
      return failure();
  }

  // Get at most nLevels of enclosing loops.
  SmallVector<LoopLikeOpInterface> reverseEnclosingLoops;
  Operation *outermostEnclosingForOp = nullptr,
            *nextEnclosingForOp =
                padTensorOp->getParentOfType<LoopLikeOpInterface>();
  while (nLevels-- > 0 && nextEnclosingForOp) {
    outermostEnclosingForOp = nextEnclosingForOp;
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingForOp =
        nextEnclosingForOp->getParentOfType<LoopLikeOpInterface>();
  }
  if (!outermostEnclosingForOp)
    return failure();

  // Get the backwards slice from `padTensorOp` that is dominated by the
  // outermost enclosing loop.
  DominanceInfo domInfo(outermostEnclosingForOp);
  getBackwardSlice(padTensorOp.getOperation(), &backwardSlice,
                   [&](Operation *op) {
                     return domInfo.dominates(outermostEnclosingForOp, op);
                   });

  // Bail on any op with a region that is not a LoopLikeInterface or a LinalgOp.
  if (llvm::any_of(backwardSlice, [](Operation *op) {
        return op->getNumRegions() > 0 && !isa<LoopLikeOpInterface>(op) &&
               !isa<LinalgOp>(op);
      }))
    return failure();

  // Filter out the loops whose induction variable is not used to compute the
  // padded result. As a first approximation, just look for IVs that have no use
  // in the backwardSlice.
  // These are the dimensions of reuse that we can exploit to reduce the amount
  // of work / memory.
  // TODO: would this optimization compose better as a canonicalization?
  for (LoopLikeOpInterface loop : llvm::reverse(reverseEnclosingLoops)) {
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

  scf::ForOp outer = cast<scf::ForOp>(outermostEnclosingForOp);

  ConstraintsSet constraints = initLoopIvsAndBounds(packingLoops.getArrayRef());
  if (failed(foldUpperBoundsIntoConstraintsSet(constraints, outer,
                                               packingLoops.getArrayRef())))
    return failure();

  unsigned numLoops = packingLoops.size();
  SmallVector<AffineMap> lbs(numLoops), ubs(numLoops);
  if (failed(computeBounds(outer, packingLoops.getArrayRef(), lbs, ubs)))
    return failure();

  SmallVector<Value> allValues;
  constraints.getAllIdValues(&allValues);
  SmallVector<Value> allNonLoopValues(allValues.begin() + numLoops,
                                      allValues.end());

  // For each packingLoop, create the extent by (ub - lb).ceilDiv(step).
  // IP just before the outermost loop considered that we hoist above.
  ImplicitLocOpBuilder b(outer->getLoc(), outer);
  assert(packingLoops.size() == lbs.size() && "expected matching lb sizes");
  assert(packingLoops.size() == ubs.size() && "expected matching ub sizes");
  for (auto it : llvm::zip(packingLoops, lbs, ubs)) {
    scf::ForOp loop = cast<scf::ForOp>(std::get<0>(it));
    AffineMap lbMap = std::get<1>(it);
    AffineMap ubMap = std::get<2>(it);
    SmallVector<Value> lbOperands(allNonLoopValues);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);
    Value lbVal = b.createOrFold<AffineMaxOp>(lbMap, lbOperands);

    SmallVector<Value> ubOperands(allNonLoopValues);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);
    Value ubVal = b.createOrFold<AffineMinOp>(ubMap, ubOperands);

    AffineExpr lb, ub, step;
    bindDims(b.getContext(), lb, ub);
    bindSymbols(b.getContext(), step);
    Value res = b.createOrFold<AffineApplyOp>(
        (ub - lb).ceilDiv(step),
        ValueRange{lbVal, ubVal, cast<scf::ForOp>(loop).step()});

    dynamicTensorSizes.push_back(res);
  }

  return success();
}

LogicalResult mlir::linalg::hoistPaddingOnTensors(PadTensorOp &padTensorOp,
                                                  unsigned nLoops) {
  SmallVector<Value> dynamicTensorSizes;
  SetVector<Operation *> backwardSlice, packingLoops;
  if (failed(hoistPaddingOnTensorsPrerequisites(padTensorOp, nLoops,
                                                backwardSlice, packingLoops,
                                                dynamicTensorSizes)))
    return failure();

  // Update actual number of loops, which may be smaller.
  nLoops = packingLoops.size();

  Location loc = padTensorOp->getLoc();
  RankedTensorType paddedTensorType = padTensorOp.getResultType();
  unsigned paddedRank = paddedTensorType.getRank();

  // Backward slice is a topologically sorted list of ops starting at
  // `outermostEnclosingForOp`.
  Operation *outermostEnclosingForOp = backwardSlice.front();
  // IP just before the outermost loop considered that we hoist above.
  OpBuilder b(outermostEnclosingForOp);

  // Create the packed tensor<?x?x..?xpadded_shape> into which we amortize
  // padding.
  SmallVector<int64_t> packedShape(nLoops, ShapedType::kDynamicSize);
  // TODO: go grab dims when necessary, for now PadTensorOp returns a static
  // tensor.
  llvm::append_range(packedShape, paddedTensorType.getShape());
  auto packedTensorType =
      RankedTensorType::get(packedShape, paddedTensorType.getElementType());
  Value packedTensor = b.create<linalg::InitTensorOp>(
      loc, dynamicTensorSizes, packedTensorType.getShape(),
      packedTensorType.getElementType());

  // Clone the operations involved in the backward slice, iteratively stepping
  // into the loops that we encounter.
  // The implementation proceeds in a stack-like fashion:
  //   1. Iteratively clone and step into the loops, pushing the `packedTensor`
  //      deeper in the stack.
  //   2. Create a InsertSliceOp at the top of the stack.
  //   3. Iteratively pop and yield the result of the InsertSliceOp across
  //     the cloned loops.
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  clonedLoopIvs.reserve(nLoops);
  leadingPackedTensorIndexings.reserve(nLoops);
  BlockAndValueMapping bvm;
  // Insert `padTensorOp` into the backwardSlice so we clone it too.
  backwardSlice.insert(padTensorOp);
  // Stack step 1. iteratively clone loops and push `packedTensor`.
  for (Operation *op : backwardSlice) {
    // Specifically sit out in the extract_slice(packedTensor) case: this is the
    // piece we seek to replace.
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op))
      if (bvm.lookupOrDefault(sliceOp.source()) == packedTensor)
        continue;
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    bool hasNoEffects = !effects || effects.hasNoEffect();
    if (hasNoEffects &&
        (op->getNumRegions() == 0 || isa<linalg::PadTensorOp>(op))) {
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
        b.create<scf::ForOp>(loc, bvm.lookupOrDefault(forOp.lowerBound()),
                             bvm.lookupOrDefault(forOp.upperBound()),
                             bvm.lookupOrDefault(forOp.step()), packedTensor);
    // Map the induction var, region args and results to the `clonedForOp`.
    bvm.map(forOp.getInductionVar(), clonedForOp.getInductionVar());
    bvm.map(forOp.getRegionIterArgs(), clonedForOp.getRegionIterArgs());
    bvm.map(forOp.getResults(), clonedForOp.getResults());
    assert(clonedForOp->getNumRegions() == 1);
    clonedLoopIvs.push_back(clonedForOp.getInductionVar());

    b.setInsertionPointToStart(&clonedForOp->getRegion(0).front());
    Value loopIndependentIterationCount = buildLoopIterationCount(
        b, cast<scf::ForOp>(outermostEnclosingForOp), clonedForOp);
    // Assert the loop-independent iteration count can be computed.
    if (!loopIndependentIterationCount)
      llvm_unreachable("loop independence prerequisite not met");
    leadingPackedTensorIndexings.push_back(loopIndependentIterationCount);
    packedTensor = clonedForOp.getRegionIterArgs().front();
  }

  // Stack step 2. create InsertSliceOp at the top of the stack.
  // offsets = [clonedLoopIvs, 0 .. 0].
  SmallVector<OpFoldResult> offsets(leadingPackedTensorIndexings.begin(),
                                    leadingPackedTensorIndexings.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape].
  SmallVector<OpFoldResult> sizes(nLoops, b.getIndexAttr(1));
  for (int64_t sz : paddedTensorType.getShape()) {
    // TODO: go grab dims when necessary, for now PadTensorOp returns a static
    // tensor.
    assert(!ShapedType::isDynamic(sz) && "padded tensor needs static sizes");
    sizes.push_back(b.getIndexAttr(sz));
  }
  // strides = [1 .. 1].
  SmallVector<OpFoldResult> strides(nLoops + paddedRank, b.getIndexAttr(1));

  Value inserted =
      b.create<tensor::InsertSliceOp>(loc, bvm.lookup(padTensorOp.result()),
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
  // 1x..x1 slice [originalLoopIvs, 0 .. 0][1 .. 1, paddedShape][1 .. 1].
  b.setInsertionPoint(padTensorOp);
  SmallVector<Value> loopIterationCounts =
      llvm::to_vector<4>(llvm::map_range(packingLoops, [&](Operation *loop) {
        return buildLoopIterationCount(
            b, cast<scf::ForOp>(outermostEnclosingForOp),
            cast<scf::ForOp>(loop));
      }));
  // Assert all loop iteration counts can be computed.
  if (llvm::any_of(loopIterationCounts, [](Value v) { return !v; }))
    llvm_unreachable("loop independence prerequisite not met");
  // offsets = [originalLoopIvs, 0 .. 0].
  offsets.assign(loopIterationCounts.begin(), loopIterationCounts.end());
  offsets.append(paddedRank, b.getIndexAttr(0));
  // sizes = [1 .. 1, paddedShape] (definedabove).
  // strides = [1 .. 1] (defined above)
  packedTensor =
      scf::getForInductionVarOwner(clonedLoopIvs.front())->getResult(0);
  padTensorOp.replaceAllUsesWith(
      b.create<tensor::ExtractSliceOp>(loc, padTensorOp.getResultType(),
                                       packedTensor, offsets, sizes, strides)
          ->getResult(0));

  Operation *toErase = padTensorOp;

  // Make the newly cloned `padTensorOp` available to the caller.
  padTensorOp =
      cast<PadTensorOp>(bvm.lookup(padTensorOp.result()).getDefiningOp());

  toErase->erase();

  return success();
}
