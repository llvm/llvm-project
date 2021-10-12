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
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
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
    if (!attr && ofr.get<Value>().getDefiningOp<arith::ConstantOp>())
      attr = ofr.get<Value>().getDefiningOp<arith::ConstantOp>().value();
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
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interputing the function walk.
    func.walk([&](LoopLikeOpInterface loopLike) {
      if (failed(moveLoopInvariantCode(loopLike)))
        llvm_unreachable(
            "Unexpected failure to move invariant code out of loop");
    });

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
        if (!loop->isAncestor(use.getOwner()))
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
