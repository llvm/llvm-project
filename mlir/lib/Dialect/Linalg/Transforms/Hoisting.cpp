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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  SubTensorInsertOp subTensorInsertOp;
};
/// Represents a unit of hoistable TransferReadOp. This may comprise other
/// instructions that need to be hoisted too.
struct HoistableRead {
  vector::TransferReadOp transferReadOp;
  SubTensorOp subTensorOp;
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
static bool sameOffsetsSizesAndStrides(SubTensorOp s, SubTensorInsertOp si) {
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
  if (write.subTensorInsertOp)
    LLVM_DEBUG(DBGS() << "findMatchingTransferRead subTensorInsertOp: "
                      << *write.subTensorInsertOp.getOperation() << "\n");

  for (Operation *user : srcTensor.getUsers()) {
    LLVM_DEBUG(DBGS() << "findMatchingTransferRead inspect user: " << *user
                      << "\n");

    // If HoistableWrite involves a SubTensorInsertOp, we need to find a
    // matching SubTensorOp.
    SubTensorOp subTensorOp;
    Operation *maybeTransferReadUser = user;
    if (write.subTensorInsertOp) {
      subTensorOp = dyn_cast<SubTensorOp>(user);
      if (!subTensorOp || subTensorOp.getResult().getType() !=
                              write.subTensorInsertOp.source().getType())
        continue;

      LLVM_DEBUG(DBGS() << "check whether sameOffsetsSizesAndStrides: "
                        << *subTensorOp << " vs " << *write.subTensorInsertOp
                        << "\n");
      if (!sameOffsetsSizesAndStrides(subTensorOp, write.subTensorInsertOp))
        continue;

      LLVM_DEBUG(DBGS() << "sameOffsetsSizesAndStrides: SUCCESS\n");
      // If we got here, subTensorOp is hoistable iff it has exactly 2 uses:
      //   1. the transfer_write we want to hoist.
      //   2. a matching transfer_read.
      // Anything else, we skip.
      bool skip = false;
      Operation *otherUser = nullptr;
      for (Operation *u : subTensorOp->getUsers()) {
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
      return HoistableRead{read, subTensorOp};
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
          user == candidateRead.subTensorOp || user == write.transferWriteOp ||
          user == write.subTensorInsertOp)
        continue;
      // Consider all transitive uses through a subtensor / subtensor_insert.
      // TODO: atm we just bail because a stronger analysis is needed for these
      // cases.
      if (isa<SubTensorOp, SubTensorInsertOp>(user))
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
/// vector.transfer_write + optional subtensor_insert or if any of the indexings
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

  if (auto subTensorInsertOp = v.getDefiningOp<SubTensorInsertOp>()) {
    // Inserted subTensor must come from vector.transfer_write.
    auto write =
        subTensorInsertOp.source().getDefiningOp<vector::TransferWriteOp>();
    if (!write)
      return HoistableWrite();

    // Tensor inserted into must be a BBArg at position matching yieldOperand's.
    auto bbArg = subTensorInsertOp.dest().dyn_cast<BlockArgument>();
    if (!bbArg || bbArg.getOwner()->getParentOp() != forOp ||
        bbArg.getArgNumber() != /*num iv=*/1 + yieldOperand.getOperandNumber())
      return HoistableWrite();

    // Indexing inserted into must not depend on `forOp`.
    for (Value operand : subTensorInsertOp->getOperands().drop_front(
             SubTensorInsertOp::getOffsetSizeAndStrideStartOperandIndex()))
      if (!forOp.isDefinedOutsideOfLoop(operand))
        return HoistableWrite();

    return HoistableWrite{write, subTensorInsertOp};
  }

  return HoistableWrite();
}

/// Mechanical hoisting of a matching HoistableRead / HoistableWrite pair.
static void hoistReadWrite(HoistableRead read, HoistableWrite write,
                           BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());
  assert(read.transferReadOp && write.transferWriteOp &&
         "expected transfer_read and transfer_write ops to be set");
  assert(((read.subTensorOp && write.subTensorInsertOp) ||
          (!read.subTensorOp && !write.subTensorInsertOp)) &&
         "expected matching subtensor / subtensor_insert");
  LLVM_DEBUG(DBGS() << "In forOp:\n"
                    << *forOp.getOperation()
                    << "\nHoist: " << *read.transferReadOp.getOperation()
                    << "\nHoist: " << *write.transferWriteOp.getOperation()
                    << "\nInvolving: " << tensorBBArg << "\n");

  // If a read subtensor is present, hoist it.
  if (read.subTensorOp && failed(forOp.moveOutOfLoop({read.subTensorOp})))
    llvm_unreachable("Unexpected failure moving subtensor out of loop");

  // Hoist the transfer_read op.
  if (failed(forOp.moveOutOfLoop({read.transferReadOp})))
    llvm_unreachable("Unexpected failure moving transfer read out of loop");

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  unsigned initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // Update the source tensor.
  if (read.subTensorOp)
    read.subTensorOp.sourceMutable().assign(forOp.initArgs()[initArgNumber]);
  else
    read.transferReadOp.sourceMutable().assign(forOp.initArgs()[initArgNumber]);

  // Hoist write after.
  if (write.subTensorInsertOp)
    write.subTensorInsertOp->moveAfter(forOp);
  write.transferWriteOp->moveAfter(forOp);

  // Update the yield.
  auto yieldOp = cast<scf::YieldOp>(forOp.region().front().getTerminator());
  if (write.subTensorInsertOp)
    yieldOp->setOperand(initArgNumber, write.subTensorInsertOp.dest());
  else
    yieldOp->setOperand(initArgNumber, write.transferWriteOp.source());

  // Rewrite `loop` with additional new yields.
  OpBuilder b(read.transferReadOp);
  auto newForOp = cloneWithNewYields(b, forOp, read.transferReadOp.vector(),
                                     write.transferWriteOp.vector());
  // Transfer write has been hoisted, need to update the vector and tensor
  // source. Replace the result of the loop to use the new tensor created
  // outside the loop.
  // Depending on whether a subtensor_insert is present or not, it carries the
  // update on the tensor operands.
  if (write.subTensorInsertOp) {
    newForOp.getResult(initArgNumber)
        .replaceAllUsesWith(write.subTensorInsertOp.getResult());
    write.transferWriteOp.sourceMutable().assign(read.subTensorOp.result());
    write.subTensorInsertOp.destMutable().assign(read.subTensorOp.source());
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
        if (write.subTensorInsertOp)
          LLVM_DEBUG(DBGS() << "Candidate subtensor_insert for hoisting: "
                            << *write.subTensorInsertOp.getOperation() << "\n");
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

      llvm::SetVector<Operation *> forwardSlice;
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
  llvm::SetVector<Operation *> backwardSlice;
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

/// Compute the tightest lower bound with quantities that are all defined
/// outside of `outer`.
/// Return null if such a bound cannot be computed.
Value computeLoopIndependentLowerBound(OpBuilder &b, scf::ForOp outer,
                                       Value v) {
  if (isDefinedOutsideOrConstant(outer, v))
    return v;
  return Value();
}

/// Compute the tightest upper bound with quantities that are all defined
/// outside of `outer`.
/// Expects all ops in the backward slice of `v` up to `outer` to be either
/// scf.for, affine.min or affine.apply.
static Value computeLoopIndependentUpperBound(OpBuilder &b, scf::ForOp outer,
                                              Value v) {
  if (isDefinedOutsideOrConstant(outer, v))
    return v;

  LLVM_DEBUG(DBGS() << "Begin loopIndependentUpperBound for: " << v << "\n");

  bool ok =
      backwardsSliceOnlyHasOpsOfType<scf::ForOp, AffineMinOp, AffineApplyOp>(
          outer, v);
  assert(ok && "expected to only be defined by scf::ForOp and AffineMinOp");
  (void)ok;

  // Compute a backward slice up to, but not including, `outer`.
  llvm::SetVector<Operation *> backwardSlice;
  getBackwardSlice(v, &backwardSlice,
                   [&](Operation *op) { return outer->isProperAncestor(op); });
  backwardSlice.insert(v.getDefiningOp());

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(outer);
  Value res = v;
  BlockAndValueMapping bvm;
  for (Operation *op : backwardSlice) {
    if (isa<scf::ForOp>(op))
      continue;
    if (isa<AffineApplyOp>(op)) {
      b.clone(*op, bvm);
      continue;
    }
    auto sliceMinOp = cast<AffineMinOp>(op);
    // Perform the substitution of the operands of AffineMinOp.
    auto mapAndOperands = substituteMin(
        sliceMinOp, [&](Operation *op) { return outer->isAncestor(op); });
    SmallVector<Value> resultOperands = mapAndOperands.dims;
    llvm::append_range(resultOperands, mapAndOperands.symbols);
    AffineMap map = mapAndOperands.map;
    canonicalizeMapAndOperands(&map, &resultOperands);
    map = simplifyAffineMap(map);
    res = b.create<AffineMinOp>(
        outer->getLoc(), map,
        llvm::to_vector<4>(llvm::map_range(resultOperands, [&](Value operand) {
          return bvm.lookupOrDefault(operand);
        })));
    bvm.map(sliceMinOp, res);
  }
  LLVM_DEBUG(DBGS() << "End loopIndependentUpperBound with: " << res << "\n");
  return res;
}

/// Return the number of iterations in the loop (ub - lb).ceilDiv(step).
/// The returned Value is guaranteed not to depend on any loop comprised in
/// [`outer`, `forOp`].
/// Return null if such a loop-independent quantity cannot be computed.
static Value buildLoopTripCount(OpBuilder &b, scf::ForOp outer,
                                scf::ForOp forOp) {
  MLIRContext *ctx = forOp->getContext();
  AffineExpr lb, ub, step;
  bindDims(ctx, lb, ub);
  bindSymbols(ctx, step);
  Value lbVal = computeLoopIndependentLowerBound(b, outer, forOp.lowerBound()),
        ubVal = computeLoopIndependentUpperBound(b, outer, forOp.upperBound()),
        stepVal = forOp.step();
  if (!lbVal || !ubVal || !stepVal)
    return Value();
  auto loc = forOp->getLoc();
  Value res = b.create<AffineApplyOp>(loc, (ub - lb).ceilDiv(step),
                                      ValueRange{lbVal, ubVal, stepVal});
  return res;
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
  Value ivVal = forOp.getInductionVar(),
        lbVal = computeLoopIndependentLowerBound(b, outer, forOp.lowerBound()),
        stepVal = forOp.step();
  if (!ivVal || !lbVal || !stepVal)
    return Value();
  auto loc = forOp->getLoc();
  return b.create<AffineApplyOp>(loc, (iv - lb).ceilDiv(step),
                                 ValueRange{ivVal, lbVal, stepVal});
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
                                   llvm::SetVector<Operation *> &backwardSlice,
                                   llvm::SetVector<Operation *> &packingLoops,
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
  if (llvm::any_of(packingLoops, [&](Operation *op) {
        scf::ForOp forOp = cast<scf::ForOp>(op);
        Value lb = forOp.lowerBound(), ub = forOp.upperBound(),
              step = forOp.step();
        return !isDefinedOutsideOrConstant(outer, lb) ||
               !(isDefinedOutsideOrConstant(outer, ub) ||
                 backwardsSliceOnlyHasOpsOfType<scf::ForOp, AffineMinOp,
                                                AffineApplyOp>(outer, ub)) ||
               !isDefinedOutsideOrConstant(outer, step);
      }))
    return failure();

  // IP just before the outermost loop considered that we hoist above.
  OpBuilder b(outermostEnclosingForOp);
  dynamicTensorSizes =
      llvm::to_vector<4>(llvm::map_range(packingLoops, [&](Operation *op) {
        return buildLoopTripCount(b, cast<scf::ForOp>(outermostEnclosingForOp),
                                  cast<scf::ForOp>(op));
      }));
  // Assert all loop trip counts can be computed.
  if (!llvm::all_of(dynamicTensorSizes, [](Value v) { return v; }))
    llvm_unreachable("loop independence prerequisite not met");
  return success();
}

LogicalResult mlir::linalg::hoistPaddingOnTensors(PadTensorOp &padTensorOp,
                                                  unsigned nLoops) {
  SmallVector<Value> dynamicTensorSizes;
  llvm::SetVector<Operation *> backwardSlice, packingLoops;
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
  //   2. Create a SubTensorInsert at the top of the stack.
  //   3. Iteratively pop and yield the result of the SubTensorInsertOp across
  //     the cloned loops.
  SmallVector<Value> clonedLoopIvs, leadingPackedTensorIndexings;
  clonedLoopIvs.reserve(nLoops);
  leadingPackedTensorIndexings.reserve(nLoops);
  BlockAndValueMapping bvm;
  // Insert `padTensorOp` into the backwardSlice so we clone it too.
  backwardSlice.insert(padTensorOp);
  // Stack step 1. iteratively clone loops and push `packedTensor`.
  for (Operation *op : backwardSlice) {
    // Specifically sit out in the subtenso(packedTensor) case: this is the
    // piece we seek to replace.
    if (auto subTensor = dyn_cast<SubTensorOp>(op))
      if (bvm.lookupOrDefault(subTensor.source()) == packedTensor)
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

  // Stack step 2. create SubTensorInsertOp at the top of the stack.
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
      b.create<SubTensorInsertOp>(loc, bvm.lookup(padTensorOp.result()),
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
      b.create<SubTensorOp>(loc, padTensorOp.getResultType(), packedTensor,
                            offsets, sizes, strides)
          ->getResult(0));

  Operation *toErase = padTensorOp;

  // Make the newly cloned `padTensorOp` available to the caller.
  padTensorOp =
      cast<PadTensorOp>(bvm.lookup(padTensorOp.result()).getDefiningOp());

  toErase->erase();

  return success();
}
