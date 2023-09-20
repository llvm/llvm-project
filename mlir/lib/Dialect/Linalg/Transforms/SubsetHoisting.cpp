//===- SubsetHoisting.cpp - Linalg hoisting transformations----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with hoisting invariant subset
// operations in the context of Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "subset-hoisting"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

/// Return true if the location of the subset defined by the op is invariant of
/// the loop iteration.
static bool
isSubsetLocationLoopInvariant(scf::ForOp forOp,
                              vector::TransferWriteOp transferWriteOp) {
  for (Value operand : transferWriteOp.getIndices())
    if (!forOp.isDefinedOutsideOfLoop(operand))
      return false;
  return true;
}

/// Return true if the location of the subset defined by the op is invariant of
/// the loop iteration.
static bool isSubsetLocationLoopInvariant(scf::ForOp forOp,
                                          tensor::InsertSliceOp insertSliceOp) {
  for (Value operand : insertSliceOp->getOperands().drop_front(
           tensor::InsertSliceOp::getOffsetSizeAndStrideStartOperandIndex()))
    if (!forOp.isDefinedOutsideOfLoop(operand))
      return false;
  return true;
}

/// Given an `srcTensor` that is a block argument belong to a loop.
/// Greedily look for the first read that can be hoisted out of the loop (i.e.
/// that satisfied the conditions):
///   - The read is of type `tensor.extract_slice`.
///   - The read is one of the uses of `srcTensor`.
///   - The read is to the same subset that `tensor.insert_slice` writes.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static FailureOr<tensor::ExtractSliceOp>
findHoistableMatchingExtractSlice(RewriterBase &rewriter,
                                  tensor::InsertSliceOp insertSliceOp,
                                  BlockArgument srcTensor) {
  assert(isa<RankedTensorType>(srcTensor.getType()) && "not a ranked tensor");

  auto forOp = cast<scf::ForOp>(srcTensor.getOwner()->getParentOp());

  LLVM_DEBUG(DBGS() << "--find matching read for: " << insertSliceOp << "\n";
             DBGS() << "--amongst users of: " << srcTensor << "\n");

  SmallVector<Operation *> users(srcTensor.getUsers());
  if (forOp.isDefinedOutsideOfLoop(insertSliceOp.getDest()))
    llvm::append_range(users, insertSliceOp.getDest().getUsers());

  for (Operation *user : users) {
    LLVM_DEBUG(DBGS() << "----inspect user: " << *user << "\n");
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    // Skip ops other than extract_slice with an exact matching of their tensor
    // subset.
    if (extractSliceOp) {
      auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
      if (extractSliceOp.getResultType() != insertSliceOp.getSourceType() ||
          !extractSliceOp.isSameAs(insertSliceOp, isSame)) {
        LLVM_DEBUG(DBGS() << "------not a matching extract_slice\n";
                   DBGS() << *user << " vs " << *insertSliceOp << "\n");
        continue;
      }

      // Skip insert_slice whose vector is defined within the loop: we need to
      // hoist that definition first otherwise dominance violations trigger.
      if (!isa<BlockArgument>(extractSliceOp.getSource()) &&
          !forOp.isDefinedOutsideOfLoop(extractSliceOp.getSource())) {
        LLVM_DEBUG(DBGS() << "------transfer_read vector is loop-dependent\n");
        continue;
      }
      return extractSliceOp;
    }

    // TODO: Look through disjoint subsets, similar to vector.transfer_write
    // and unify implementations.
  }

  LLVM_DEBUG(DBGS() << "----no matching extract_slice");
  return failure();
}

/// Given an `srcTensor` that is a block argument belong to a loop.
/// Greedily look for the first read that can be hoisted out of the loop (i.e.
/// that satisfied the conditions):
///   - The read is of type `tensor.transfer_read`.
///   - The read is one of the uses of `srcTensor`.
///   - The read is to the same subset that `tensor.transfer_write` writes.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static FailureOr<vector::TransferReadOp>
findHoistableMatchingTransferRead(RewriterBase &rewriter,
                                  vector::TransferWriteOp transferWriteOp,
                                  BlockArgument srcTensor) {
  if (!isa<RankedTensorType>(srcTensor.getType()))
    return failure();

  auto forOp = cast<scf::ForOp>(srcTensor.getOwner()->getParentOp());

  LLVM_DEBUG(DBGS() << "--find matching read for: " << transferWriteOp << "\n";
             DBGS() << "--amongst users of: " << srcTensor << "\n";);

  // vector.transfer_write is a bit peculiar: we look through dependencies
  // to disjoint tensor subsets. This requires a while loop.
  // TODO: Look through disjoint subsets for tensor.insert_slice and unify
  // implementations.
  SmallVector<Operation *> users(srcTensor.getUsers());
  // TODO: transferWriteOp.getSource is actually the destination tensor!!
  if (forOp.isDefinedOutsideOfLoop(transferWriteOp.getSource()))
    llvm::append_range(users, transferWriteOp.getSource().getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    LLVM_DEBUG(DBGS() << "----inspect user: " << *user << "\n");
    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (read) {
      // Skip ops other than transfer_read with an exact matching subset.
      if (read.getIndices() != transferWriteOp.getIndices() ||
          read.getVectorType() != transferWriteOp.getVectorType()) {
        LLVM_DEBUG(DBGS() << "------not a transfer_read that matches the "
                             "transfer_write: "
                          << *user << "\n\t(vs " << *transferWriteOp << ")\n");
        continue;
      }

      // transfer_read may be of a vector that is defined within the loop: we
      // traverse it by virtue of bypassing disjoint subset operations rooted at
      // a bbArg and yielding a matching yield.
      if (!isa<BlockArgument>(read.getSource()) &&
          !forOp.isDefinedOutsideOfLoop(read.getSource())) {
        LLVM_DEBUG(DBGS() << "------transfer_read vector appears loop "
                             "dependent but will be tested for disjointness as "
                             "part of the bypass analysis\n");
      }
      LLVM_DEBUG(DBGS() << "------found match\n");
      return read;
    }

    // As an optimization, we look further through dependencies to disjoint
    // tensor subsets. This creates more opportunities to find a matching read.
    if (isa<vector::TransferWriteOp>(user)) {
      // If we find a write with disjoint indices append all its uses.
      // TODO: Generalize areSubsetsDisjoint and allow other bypass than
      // just vector.transfer_write - vector.transfer_write.
      if (vector::isDisjointTransferIndices(
              cast<VectorTransferOpInterface>(user),
              cast<VectorTransferOpInterface>(
                  transferWriteOp.getOperation()))) {
        LLVM_DEBUG(DBGS() << "----follow through disjoint write\n");
        users.append(user->getUsers().begin(), user->getUsers().end());
      } else {
        LLVM_DEBUG(DBGS() << "----skip non-disjoint write\n");
      }
    }
  }

  LLVM_DEBUG(DBGS() << "--no matching transfer_read\n");
  return rewriter.notifyMatchFailure(transferWriteOp,
                                     "no matching transfer_read");
}

/// Return the `vector.transfer_write` that produces `yieldOperand`, if:
///   - The write operates on tensors.
///   - All indices are defined outside of the loop.
/// Return failure otherwise.
///
/// This is sufficient condition to hoist the `vector.transfer_write`; other
/// operands can always be yielded by the loop where needed.
// TODO: generalize beyond scf::ForOp.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static FailureOr<vector::TransferWriteOp>
getLoopInvariantTransferWriteDefining(RewriterBase &rewriter, scf::ForOp forOp,
                                      BlockArgument bbArg,
                                      OpOperand &yieldOperand) {
  assert(bbArg.getArgNumber() ==
             forOp.getNumInductionVars() + yieldOperand.getOperandNumber() &&
         "bbArg and yieldOperand must match");
  assert(isa<scf::YieldOp>(yieldOperand.getOwner()) && "must be an scf.yield");

  Value v = yieldOperand.get();
  auto transferWriteOp = v.getDefiningOp<vector::TransferWriteOp>();
  if (!transferWriteOp)
    return rewriter.notifyMatchFailure(v.getLoc(), "not a transfer_write");

  if (transferWriteOp->getNumResults() == 0) {
    return rewriter.notifyMatchFailure(v.getLoc(),
                                       "unsupported transfer_write on buffers");
  }

  // We do not explicitly check that the destination is a BBarg that matches the
  // yield operand as this would prevent us from bypassing other non-conflicting
  // writes.

  // Indexing must not depend on `forOp`.
  if (!isSubsetLocationLoopInvariant(forOp, transferWriteOp))
    return rewriter.notifyMatchFailure(
        v.getLoc(), "transfer_write indexing is loop-dependent");

  return transferWriteOp;
}

/// Return the `tensor.insert_slice` that produces `yieldOperand`, if:
///   1. Its destination tensor is a block argument of the `forOp`.
///   2. The unique use of its result is a yield with operand number matching
///   the block argument.
///   3. All indices are defined outside of the loop.
/// Return failure otherwise.
///
/// This is sufficient condition to hoist the `tensor.insert_slice`; other
/// operands can always be yielded by the loop where needed.
/// Note: 1. + 2. ensure that the yield / iter_args cycle results in proper
/// semantics (i.e. no ping-ping between iter_args across iterations).
// TODO: generalize beyond scf::ForOp.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static FailureOr<tensor::InsertSliceOp>
getLoopInvariantInsertSliceDefining(RewriterBase &rewriter, scf::ForOp forOp,
                                    BlockArgument bbArg,
                                    OpOperand &yieldOperand) {
  assert(bbArg.getArgNumber() ==
             forOp.getNumInductionVars() + yieldOperand.getOperandNumber() &&
         "bbArg and yieldOperand must match");
  assert(isa<scf::YieldOp>(yieldOperand.getOwner()) && "must be an scf.yield");

  Value v = yieldOperand.get();
  auto insertSliceOp = v.getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSliceOp)
    return rewriter.notifyMatchFailure(v.getLoc(), "not an insert_slice");

  // Tensor inserted into must be a BBArg at position matching yield operand.
  // TODO: In the future we should not perform this check if we want to bypass
  // other non-conflicting writes.
  if (bbArg != insertSliceOp.getDest())
    return rewriter.notifyMatchFailure(v.getLoc(), "not a matching bbarg");

  // Indexing inserted into must not depend on `forOp`.
  if (!isSubsetLocationLoopInvariant(forOp, insertSliceOp))
    return rewriter.notifyMatchFailure(
        v.getLoc(), "insert_slice indexing is loop-dependent");

  return insertSliceOp;
}

/// Check if the chunk of data inserted by the `writeOp` is read by any other
/// op than the candidateReadOp. This conflicting operation prevents hoisting,
/// return it or nullptr if none is found.
// TODO: Generalize subset disjunction analysis/interface.
// TODO: Support more subset op types.
static Operation *isTensorChunkAccessedByUnknownOp(Operation *writeOp,
                                                   Operation *candidateReadOp,
                                                   BlockArgument tensorArg) {
  // Make sure none of the other uses read the part of the tensor modified
  // by the transfer_write.
  llvm::SmallVector<Value::use_range, 1> uses;
  uses.push_back(tensorArg.getUses());
  while (!uses.empty()) {
    for (OpOperand &use : uses.pop_back_val()) {
      Operation *user = use.getOwner();
      // Skip the candidate use, only inspect the "other" uses.
      if (user == candidateReadOp || user == writeOp)
        continue;

      // TODO: Consider all transitive uses through
      // extract_slice/insert_slice. Atm we just bail because a stronger
      // analysis is needed for these cases.
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user))
        return user;

      // Consider all transitive uses through a vector.transfer_write.
      if (isa<vector::TransferWriteOp>(writeOp)) {
        if (auto writeUser = dyn_cast<vector::TransferWriteOp>(user)) {
          uses.push_back(writeUser->getResult(0).getUses());
          continue;
        }
      }

      // Consider all nested uses through an scf::ForOp. We may have
      // pass-through tensor arguments left from previous level of
      // hoisting.
      if (auto forUser = dyn_cast<scf::ForOp>(user)) {
        Value arg = forUser.getBody()->getArgument(
            use.getOperandNumber() - forUser.getNumControlOperands() +
            /*iv value*/ 1);
        uses.push_back(arg.getUses());
        continue;
      }

      // Follow the use yield, only if it doesn't escape the original region.
      scf::YieldOp yieldUser = dyn_cast<scf::YieldOp>(user);
      if (yieldUser &&
          writeOp->getParentOp()->isAncestor(yieldUser->getParentOp())) {
        Value ret = yieldUser->getParentOp()->getResult(use.getOperandNumber());
        uses.push_back(ret.getUses());
        continue;
      }

      // If the write is a vector::TransferWriteOp, it may have been bypassed
      // and we need to check subset disjunction
      if (isa<vector::TransferWriteOp>(writeOp)) {
        auto read = dyn_cast<vector::TransferReadOp>(user);
        if (!read || !vector::isDisjointTransferIndices(
                         cast<VectorTransferOpInterface>(read.getOperation()),
                         cast<VectorTransferOpInterface>(writeOp))) {
          return user;
        }
      }
    }
  }
  return nullptr;
}

/// Mechanical hoisting of a matching read / write pair.
/// Return the newly created scf::ForOp with an extra yields.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static scf::ForOp hoistTransferReadWrite(
    RewriterBase &rewriter, vector::TransferReadOp transferReadOp,
    vector::TransferWriteOp transferWriteOp, BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());
  LLVM_DEBUG(DBGS() << "--Start hoisting\n";
             DBGS() << "--Hoist read : " << transferReadOp << "\n";
             DBGS() << "--Hoist write: " << transferWriteOp << "\n";
             DBGS() << "--Involving  : " << tensorBBArg << "\n");

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  int64_t initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // 1. Hoist the read op. Thanks to our previous checks we know this will not
  // trigger dominance violations once BBArgs are updated.
  // TODO: should the rewriter ever want to track this move ?
  transferReadOp->moveBefore(forOp);
  if (!forOp.isDefinedOutsideOfLoop(transferReadOp.getSource())) {
    rewriter.startRootUpdate(transferReadOp);
    transferReadOp.getSourceMutable().assign(
        forOp.getInitArgs()[initArgNumber]);
    rewriter.finalizeRootUpdate(transferReadOp);
  }

  // 2. Rewrite `loop` with an additional yield. This is the quantity that is
  // computed iteratively but whose storage has become loop-invariant.
  NewYieldValueFn yieldFn = [&](OpBuilder &b, Location loc,
                                ArrayRef<BlockArgument> newBBArgs) {
    return SmallVector<Value>{transferWriteOp.getVector()};
  };
  auto newForOp = replaceLoopWithNewYields(
      rewriter, forOp, {transferReadOp.getVector()}, yieldFn);
  rewriter.eraseOp(forOp);

  // 3. Update the yield. Invariant: initArgNumber is the destination tensor.
  auto yieldOp =
      cast<scf::YieldOp>(newForOp.getRegion().front().getTerminator());
  // TODO: transferWriteOp.getSource is actually the destination tensor!!
  rewriter.startRootUpdate(yieldOp);
  yieldOp->setOperand(initArgNumber, transferWriteOp.getSource());
  rewriter.finalizeRootUpdate(yieldOp);

  // 4. Hoist write after and make uses of newForOp.getResult(initArgNumber)
  // flow through it.
  // TODO: should the rewriter ever want to track this move ?
  transferWriteOp->moveAfter(newForOp);
  rewriter.startRootUpdate(transferWriteOp);
  transferWriteOp.getVectorMutable().assign(newForOp.getResults().back());
  // TODO: transferWriteOp.getSource is actually the destination tensor!!
  transferWriteOp.getSourceMutable().assign(newForOp.getResult(initArgNumber));
  rewriter.finalizeRootUpdate(transferWriteOp);
  rewriter.replaceAllUsesExcept(newForOp.getResult(initArgNumber),
                                transferWriteOp.getResult(), transferWriteOp);
  return newForOp;
}

/// Mechanical hoisting of a matching read / write pair.
/// Return the newly created scf::ForOp with an extra yields.
// TODO: Unify implementations once the "bypassing behavior" is the same.
static scf::ForOp hoistExtractInsertSlice(RewriterBase &rewriter,
                                          tensor::ExtractSliceOp extractSliceOp,
                                          tensor::InsertSliceOp insertSliceOp,
                                          BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());
  LLVM_DEBUG(DBGS() << "--Start hoisting\n";
             DBGS() << "--Hoist read : " << extractSliceOp << "\n";
             DBGS() << "--Hoist write: " << insertSliceOp << "\n";
             DBGS() << "--Involving  : " << tensorBBArg << "\n");

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  int64_t initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // 1. Hoist the read op. Thanks to our previous checks we know this will not
  // trigger dominance violations once BBArgs are updated.
  // TODO: should the rewriter ever want to track this move ?
  extractSliceOp->moveBefore(forOp);
  if (!forOp.isDefinedOutsideOfLoop(extractSliceOp.getSource())) {
    assert(extractSliceOp.getSource() == tensorBBArg &&
           "extractSlice source not defined above must be the tracked bbArg");
    rewriter.startRootUpdate(extractSliceOp);
    extractSliceOp.getSourceMutable().assign(
        forOp.getInitArgs()[initArgNumber]);
    rewriter.finalizeRootUpdate(extractSliceOp);
  }

  // 2. Rewrite `loop` with an additional yield. This is the quantity that is
  // computed iteratively but whose storage has become loop-invariant.
  NewYieldValueFn yieldFn = [&](OpBuilder &b, Location loc,
                                ArrayRef<BlockArgument> newBBArgs) {
    return SmallVector<Value>{insertSliceOp.getSource()};
  };
  auto newForOp = replaceLoopWithNewYields(rewriter, forOp,
                                           extractSliceOp.getResult(), yieldFn);
  rewriter.eraseOp(forOp);

  // 3. Update the yield. Invariant: initArgNumber is the destination tensor.
  auto yieldOp =
      cast<scf::YieldOp>(newForOp.getRegion().front().getTerminator());
  // TODO: should the rewriter ever want to track this ?
  rewriter.startRootUpdate(yieldOp);
  yieldOp->setOperand(initArgNumber, insertSliceOp.getDest());
  rewriter.finalizeRootUpdate(yieldOp);

  // 4. Hoist write after and make uses of newForOp.getResult(initArgNumber)
  // flow through it.
  // TODO: should the rewriter ever want to track this move ?
  insertSliceOp->moveAfter(newForOp);
  rewriter.startRootUpdate(insertSliceOp);
  insertSliceOp.getSourceMutable().assign(newForOp.getResults().back());
  insertSliceOp.getDestMutable().assign(newForOp.getResult(initArgNumber));
  rewriter.finalizeRootUpdate(insertSliceOp);
  rewriter.replaceAllUsesExcept(newForOp.getResult(initArgNumber),
                                insertSliceOp.getResult(), insertSliceOp);
  return newForOp;
}

/// Greedily hoist redundant subset extract/insert operations on tensors
/// outside `forOp`.
/// Return the unmodified `forOp` if no hoisting occurred.
/// Return a new scf::ForOp if hoisting on tensors occurred.
scf::ForOp
mlir::linalg::hoistRedundantSubsetExtractInsert(RewriterBase &rewriter,
                                                scf::ForOp forOp) {
  LLVM_DEBUG(DBGS() << "Enter hoistRedundantSubsetExtractInsert scf.for\n");
  Operation *yield = forOp.getBody()->getTerminator();

  LLVM_DEBUG(DBGS() << "\n"; DBGS() << "Consider " << forOp << "\n");

  scf::ForOp newForOp = forOp;
  do {
    forOp = newForOp;
    for (const auto &it : llvm::enumerate(forOp.getRegionIterArgs())) {
      LLVM_DEBUG(DBGS() << "Consider " << it.value() << "\n");

      // 1. Find a loop invariant subset write yielding `ret` that we can
      // consider for hoisting.
      // TODO: TypeSwitch when we add more cases.
      OpOperand &ret = yield->getOpOperand(it.index());
      FailureOr<vector::TransferWriteOp> transferWriteOp =
          getLoopInvariantTransferWriteDefining(rewriter, forOp, it.value(),
                                                ret);
      FailureOr<tensor::InsertSliceOp> insertSliceOp =
          getLoopInvariantInsertSliceDefining(rewriter, forOp, it.value(), ret);
      if (failed(transferWriteOp) && failed(insertSliceOp)) {
        LLVM_DEBUG(DBGS() << "no loop invariant write defining iter_args "
                          << it.value() << "\n");
        continue;
      }

      Operation *writeOp = succeeded(transferWriteOp)
                               ? transferWriteOp->getOperation()
                               : insertSliceOp->getOperation();

      // 2. Only accept writes with a single use (i.e. the yield).
      if (!writeOp->hasOneUse()) {
        LLVM_DEBUG(DBGS() << "write with more than 1 use " << *writeOp << "\n");
        continue;
      }

      LLVM_DEBUG(DBGS() << "Write to hoist: " << *writeOp << "\n");

      // 3. Find a matching read that can also be hoisted.
      Operation *matchingReadOp = nullptr;
      // TODO: TypeSwitch.
      if (succeeded(transferWriteOp)) {
        auto maybeTransferRead = findHoistableMatchingTransferRead(
            rewriter, *transferWriteOp, it.value());
        if (succeeded(maybeTransferRead))
          matchingReadOp = maybeTransferRead->getOperation();
      } else if (succeeded(insertSliceOp)) {
        auto maybeExtractSlice = findHoistableMatchingExtractSlice(
            rewriter, *insertSliceOp, it.value());
        if (succeeded(maybeExtractSlice))
          matchingReadOp = maybeExtractSlice->getOperation();
      } else {
        llvm_unreachable("unexpected case");
      }
      if (!matchingReadOp) {
        LLVM_DEBUG(DBGS() << "No matching read\n");
        continue;
      }

      // 4. Make sure no other use reads the part of the modified tensor.
      // This is necessary to guard against hazards when non-conflicting subset
      // ops are bypassed.
      Operation *maybeUnknownOp =
          isTensorChunkAccessedByUnknownOp(writeOp, matchingReadOp, it.value());
      if (maybeUnknownOp) {
        LLVM_DEBUG(DBGS() << "Tensor chunk accessed by unknown op, skip: "
                          << *maybeUnknownOp << "\n");
        continue;
      }

      // 5. Perform the actual mechanical hoisting.
      // TODO: TypeSwitch.
      LLVM_DEBUG(DBGS() << "Read to hoist: " << *matchingReadOp << "\n");
      if (succeeded(transferWriteOp)) {
        newForOp = hoistTransferReadWrite(
            rewriter, cast<vector::TransferReadOp>(matchingReadOp),
            *transferWriteOp, it.value());
      } else if (succeeded(insertSliceOp)) {
        newForOp = hoistExtractInsertSlice(
            rewriter, cast<tensor::ExtractSliceOp>(matchingReadOp),
            *insertSliceOp, it.value());
      } else {
        llvm_unreachable("unexpected case");
      }
      break;
    }
  } while (forOp != newForOp);

  return newForOp;
}
