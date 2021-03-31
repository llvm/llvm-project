//===- VectorTransferOpTransforms.cpp - transfer op transforms ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with optimizing transfer_read and
// transfer_write ops.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-transfer-opt"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

/// Return the ancestor op in the region or nullptr if the region is not
/// an ancestor of the op.
static Operation *findAncestorOpInRegion(Region *region, Operation *op) {
  for (; op != nullptr && op->getParentRegion() != region;
       op = op->getParentOp())
    ;
  return op;
}

/// Return true if the transfer_write fully writes the data accessed by the
/// transfer_read.
static bool transferEncompasses(vector::TransferWriteOp defWrite,
                                vector::TransferReadOp read) {
  return !defWrite.hasOutOfBoundsDim() &&
         defWrite.indices() == read.indices() &&
         defWrite.getVectorType() == read.getVectorType() &&
         defWrite.permutation_map() == read.permutation_map();
}

/// Return true if the write op fully over-write the priorWrite transfer_write
/// op.
static bool transferEncompasses(vector::TransferWriteOp write,
                                vector::TransferWriteOp priorWrite) {
  return priorWrite.indices() == write.indices() &&
         priorWrite.getVectorType() == write.getVectorType() &&
         priorWrite.permutation_map() == write.permutation_map();
}

namespace {

class TransferOptimization {
public:
  TransferOptimization(FuncOp func) : dominators(func), postDominators(func) {}
  void deadStoreOp(vector::TransferWriteOp);
  void deadStoreOpTensor(vector::TransferWriteOp);
  void storeToLoadForwarding(vector::TransferReadOp);
  void storeToLoadForwardingTensor(vector::TransferReadOp);
  void removeDeadOp() {
    for (Operation *op : opToErase)
      op->erase();
    opToErase.clear();
  }

private:
  bool isReachable(Operation *start, Operation *dest);
  DominanceInfo dominators;
  PostDominanceInfo postDominators;
  std::vector<Operation *> opToErase;
};

/// Return true if there is a path from start operation to dest operation,
/// otherwise return false. The operations have to be in the same region.
bool TransferOptimization::isReachable(Operation *start, Operation *dest) {
  assert(start->getParentRegion() == dest->getParentRegion() &&
         "This function only works for ops i the same region");
  // Simple case where the start op dominate the destination.
  if (dominators.dominates(start, dest))
    return true;
  Block *startBlock = start->getBlock();
  Block *destBlock = dest->getBlock();
  SmallVector<Block *, 32> worklist(startBlock->succ_begin(),
                                    startBlock->succ_end());
  SmallPtrSet<Block *, 32> visited;
  while (!worklist.empty()) {
    Block *bb = worklist.pop_back_val();
    if (!visited.insert(bb).second)
      continue;
    if (dominators.dominates(bb, destBlock))
      return true;
    worklist.append(bb->succ_begin(), bb->succ_end());
  }
  return false;
}

/// For transfer_write to overwrite fully another transfer_write must:
/// 1. Access the same memref with the same indices and vector type.
/// 2. Post-dominate the other transfer_write operation.
/// If several candidates are available, one must be post-dominated by all the
/// others since they are all post-dominating the same transfer_write. We only
/// consider the transfer_write post-dominated by all the other candidates as
/// this will be the first transfer_write executed after the potentially dead
/// transfer_write.
/// If we found such an overwriting transfer_write we know that the original
/// transfer_write is dead if all reads that can be reached from the potentially
/// dead transfer_write are dominated by the overwriting transfer_write.
void TransferOptimization::deadStoreOp(vector::TransferWriteOp write) {
  LLVM_DEBUG(DBGS() << "Candidate for dead store: " << *write.getOperation()
                    << "\n");
  llvm::SmallVector<Operation *, 8> reads;
  Operation *firstOverwriteCandidate = nullptr;
  for (auto *user : write.source().getUsers()) {
    if (user == write.getOperation())
      continue;
    if (auto nextWrite = dyn_cast<vector::TransferWriteOp>(user)) {
      // Check candidate that can override the store.
      if (transferEncompasses(nextWrite, write) &&
          postDominators.postDominates(nextWrite, write)) {
        if (firstOverwriteCandidate == nullptr ||
            postDominators.postDominates(firstOverwriteCandidate, nextWrite))
          firstOverwriteCandidate = nextWrite;
        else
          assert(
              postDominators.postDominates(nextWrite, firstOverwriteCandidate));
      }
    } else {
      if (auto read = dyn_cast<vector::TransferReadOp>(user)) {
        // Don't need to consider disjoint reads.
        if (isDisjointTransferSet(
                cast<VectorTransferOpInterface>(write.getOperation()),
                cast<VectorTransferOpInterface>(read.getOperation())))
          continue;
      }
      reads.push_back(user);
    }
  }
  if (firstOverwriteCandidate == nullptr)
    return;
  Region *topRegion = firstOverwriteCandidate->getParentRegion();
  Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
  assert(writeAncestor &&
         "write op should be recursively part of the top region");

  for (Operation *read : reads) {
    Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
    // TODO: if the read and write have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (readAncestor == nullptr || !isReachable(writeAncestor, readAncestor))
      continue;
    if (!dominators.dominates(firstOverwriteCandidate, read)) {
      LLVM_DEBUG(DBGS() << "Store may not be dead due to op: " << *read
                        << "\n");
      return;
    }
  }
  LLVM_DEBUG(DBGS() << "Found dead store: " << *write.getOperation()
                    << " overwritten by: " << *firstOverwriteCandidate << "\n");
  opToErase.push_back(write.getOperation());
}

/// A transfer_write candidate to storeToLoad forwarding must:
/// 1. Access the same memref with the same indices and vector type as the
/// transfer_read.
/// 2. Dominate the transfer_read operation.
/// If several candidates are available, one must be dominated by all the others
/// since they are all dominating the same transfer_read. We only consider the
/// transfer_write dominated by all the other candidates as this will be the
/// last transfer_write executed before the transfer_read.
/// If we found such a candidate we can do the forwarding if all the other
/// potentially aliasing ops that may reach the transfer_read are post-dominated
/// by the transfer_write.
void TransferOptimization::storeToLoadForwarding(vector::TransferReadOp read) {
  if (read.hasOutOfBoundsDim())
    return;
  LLVM_DEBUG(DBGS() << "Candidate for Forwarding: " << *read.getOperation()
                    << "\n");
  SmallVector<Operation *, 8> blockingWrites;
  vector::TransferWriteOp lastwrite = nullptr;
  for (Operation *user : read.source().getUsers()) {
    if (isa<vector::TransferReadOp>(user))
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      // If there is a write, but we can prove that it is disjoint we can ignore
      // the write.
      if (isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(read.getOperation())))
        continue;
      if (dominators.dominates(write, read) &&
          transferEncompasses(write, read)) {
        if (lastwrite == nullptr || dominators.dominates(lastwrite, write))
          lastwrite = write;
        else
          assert(dominators.dominates(write, lastwrite));
        continue;
      }
    }
    blockingWrites.push_back(user);
  }

  if (lastwrite == nullptr)
    return;

  Region *topRegion = lastwrite->getParentRegion();
  Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
  assert(readAncestor &&
         "read op should be recursively part of the top region");

  for (Operation *write : blockingWrites) {
    Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
    // TODO: if the store and read have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (writeAncestor == nullptr || !isReachable(writeAncestor, readAncestor))
      continue;
    if (!postDominators.postDominates(lastwrite, write)) {
      LLVM_DEBUG(DBGS() << "Fail to do write to read forwarding due to op: "
                        << *write << "\n");
      return;
    }
  }

  LLVM_DEBUG(DBGS() << "Forward value from " << *lastwrite.getOperation()
                    << " to: " << *read.getOperation() << "\n");
  read.replaceAllUsesWith(lastwrite.vector());
  opToErase.push_back(read.getOperation());
}

/// Walk up the SSA links, if any write gets fully overwritten we can skip it.
/// If it has no more uses it becomes dead.
void TransferOptimization::deadStoreOpTensor(vector::TransferWriteOp write) {
  auto defWrite = write.source().getDefiningOp<vector::TransferWriteOp>();
  while (defWrite) {
    if (transferEncompasses(write, defWrite)) {
      write.sourceMutable().assign(defWrite.source());
      if (defWrite->use_empty())
        opToErase.push_back(defWrite.getOperation());
      return;
    }
    if (!isDisjointTransferIndices(
            cast<VectorTransferOpInterface>(defWrite.getOperation()),
            cast<VectorTransferOpInterface>(write.getOperation())))
      break;
    defWrite = defWrite.source().getDefiningOp<vector::TransferWriteOp>();
  }
}

/// Walk up the SSA links, if any write fully match the written vector we can
/// replace the read by the vector. The read becomes dead and can be removed.
void TransferOptimization::storeToLoadForwardingTensor(
    vector::TransferReadOp read) {
  auto defWrite = read.source().getDefiningOp<vector::TransferWriteOp>();
  while (defWrite) {
    if (transferEncompasses(defWrite, read)) {
      read.replaceAllUsesWith(defWrite.vector());
      opToErase.push_back(read.getOperation());
      return;
    }
    if (!isDisjointTransferIndices(
            cast<VectorTransferOpInterface>(defWrite.getOperation()),
            cast<VectorTransferOpInterface>(read.getOperation())))
      break;
    defWrite = defWrite.source().getDefiningOp<vector::TransferWriteOp>();
  }
}

} // namespace

void mlir::vector::transferOpflowOpt(FuncOp func) {
  TransferOptimization opt(func);
  // Run store to load forwarding first since it can expose more dead store
  // opportunity.
  func.walk([&](vector::TransferReadOp read) {
    if (read.getShapedType().isa<MemRefType>())
      opt.storeToLoadForwarding(read);
    else
      opt.storeToLoadForwardingTensor(read);
  });
  opt.removeDeadOp();
  func.walk([&](vector::TransferWriteOp write) {
    if (write.getShapedType().isa<MemRefType>())
      opt.deadStoreOp(write);
    else
      opt.deadStoreOpTensor(write);
  });
  opt.removeDeadOp();
}
