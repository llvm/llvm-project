//===- CopyRemoval.cpp - Removing the redundant copies --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace MemoryEffects;

namespace {

//===----------------------------------------------------------------------===//
// CopyRemovalPass
//===----------------------------------------------------------------------===//

/// This pass removes the redundant Copy operations. Additionally, it
/// removes the leftover definition and deallocation operations by erasing the
/// copy operation.
class CopyRemovalPass : public PassWrapper<CopyRemovalPass, OperationPass<>> {
public:
  void runOnOperation() override {
    getOperation()->walk([&](CopyOpInterface copyOp) {
      reuseCopySourceAsTarget(copyOp);
      reuseCopyTargetAsSource(copyOp);
    });
    for (Operation *op : eraseList)
      op->erase();
  }

private:
  /// List of operations that need to be removed.
  DenseSet<Operation *> eraseList;

  /// Returns the deallocation operation for `value` in `block` if it exists.
  Operation *getDeallocationInBlock(Value value, Block *block) {
    assert(block && "Block cannot be null");
    auto valueUsers = value.getUsers();
    auto it = llvm::find_if(valueUsers, [&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      return effects && op->getBlock() == block && effects.hasEffect<Free>();
    });
    return (it == valueUsers.end() ? nullptr : *it);
  }

  /// Returns true if an operation between start and end operations has memory
  /// effect.
  bool hasMemoryEffectOpBetween(Operation *start, Operation *end) {
    assert((start || end) && "Start and end operations cannot be null");
    assert(start->getBlock() == end->getBlock() &&
           "Start and end operations should be in the same block.");
    Operation *op = start->getNextNode();
    while (op->isBeforeInBlock(end)) {
      if (isa<MemoryEffectOpInterface>(op))
        return true;
      op = op->getNextNode();
    }
    return false;
  };

  /// Returns true if `val` value has at least a user between `start` and
  /// `end` operations.
  bool hasUsersBetween(Value val, Operation *start, Operation *end) {
    assert((start || end) && "Start and end operations cannot be null");
    Block *block = start->getBlock();
    assert(block == end->getBlock() &&
           "Start and end operations should be in the same block.");
    return llvm::any_of(val.getUsers(), [&](Operation *op) {
      return op->getBlock() == block && start->isBeforeInBlock(op) &&
             op->isBeforeInBlock(end);
    });
  };

  bool areOpsInTheSameBlock(ArrayRef<Operation *> operations) {
    assert(!operations.empty() &&
           "The operations list should contain at least a single operation");
    Block *block = operations.front()->getBlock();
    return llvm::none_of(
        operations, [&](Operation *op) { return block != op->getBlock(); });
  }

  /// Input:
  /// func(){
  ///   %from = alloc()
  ///   write_to(%from)
  ///   %to = alloc()
  ///   copy(%from,%to)
  ///   dealloc(%from)
  ///   return %to
  /// }
  ///
  /// Output:
  /// func(){
  ///   %from = alloc()
  ///   write_to(%from)
  ///   return %from
  /// }
  /// Constraints:
  /// 1) %to, copy and dealloc must all be defined and lie in the same block.
  /// 2) This transformation cannot be applied if there is a single user/alias
  /// of `to` value between the defining operation of `to` and the copy
  /// operation.
  /// 3) This transformation cannot be applied if there is a single user/alias
  /// of `from` value between the copy operation and the deallocation of `from`.
  /// TODO: Alias analysis is not available at the moment. Currently, we check
  /// if there are any operations with memory effects between copy and
  /// deallocation operations.
  void reuseCopySourceAsTarget(CopyOpInterface copyOp) {
    if (eraseList.count(copyOp))
      return;

    Value from = copyOp.getSource();
    Value to = copyOp.getTarget();

    Operation *copy = copyOp.getOperation();
    Operation *fromDefiningOp = from.getDefiningOp();
    Operation *fromFreeingOp = getDeallocationInBlock(from, copy->getBlock());
    Operation *toDefiningOp = to.getDefiningOp();
    if (!fromDefiningOp || !fromFreeingOp || !toDefiningOp ||
        !areOpsInTheSameBlock({fromFreeingOp, toDefiningOp, copy}) ||
        hasUsersBetween(to, toDefiningOp, copy) ||
        hasUsersBetween(from, copy, fromFreeingOp) ||
        hasMemoryEffectOpBetween(copy, fromFreeingOp))
      return;

    to.replaceAllUsesWith(from);
    eraseList.insert(copy);
    eraseList.insert(toDefiningOp);
    eraseList.insert(fromFreeingOp);
  }

  /// Input:
  /// func(){
  ///   %to = alloc()
  ///   %from = alloc()
  ///   write_to(%from)
  ///   copy(%from,%to)
  ///   dealloc(%from)
  ///   return %to
  /// }
  ///
  /// Output:
  /// func(){
  ///   %to = alloc()
  ///   write_to(%to)
  ///   return %to
  /// }
  /// Constraints:
  /// 1) %from, copy and dealloc must all be defined and lie in the same block.
  /// 2) This transformation cannot be applied if there is a single user/alias
  /// of `to` value between the defining operation of `from` and the copy
  /// operation.
  /// 3) This transformation cannot be applied if there is a single user/alias
  /// of `from` value between the copy operation and the deallocation of `from`.
  /// TODO: Alias analysis is not available at the moment. Currently, we check
  /// if there are any operations with memory effects between copy and
  /// deallocation operations.
  void reuseCopyTargetAsSource(CopyOpInterface copyOp) {
    if (eraseList.count(copyOp))
      return;

    Value from = copyOp.getSource();
    Value to = copyOp.getTarget();

    Operation *copy = copyOp.getOperation();
    Operation *fromDefiningOp = from.getDefiningOp();
    Operation *fromFreeingOp = getDeallocationInBlock(from, copy->getBlock());
    if (!fromDefiningOp || !fromFreeingOp ||
        !areOpsInTheSameBlock({fromFreeingOp, fromDefiningOp, copy}) ||
        hasUsersBetween(to, fromDefiningOp, copy) ||
        hasUsersBetween(from, copy, fromFreeingOp) ||
        hasMemoryEffectOpBetween(copy, fromFreeingOp))
      return;

    from.replaceAllUsesWith(to);
    eraseList.insert(copy);
    eraseList.insert(fromDefiningOp);
    eraseList.insert(fromFreeingOp);
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// CopyRemovalPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createCopyRemovalPass() {
  return std::make_unique<CopyRemovalPass>();
}
