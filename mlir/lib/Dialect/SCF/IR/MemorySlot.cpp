//===- MemorySlot.cpp - Memory Slot interface implementations for SCF -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::scf;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Creates a shallow copy of an operation with new result types moving the
/// regions out of the original operation, then deletes the original operation.
template <typename OpTy>
static OpTy replaceWithNewResults(OpBuilder &builder, Operation *op,
                                  TypeRange resultTypes) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  builder.
  auto newOp = OpTy::create(builder, op->getLoc(), resultTypes,
                            op->getOperands(), op->getProperties(),
                            op->getSuccessors(), op->getNumRegions());
                            builder.create()
  op.erase();
  return newOp;
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

bool ExecuteRegionOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                         bool hasValueStores) {
  return true;
}

void ExecuteRegionOp::propagateLiveIn(
    const MemorySlot &slot, Region *regionLiveIn,
    SmallPtrSetImpl<Operation *> &operationsLiveIn) {
  assert(regionLiveIn == &getRegion() &&
         "regionLiveIn must be the region of the ExecuteRegionOp");
  operationsLiveIn.insert(getOperation());
}

void ExecuteRegionOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  regionsToProcess.insert({&getRegion(), reachingDef});
}

Value ExecuteRegionOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::DenseMap<Block *, Value> &reachingAtBlockEnd, OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  // Update the yield terminators to return the newly defined reaching
  // definition.
  for (Block &block : getRegion().getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (!isa<YieldOp>(terminator))
      continue;
    Value blockReachingDef = reachingAtBlockEnd[block];
    if (!blockReachingDef) {
      // Block is dead code or the region is not using the slot, so the reaching
      // definition is the entry reaching definition.
      blockReachingDef = reachingDef;
    }
    terminator->insertOperands(terminator->getNumOperands(),
                               {blockReachingDef});
  }

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  auto newOp = replaceWithNewResults<ExecuteRegionOp>(builder, getOperation(),
                                                      resultTypes);

  return reachingDef;
}
