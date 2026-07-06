//===- MemorySlot.cpp - Memory Slot interface implementations for SCF -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/Utils/MemorySlotUtils.h"

using namespace mlir;
using namespace mlir::scf;

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

bool ExecuteRegionOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                         bool hasValueStores) {
  return true;
}

void ExecuteRegionOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  regionsToProcess.insert({&getRegion(), reachingDef});
}

Value ExecuteRegionOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  // Update the yield terminators to return the newly defined reaching
  // definition.
  for (Block &block : getRegion().getBlocks())
    if (isa<YieldOp>(block.getTerminator()))
      memoryslot::updateTerminator(&block, reachingDef, reachingAtBlockEnd);

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  IRRewriter rewriter(builder);
  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, getOperation(), resultTypes);
  return newOp->getResults().back();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

bool ForOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                               bool hasValueStores) {
  return true;
}

void ForOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  Region &bodyRegion = getBodyRegion();
  if (!hasValueStores) {
    regionsToProcess.insert({&bodyRegion, reachingDef});
    return;
  }

  getInitArgsMutable().append(reachingDef);
  bodyRegion.addArgument(slot.elemType, slot.ptr.getLoc());
  regionsToProcess.insert({&bodyRegion, bodyRegion.getArguments().back()});
}

Value ForOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  // Update the yield terminator to return the newly defined reaching
  // definition.
  memoryslot::updateTerminator(getBody(), reachingDef, reachingAtBlockEnd);

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  IRRewriter rewriter(builder);
  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, getOperation(), resultTypes);
  return newOp->getResults().back();
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

bool ForallOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                  bool hasValueStores) {
  // The ForallOp body can be ran in parallel, thus does not support sequenced
  // value passing. Therefore only loads can be handled.
  return !hasValueStores;
}

void ForallOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  assert(!hasValueStores && "ForallOp does not support stores");
  regionsToProcess.insert({&getBodyRegion(), reachingDef});
}

Value ForallOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  assert(!hasValueStores && "ForallOp does not support stores");
  return reachingDef;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

bool IfOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                              bool hasValueStores) {
  return true;
}

void IfOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  regionsToProcess.insert({&getThenRegion(), reachingDef});
  regionsToProcess.insert({&getElseRegion(), reachingDef});
}

Value IfOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  IRRewriter rewriter(builder);

  // Update the yield terminators to return the newly defined reaching
  // definition.
  memoryslot::updateTerminator(&getThenRegion().back(), reachingDef,
                               reachingAtBlockEnd);
  if (getElseRegion().hasOneBlock()) {
    memoryslot::updateTerminator(&getElseRegion().back(), reachingDef,
                                 reachingAtBlockEnd);
  } else {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&getElseRegion());
    YieldOp::create(rewriter, getOperation()->getLoc(), reachingDef);
  }

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, getOperation(), resultTypes);
  return newOp->getResults().back();
}

//===----------------------------------------------------------------------===//
// IndexSwitchOp
//===----------------------------------------------------------------------===//

bool IndexSwitchOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                       bool hasValueStores) {
  return true;
}

void IndexSwitchOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  regionsToProcess.insert({&getDefaultRegion(), reachingDef});
  for (Region &caseRegion : getCaseRegions())
    regionsToProcess.insert({&caseRegion, reachingDef});
}

Value IndexSwitchOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  IRRewriter rewriter(builder);

  // Update the yield terminators to return the newly defined reaching
  // definition.
  memoryslot::updateTerminator(&getDefaultRegion().back(), reachingDef,
                               reachingAtBlockEnd);
  for (Region &caseRegion : getCaseRegions())
    memoryslot::updateTerminator(&caseRegion.back(), reachingDef,
                                 reachingAtBlockEnd);

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, getOperation(), resultTypes);
  return newOp->getResults().back();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

bool ParallelOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                    bool hasValueStores) {
  // The ParallelOp body can be ran in parallel, thus does not support sequenced
  // value passing. Therefore only loads can be handled.
  return !hasValueStores;
}

void ParallelOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  assert(!hasValueStores && "ParallelOp does not support stores");
  regionsToProcess.insert({&getBodyRegion(), reachingDef});
}

Value ParallelOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  assert(!hasValueStores && "ParallelOp does not support stores");
  return reachingDef;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

bool ReduceOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                  bool hasValueStores) {
  // The ReduceOp body can be ran in parallel, thus does not support sequenced
  // value passing. Therefore only loads can be handled.
  return !hasValueStores;
}

void ReduceOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  assert(!hasValueStores && "ReduceOp does not support stores");
  for (Region &reduction : getReductions())
    regionsToProcess.insert({&reduction, reachingDef});
}

Value ReduceOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  assert(!hasValueStores && "ReduceOp does not support stores");
  return reachingDef;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

bool WhileOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                 bool hasValueStores) {
  return true;
}

void WhileOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  Region &beforeRegion = getBefore();
  Region &afterRegion = getAfter();
  if (!hasValueStores) {
    regionsToProcess.insert({&beforeRegion, reachingDef});
    regionsToProcess.insert({&afterRegion, reachingDef});
    return;
  }

  getInitsMutable().append(reachingDef);

  beforeRegion.addArgument(slot.elemType, slot.ptr.getLoc());
  regionsToProcess.insert({&beforeRegion, beforeRegion.getArguments().back()});

  afterRegion.addArgument(slot.elemType, slot.ptr.getLoc());
  regionsToProcess.insert({&afterRegion, afterRegion.getArguments().back()});
}

Value WhileOp::finalizePromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    const llvm::DenseMap<Block *, Value> &reachingAtBlockEnd,
    OpBuilder &builder) {
  if (!hasValueStores)
    return reachingDef;

  // Update the yield terminators to return the newly defined reaching
  // definition.
  memoryslot::updateTerminator(&getBefore().back(),
                               getBefore().getArguments().back(),
                               reachingAtBlockEnd);
  memoryslot::updateTerminator(
      &getAfter().back(), getAfter().getArguments().back(), reachingAtBlockEnd);

  SmallVector<Type> resultTypes(getResultTypes());
  resultTypes.push_back(slot.elemType);

  IRRewriter rewriter(builder);
  Operation *newOp =
      memoryslot::replaceWithNewResults(rewriter, getOperation(), resultTypes);
  return newOp->getResults().back();
}
