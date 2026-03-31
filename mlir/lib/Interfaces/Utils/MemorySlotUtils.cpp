//===- MemorySlotUtils.cpp - Utilities for MemorySlot interfaces ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities for implementing MemorySlot interfaces,
// in particular PromotableRegionOpInterface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/Utils/MemorySlotUtils.h"

using namespace mlir;

void mlir::memoryslot::updateTerminator(
    Block *block, Value defaultReachingDef,
    const DenseMap<Block *, Value> &reachingAtBlockEnd) {
  Value blockReachingDef = reachingAtBlockEnd.lookup(block);
  if (!blockReachingDef)
    blockReachingDef = defaultReachingDef;
  Operation *terminator = block->getTerminator();
  terminator->insertOperands(terminator->getNumOperands(), {blockReachingDef});
}

Operation *mlir::memoryslot::replaceWithNewResults(RewriterBase &rewriter,
                                                   Operation *op,
                                                   TypeRange resultTypes) {
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       resultTypes, op->getAttrs());
  state.propertiesAttr = op->getPropertiesAsAttribute();
  unsigned numRegions = op->getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i)
    state.addRegion();
  Operation *newOp = rewriter.create(state);
  rewriter.startOpModification(newOp);
  rewriter.startOpModification(op);
  for (unsigned i = 0; i < numRegions; ++i)
    newOp->getRegion(i).takeBody(op->getRegion(i));
  rewriter.finalizeOpModification(op);
  rewriter.finalizeOpModification(newOp);

  rewriter.replaceAllOpUsesWith(
      op, newOp->getResults().take_front(op->getNumResults()));
  rewriter.eraseOp(op);
  return newOp;
}
