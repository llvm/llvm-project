//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/Utils/MemorySlotUtils.h"

using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// AffineForOp
//===----------------------------------------------------------------------===//

bool AffineForOp::isRegionPromotable(const MemorySlot &slot, Region *region,
                                     bool hasValueStores) {
  return true;
}

void AffineForOp::setupPromotion(
    const MemorySlot &slot, Value reachingDef, bool hasValueStores,
    llvm::SmallMapVector<Region *, Value, 2> &regionsToProcess) {
  Region &bodyRegion = getBodyRegion();
  if (!hasValueStores) {
    regionsToProcess.insert({&bodyRegion, reachingDef});
    return;
  }

  getInitsMutable().append(reachingDef);
  bodyRegion.addArgument(slot.elemType, slot.ptr.getLoc());
  regionsToProcess.insert({&bodyRegion, bodyRegion.getArguments().back()});
}

Value AffineForOp::finalizePromotion(
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
