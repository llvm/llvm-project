//===- BufferUtils.cpp - buffer transformation utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for buffer optimization passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/BufferUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BufferPlacementAllocs
//===----------------------------------------------------------------------===//

/// Get the start operation to place the given alloc value withing the
// specified placement block.
Operation *BufferPlacementAllocs::getStartOperation(Value allocValue,
                                                    Block *placementBlock,
                                                    const Liveness &liveness) {
  // We have to ensure that we place the alloc before its first use in this
  // block.
  const LivenessBlockInfo &livenessInfo = *liveness.getLiveness(placementBlock);
  Operation *startOperation = livenessInfo.getStartOperation(allocValue);
  // Check whether the start operation lies in the desired placement block.
  // If not, we will use the terminator as this is the last operation in
  // this block.
  if (startOperation->getBlock() != placementBlock) {
    Operation *opInPlacementBlock =
        placementBlock->findAncestorOpInBlock(*startOperation);
    startOperation = opInPlacementBlock ? opInPlacementBlock
                                        : placementBlock->getTerminator();
  }

  return startOperation;
}

/// Initializes the internal list by discovering all supported allocation
/// nodes.
BufferPlacementAllocs::BufferPlacementAllocs(Operation *op) { build(op); }

/// Searches for and registers all supported allocation entries.
void BufferPlacementAllocs::build(Operation *op) {
  op->walk([&](MemoryEffectOpInterface opInterface) {
    // Try to find a single allocation result.
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    opInterface.getEffects(effects);

    SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
    llvm::copy_if(
        effects, std::back_inserter(allocateResultEffects),
        [=](MemoryEffects::EffectInstance &it) {
          Value value = it.getValue();
          return isa<MemoryEffects::Allocate>(it.getEffect()) && value &&
                 value.isa<OpResult>() &&
                 it.getResource() !=
                     SideEffects::AutomaticAllocationScopeResource::get();
        });
    // If there is one result only, we will be able to move the allocation and
    // (possibly existing) deallocation ops.
    if (allocateResultEffects.size() != 1)
      return;
    // Get allocation result.
    Value allocValue = allocateResultEffects[0].getValue();
    // Find the associated dealloc value and register the allocation entry.
    llvm::Optional<Operation *> dealloc = findDealloc(allocValue);
    // If the allocation has > 1 dealloc associated with it, skip handling it.
    if (!dealloc.hasValue())
      return;
    allocs.push_back(std::make_tuple(allocValue, *dealloc));
  });
}

//===----------------------------------------------------------------------===//
// BufferPlacementTransformationBase
//===----------------------------------------------------------------------===//

/// Constructs a new transformation base using the given root operation.
BufferPlacementTransformationBase::BufferPlacementTransformationBase(
    Operation *op)
    : aliases(op), allocs(op), liveness(op) {}

/// Returns true if the given operation represents a loop by testing whether it
/// implements the `LoopLikeOpInterface` or the `RegionBranchOpInterface`. In
/// the case of a `RegionBranchOpInterface`, it checks all region-based control-
/// flow edges for cycles.
bool BufferPlacementTransformationBase::isLoop(Operation *op) {
  // If the operation implements the `LoopLikeOpInterface` it can be considered
  // a loop.
  if (isa<LoopLikeOpInterface>(op))
    return true;

  // If the operation does not implement the `RegionBranchOpInterface`, it is
  // (currently) not possible to detect a loop.
  RegionBranchOpInterface regionInterface;
  if (!(regionInterface = dyn_cast<RegionBranchOpInterface>(op)))
    return false;

  // Recurses into a region using the current region interface to find potential
  // cycles.
  SmallPtrSet<Region *, 4> visitedRegions;
  std::function<bool(Region *)> recurse = [&](Region *current) {
    if (!current)
      return false;
    // If we have found a back edge, the parent operation induces a loop.
    if (!visitedRegions.insert(current).second)
      return true;
    // Recurses into all region successors.
    SmallVector<RegionSuccessor, 2> successors;
    regionInterface.getSuccessorRegions(current->getRegionNumber(), successors);
    for (RegionSuccessor &regionEntry : successors)
      if (recurse(regionEntry.getSuccessor()))
        return true;
    return false;
  };

  // Start with all entry regions and test whether they induce a loop.
  SmallVector<RegionSuccessor, 2> successorRegions;
  regionInterface.getSuccessorRegions(/*index=*/llvm::None, successorRegions);
  for (RegionSuccessor &regionEntry : successorRegions) {
    if (recurse(regionEntry.getSuccessor()))
      return true;
    visitedRegions.clear();
  }

  return false;
}
