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

#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallString.h"
#include <optional>

using namespace mlir;
using namespace mlir::bufferization;

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
                 isa<OpResult>(value) &&
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
    std::optional<Operation *> dealloc = memref::findDealloc(allocValue);
    // If the allocation has > 1 dealloc associated with it, skip handling it.
    if (!dealloc)
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
    regionInterface.getSuccessorRegions(current, successors);
    for (RegionSuccessor &regionEntry : successors)
      if (recurse(regionEntry.getSuccessor()))
        return true;
    return false;
  };

  // Start with all entry regions and test whether they induce a loop.
  SmallVector<RegionSuccessor, 2> successorRegions;
  regionInterface.getSuccessorRegions(/*point=*/RegionBranchPoint::parent(),
                                      successorRegions);
  for (RegionSuccessor &regionEntry : successorRegions) {
    if (recurse(regionEntry.getSuccessor()))
      return true;
    visitedRegions.clear();
  }

  return false;
}

//===----------------------------------------------------------------------===//
// BufferPlacementTransformationBase
//===----------------------------------------------------------------------===//

FailureOr<memref::GlobalOp>
bufferization::getGlobalFor(arith::ConstantOp constantOp, uint64_t alignment,
                            Attribute memorySpace) {
  auto type = cast<RankedTensorType>(constantOp.getType());
  auto moduleOp = constantOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  // If we already have a global for this constant value, no need to do
  // anything else.
  for (Operation &op : moduleOp.getRegion().getOps()) {
    auto globalOp = dyn_cast<memref::GlobalOp>(&op);
    if (!globalOp)
      continue;
    if (!globalOp.getInitialValue().has_value())
      continue;
    uint64_t opAlignment = globalOp.getAlignment().value_or(0);
    Attribute initialValue = globalOp.getInitialValue().value();
    if (opAlignment == alignment && initialValue == constantOp.getValue())
      return globalOp;
  }

  // Create a builder without an insertion point. We will insert using the
  // symbol table to guarantee unique names.
  OpBuilder globalBuilder(moduleOp.getContext());
  SymbolTable symbolTable(moduleOp);

  // Create a pretty name.
  SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  interleave(type.getShape(), os, "x");
  os << "x" << type.getElementType();

  // Add an optional alignment to the global memref.
  IntegerAttr memrefAlignment =
      alignment > 0 ? IntegerAttr::get(globalBuilder.getI64Type(), alignment)
                    : IntegerAttr();

  BufferizeTypeConverter typeConverter;
  auto memrefType = cast<MemRefType>(typeConverter.convertType(type));
  if (memorySpace)
    memrefType = MemRefType::Builder(memrefType).setMemorySpace(memorySpace);
  auto global = globalBuilder.create<memref::GlobalOp>(
      constantOp.getLoc(), (Twine("__constant_") + os.str()).str(),
      /*sym_visibility=*/globalBuilder.getStringAttr("private"),
      /*type=*/memrefType,
      /*initial_value=*/cast<ElementsAttr>(constantOp.getValue()),
      /*constant=*/true,
      /*alignment=*/memrefAlignment);
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());
  return global;
}
