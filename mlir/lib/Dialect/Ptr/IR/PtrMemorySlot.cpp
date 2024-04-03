//===- LLVMMemorySlot.cpp - MemorySlot interfaces ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MemorySlot-related interfaces for LLVM dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

namespace {
/// Checks if `slot` can be accessed through the provided access type.
bool isValidAccessType(const MemorySlot &slot, Type accessType,
                       const DataLayout &dataLayout) {
  return dataLayout.getTypeSize(accessType) <=
         dataLayout.getTypeSize(slot.elemType);
}

/// Returns the subslot's type at the requested index.
Type getTypeAtIndex(const DestructurableMemorySlot &slot, Attribute index) {
  auto subelementIndexMap =
      slot.elemType.cast<DestructurableTypeInterface>().getSubelementIndexMap();
  if (!subelementIndexMap)
    return {};
  assert(!subelementIndexMap->empty());

  // Note: Returns a null-type when no entry was found.
  return subelementIndexMap->lookup(index);
}

/// Conditions the deletion of the operation to the removal of all its uses.
bool forwardToUsers(Operation *op,
                    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}
} // namespace

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//

bool AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const ::mlir::DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

Value ptr::LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool ptr::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool ptr::LoadOp::storesTo(const MemorySlot &slot) { return false; }

bool LoadOp::canUsesBeRemoved(const MemorySlot &slot,
                              const SmallPtrSetImpl<OpOperand *> &blockingUses,
                              SmallVectorImpl<OpOperand *> &newBlockingUses,
                              const ::mlir::DataLayout &datalayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it loads the exact stored value and is not
  // volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getResult().getType() == slot.elemType && !getVolatile_();
}

DeletionKind
LoadOp::removeBlockingUses(const MemorySlot &slot,
                           const SmallPtrSetImpl<OpOperand *> &blockingUses,
                           RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

LogicalResult
LoadOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                               SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                               const ::mlir::DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getType(), dataLayout));
}

bool LoadOp::canRewire(const DestructurableMemorySlot &slot,
                       SmallPtrSetImpl<Attribute> &usedIndices,
                       SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                       const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // A load always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getType()) > dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind LoadOp::rewire(const DestructurableMemorySlot &slot,
                            DenseMap<Attribute, MemorySlot> &subslots,
                            RewriterBase &rewriter,
                            const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  rewriter.modifyOpInPlace(
      *this, [&]() { getAddrMutable().set(it->getSecond().ptr); });
  return DeletionKind::Keep;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

bool StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool StoreOp::storesTo(const MemorySlot &slot) { return getAddr() == slot.ptr; }

Value StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  return getValue();
}

bool StoreOp::canUsesBeRemoved(const MemorySlot &slot,
                               const SmallPtrSetImpl<OpOperand *> &blockingUses,
                               SmallVectorImpl<OpOperand *> &newBlockingUses,
                               const ::mlir::DataLayout &datalayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType &&
         !getVolatile_();
}

DeletionKind
StoreOp::removeBlockingUses(const MemorySlot &slot,
                            const SmallPtrSetImpl<OpOperand *> &blockingUses,
                            RewriterBase &rewriter, Value reachingDefinition) {
  return DeletionKind::Delete;
}

LogicalResult
StoreOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const ::mlir::DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getValue().getType(), dataLayout));
}

bool StoreOp::canRewire(const DestructurableMemorySlot &slot,
                        SmallPtrSetImpl<Attribute> &usedIndices,
                        SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                        const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // Storing the pointer to memory cannot be dealt with.
  if (getValue() == slot.ptr)
    return false;

  // A store always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getValue().getType()) >
      dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind StoreOp::rewire(const DestructurableMemorySlot &slot,
                             DenseMap<Attribute, MemorySlot> &subslots,
                             RewriterBase &rewriter,
                             const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  rewriter.modifyOpInPlace(
      *this, [&]() { getAddrMutable().set(it->getSecond().ptr); });
  return DeletionKind::Keep;
}
