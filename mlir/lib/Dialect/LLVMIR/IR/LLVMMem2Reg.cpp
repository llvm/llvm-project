//===- LLVMMem2Reg.cpp - Mem2Reg Interfaces ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces for LLVM dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/Mem2RegInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> LLVM::AllocaOp::getPromotableSlots() {
  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  Type elemType =
      getElemType() ? *getElemType() : getResult().getType().getElementType();
  return {MemorySlot{getResult(), elemType}};
}

Value LLVM::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                      OpBuilder &builder) {
  return builder.create<LLVM::UndefOp>(getLoc(), slot.elemType);
}

void LLVM::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         OpBuilder &builder) {
  for (Operation *user : getOperation()->getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      builder.create<LLVM::DbgValueOp>(declareOp.getLoc(), argument,
                                       declareOp.getVarInfo());
}

void LLVM::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                             Value defaultValue) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  erase();
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp/StoreOp
//===----------------------------------------------------------------------===//

bool LLVM::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value LLVM::LoadOp::getStored(const MemorySlot &slot) { return {}; }

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

Value LLVM::StoreOp::getStored(const MemorySlot &slot) {
  return getAddr() == slot.ptr ? getValue() : Value();
}

bool LLVM::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
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

DeletionKind LLVM::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

bool LLVM::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
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

DeletionKind LLVM::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the stored slot
  // pointer.
  for (Operation *user : slot.ptr.getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      builder.create<LLVM::DbgValueOp>(declareOp->getLoc(), getValue(),
                                       declareOp.getVarInfo());
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for discardable OPs
//===----------------------------------------------------------------------===//

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

bool LLVM::BitcastOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::BitcastOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::AddrSpaceCastOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::AddrSpaceCastOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeStartOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeStartOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::LifetimeEndOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::LifetimeEndOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}

static bool hasAllZeroIndices(LLVM::GEPOp gepOp) {
  return llvm::all_of(gepOp.getIndices(), [](auto index) {
    auto indexAttr = index.template dyn_cast<IntegerAttr>();
    return indexAttr && indexAttr.getValue() == 0;
  });
}

bool LLVM::GEPOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  // GEP can be removed as long as it is a no-op and its users can be removed.
  if (!hasAllZeroIndices(*this))
    return false;
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::GEPOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder) {
  return DeletionKind::Delete;
}
