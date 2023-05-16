//===- MemRefMem2Reg.cpp - Mem2Reg Interfaces -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Mem2Reg-related interfaces for MemRef dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
//  AllocaOp interfaces
//===----------------------------------------------------------------------===//

static bool isSupportedElementType(Type type) {
  return llvm::isa<MemRefType>(type) ||
         OpBuilder(type.getContext()).getZeroAttr(type);
}

SmallVector<MemorySlot> memref::AllocaOp::getPromotableSlots() {
  MemRefType type = getType();
  if (!isSupportedElementType(type.getElementType()))
    return {};
  if (!type.hasStaticShape())
    return {};
  // Make sure the memref contains only a single element.
  if (any_of(type.getShape(), [](uint64_t dim) { return dim != 1; }))
    return {};

  return {MemorySlot{getResult(), type.getElementType()}};
}

Value memref::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                        OpBuilder &builder) {
  assert(isSupportedElementType(slot.elemType));
  // TODO: support more types.
  return TypeSwitch<Type, Value>(slot.elemType)
      .Case([&](MemRefType t) {
        return builder.create<memref::AllocaOp>(getLoc(), t);
      })
      .Default([&](Type t) {
        return builder.create<arith::ConstantOp>(getLoc(), t,
                                                 builder.getZeroAttr(t));
      });
}

void memref::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                               Value defaultValue) {
  if (defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  erase();
}

void memref::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                           BlockArgument argument,
                                           OpBuilder &builder) {}

//===----------------------------------------------------------------------===//
//  LoadOp/StoreOp interfaces
//===----------------------------------------------------------------------===//

bool memref::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getMemRef() == slot.ptr;
}

Value memref::LoadOp::getStored(const MemorySlot &slot) { return {}; }

bool memref::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getMemRef() == slot.ptr &&
         getResult().getType() == slot.elemType;
}

DeletionKind memref::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

bool memref::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

Value memref::StoreOp::getStored(const MemorySlot &slot) {
  if (getMemRef() != slot.ptr)
    return {};
  return getValue();
}

bool memref::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getMemRef() == slot.ptr &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType;
}

DeletionKind memref::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition) {
  return DeletionKind::Delete;
}
