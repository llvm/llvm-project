//===- Value.cpp - MLIR Value Classes -------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

/// If this value is the result of an Operation, return the operation that
/// defines it.
Operation *Value::getDefiningOp() const {
  if (auto result = dyn_cast<OpResult>())
    return result->getOwner();
  return nullptr;
}

Location Value::getLoc() {
  if (auto *op = getDefiningOp())
    return op->getLoc();
  return UnknownLoc::get(getContext());
}

/// Return the Region in which this Value is defined.
Region *Value::getParentRegion() {
  if (auto *op = getDefiningOp())
    return op->getParentRegion();
  return cast<BlockArgument>()->getOwner()->getParent();
}

//===----------------------------------------------------------------------===//
// Value::UseLists
//===----------------------------------------------------------------------===//

/// Provide the use list that is attached to this value.
IRObjectWithUseList<OpOperand> *Value::getUseList() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl();
  return cast<OpResult>().getOwner();
}

/// Drop all uses of this object from their respective owners.
void Value::dropAllUses() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl()->dropAllUses();
  return cast<OpResult>().getOwner()->dropAllUses(*this);
}

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead.  When this returns
/// there are zero uses of 'this'.
void Value::replaceAllUsesWith(Value newValue) const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl()->replaceAllUsesWith(newValue);
  IRMultiObjectWithUseList<OpOperand> *useList = cast<OpResult>().getOwner();
  useList->replaceAllUsesWith(*this, newValue);
}

//===--------------------------------------------------------------------===//
// Uses

auto Value::use_begin() const -> use_iterator {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl()->use_begin();
  return cast<OpResult>().getOwner()->use_begin(*this);
}

/// Returns true if this value has exactly one use.
bool Value::hasOneUse() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl()->hasOneUse();
  return cast<OpResult>().getOwner()->hasOneUse(*this);
}

/// Returns true if this value has no uses.
bool Value::use_empty() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl()->use_empty();
  return cast<OpResult>().getOwner()->use_empty(*this);
}

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// Provide the use list that is attached to the given block.
IRObjectWithUseList<BlockOperand> *BlockOperand::getUseList(Block *value) {
  return value;
}

/// Return which operand this is in the operand list.
unsigned BlockOperand::getOperandNumber() {
  return this - &getOwner()->getBlockOperands()[0];
}

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

/// Provide the use list that is attached to the given value.
IRObjectWithUseList<OpOperand> *OpOperand::getUseList(Value value) {
  return value.getUseList();
}

/// Return the current value being used by this operand.
Value OpOperand::get() const {
  return IROperand<OpOperand, detail::OpaqueValue>::get();
}

/// Set the operand to the given value.
void OpOperand::set(Value value) {
  IROperand<OpOperand, detail::OpaqueValue>::set(value);
}

/// Return which operand this is in the operand list.
unsigned OpOperand::getOperandNumber() {
  return this - &getOwner()->getOpOperands()[0];
}

//===----------------------------------------------------------------------===//
// detail::OpaqueValue
//===----------------------------------------------------------------------===//

/// Implicit conversion from 'Value'.
detail::OpaqueValue::OpaqueValue(Value value)
    : impl(value.getAsOpaquePointer()) {}

/// Implicit conversion back to 'Value'.
detail::OpaqueValue::operator Value() const {
  return Value::getFromOpaquePointer(impl);
}
