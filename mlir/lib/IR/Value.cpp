//===- Value.cpp - MLIR Value Classes -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::detail;

/// Construct a value.
Value::Value(BlockArgumentImpl *impl)
    : ownerAndKind(impl, Kind::BlockArgument) {}
Value::Value(Operation *op, unsigned resultNo) {
  assert(op->getNumResults() > resultNo && "invalid result number");
  if (LLVM_LIKELY(canPackResultInline(resultNo))) {
    ownerAndKind = {op, static_cast<Kind>(resultNo)};
    return;
  }

  // If we can't pack the result directly, grab the use list from the parent op.
  unsigned trailingNo = resultNo - OpResult::getMaxInlineResults();
  ownerAndKind = {op->getTrailingResult(trailingNo), Kind::TrailingOpResult};
}

/// Return the type of this value.
Type Value::getType() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getType();

  // If this is an operation result, query the parent operation.
  OpResult result = cast<OpResult>();
  Operation *owner = result.getOwner();
  if (owner->hasSingleResult)
    return owner->resultTypeOrSize.type;
  return owner->getResultTypes()[result.getResultNumber()];
}

/// Mutate the type of this Value to be of the specified type.
void Value::setType(Type newType) {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.setType(newType);

  OpResult result = cast<OpResult>();
  Operation *owner = result.getOwner();
  if (owner->hasSingleResult)
    owner->resultTypeOrSize.type = newType;
  else
    owner->getTrailingObjects<Type>()[result.getResultNumber()] = newType;
}

/// If this value is the result of an Operation, return the operation that
/// defines it.
Operation *Value::getDefiningOp() const {
  if (auto result = dyn_cast<OpResult>())
    return result.getOwner();
  return nullptr;
}

Location Value::getLoc() const {
  if (auto *op = getDefiningOp())
    return op->getLoc();

  // Use the location of the parent operation if this is a block argument.
  // TODO: Should we just add locations to block arguments?
  Operation *parentOp = cast<BlockArgument>().getOwner()->getParentOp();
  return parentOp ? parentOp->getLoc() : UnknownLoc::get(getContext());
}

/// Return the Region in which this Value is defined.
Region *Value::getParentRegion() {
  if (auto *op = getDefiningOp())
    return op->getParentRegion();
  return cast<BlockArgument>().getOwner()->getParent();
}

/// Return the Block in which this Value is defined.
Block *Value::getParentBlock() {
  if (Operation *op = getDefiningOp())
    return op->getBlock();
  return cast<BlockArgument>().getOwner();
}

//===----------------------------------------------------------------------===//
// Value::UseLists
//===----------------------------------------------------------------------===//

/// Provide the use list that is attached to this value.
IRObjectWithUseList<OpOperand> *Value::getUseList() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getImpl();
  if (getKind() != Kind::TrailingOpResult) {
    OpResult result = cast<OpResult>();
    return result.getOwner()->getInlineResult(result.getResultNumber());
  }

  // Otherwise this is a trailing operation result, which contains a use list.
  return reinterpret_cast<TrailingOpResult *>(ownerAndKind.getPointer());
}

/// Drop all uses of this object from their respective owners.
void Value::dropAllUses() const { return getUseList()->dropAllUses(); }

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead.  When this returns
/// there are zero uses of 'this'.
void Value::replaceAllUsesWith(Value newValue) const {
  return getUseList()->replaceAllUsesWith(newValue);
}

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead except if the user is
/// listed in 'exceptions' .
void Value::replaceAllUsesExcept(
    Value newValue, const SmallPtrSetImpl<Operation *> &exceptions) const {
  for (auto &use : llvm::make_early_inc_range(getUses())) {
    if (exceptions.count(use.getOwner()) == 0)
      use.set(newValue);
  }
}

/// Replace all uses of 'this' value with 'newValue' if the given callback
/// returns true.
void Value::replaceUsesWithIf(Value newValue,
                              function_ref<bool(OpOperand &)> shouldReplace) {
  for (OpOperand &use : llvm::make_early_inc_range(getUses()))
    if (shouldReplace(use))
      use.set(newValue);
}

/// Returns true if the value is used outside of the given block.
bool Value::isUsedOutsideOfBlock(Block *block) {
  return llvm::any_of(getUsers(), [block](Operation *user) {
    return user->getBlock() != block;
  });
}

//===--------------------------------------------------------------------===//
// Uses

auto Value::use_begin() const -> use_iterator {
  return getUseList()->use_begin();
}

/// Returns true if this value has exactly one use.
bool Value::hasOneUse() const { return getUseList()->hasOneUse(); }

/// Returns true if this value has no uses.
bool Value::use_empty() const { return getUseList()->use_empty(); }

//===----------------------------------------------------------------------===//
// OpResult
//===----------------------------------------------------------------------===//

/// Returns the operation that owns this result.
Operation *OpResult::getOwner() const {
  // If the result is in-place, the `owner` is the operation.
  void *owner = ownerAndKind.getPointer();
  if (LLVM_LIKELY(getKind() != Kind::TrailingOpResult))
    return static_cast<Operation *>(owner);

  // Otherwise, query the trailing result for the owner.
  return static_cast<TrailingOpResult *>(owner)->getOwner();
}

/// Return the result number of this result.
unsigned OpResult::getResultNumber() const {
  // If the result is in-place, we can use the kind directly.
  if (LLVM_LIKELY(getKind() != Kind::TrailingOpResult))
    return static_cast<unsigned>(ownerAndKind.getInt());
  // Otherwise, query the trailing result.
  auto *result = static_cast<TrailingOpResult *>(ownerAndKind.getPointer());
  return result->getResultNumber();
}

/// Given a number of operation results, returns the number that need to be
/// stored inline.
unsigned OpResult::getNumInline(unsigned numResults) {
  return std::min(numResults, getMaxInlineResults());
}

/// Given a number of operation results, returns the number that need to be
/// stored as trailing.
unsigned OpResult::getNumTrailing(unsigned numResults) {
  // If we can pack all of the results, there is no need for additional storage.
  unsigned maxInline = getMaxInlineResults();
  return numResults <= maxInline ? 0 : numResults - maxInline;
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
  return IROperand<OpOperand, OpaqueValue>::get();
}

/// Set the operand to the given value.
void OpOperand::set(Value value) {
  IROperand<OpOperand, OpaqueValue>::set(value);
}

/// Return which operand this is in the operand list.
unsigned OpOperand::getOperandNumber() {
  return this - &getOwner()->getOpOperands()[0];
}

//===----------------------------------------------------------------------===//
// OpaqueValue
//===----------------------------------------------------------------------===//

/// Implicit conversion from 'Value'.
OpaqueValue::OpaqueValue(Value value) : impl(value.getAsOpaquePointer()) {}

/// Implicit conversion back to 'Value'.
OpaqueValue::operator Value() const {
  return Value::getFromOpaquePointer(impl);
}
