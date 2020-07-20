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
#include "mlir/IR/StandardTypes.h"
using namespace mlir;

/// Construct a value.
Value::Value(detail::BlockArgumentImpl *impl)
    : ownerAndKind(impl, Kind::BlockArgument) {}
Value::Value(Operation *op, unsigned resultNo) {
  assert(op->getNumResults() > resultNo && "invalid result number");
  if (LLVM_LIKELY(canPackResultInline(resultNo))) {
    ownerAndKind = {op, static_cast<Kind>(resultNo)};
    return;
  }

  // If we can't pack the result directly, we need to represent this as a
  // trailing result.
  unsigned trailingResultNo =
      resultNo - static_cast<unsigned>(Kind::TrailingOpResult);
  ownerAndKind = {op->getTrailingResult(trailingResultNo),
                  Kind::TrailingOpResult};
}

/// Return the type of this value.
Type Value::getType() const {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.getType();

  // If this is an operation result, query the parent operation.
  OpResult result = cast<OpResult>();
  Operation *owner = result.getOwner();
  if (owner->hasSingleResult)
    return owner->resultType;
  return owner->resultType.cast<TupleType>().getType(result.getResultNumber());
}

/// Mutate the type of this Value to be of the specified type.
void Value::setType(Type newType) {
  if (BlockArgument arg = dyn_cast<BlockArgument>())
    return arg.setType(newType);
  OpResult result = cast<OpResult>();

  // If the owner has a single result, simply update it directly.
  Operation *owner = result.getOwner();
  if (owner->hasSingleResult) {
    owner->resultType = newType;
    return;
  }
  unsigned resultNo = result.getResultNumber();

  // Otherwise, rebuild the tuple if the new type is different from the current.
  auto curTypes = owner->resultType.cast<TupleType>().getTypes();
  if (curTypes[resultNo] == newType)
    return;
  auto newTypes = llvm::to_vector<4>(curTypes);
  newTypes[resultNo] = newType;
  owner->resultType = TupleType::get(newTypes, newType.getContext());
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
  return UnknownLoc::get(getContext());
}

/// Return the Region in which this Value is defined.
Region *Value::getParentRegion() {
  if (auto *op = getDefiningOp())
    return op->getParentRegion();
  return cast<BlockArgument>().getOwner()->getParent();
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
// OpResult
//===----------------------------------------------------------------------===//

/// Returns the operation that owns this result.
Operation *OpResult::getOwner() const {
  // If the result is in-place, the `owner` is the operation.
  if (LLVM_LIKELY(getKind() != Kind::TrailingOpResult))
    return reinterpret_cast<Operation *>(ownerAndKind.getPointer());

  // Otherwise, we need to do some arithmetic to get the operation pointer.
  // Move the trailing owner to the start of the array.
  auto *trailingIt =
      static_cast<detail::TrailingOpResult *>(ownerAndKind.getPointer());
  trailingIt -= trailingIt->trailingResultNumber;

  // This point is the first trailing object after the operation. So all we need
  // to do here is adjust for the operation size.
  return reinterpret_cast<Operation *>(trailingIt) - 1;
}

/// Return the result number of this result.
unsigned OpResult::getResultNumber() const {
  // If the result is in-place, we can use the kind directly.
  if (LLVM_LIKELY(getKind() != Kind::TrailingOpResult))
    return static_cast<unsigned>(ownerAndKind.getInt());
  // Otherwise, we add the number of inline results to the trailing owner.
  auto *trailingIt =
      static_cast<detail::TrailingOpResult *>(ownerAndKind.getPointer());
  unsigned trailingNumber = trailingIt->trailingResultNumber;
  return trailingNumber + static_cast<unsigned>(Kind::TrailingOpResult);
}

/// Given a number of operation results, returns the number that need to be
/// stored as trailing.
unsigned OpResult::getNumTrailing(unsigned numResults) {
  // If we can pack all of the results, there is no need for additional storage.
  if (numResults <= static_cast<unsigned>(Kind::TrailingOpResult))
    return 0;
  return numResults - static_cast<unsigned>(Kind::TrailingOpResult);
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
