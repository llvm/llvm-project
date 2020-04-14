//===- OperationSupport.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains out-of-line implementations of the support types that
// Operation and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// OperationState
//===----------------------------------------------------------------------===//

OperationState::OperationState(Location location, StringRef name)
    : location(location), name(name, location->getContext()) {}

OperationState::OperationState(Location location, OperationName name)
    : location(location), name(name) {}

OperationState::OperationState(Location location, StringRef name,
                               ValueRange operands, ArrayRef<Type> types,
                               ArrayRef<NamedAttribute> attributes,
                               ArrayRef<Block *> successors,
                               MutableArrayRef<std::unique_ptr<Region>> regions,
                               bool resizableOperandList)
    : location(location), name(name, location->getContext()),
      operands(operands.begin(), operands.end()),
      types(types.begin(), types.end()),
      attributes(attributes.begin(), attributes.end()),
      successors(successors.begin(), successors.end()) {
  for (std::unique_ptr<Region> &r : regions)
    this->regions.push_back(std::move(r));
}

void OperationState::addOperands(ValueRange newOperands) {
  operands.append(newOperands.begin(), newOperands.end());
}

void OperationState::addSuccessors(SuccessorRange newSuccessors) {
  successors.append(newSuccessors.begin(), newSuccessors.end());
}

Region *OperationState::addRegion() {
  regions.emplace_back(new Region);
  return regions.back().get();
}

void OperationState::addRegion(std::unique_ptr<Region> &&region) {
  regions.push_back(std::move(region));
}

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

/// Replace the operands contained in the storage with the ones provided in
/// 'operands'.
void detail::OperandStorage::setOperands(Operation *owner,
                                         ValueRange operands) {
  // If the number of operands is less than or equal to the current amount, we
  // can just update in place.
  if (operands.size() <= numOperands) {
    auto opOperands = getOperands();

    // If the number of new operands is less than the current count, then remove
    // any extra operands.
    for (unsigned i = operands.size(); i != numOperands; ++i)
      opOperands[i].~OpOperand();

    // Set the operands in place.
    numOperands = operands.size();
    for (unsigned i = 0; i != numOperands; ++i)
      opOperands[i].set(operands[i]);
    return;
  }

  // Otherwise, we need to be resizable.
  assert(resizable && "Only resizable operations may add operands");

  // Grow the capacity if necessary.
  auto &resizeUtil = getResizableStorage();
  if (resizeUtil.capacity < operands.size())
    grow(resizeUtil, operands.size());

  // Set the operands.
  OpOperand *opBegin = getRawOperands();
  for (unsigned i = 0; i != numOperands; ++i)
    opBegin[i].set(operands[i]);
  for (unsigned e = operands.size(); numOperands != e; ++numOperands)
    new (&opBegin[numOperands]) OpOperand(owner, operands[numOperands]);
}

/// Erase an operand held by the storage.
void detail::OperandStorage::eraseOperand(unsigned index) {
  assert(index < size());
  auto operands = getOperands();
  --numOperands;

  // Shift all operands down by 1 if the operand to remove is not at the end.
  auto indexIt = std::next(operands.begin(), index);
  if (index != numOperands)
    std::rotate(indexIt, std::next(indexIt), operands.end());
  operands[numOperands].~OpOperand();
}

/// Grow the internal operand storage.
void detail::OperandStorage::grow(ResizableStorage &resizeUtil,
                                  size_t minSize) {
  // Allocate a new storage array.
  resizeUtil.capacity =
      std::max(size_t(llvm::NextPowerOf2(resizeUtil.capacity + 2)), minSize);
  OpOperand *newStorage = static_cast<OpOperand *>(
      llvm::safe_malloc(resizeUtil.capacity * sizeof(OpOperand)));

  // Move the current operands to the new storage.
  auto operands = getOperands();
  std::uninitialized_copy(std::make_move_iterator(operands.begin()),
                          std::make_move_iterator(operands.end()), newStorage);

  // Destroy the original operands and update the resizable storage pointer.
  for (auto &operand : operands)
    operand.~OpOperand();
  resizeUtil.setDynamicStorage(newStorage);
}

//===----------------------------------------------------------------------===//
// Operation Value-Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TypeRange

TypeRange::TypeRange(ArrayRef<Type> types)
    : TypeRange(types.data(), types.size()) {}
TypeRange::TypeRange(OperandRange values)
    : TypeRange(values.begin().getBase(), values.size()) {}
TypeRange::TypeRange(ResultRange values)
    : TypeRange(values.getBase()->getResultTypes().slice(values.getStartIndex(),
                                                         values.size())) {}
TypeRange::TypeRange(ArrayRef<Value> values)
    : TypeRange(values.data(), values.size()) {}
TypeRange::TypeRange(ValueRange values) : TypeRange(OwnerT(), values.size()) {
  detail::ValueRangeOwner owner = values.begin().getBase();
  if (auto *op = reinterpret_cast<Operation *>(owner.ptr.dyn_cast<void *>()))
    this->base = &op->getResultTypes()[owner.startIndex];
  else if (auto *operand = owner.ptr.dyn_cast<OpOperand *>())
    this->base = operand;
  else
    this->base = owner.ptr.get<const Value *>();
}

/// See `llvm::detail::indexed_accessor_range_base` for details.
TypeRange::OwnerT TypeRange::offset_base(OwnerT object, ptrdiff_t index) {
  if (auto *value = object.dyn_cast<const Value *>())
    return {value + index};
  if (auto *operand = object.dyn_cast<OpOperand *>())
    return {operand + index};
  return {object.dyn_cast<const Type *>() + index};
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Type TypeRange::dereference_iterator(OwnerT object, ptrdiff_t index) {
  if (auto *value = object.dyn_cast<const Value *>())
    return (value + index)->getType();
  if (auto *operand = object.dyn_cast<OpOperand *>())
    return (operand + index)->get().getType();
  return object.dyn_cast<const Type *>()[index];
}

//===----------------------------------------------------------------------===//
// OperandRange

OperandRange::OperandRange(Operation *op)
    : OperandRange(op->getOpOperands().data(), op->getNumOperands()) {}

/// Return the operand index of the first element of this range. The range
/// must not be empty.
unsigned OperandRange::getBeginOperandIndex() const {
  assert(!empty() && "range must not be empty");
  return base->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// ResultRange

ResultRange::ResultRange(Operation *op)
    : ResultRange(op, /*startIndex=*/0, op->getNumResults()) {}

ArrayRef<Type> ResultRange::getTypes() const {
  return getBase()->getResultTypes();
}

/// See `llvm::indexed_accessor_range` for details.
OpResult ResultRange::dereference(Operation *op, ptrdiff_t index) {
  return op->getResult(index);
}

//===----------------------------------------------------------------------===//
// ValueRange

ValueRange::ValueRange(ArrayRef<Value> values)
    : ValueRange(values.data(), values.size()) {}
ValueRange::ValueRange(OperandRange values)
    : ValueRange(values.begin().getBase(), values.size()) {}
ValueRange::ValueRange(ResultRange values)
    : ValueRange(
          {values.getBase(), static_cast<unsigned>(values.getStartIndex())},
          values.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
ValueRange::OwnerT ValueRange::offset_base(const OwnerT &owner,
                                           ptrdiff_t index) {
  if (auto *value = owner.ptr.dyn_cast<const Value *>())
    return {value + index};
  if (auto *operand = owner.ptr.dyn_cast<OpOperand *>())
    return {operand + index};
  Operation *operation = reinterpret_cast<Operation *>(owner.ptr.get<void *>());
  return {operation, owner.startIndex + static_cast<unsigned>(index)};
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Value ValueRange::dereference_iterator(const OwnerT &owner, ptrdiff_t index) {
  if (auto *value = owner.ptr.dyn_cast<const Value *>())
    return value[index];
  if (auto *operand = owner.ptr.dyn_cast<OpOperand *>())
    return operand[index].get();
  Operation *operation = reinterpret_cast<Operation *>(owner.ptr.get<void *>());
  return operation->getResult(owner.startIndex + index);
}
