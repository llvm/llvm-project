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
                               MutableArrayRef<std::unique_ptr<Region>> regions)
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

detail::OperandStorage::OperandStorage(Operation *owner, ValueRange values)
    : representation(0) {
  auto &inlineStorage = getInlineStorage();
  inlineStorage.numOperands = inlineStorage.capacity = values.size();
  auto *operandPtrBegin = getTrailingObjects<OpOperand>();
  for (unsigned i = 0, e = inlineStorage.numOperands; i < e; ++i)
    new (&operandPtrBegin[i]) OpOperand(owner, values[i]);
}

detail::OperandStorage::~OperandStorage() {
  // Destruct the current storage container.
  if (isDynamicStorage()) {
    TrailingOperandStorage &storage = getDynamicStorage();
    storage.~TrailingOperandStorage();
    free(&storage);
  } else {
    getInlineStorage().~TrailingOperandStorage();
  }
}

/// Replace the operands contained in the storage with the ones provided in
/// 'values'.
void detail::OperandStorage::setOperands(Operation *owner, ValueRange values) {
  MutableArrayRef<OpOperand> storageOperands = resize(owner, values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i)
    storageOperands[i].set(values[i]);
}

/// Resize the storage to the given size. Returns the array containing the new
/// operands.
MutableArrayRef<OpOperand> detail::OperandStorage::resize(Operation *owner,
                                                          unsigned newSize) {
  TrailingOperandStorage &storage = getStorage();

  // If the number of operands is less than or equal to the current amount, we
  // can just update in place.
  unsigned &numOperands = storage.numOperands;
  MutableArrayRef<OpOperand> operands = storage.getOperands();
  if (newSize <= numOperands) {
    // If the number of new size is less than the current, remove any extra
    // operands.
    for (unsigned i = newSize; i != numOperands; ++i)
      operands[i].~OpOperand();
    numOperands = newSize;
    return operands.take_front(newSize);
  }

  // If the new size is within the original inline capacity, grow inplace.
  if (newSize <= storage.capacity) {
    OpOperand *opBegin = operands.data();
    for (unsigned e = newSize; numOperands != e; ++numOperands)
      new (&opBegin[numOperands]) OpOperand(owner);
    return MutableArrayRef<OpOperand>(opBegin, newSize);
  }

  // Otherwise, we need to allocate a new storage.
  unsigned newCapacity =
      std::max(unsigned(llvm::NextPowerOf2(storage.capacity + 2)), newSize);
  auto *newStorageMem =
      malloc(TrailingOperandStorage::totalSizeToAlloc<OpOperand>(newCapacity));
  auto *newStorage = ::new (newStorageMem) TrailingOperandStorage();
  newStorage->numOperands = newSize;
  newStorage->capacity = newCapacity;

  // Move the current operands to the new storage.
  MutableArrayRef<OpOperand> newOperands = newStorage->getOperands();
  std::uninitialized_copy(std::make_move_iterator(operands.begin()),
                          std::make_move_iterator(operands.end()),
                          newOperands.begin());

  // Destroy the original operands.
  for (auto &operand : operands)
    operand.~OpOperand();

  // Initialize any new operands.
  for (unsigned e = newSize; numOperands != e; ++numOperands)
    new (&newOperands[numOperands]) OpOperand(owner);

  // If the current storage is also dynamic, free it.
  if (isDynamicStorage())
    free(&storage);

  // Update the storage representation to use the new dynamic storage.
  representation = reinterpret_cast<intptr_t>(newStorage);
  representation |= DynamicStorageBit;
  return newOperands;
}

/// Erase an operand held by the storage.
void detail::OperandStorage::eraseOperand(unsigned index) {
  assert(index < size());
  TrailingOperandStorage &storage = getStorage();
  MutableArrayRef<OpOperand> operands = storage.getOperands();
  --storage.numOperands;

  // Shift all operands down by 1 if the operand to remove is not at the end.
  auto indexIt = std::next(operands.begin(), index);
  if (index != storage.numOperands)
    std::rotate(indexIt, std::next(indexIt), operands.end());
  operands[storage.numOperands].~OpOperand();
}

//===----------------------------------------------------------------------===//
// ResultStorage
//===----------------------------------------------------------------------===//

/// Returns the parent operation of this trailing result.
Operation *detail::TrailingOpResult::getOwner() {
  // We need to do some arithmetic to get the operation pointer. Move the
  // trailing owner to the start of the array.
  TrailingOpResult *trailingIt = this - trailingResultNumber;

  // Move the owner past the inline op results to get to the operation.
  auto *inlineResultIt = reinterpret_cast<InLineOpResult *>(trailingIt) -
                         OpResult::getMaxInlineResults();
  return reinterpret_cast<Operation *>(inlineResultIt) - 1;
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
    this->base = op->getResultTypes().drop_front(owner.startIndex).data();
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
  return getBase()->getResultTypes().slice(getStartIndex(), size());
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
