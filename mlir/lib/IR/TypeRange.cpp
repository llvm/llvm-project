//===- TypeRange.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TypeRange

TypeRange::TypeRange(ArrayRef<Type> types)
    : TypeRange(types.data(), types.size()) {
  assert(llvm::all_of(types, [](Type t) { return t; }) &&
         "attempting to construct a TypeRange with null types");
}
TypeRange::TypeRange(OperandRange values)
    : TypeRange(values.begin().getBase(), values.size()) {}
TypeRange::TypeRange(ResultRange values)
    : TypeRange(values.getBase(), values.size()) {}
TypeRange::TypeRange(ValueRange values) : TypeRange(OwnerT(), values.size()) {
  if (count == 0)
    return;
  ValueRange::OwnerT owner = values.begin().getBase();
  if (auto *result = llvm::dyn_cast_if_present<detail::OpResultImpl *>(owner))
    this->base = result;
  else if (auto *operand = llvm::dyn_cast_if_present<OpOperand *>(owner))
    this->base = operand;
  else if (auto value = llvm::dyn_cast_if_present<Value>(owner))
    this->base = value.getType();
  else
    this->base = cast<const Value *>(owner);
}

/// See `llvm::detail::indexed_accessor_range_base` for details.
TypeRange::OwnerT TypeRange::offset_base(OwnerT object, ptrdiff_t index) {
  if (llvm::isa_and_nonnull<Type>(object)) {
    // Prevent out-of-bounds indexing for single values.
    // Note that we do allow an index of 1 as is required by 'slice'ing that
    // returns an empty range. This also matches the usual rules of C++ of being
    // allowed to index past the last element of an array.
    assert(index <= 1 && "out-of-bound offset into single-value 'ValueRange'");
    // Return nullptr to quickly cause segmentation faults on misuse.
    return index == 0 ? object : nullptr;
  }
  if (const auto *value = llvm::dyn_cast_if_present<const Value *>(object))
    return {value + index};
  if (auto *operand = llvm::dyn_cast_if_present<OpOperand *>(object))
    return {operand + index};
  if (auto *result = llvm::dyn_cast_if_present<detail::OpResultImpl *>(object))
    return {result->getNextResultAtOffset(index)};
  return {llvm::dyn_cast_if_present<const Type *>(object) + index};
}

/// See `llvm::detail::indexed_accessor_range_base` for details.
Type TypeRange::dereference_iterator(OwnerT object, ptrdiff_t index) {
  if (auto type = llvm::dyn_cast_if_present<Type>(object)) {
    assert(index == 0 && "cannot offset into single-value 'TypeRange'");
    return type;
  }
  if (const auto *value = llvm::dyn_cast_if_present<const Value *>(object))
    return (value + index)->getType();
  if (auto *operand = llvm::dyn_cast_if_present<OpOperand *>(object))
    return (operand + index)->get().getType();
  if (auto *result = llvm::dyn_cast_if_present<detail::OpResultImpl *>(object))
    return result->getNextResultAtOffset(index)->getType();
  return llvm::dyn_cast_if_present<const Type *>(object)[index];
}
