//===- UseDefLists.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic use/def list machinery and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_USEDEFLISTS_H
#define MLIR_IR_USEDEFLISTS_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {

class Block;
class Operation;
class Value;
template <typename OperandType> class ValueUseIterator;
template <typename OperandType> class FilteredValueUseIterator;
template <typename UseIteratorT, typename OperandType> class ValueUserIterator;

//===----------------------------------------------------------------------===//
// IRObjectWithUseList
//===----------------------------------------------------------------------===//

/// This class represents a single IR object that contains a use list.
template <typename OperandType> class IRObjectWithUseList {
public:
  ~IRObjectWithUseList() {
    assert(use_empty() && "Cannot destroy a value that still has uses!");
  }

  /// Drop all uses of this object from their respective owners.
  void dropAllUses() {
    while (!use_empty())
      use_begin()->drop();
  }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(typename OperandType::ValueType newValue) {
    assert((!newValue || this != OperandType::getUseList(newValue)) &&
           "cannot RAUW a value with itself");
    while (!use_empty())
      use_begin()->set(newValue);
  }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  using use_iterator = ValueUseIterator<OperandType>;
  using use_range = iterator_range<use_iterator>;

  use_iterator use_begin() const { return use_iterator(firstUse); }
  use_iterator use_end() const { return use_iterator(nullptr); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() const { return {use_begin(), use_end()}; }

  /// Returns true if this value has exactly one use.
  bool hasOneUse() const {
    return firstUse && firstUse->getNextOperandUsingThisValue() == nullptr;
  }

  /// Returns true if this value has no uses.
  bool use_empty() const { return firstUse == nullptr; }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OperandType>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() const { return user_iterator(use_begin()); }
  user_iterator user_end() const { return user_iterator(use_end()); }

  /// Returns a range of all users.
  user_range getUsers() const { return {user_begin(), user_end()}; }

protected:
  IRObjectWithUseList() {}

  /// Return the first operand that is using this value, for use by custom
  /// use/def iterators.
  OperandType *getFirstUse() const { return firstUse; }

private:
  template <typename DerivedT, typename IRValueTy> friend class IROperand;
  OperandType *firstUse = nullptr;
};

//===----------------------------------------------------------------------===//
// IRMultiObjectWithUseList
//===----------------------------------------------------------------------===//

/// This class represents multiple IR objects with a single use list. This class
/// provides wrapper functionality for manipulating the uses of a single object.
template <typename OperandType>
class IRMultiObjectWithUseList : public IRObjectWithUseList<OperandType> {
public:
  using BaseType = IRObjectWithUseList<OperandType>;
  using ValueType = typename OperandType::ValueType;

  /// Drop all uses of `value` from their respective owners.
  void dropAllUses(ValueType value) {
    assert(this == OperandType::getUseList(value) &&
           "value not attached to this use list");
    for (OperandType &use : llvm::make_early_inc_range(getUses(value)))
      use.drop();
  }
  using BaseType::dropAllUses;

  /// Replace all uses of `oldValue` with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(ValueType oldValue, ValueType newValue) {
    assert(this == OperandType::getUseList(oldValue) &&
           "value not attached to this use list");
    assert((!newValue || this != OperandType::getUseList(newValue)) &&
           "cannot RAUW a value with itself");
    for (OperandType &use : llvm::make_early_inc_range(getUses(oldValue)))
      use.set(newValue);
  }
  using BaseType::replaceAllUsesWith;

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  using filtered_use_iterator = FilteredValueUseIterator<OperandType>;
  using filtered_use_range = iterator_range<filtered_use_iterator>;

  filtered_use_iterator use_begin(ValueType value) const {
    return filtered_use_iterator(this->getFirstUse(), value);
  }
  filtered_use_iterator use_end(ValueType) const { return use_end(); }
  filtered_use_range getUses(ValueType value) const {
    return {use_begin(value), use_end(value)};
  }
  bool hasOneUse(ValueType value) const {
    return mlir::has_single_element(getUses(value));
  }
  bool use_empty(ValueType value) const {
    return use_begin(value) == use_end(value);
  }
  using BaseType::getUses;
  using BaseType::hasOneUse;
  using BaseType::use_begin;
  using BaseType::use_empty;
  using BaseType::use_end;

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using filtered_user_iterator =
      ValueUserIterator<filtered_use_iterator, OperandType>;
  using filtered_user_range = iterator_range<filtered_user_iterator>;

  filtered_user_iterator user_begin(ValueType value) const {
    return {use_begin(value)};
  }
  filtered_user_iterator user_end(ValueType value) const { return {use_end()}; }
  filtered_user_range getUsers(ValueType value) const {
    return {user_begin(value), user_end(value)};
  }
  using BaseType::getUsers;
  using BaseType::user_begin;
  using BaseType::user_end;
};

//===----------------------------------------------------------------------===//
// IROperand
//===----------------------------------------------------------------------===//

/// A reference to a value, suitable for use as an operand of an operation.
/// IRValueTy is the root type to use for values this tracks. Derived operand
/// types are expected to provide the following:
///  * static IRObjectWithUseList *getUseList(IRValueTy value);
///    - Provide the use list that is attached to the given value.
template <typename DerivedT, typename IRValueTy> class IROperand {
public:
  using ValueType = IRValueTy;

  IROperand(Operation *owner) : owner(owner) {}
  IROperand(Operation *owner, ValueType value) : value(value), owner(owner) {
    insertIntoCurrent();
  }

  /// Return the current value being used by this operand.
  ValueType get() const { return value; }

  /// Set the current value being used by this operand.
  void set(ValueType newValue) {
    // It isn't worth optimizing for the case of switching operands on a single
    // value.
    removeFromCurrent();
    value = newValue;
    insertIntoCurrent();
  }

  /// Return the owner of this operand.
  Operation *getOwner() { return owner; }
  Operation *getOwner() const { return owner; }

  /// \brief Remove this use of the operand.
  void drop() {
    removeFromCurrent();
    value = nullptr;
    nextUse = nullptr;
    back = nullptr;
  }

  ~IROperand() { removeFromCurrent(); }

  /// Return the next operand on the use-list of the value we are referring to.
  /// This should generally only be used by the internal implementation details
  /// of the SSA machinery.
  DerivedT *getNextOperandUsingThisValue() { return nextUse; }

  /// We support a move constructor so IROperand's can be in vectors, but this
  /// shouldn't be used by general clients.
  IROperand(IROperand &&other) : owner(other.owner) {
    *this = std::move(other);
  }
  IROperand &operator=(IROperand &&other) {
    removeFromCurrent();
    other.removeFromCurrent();
    value = other.value;
    other.value = nullptr;
    other.back = nullptr;
    nextUse = nullptr;
    back = nullptr;
    insertIntoCurrent();
    return *this;
  }

private:
  /// The value used as this operand. This can be null when in a 'dropAllUses'
  /// state.
  ValueType value = {};

  /// The next operand in the use-chain.
  DerivedT *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  DerivedT **back = nullptr;

  /// The operation owner of this operand.
  Operation *const owner;

  /// Operands are not copyable or assignable.
  IROperand(const IROperand &use) = delete;
  IROperand &operator=(const IROperand &use) = delete;

  void removeFromCurrent() {
    if (!back)
      return;
    *back = nextUse;
    if (nextUse)
      nextUse->back = back;
  }

  void insertIntoCurrent() {
    auto *useList = DerivedT::getUseList(value);
    back = &useList->firstUse;
    nextUse = useList->firstUse;
    if (nextUse)
      nextUse->back = &nextUse;
    useList->firstUse = static_cast<DerivedT *>(this);
  }
};

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// Terminator operations can have Block operands to represent successors.
class BlockOperand : public IROperand<BlockOperand, Block *> {
public:
  using IROperand<BlockOperand, Block *>::IROperand;

  /// Provide the use list that is attached to the given block.
  static IRObjectWithUseList<BlockOperand> *getUseList(Block *value);

  /// Return which operand this is in the operand list of the User.
  unsigned getOperandNumber();

private:
  /// The number of OpOperands that correspond with this block operand.
  unsigned numSuccessorOperands = 0;

  /// Allow access to 'numSuccessorOperands'.
  friend Operation;
};

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides an opaque type erased wrapper around a `Value`.
class OpaqueValue {
public:
  /// Implicit conversion from 'Value'.
  OpaqueValue(Value value);
  OpaqueValue(std::nullptr_t = nullptr) : impl(nullptr) {}
  OpaqueValue(const OpaqueValue &) = default;
  OpaqueValue &operator=(const OpaqueValue &) = default;
  operator bool() const { return impl; }

  /// Implicit conversion back to 'Value'.
  operator Value() const;

private:
  void *impl;
};
} // namespace detail

/// A reference to a value, suitable for use as an operand of an operation.
class OpOperand : public IROperand<OpOperand, detail::OpaqueValue> {
public:
  using IROperand<OpOperand, detail::OpaqueValue>::IROperand;

  /// Provide the use list that is attached to the given value.
  static IRObjectWithUseList<OpOperand> *getUseList(Value value);

  /// Return the current value being used by this operand.
  Value get() const;

  /// Set the operand to the given value.
  void set(Value value);

  /// Return which operand this is in the operand list of the User.
  unsigned getOperandNumber();
};

//===----------------------------------------------------------------------===//
// ValueUseIterator
//===----------------------------------------------------------------------===//

namespace detail {
/// A base iterator class that allows for iterating over the uses of a value.
/// This is templated to allow for derived iterators to override specific
/// iterator methods.
template <typename DerivedT, typename OperandType>
class ValueUseIteratorImpl
    : public llvm::iterator_facade_base<DerivedT, std::forward_iterator_tag,
                                        OperandType> {
public:
  template <typename T>
  ValueUseIteratorImpl(const ValueUseIteratorImpl<T, OperandType> &other)
      : current(other.getOperand()) {}
  ValueUseIteratorImpl(OperandType *current = nullptr) : current(current) {}

  Operation *getUser() const { return current->getOwner(); }
  OperandType *getOperand() const { return current; }

  OperandType &operator*() const { return *current; }

  using llvm::iterator_facade_base<DerivedT, std::forward_iterator_tag,
                                   OperandType>::operator++;
  ValueUseIteratorImpl &operator++() {
    assert(current && "incrementing past end()!");
    current = (OperandType *)current->getNextOperandUsingThisValue();
    return *this;
  }

  bool operator==(const ValueUseIteratorImpl &rhs) const {
    return current == rhs.current;
  }

protected:
  OperandType *current;
};

} // end namespace detail

/// An iterator over all of the uses of an IR object.
template <typename OperandType>
class ValueUseIterator
    : public detail::ValueUseIteratorImpl<ValueUseIterator<OperandType>,
                                          OperandType> {
public:
  using detail::ValueUseIteratorImpl<ValueUseIterator<OperandType>,
                                     OperandType>::ValueUseIteratorImpl;
};

/// This class represents an iterator of the uses of a IR object that optionally
/// filters on a specific sub-value. This allows for filtering the uses of an
/// IRMultiObjectWithUseList.
template <typename OperandType>
class FilteredValueUseIterator
    : public detail::ValueUseIteratorImpl<FilteredValueUseIterator<OperandType>,
                                          OperandType> {
public:
  using BaseT =
      detail::ValueUseIteratorImpl<FilteredValueUseIterator<OperandType>,
                                   OperandType>;

  FilteredValueUseIterator() = default;
  FilteredValueUseIterator(const ValueUseIterator<OperandType> &it)
      : BaseT(it), filterVal(nullptr) {}
  FilteredValueUseIterator(OperandType *current,
                           typename OperandType::ValueType filterVal)
      : BaseT(current), filterVal(filterVal) {
    findNextValid();
  }

  using BaseT::operator++;
  FilteredValueUseIterator<OperandType> &operator++() {
    BaseT::operator++();
    findNextValid();
    return *this;
  }

private:
  void findNextValid() {
    if (!filterVal)
      return;
    while (this->current && ((OperandType *)this->current)->get() != filterVal)
      BaseT::operator++();
  }

  /// An optional value to use to filter specific uses.
  typename OperandType::ValueType filterVal;
};

//===----------------------------------------------------------------------===//
// ValueUserIterator
//===----------------------------------------------------------------------===//

/// An iterator over all users of a ValueBase.
template <typename UseIteratorT, typename OperandType>
class ValueUserIterator final
    : public llvm::mapped_iterator<UseIteratorT,
                                   Operation *(*)(OperandType &)> {
  static Operation *unwrap(OperandType &value) { return value.getOwner(); }

public:
  using pointer = Operation *;
  using reference = Operation *;

  /// Initializes the result type iterator to the specified result iterator.
  ValueUserIterator(UseIteratorT it)
      : llvm::mapped_iterator<UseIteratorT, Operation *(*)(OperandType &)>(
            it, &unwrap) {}
  Operation *operator->() { return **this; }
};

} // namespace mlir

#endif
