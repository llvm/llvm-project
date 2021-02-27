//===- Value.h - Base of the SSA Value hierarchy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic Value type and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VALUE_H
#define MLIR_IR_VALUE_H

#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class AsmState;
class BlockArgument;
class Operation;
class OpResult;
class Region;
class Value;

namespace detail {
/// The internal implementation of a BlockArgument.
class BlockArgumentImpl;
} // end namespace detail

/// This class represents an instance of an SSA value in the MLIR system,
/// representing a computable value that has a type and a set of users. An SSA
/// value is either a BlockArgument or the result of an operation. Note: This
/// class has value-type semantics and is just a simple wrapper around a
/// ValueImpl that is either owner by a block(in the case of a BlockArgument) or
/// an Operation(in the case of an OpResult).
class Value {
public:
  /// The enumeration represents the various different kinds of values the
  /// internal representation may take. We steal 2 bits to support a total of 4
  /// possible values.
  enum class Kind {
    /// The first N kinds are all inline operation results. An inline operation
    /// result means that the kind represents the result number, and the owner
    /// pointer is the owning `Operation*`. Note: These are packed first to make
    /// result number lookups more efficient.
    OpResult0 = 0,
    OpResult1 = 1,

    /// The next kind represents a 'trailing' operation result. This is for
    /// results with numbers larger than we can represent inline. The owner here
    /// is an `TrailingOpResult*` that points to a trailing storage on the
    /// parent operation.
    TrailingOpResult = 2,

    /// The last kind represents a block argument. The owner here is a
    /// `BlockArgumentImpl*`.
    BlockArgument = 3
  };

  /// This value represents the 'owner' of the value and its kind. See the
  /// 'Kind' enumeration above for a more detailed description of each kind of
  /// owner.
  struct ImplTypeTraits : public llvm::PointerLikeTypeTraits<void *> {
    // We know that all pointers within the ImplType are aligned by 8-bytes,
    // meaning that we can steal up to 3 bits for the different values.
    static constexpr int NumLowBitsAvailable = 3;
  };
  using ImplType = llvm::PointerIntPair<void *, 2, Kind, ImplTypeTraits>;

public:
  constexpr Value(std::nullptr_t) : ownerAndKind() {}
  Value(ImplType ownerAndKind = {}) : ownerAndKind(ownerAndKind) {}
  Value(const Value &) = default;
  Value &operator=(const Value &) = default;

  template <typename U> bool isa() const {
    assert(*this && "isa<> used on a null type.");
    return U::classof(*this);
  }

  template <typename First, typename Second, typename... Rest>
  bool isa() const {
    return isa<First>() || isa<Second, Rest...>();
  }

  template <typename U> U dyn_cast() const {
    return isa<U>() ? U(ownerAndKind) : U(nullptr);
  }
  template <typename U> U dyn_cast_or_null() const {
    return (*this && isa<U>()) ? U(ownerAndKind) : U(nullptr);
  }
  template <typename U> U cast() const {
    assert(isa<U>());
    return U(ownerAndKind);
  }

  explicit operator bool() const { return ownerAndKind.getPointer(); }
  bool operator==(const Value &other) const {
    return ownerAndKind == other.ownerAndKind;
  }
  bool operator!=(const Value &other) const { return !(*this == other); }

  /// Return the type of this value.
  Type getType() const;

  /// Utility to get the associated MLIRContext that this value is defined in.
  MLIRContext *getContext() const { return getType().getContext(); }

  /// Mutate the type of this Value to be of the specified type.
  ///
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void setType(Type newType);

  /// If this value is the result of an operation, return the operation that
  /// defines it.
  Operation *getDefiningOp() const;

  /// If this value is the result of an operation of type OpTy, return the
  /// operation that defines it.
  template <typename OpTy>
  OpTy getDefiningOp() const {
    return llvm::dyn_cast_or_null<OpTy>(getDefiningOp());
  }

  /// If this value is the result of an operation, use it as a location,
  /// otherwise return an unknown location.
  Location getLoc() const;

  /// Return the Region in which this Value is defined.
  Region *getParentRegion();

  /// Return the Block in which this Value is defined.
  Block *getParentBlock();

  //===--------------------------------------------------------------------===//
  // UseLists
  //===--------------------------------------------------------------------===//

  /// Provide the use list that is attached to this value.
  IRObjectWithUseList<OpOperand> *getUseList() const;

  /// Drop all uses of this object from their respective owners.
  void dropAllUses() const;

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(Value newValue) const;

  /// Replace all uses of 'this' value with 'newValue', updating anything in the
  /// IR that uses 'this' to use the other value instead except if the user is
  /// listed in 'exceptions' .
  void
  replaceAllUsesExcept(Value newValue,
                       const SmallPtrSetImpl<Operation *> &exceptions) const;

  /// Replace all uses of 'this' value with 'newValue' if the given callback
  /// returns true.
  void replaceUsesWithIf(Value newValue,
                         function_ref<bool(OpOperand &)> shouldReplace);

  /// Returns true if the value is used outside of the given block.
  bool isUsedOutsideOfBlock(Block *block);

  //===--------------------------------------------------------------------===//
  // Uses

  /// This class implements an iterator over the uses of a value.
  using use_iterator = ValueUseIterator<OpOperand>;
  using use_range = iterator_range<use_iterator>;

  use_iterator use_begin() const;
  use_iterator use_end() const { return use_iterator(); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() const { return {use_begin(), use_end()}; }

  /// Returns true if this value has exactly one use.
  bool hasOneUse() const;

  /// Returns true if this value has no uses.
  bool use_empty() const;

  //===--------------------------------------------------------------------===//
  // Users

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() const { return use_begin(); }
  user_iterator user_end() const { return use_end(); }
  user_range getUsers() const { return {user_begin(), user_end()}; }

  //===--------------------------------------------------------------------===//
  // Utilities

  /// Returns the kind of this value.
  Kind getKind() const { return ownerAndKind.getInt(); }

  void print(raw_ostream &os);
  void print(raw_ostream &os, AsmState &state);
  void dump();

  /// Print this value as if it were an operand.
  void printAsOperand(raw_ostream &os, AsmState &state);

  /// Methods for supporting PointerLikeTypeTraits.
  void *getAsOpaquePointer() const { return ownerAndKind.getOpaqueValue(); }
  static Value getFromOpaquePointer(const void *pointer) {
    Value value;
    value.ownerAndKind.setFromOpaqueValue(const_cast<void *>(pointer));
    return value;
  }

  friend ::llvm::hash_code hash_value(Value arg);

protected:
  /// Returns true if the given operation result can be packed inline.
  static bool canPackResultInline(unsigned resultNo) {
    return resultNo < static_cast<unsigned>(Kind::TrailingOpResult);
  }

  /// Construct a value.
  Value(detail::BlockArgumentImpl *impl);
  Value(Operation *op, unsigned resultNo);

  /// This value represents the 'owner' of the value and its kind. See the
  /// 'Kind' enumeration above for a more detailed description of each kind of
  /// owner.
  ImplType ownerAndKind;
};

inline raw_ostream &operator<<(raw_ostream &os, Value value) {
  value.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// BlockArgument
//===----------------------------------------------------------------------===//

namespace detail {
/// The internal implementation of a BlockArgument.
class BlockArgumentImpl : public IRObjectWithUseList<OpOperand> {
  BlockArgumentImpl(Type type, Block *owner, int64_t index)
      : type(type), owner(owner), index(index) {}

  /// The type of this argument.
  Type type;

  /// The owner of this argument.
  Block *owner;

  /// The position in the argument list.
  int64_t index;

  /// Allow access to owner and constructor.
  friend BlockArgument;
};
} // end namespace detail

/// Block arguments are values.
class BlockArgument : public Value {
public:
  using Value::Value;

  static bool classof(Value value) {
    return value.getKind() == Kind::BlockArgument;
  }

  /// Returns the block that owns this argument.
  Block *getOwner() const { return getImpl()->owner; }

  /// Return the type of this value.
  Type getType() const { return getImpl()->type; }

  /// Set the type of this value.
  void setType(Type newType) { getImpl()->type = newType; }

  /// Returns the number of this argument.
  unsigned getArgNumber() const { return getImpl()->index; }

private:
  /// Allocate a new argument with the given type and owner.
  static BlockArgument create(Type type, Block *owner, int64_t index) {
    return new detail::BlockArgumentImpl(type, owner, index);
  }

  /// Destroy and deallocate this argument.
  void destroy() { delete getImpl(); }

  /// Get a raw pointer to the internal implementation.
  detail::BlockArgumentImpl *getImpl() const {
    return reinterpret_cast<detail::BlockArgumentImpl *>(
        ownerAndKind.getPointer());
  }

  /// Cache the position in the block argument list.
  void setArgNumber(int64_t index) { getImpl()->index = index; }

  /// Allow access to `create`, `destroy` and `setArgNumber`.
  friend Block;

  /// Allow access to 'getImpl'.
  friend Value;
};

//===----------------------------------------------------------------------===//
// OpResult
//===----------------------------------------------------------------------===//

/// This is a value defined by a result of an operation.
class OpResult : public Value {
public:
  using Value::Value;

  static bool classof(Value value) {
    return value.getKind() != Kind::BlockArgument;
  }

  /// Returns the operation that owns this result.
  Operation *getOwner() const;

  /// Returns the number of this result.
  unsigned getResultNumber() const;

  /// Returns the maximum number of results that can be stored inline.
  static unsigned getMaxInlineResults() {
    return static_cast<unsigned>(Kind::TrailingOpResult);
  }

private:
  /// Given a number of operation results, returns the number that need to be
  /// stored inline.
  static unsigned getNumInline(unsigned numResults);

  /// Given a number of operation results, returns the number that need to be
  /// stored as trailing.
  static unsigned getNumTrailing(unsigned numResults);

  /// Allow access to constructor.
  friend Operation;
};

/// Make Value hashable.
inline ::llvm::hash_code hash_value(Value arg) {
  return ::llvm::hash_value(arg.ownerAndKind.getOpaqueValue());
}

} // namespace mlir

namespace llvm {

template <> struct DenseMapInfo<mlir::Value> {
  static mlir::Value getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Value::getFromOpaquePointer(pointer);
  }
  static mlir::Value getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Value::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::Value val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Value lhs, mlir::Value rhs) { return lhs == rhs; }
};

/// Allow stealing the low bits of a value.
template <> struct PointerLikeTypeTraits<mlir::Value> {
public:
  static inline void *getAsVoidPointer(mlir::Value I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Value getFromVoidPointer(void *P) {
    return mlir::Value::getFromOpaquePointer(P);
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value::ImplType>::NumLowBitsAvailable
  };
};

template <> struct DenseMapInfo<mlir::BlockArgument> {
  static mlir::BlockArgument getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::BlockArgument(
        mlir::Value::ImplType::getFromOpaqueValue(pointer));
  }
  static mlir::BlockArgument getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::BlockArgument(
        mlir::Value::ImplType::getFromOpaqueValue(pointer));
  }
  static unsigned getHashValue(mlir::BlockArgument val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::BlockArgument LHS, mlir::BlockArgument RHS) {
    return LHS == RHS;
  }
};

/// Allow stealing the low bits of a value.
template <> struct PointerLikeTypeTraits<mlir::BlockArgument> {
public:
  static inline void *getAsVoidPointer(mlir::Value I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::BlockArgument getFromVoidPointer(void *P) {
    return mlir::Value::getFromOpaquePointer(P).cast<mlir::BlockArgument>();
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};
} // end namespace llvm

#endif
