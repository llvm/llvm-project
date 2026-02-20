//===--- Value.h - Value Representation for llubi ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_VALUE_H
#define LLVM_TOOLS_LLUBI_VALUE_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::ubi {

class MemoryObject;
class Context;
class AnyValue;

enum class ByteKind : uint8_t {
  // A concrete byte with a known value.
  Concrete,
  // A uninitialized byte. Each load from an uninitialized byte yields
  // a nondeterministic value.
  Undef,
  // A poisoned byte. It occurs when the program stores a poison value to
  // memory,
  // or when a memory object is dead.
  Poison,
};

struct Byte {
  uint8_t Value;
  ByteKind Kind : 2;
  // TODO: provenance

  void set(uint8_t V) {
    Value = V;
    Kind = ByteKind::Concrete;
  }
};

// TODO: Byte
enum class StorageKind {
  Integer,
  Float,
  Pointer,
  Poison,
  None,      // Placeholder for void type
  Aggregate, // Struct, Array or Vector
};

/// Tri-state boolean value.
enum class BooleanKind { False, True, Poison };

class Pointer {
  // The underlying memory object. It can be null for invalid or dangling
  // pointers.
  IntrusiveRefCntPtr<MemoryObject> Obj;
  // The address of the pointer. The bit width is determined by
  // DataLayout::getPointerSizeInBits.
  APInt Address;
  // TODO: modeling inrange(Start, End) attribute

public:
  explicit Pointer(const APInt &Address) : Obj(nullptr), Address(Address) {}
  explicit Pointer(IntrusiveRefCntPtr<MemoryObject> Obj, const APInt &Address)
      : Obj(std::move(Obj)), Address(Address) {}
  Pointer getWithNewAddr(const APInt &NewAddr) const {
    return Pointer(Obj, NewAddr);
  }
  static AnyValue null(unsigned BitWidth);
  void print(raw_ostream &OS) const;
  const APInt &address() const { return Address; }
  MemoryObject *getMemoryObject() const { return Obj.get(); }
};

// Value representation for actual values of LLVM values.
// We don't model undef values here (except for byte types).
class [[nodiscard]] AnyValue {
  StorageKind Kind;
  union {
    APInt IntVal;
    APFloat FloatVal;
    Pointer PtrVal;
    std::vector<AnyValue> AggVal;
  };

  struct PoisonTag {};
  void destroy();

public:
  AnyValue() : Kind(StorageKind::None) {}
  explicit AnyValue(PoisonTag) : Kind(StorageKind::Poison) {}
  AnyValue(APInt Val) : Kind(StorageKind::Integer), IntVal(std::move(Val)) {}
  AnyValue(APFloat Val) : Kind(StorageKind::Float), FloatVal(std::move(Val)) {}
  AnyValue(Pointer Val) : Kind(StorageKind::Pointer), PtrVal(std::move(Val)) {}
  AnyValue(std::vector<AnyValue> Val)
      : Kind(StorageKind::Aggregate), AggVal(std::move(Val)) {}
  AnyValue(const AnyValue &Other);
  AnyValue(AnyValue &&Other);
  AnyValue &operator=(const AnyValue &);
  AnyValue &operator=(AnyValue &&);
  ~AnyValue() { destroy(); }

  void print(raw_ostream &OS) const;

  static AnyValue poison() { return AnyValue(PoisonTag{}); }
  static AnyValue boolean(bool Val) { return AnyValue(APInt(1, Val)); }
  static AnyValue getPoisonValue(Context &Ctx, Type *Ty);
  static AnyValue getNullValue(Context &Ctx, Type *Ty);
  static AnyValue getVectorSplat(const AnyValue &Scalar, size_t NumElements);

  bool isNone() const { return Kind == StorageKind::None; }
  bool isPoison() const { return Kind == StorageKind::Poison; }
  bool isInteger() const { return Kind == StorageKind::Integer; }
  bool isFloat() const { return Kind == StorageKind::Float; }
  bool isPointer() const { return Kind == StorageKind::Pointer; }
  bool isAggregate() const { return Kind == StorageKind::Aggregate; }

  const APInt &asInteger() const {
    assert(Kind == StorageKind::Integer && "Expect an integer value");
    return IntVal;
  }

  const APFloat &asFloat() const {
    assert(Kind == StorageKind::Float && "Expect a float value");
    return FloatVal;
  }

  const Pointer &asPointer() const {
    assert(Kind == StorageKind::Pointer && "Expect a pointer value");
    return PtrVal;
  }

  const std::vector<AnyValue> &asAggregate() const {
    assert(Kind == StorageKind::Aggregate &&
           "Expect an aggregate/vector value");
    return AggVal;
  }

  std::vector<AnyValue> &asAggregate() {
    assert(Kind == StorageKind::Aggregate &&
           "Expect an aggregate/vector value");
    return AggVal;
  }

  // Helper function for C++ 17 structured bindings.
  template <size_t I> const AnyValue &get() const {
    assert(Kind == StorageKind::Aggregate &&
           "Expect an aggregate/vector value");
    assert(I < AggVal.size() && "Index out of bounds");
    return AggVal[I];
  }

  BooleanKind asBoolean() const {
    if (isPoison())
      return BooleanKind::Poison;
    return asInteger().isZero() ? BooleanKind::False : BooleanKind::True;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const AnyValue &V) {
  V.print(OS);
  return OS;
}

} // namespace llvm::ubi

#endif
