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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::ubi {

class MemoryObject;
class Context;
class AnyValue;

/// Representation of a byte in memory.
/// How to interpret the byte per bit:
/// - If the concrete mask bit is 0, the bit is either undef or poison. The
/// value bit indicates whether it is undef.
/// - If the concrete mask bit is 1, the bit is a concrete value. The value bit
/// stores the concrete bit value. The tag mask bit indicates whether it is a
/// pointer bit, and the tag value bit is used for provenance tracking of
/// pointers.
struct Byte {
  uint8_t ConcreteMask;
  uint8_t Value;
  uint8_t TagMask;  // A mask to indicate which bits are pointer bits.
  uint8_t TagValue; // Part of the tag for provenance tracking of pointers.

  static Byte poison() { return Byte{0, 0, 0, 0}; }
  static Byte undef() { return Byte{0, 255, 0, 0}; }
  static Byte concrete(uint8_t Val) { return Byte{255, Val, 0, 0}; }

  void zeroBits(uint8_t Mask) {
    ConcreteMask |= Mask;
    Value &= ~Mask;
  }

  void poisonBits(uint8_t Mask) {
    ConcreteMask &= ~Mask;
    Value &= ~Mask;
  }

  void undefBits(uint8_t Mask) {
    ConcreteMask &= ~Mask;
    Value |= Mask;
  }

  void writeBits(uint8_t Mask, uint8_t Val) {
    ConcreteMask |= Mask;
    Value = (Value & ~Mask) | (Val & Mask);
    TagMask &= ~Mask;
  }

  void writeTagBits(uint8_t Mask, uint8_t Tag) {
    assert(
        (ConcreteMask & Mask) == Mask &&
        "Please ensure pointer bits are concrete before calling writeTagBits.");
    TagMask |= Mask;
    TagValue = (TagValue & ~Mask) | (Tag & Mask);
  }

  /// Returns a logical byte that is part of two adjacent bytes.
  /// Example with ShAmt = 5:
  ///     |       Low       |      High       |
  /// LSB | 0 1 0 1 0 1 0 1 | 0 0 0 0 1 1 1 1 | MSB
  ///     Result =  | 1 0 1   0 0 0 0 1 |
  static Byte fshr(const Byte &Low, const Byte &High, uint32_t ShAmt) {
    return Byte{
        static_cast<uint8_t>((Low.ConcreteMask | (High.ConcreteMask << 8)) >>
                             ShAmt),
        static_cast<uint8_t>((Low.Value | (High.Value << 8)) >> ShAmt),
        static_cast<uint8_t>((Low.TagMask | (High.TagMask << 8)) >> ShAmt),
        static_cast<uint8_t>((Low.TagValue | (High.TagValue << 8)) >> ShAmt)};
  }

  Byte lshr(uint8_t Shift) const {
    return Byte{static_cast<uint8_t>(ConcreteMask >> Shift),
                static_cast<uint8_t>(Value >> Shift),
                static_cast<uint8_t>(TagMask >> Shift),
                static_cast<uint8_t>(TagValue >> Shift)};
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

/// Components of a pointer excluding address. They are shared between pointer
/// values, as most of operations don't change the provenance.
/// Each node will be assigned a unique, pointer-sized tag, which is used to
/// represent the pointer in the memory.
class Provenance : public RefCountedBase<Provenance> {
  // TODO: store reference to the provenance of the pointer it is derived from

  // The underlying memory object. It can be null for invalid or dangling
  // pointers.
  IntrusiveRefCntPtr<MemoryObject> Obj;

  // A tag is a randomly generated unique identifier to recover the provenance
  // of a pointer. The length of tag is equal to the store size of the pointer
  // type, in bits. It may produce false negatives in some corner cases. But in
  // real practice the false negative rate should be negligible.
  // A zero tag is invalid.
  // TODO: we need a special tag for wildcard provenance, which is introduced by
  // inttoptr.
  APInt Tag;

  // TODO: modeling nofree
  // TODO: modeling captures
  // TODO: modeling inrange(Start, End) attribute

  const APInt &getTag() const { return Tag; }
  void setTag(const APInt &T) { Tag = T; }

  friend class Context;

public:
  Provenance(IntrusiveRefCntPtr<MemoryObject> Obj) : Obj(std::move(Obj)) {}
  static IntrusiveRefCntPtr<Provenance> nullary();
  MemoryObject *getMemoryObject() const { return Obj.get(); }
};

class Pointer {
  // The provenance of the pointer.
  IntrusiveRefCntPtr<Provenance> Prov;
  // The address of the pointer. The bit width is determined by
  // DataLayout::getPointerSizeInBits.
  APInt Address;

public:
  explicit Pointer(const APInt &Address)
      : Prov(Provenance::nullary()), Address(Address) {}
  explicit Pointer(IntrusiveRefCntPtr<Provenance> Prov, const APInt &Address)
      : Prov(std::move(Prov)), Address(Address) {
    assert(this->Prov && "Invalid provenance.");
  }
  Pointer getWithNewAddr(const APInt &NewAddr) const {
    return Pointer(Prov, NewAddr);
  }
  static AnyValue null(unsigned AS, const DataLayout &DL);
  bool isNullPtr(unsigned AS, const DataLayout &DL) const;
  void print(raw_ostream &OS) const;
  const APInt &address() const { return Address; }
  Provenance &provenance() const { return *Prov; }
  MemoryObject *getMemoryObject() const { return Prov->getMemoryObject(); }
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

  bool isCompatibleWith(Type *Ty) const {
    switch (Kind) {
    case StorageKind::None:
      return Ty->isVoidTy();
    case StorageKind::Poison:
      return Ty->isFloatingPointTy() || Ty->isIntegerTy() || Ty->isPointerTy();
    case StorageKind::Integer:
      return Ty->isIntegerTy();
    case StorageKind::Float:
      return Ty->isFloatingPointTy();
    case StorageKind::Pointer:
      return Ty->isPointerTy();
    // We don't check elements recursively.
    case StorageKind::Aggregate:
      return Ty->isAggregateType() || Ty->isVectorTy();
    }
    llvm_unreachable("Unhandled storage kind.");
  }

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

inline raw_ostream &operator<<(raw_ostream &OS, const Pointer &P) {
  P.print(OS);
  return OS;
}

} // namespace llvm::ubi

#endif
