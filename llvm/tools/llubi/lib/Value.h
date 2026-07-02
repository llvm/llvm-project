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
///
/// Note that the idealized interpreter would store a full pointer tag for every
/// single pointer bit, as well as the position of that bit in the pointer. The
/// provenance is preserved as the bit is copied around, and when a sequence of
/// bytes is eventually converted back to a pointer, all bits must be in the
/// original order and have the same provenance. However, that would be
/// prohibitively expensive. So instead, we rely on randomized ptr-sized tags.
/// This means that if bits get reordered, or if bits from different pointers
/// get mixed, then the result is unlikely to be a valid tag.
struct Byte {
  uint8_t ConcreteMask;
  uint8_t Value;
  uint8_t TagMask;  // A mask to indicate which bits are pointer bits.
  uint8_t TagValue; // For each pointer bit, the corresponding bit of the tag
                    // for provenance tracking.

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

/// A set of previously exposed provenances. It is originally yielded by
/// inttoptr, and shared by pointers derived from the result.
///
/// Each capability check may invalidate some provenances. If we cannot
/// pick one, it is UB. That is, from the angelic non-determinism view,
/// we cannot pick a provenance to make the program reach this point.
///
/// For efficiency, this class has different forms in two stages:
/// 1. Before any memory access is performed, ActiveMask is set to zero and
/// Generation represents the global generation number of the snapshot.
/// 2. After a memory access is performed, we can determine exactly one memory
/// object to be accessed (address ranges are distinct). In this case,
/// BaseAddress is set and ActiveMask is non-zero. ActiveMask represents the
/// validity of the first N exposed provenances associated with the memory
/// object. The bitwidth N is the number of provenances in the list with
/// List[I].Generation <= WildcardProvenance::Generation (The generation field
/// in the list is monotonically increasing). That is, we can only access
/// through exposed provenances before inttoptr executes. Note that if
/// ActiveMask becomes zero again, UB must be triggered.
class WildcardProvenance : public RefCountedBase<WildcardProvenance> {
  APInt ActiveMask;
  union {
    uint64_t Generation;
    uint64_t BaseAddress;
  };

  friend class Context;

public:
  explicit WildcardProvenance(uint64_t Generation)
      : ActiveMask(), Generation(Generation) {}
};

/// Components of a pointer excluding address. They are shared between pointer
/// values, as most of operations don't change the provenance.
/// Each node will be assigned a unique, pointer-sized tag, which is used to
/// represent the pointer in the memory.
/// The provenance can be either concrete or wildcard, as determined by the
/// cases below:
///  Obj        Wildcard      State
///  Null       Null          Invalid
///  Null       NonNull       Wildcard
///  NonNull    Null          Concrete
///  NonNull    NonNull       Wildcard (associated with a specific MO)
class Provenance : public RefCountedBase<Provenance> {
  // TODO: store reference to the provenance of the pointer it is derived from

  // The underlying memory object. It can be null for invalid or dangling
  // pointers. Besides, for pointers with wildcard provenance, it can be null
  // until the memory object is resolved by gep inbounds.
  IntrusiveRefCntPtr<MemoryObject> Obj;

  // A tag is a randomly generated unique identifier to recover the provenance
  // of a pointer. The length of tag is equal to the store size of the pointer
  // type, in bits. It may produce false negatives in some corner cases. But in
  // real practice the false negative rate should be negligible.
  // A zero tag is invalid.
  APInt Tag;

  // Null if it is concrete.
  IntrusiveRefCntPtr<WildcardProvenance> Wildcard;

  // TODO: modeling nofree
  // TODO: modeling captures
  // TODO: modeling inrange(Start, End) attribute

  const APInt &getTag() const { return Tag; }
  void setTag(const APInt &T) { Tag = T; }

  friend class Context;

public:
  Provenance(IntrusiveRefCntPtr<MemoryObject> Obj) : Obj(std::move(Obj)) {}
  static IntrusiveRefCntPtr<Provenance> nullary();
  IntrusiveRefCntPtr<Provenance> getWithKnownMemoryObject(MemoryObject &Obj);
  MemoryObject *getMemoryObject() const { return Obj.get(); }
  bool isWildcard() const { return Wildcard != nullptr; }
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
  Pointer getWithNewProvenance(IntrusiveRefCntPtr<Provenance> NewProv) const {
    return Pointer(NewProv, Address);
  }
  static AnyValue null(unsigned AS, const DataLayout &DL);
  bool isNullPtr(unsigned AS, const DataLayout &DL) const;
  void print(raw_ostream &OS) const;
  const APInt &address() const { return Address; }
  Provenance &provenance() const { return *Prov; }
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

inline AnyValue addNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  APInt Res = LHS + RHS;
  if (HasNUW && Res.ult(RHS))
    return AnyValue::poison();
  if (HasNSW && LHS.isNonNegative() == RHS.isNonNegative() &&
      LHS.isNonNegative() != Res.isNonNegative())
    return AnyValue::poison();
  return Res;
}

inline AnyValue subNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  APInt Res = LHS - RHS;
  if (HasNUW && Res.ugt(LHS))
    return AnyValue::poison();
  if (HasNSW && LHS.isNonNegative() != RHS.isNonNegative() &&
      LHS.isNonNegative() != Res.isNonNegative())
    return AnyValue::poison();
  return Res;
}

inline AnyValue mulNoWrap(const APInt &LHS, const APInt &RHS, bool HasNSW,
                          bool HasNUW) {
  bool Overflow = false;
  APInt Res = LHS.smul_ov(RHS, Overflow);
  if (HasNSW && Overflow)
    return AnyValue::poison();
  if (HasNUW) {
    (void)LHS.umul_ov(RHS, Overflow);
    if (Overflow)
      return AnyValue::poison();
  }
  return Res;
}

} // namespace llvm::ubi

#endif
