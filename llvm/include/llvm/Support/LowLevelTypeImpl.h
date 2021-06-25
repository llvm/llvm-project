//== llvm/Support/LowLevelTypeImpl.h --------------------------- -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Implement a low-level type suitable for MachineInstr level instruction
/// selection.
///
/// For a type attached to a MachineInstr, we only care about 2 details: total
/// size and the number of vector lanes (if any). Accordingly, there are 4
/// possible valid type-kinds:
///
///    * `sN` for scalars and aggregates
///    * `<N x sM>` for vectors, which must have at least 2 elements.
///    * `pN` for pointers
///
/// Other information required for correct selection is expected to be carried
/// by the opcode, or non-type flags. For example the distinction between G_ADD
/// and G_FADD for int/float or fast-math flags.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LOWLEVELTYPEIMPL_H
#define LLVM_SUPPORT_LOWLEVELTYPEIMPL_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MachineValueType.h"
#include <cassert>

namespace llvm {

class DataLayout;
class Type;
class raw_ostream;

class LLT {
public:
  /// Get a low-level scalar or aggregate "bag of bits".
  static LLT scalar(unsigned SizeInBits) {
    assert(SizeInBits > 0 && "invalid scalar size");
    return LLT{/*isPointer=*/false, /*isVector=*/false,
               ElementCount::getFixed(0), SizeInBits,
               /*AddressSpace=*/0};
  }

  /// Get a low-level pointer in the given address space.
  static LLT pointer(unsigned AddressSpace, unsigned SizeInBits) {
    assert(SizeInBits > 0 && "invalid pointer size");
    return LLT{/*isPointer=*/true, /*isVector=*/false,
               ElementCount::getFixed(0), SizeInBits, AddressSpace};
  }

  /// Get a low-level vector of some number of elements and element width.
  static LLT vector(ElementCount EC, unsigned ScalarSizeInBits) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    assert(ScalarSizeInBits > 0 && "invalid vector element size");
    return LLT{/*isPointer=*/false, /*isVector=*/true, EC, ScalarSizeInBits,
               /*AddressSpace=*/0};
  }

  /// Get a low-level vector of some number of elements and element type.
  static LLT vector(ElementCount EC, LLT ScalarTy) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    assert(!ScalarTy.isVector() && "invalid vector element type");
    return LLT{ScalarTy.isPointer(), /*isVector=*/true, EC,
               ScalarTy.getSizeInBits(),
               ScalarTy.isPointer() ? ScalarTy.getAddressSpace() : 0};
  }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// width.
  static LLT fixed_vector(unsigned NumElements, unsigned ScalarSizeInBits) {
    return vector(ElementCount::getFixed(NumElements), ScalarSizeInBits);
  }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// type.
  static LLT fixed_vector(unsigned NumElements, LLT ScalarTy) {
    return vector(ElementCount::getFixed(NumElements), ScalarTy);
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// width.
  static LLT scalable_vector(unsigned MinNumElements,
                             unsigned ScalarSizeInBits) {
    return vector(ElementCount::getScalable(MinNumElements), ScalarSizeInBits);
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// type.
  static LLT scalable_vector(unsigned MinNumElements, LLT ScalarTy) {
    return vector(ElementCount::getScalable(MinNumElements), ScalarTy);
  }

  static LLT scalarOrVector(ElementCount EC, LLT ScalarTy) {
    return EC.isScalar() ? ScalarTy : LLT::vector(EC, ScalarTy);
  }

  static LLT scalarOrVector(ElementCount EC, unsigned ScalarSize) {
    return scalarOrVector(EC, LLT::scalar(ScalarSize));
  }

  explicit LLT(bool isPointer, bool isVector, ElementCount EC,
               unsigned SizeInBits, unsigned AddressSpace) {
    init(isPointer, isVector, EC, SizeInBits, AddressSpace);
  }
  explicit LLT() : IsPointer(false), IsVector(false), RawData(0) {}

  explicit LLT(MVT VT);

  bool isValid() const { return RawData != 0; }

  bool isScalar() const { return isValid() && !IsPointer && !IsVector; }

  bool isPointer() const { return isValid() && IsPointer && !IsVector; }

  bool isVector() const { return isValid() && IsVector; }

  /// Returns the number of elements in a vector LLT. Must only be called on
  /// vector types.
  uint16_t getNumElements() const {
    if (isScalable())
      llvm::reportInvalidSizeRequest(
          "Possible incorrect use of LLT::getNumElements() for "
          "scalable vector. Scalable flag may be dropped, use "
          "LLT::getElementCount() instead");
    return getElementCount().getKnownMinValue();
  }

  /// Returns true if the LLT is a scalable vector. Must only be called on
  /// vector types.
  bool isScalable() const {
    assert(isVector() && "Expected a vector type");
    return IsPointer ? getFieldValue(PointerVectorScalableFieldInfo)
                     : getFieldValue(VectorScalableFieldInfo);
  }

  ElementCount getElementCount() const {
    assert(IsVector && "cannot get number of elements on scalar/aggregate");
    return ElementCount::get(IsPointer
                                 ? getFieldValue(PointerVectorElementsFieldInfo)
                                 : getFieldValue(VectorElementsFieldInfo),
                             isScalable());
  }

  /// Returns the total size of the type. Must only be called on sized types.
  unsigned getSizeInBits() const {
    if (isPointer() || isScalar())
      return getScalarSizeInBits();
    // FIXME: This should return a TypeSize in order to work for scalable
    // vectors.
    return getScalarSizeInBits() * getElementCount().getKnownMinValue();
  }

  /// Returns the total size of the type in bytes, i.e. number of whole bytes
  /// needed to represent the size in bits. Must only be called on sized types.
  unsigned getSizeInBytes() const {
    return (getSizeInBits() + 7) / 8;
  }

  LLT getScalarType() const {
    return isVector() ? getElementType() : *this;
  }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element type. Otherwise, return the new element type.
  LLT changeElementType(LLT NewEltTy) const {
    return isVector() ? LLT::vector(getElementCount(), NewEltTy) : NewEltTy;
  }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element size. Otherwise, return the new element type. Invalid
  /// for pointer types. For pointer types, use changeElementType.
  LLT changeElementSize(unsigned NewEltSize) const {
    assert(!getScalarType().isPointer() &&
           "invalid to directly change element size for pointers");
    return isVector() ? LLT::vector(getElementCount(), NewEltSize)
                      : LLT::scalar(NewEltSize);
  }

  /// Return a vector or scalar with the same element type and the new element
  /// count.
  LLT changeElementCount(ElementCount EC) const {
    return LLT::scalarOrVector(EC, getScalarType());
  }

  /// Return a type that is \p Factor times smaller. Reduces the number of
  /// elements if this is a vector, or the bitwidth for scalar/pointers. Does
  /// not attempt to handle cases that aren't evenly divisible.
  LLT divide(int Factor) const {
    assert(Factor != 1);
    if (isVector()) {
      assert(getElementCount().isKnownMultipleOf(Factor));
      return scalarOrVector(getElementCount().divideCoefficientBy(Factor),
                            getElementType());
    }

    assert(getSizeInBits() % Factor == 0);
    return scalar(getSizeInBits() / Factor);
  }

  bool isByteSized() const { return (getSizeInBits() & 7) == 0; }

  unsigned getScalarSizeInBits() const {
    assert(RawData != 0 && "Invalid Type");
    if (!IsVector) {
      if (!IsPointer)
        return getFieldValue(ScalarSizeFieldInfo);
      else
        return getFieldValue(PointerSizeFieldInfo);
    } else {
      if (!IsPointer)
        return getFieldValue(VectorSizeFieldInfo);
      else
        return getFieldValue(PointerVectorSizeFieldInfo);
    }
  }

  unsigned getAddressSpace() const {
    assert(RawData != 0 && "Invalid Type");
    assert(IsPointer && "cannot get address space of non-pointer type");
    if (!IsVector)
      return getFieldValue(PointerAddressSpaceFieldInfo);
    else
      return getFieldValue(PointerVectorAddressSpaceFieldInfo);
  }

  /// Returns the vector's element type. Only valid for vector types.
  LLT getElementType() const {
    assert(isVector() && "cannot get element type of scalar/aggregate");
    if (IsPointer)
      return pointer(getAddressSpace(), getScalarSizeInBits());
    else
      return scalar(getScalarSizeInBits());
  }

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << '\n';
  }
#endif

  bool operator==(const LLT &RHS) const {
    return IsPointer == RHS.IsPointer && IsVector == RHS.IsVector &&
           RHS.RawData == RawData;
  }

  bool operator!=(const LLT &RHS) const { return !(*this == RHS); }

  friend struct DenseMapInfo<LLT>;
  friend class GISelInstProfileBuilder;

private:
  /// LLT is packed into 64 bits as follows:
  /// isPointer : 1
  /// isVector  : 1
  /// with 62 bits remaining for Kind-specific data, packed in bitfields
  /// as described below. As there isn't a simple portable way to pack bits
  /// into bitfields, here the different fields in the packed structure is
  /// described in static const *Field variables. Each of these variables
  /// is a 2-element array, with the first element describing the bitfield size
  /// and the second element describing the bitfield offset.
  typedef int BitFieldInfo[2];
  ///
  /// This is how the bitfields are packed per Kind:
  /// * Invalid:
  ///   gets encoded as RawData == 0, as that is an invalid encoding, since for
  ///   valid encodings, SizeInBits/SizeOfElement must be larger than 0.
  /// * Non-pointer scalar (isPointer == 0 && isVector == 0):
  ///   SizeInBits: 32;
  static const constexpr BitFieldInfo ScalarSizeFieldInfo{32, 0};
  /// * Pointer (isPointer == 1 && isVector == 0):
  ///   SizeInBits: 16;
  ///   AddressSpace: 24;
  static const constexpr BitFieldInfo PointerSizeFieldInfo{16, 0};
  static const constexpr BitFieldInfo PointerAddressSpaceFieldInfo{
      24, PointerSizeFieldInfo[0] + PointerSizeFieldInfo[1]};
  static_assert((PointerAddressSpaceFieldInfo[0] +
                 PointerAddressSpaceFieldInfo[1]) <= 62,
                "Insufficient bits to encode all data");
  /// * Vector-of-non-pointer (isPointer == 0 && isVector == 1):
  ///   NumElements: 16;
  ///   SizeOfElement: 32;
  ///   Scalable: 1;
  static const constexpr BitFieldInfo VectorElementsFieldInfo{16, 0};
  static const constexpr BitFieldInfo VectorSizeFieldInfo{
      32, VectorElementsFieldInfo[0] + VectorElementsFieldInfo[1]};
  static const constexpr BitFieldInfo VectorScalableFieldInfo{
      1, VectorSizeFieldInfo[0] + VectorSizeFieldInfo[1]};
  static_assert((VectorSizeFieldInfo[0] + VectorSizeFieldInfo[1]) <= 62,
                "Insufficient bits to encode all data");
  /// * Vector-of-pointer (isPointer == 1 && isVector == 1):
  ///   NumElements: 16;
  ///   SizeOfElement: 16;
  ///   AddressSpace: 24;
  ///   Scalable: 1;
  static const constexpr BitFieldInfo PointerVectorElementsFieldInfo{16, 0};
  static const constexpr BitFieldInfo PointerVectorSizeFieldInfo{
      16,
      PointerVectorElementsFieldInfo[1] + PointerVectorElementsFieldInfo[0]};
  static const constexpr BitFieldInfo PointerVectorAddressSpaceFieldInfo{
      24, PointerVectorSizeFieldInfo[1] + PointerVectorSizeFieldInfo[0]};
  static const constexpr BitFieldInfo PointerVectorScalableFieldInfo{
      1, PointerVectorAddressSpaceFieldInfo[0] +
             PointerVectorAddressSpaceFieldInfo[1]};
  static_assert((PointerVectorAddressSpaceFieldInfo[0] +
                 PointerVectorAddressSpaceFieldInfo[1]) <= 62,
                "Insufficient bits to encode all data");

  uint64_t IsPointer : 1;
  uint64_t IsVector : 1;
  uint64_t RawData : 62;

  static uint64_t getMask(const BitFieldInfo FieldInfo) {
    const int FieldSizeInBits = FieldInfo[0];
    return (((uint64_t)1) << FieldSizeInBits) - 1;
  }
  static uint64_t maskAndShift(uint64_t Val, uint64_t Mask, uint8_t Shift) {
    assert(Val <= Mask && "Value too large for field");
    return (Val & Mask) << Shift;
  }
  static uint64_t maskAndShift(uint64_t Val, const BitFieldInfo FieldInfo) {
    return maskAndShift(Val, getMask(FieldInfo), FieldInfo[1]);
  }
  uint64_t getFieldValue(const BitFieldInfo FieldInfo) const {
    return getMask(FieldInfo) & (RawData >> FieldInfo[1]);
  }

  void init(bool IsPointer, bool IsVector, ElementCount EC, unsigned SizeInBits,
            unsigned AddressSpace) {
    this->IsPointer = IsPointer;
    this->IsVector = IsVector;
    if (!IsVector) {
      if (!IsPointer)
        RawData = maskAndShift(SizeInBits, ScalarSizeFieldInfo);
      else
        RawData = maskAndShift(SizeInBits, PointerSizeFieldInfo) |
                  maskAndShift(AddressSpace, PointerAddressSpaceFieldInfo);
    } else {
      assert(EC.isVector() && "invalid number of vector elements");
      if (!IsPointer)
        RawData =
            maskAndShift(EC.getKnownMinValue(), VectorElementsFieldInfo) |
            maskAndShift(SizeInBits, VectorSizeFieldInfo) |
            maskAndShift(EC.isScalable() ? 1 : 0, VectorScalableFieldInfo);
      else
        RawData =
            maskAndShift(EC.getKnownMinValue(),
                         PointerVectorElementsFieldInfo) |
            maskAndShift(SizeInBits, PointerVectorSizeFieldInfo) |
            maskAndShift(AddressSpace, PointerVectorAddressSpaceFieldInfo) |
            maskAndShift(EC.isScalable() ? 1 : 0,
                         PointerVectorScalableFieldInfo);
    }
  }

  uint64_t getUniqueRAWLLTData() const {
    return ((uint64_t)RawData) << 2 | ((uint64_t)IsPointer) << 1 |
           ((uint64_t)IsVector);
  }
};

inline raw_ostream& operator<<(raw_ostream &OS, const LLT &Ty) {
  Ty.print(OS);
  return OS;
}

template<> struct DenseMapInfo<LLT> {
  static inline LLT getEmptyKey() {
    LLT Invalid;
    Invalid.IsPointer = true;
    return Invalid;
  }
  static inline LLT getTombstoneKey() {
    LLT Invalid;
    Invalid.IsVector = true;
    return Invalid;
  }
  static inline unsigned getHashValue(const LLT &Ty) {
    uint64_t Val = Ty.getUniqueRAWLLTData();
    return DenseMapInfo<uint64_t>::getHashValue(Val);
  }
  static bool isEqual(const LLT &LHS, const LLT &RHS) {
    return LHS == RHS;
  }
};

}

#endif // LLVM_SUPPORT_LOWLEVELTYPEIMPL_H
