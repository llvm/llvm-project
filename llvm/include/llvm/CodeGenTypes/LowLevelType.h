//== llvm/CodeGenTypes/LowLevelType.h -------------------------- -*- C++ -*-==//
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

#ifndef LLVM_CODEGEN_LOWLEVELTYPE_H
#define LLVM_CODEGEN_LOWLEVELTYPE_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {

extern cl::opt<bool> EnableFPInfo;

class Type;
class raw_ostream;

class LLT {
public:
  enum class FPInfo {
    IEEE_FLOAT = 0x0,
    VARIANT_FLOAT_1 = 0x1,
    VARIANT_FLOAT_2 = 0x2,
    VARIANT_FLOAT_3 = 0x3,
  };

  enum class Kind : uint64_t {
    INVALID = 0b000,
    INTEGER = 0b001,
    FLOAT = 0b010,
    POINTER = 0b011,
    VECTOR_INTEGER = 0b101,
    VECTOR_FLOAT = 0b110,
    VECTOR_POINTER = 0b111,
  };

  constexpr static Kind toVector(Kind Ty) {
    if (Ty == Kind::POINTER)
      return Kind::VECTOR_POINTER;

    if (Ty == Kind::INTEGER)
      return Kind::VECTOR_INTEGER;

    if (Ty == Kind::FLOAT)
      return Kind::VECTOR_FLOAT;

    assert(false && "Type is already a vector type");
    return Ty;
  }

  constexpr static Kind toScalar(Kind Ty) {
    if (Ty == Kind::VECTOR_POINTER)
      return Kind::POINTER;

    if (Ty == Kind::VECTOR_INTEGER)
      return Kind::INTEGER;

    if (Ty == Kind::VECTOR_FLOAT)
      return Kind::FLOAT;

    assert(false && "Type is already a scalar type");
    return Ty;
  }

  /// Get a low-level scalar or aggregate "bag of bits".
  [[deprecated("Use LLT::integer(unsigned) instead.")]]
  static constexpr LLT scalar(unsigned SizeInBits) {
    return LLT{Kind::INTEGER, ElementCount::getFixed(0), SizeInBits,
               /*AddressSpace=*/0, static_cast<FPInfo>(0)};
  }

  static constexpr LLT integer(unsigned SizeInBits) {
    return LLT{Kind::INTEGER, ElementCount::getFixed(0), SizeInBits,
               /*AddressSpace=*/0, static_cast<FPInfo>(0)};
  }

  static constexpr LLT floatingPoint(unsigned SizeInBits, FPInfo FP) {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), SizeInBits,
               /*AddressSpace=*/0, FP};
  }

  /// Get a low-level token; just a scalar with zero bits (or no size).
  static constexpr LLT token() {
    return LLT{Kind::INTEGER, ElementCount::getFixed(0),
               /*SizeInBits=*/0,
               /*AddressSpace=*/0, static_cast<FPInfo>(0)};
  }

  /// Get a low-level pointer in the given address space.
  static constexpr LLT pointer(unsigned AddressSpace, unsigned SizeInBits) {
    assert(SizeInBits > 0 && "invalid pointer size");
    return LLT{Kind::POINTER, ElementCount::getFixed(0), SizeInBits,
               AddressSpace, static_cast<FPInfo>(0)};
  }

  /// Get a low-level vector of some number of elements and element width.
  [[deprecated("Use LLT::vector(EC, LLT) instead.")]]
  static constexpr LLT vector(ElementCount EC, unsigned ScalarSizeInBits) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    return LLT{Kind::VECTOR_INTEGER, EC, ScalarSizeInBits,
               /*AddressSpace=*/0, static_cast<FPInfo>(0)};
  }

  /// Get a low-level vector of some number of elements and element type.
  static constexpr LLT vector(ElementCount EC, LLT ScalarTy) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    assert(!ScalarTy.isVector() && "invalid vector element type");

    Kind Info = toVector(ScalarTy.Info);
    return LLT{Info, EC, ScalarTy.getSizeInBits().getFixedValue(),
               ScalarTy.isPointer() ? ScalarTy.getAddressSpace() : 0,
               ScalarTy.isFloat() ? ScalarTy.getFPInfo()
                                  : static_cast<FPInfo>(0)};
  }

  // Get a 16-bit brain float value.
  static constexpr LLT bfloat() { return floatingPoint(16, FPInfo::VARIANT_FLOAT_1); }

  /// Get a 16-bit IEEE half value.
  static constexpr LLT float16() { return floatingPoint(16, FPInfo::IEEE_FLOAT); }

  /// Get a 32-bit IEEE float value.
  static constexpr LLT float32() { return floatingPoint(32, FPInfo::IEEE_FLOAT); }

  /// Get a 64-bit IEEE double value.
  static constexpr LLT float64() { return floatingPoint(64, FPInfo::IEEE_FLOAT); }

  /// Get a 80-bit X86 floating point value.
  static constexpr LLT x86fp80() { return floatingPoint(80, FPInfo::VARIANT_FLOAT_1); }

  /// Get a 128-bit IEEE quad value.
  static constexpr LLT float128() { return floatingPoint(128, FPInfo::IEEE_FLOAT); }

  /// Get a 128-bit PowerPC double double value.
  static constexpr LLT ppcf128() { return floatingPoint(128, FPInfo::VARIANT_FLOAT_1); }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// width.
  [[deprecated("Use LLT::fixed_vector(unsigned, LLT) instead.")]]
  static constexpr LLT fixed_vector(unsigned NumElements,
                                    unsigned ScalarSizeInBits) {
    return vector(ElementCount::getFixed(NumElements),
                  LLT::integer(ScalarSizeInBits));
  }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// type.
  static constexpr LLT fixed_vector(unsigned NumElements, LLT ScalarTy) {
    return vector(ElementCount::getFixed(NumElements), ScalarTy);
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// width.
  [[deprecated("Use LLT::scalable_vector(unsigned, LLT) instead.")]]
  static constexpr LLT scalable_vector(unsigned MinNumElements,
                                       unsigned ScalarSizeInBits) {
    return vector(ElementCount::getScalable(MinNumElements),
                  LLT::integer(ScalarSizeInBits));
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// type.
  static constexpr LLT scalable_vector(unsigned MinNumElements, LLT ScalarTy) {
    return vector(ElementCount::getScalable(MinNumElements), ScalarTy);
  }

  static constexpr LLT scalarOrVector(ElementCount EC, LLT ScalarTy) {
    return EC.isScalar() ? ScalarTy : LLT::vector(EC, ScalarTy);
  }

  [[deprecated("Use LLT::scalarOrVector(EC, LLT) instead.")]]
  static constexpr LLT scalarOrVector(ElementCount EC, uint64_t ScalarSize) {
    assert(ScalarSize <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    return scalarOrVector(EC, LLT::integer(static_cast<unsigned>(ScalarSize)));
  }

  explicit constexpr LLT(Kind Info, ElementCount EC, uint64_t SizeInBits,
                         unsigned AddressSpace, FPInfo FP)
      : LLT() {
    init(Info, EC, SizeInBits, AddressSpace, FP);
  }

  explicit LLT(MVT VT, bool EnableFPInfo = false);
  explicit constexpr LLT() : Info(static_cast<Kind>(0)), RawData(0) {}

  constexpr bool isValid() const {
    return isToken() || RawData != 0;
  }
  constexpr bool isScalar() const {
    return Info == Kind::INTEGER || Info == Kind::FLOAT;
  }
  constexpr bool isScalar(unsigned Size) const {
    return isScalar() && getScalarSizeInBits() == Size;
  }
  constexpr bool isFloat() const { return isValid() && Info == Kind::FLOAT; }
  constexpr bool isFloat(unsigned Size) const {
    return isFloat() && getScalarSizeInBits() == Size;
  }
  constexpr bool isVariantFloat() const {
    return isFloat() && (getFPInfo() == FPInfo::VARIANT_FLOAT_1 ||
                         getFPInfo() == FPInfo::VARIANT_FLOAT_2 ||
                         getFPInfo() == FPInfo::VARIANT_FLOAT_3);
  }
  constexpr bool isVariantFloat(FPInfo Variant) const {
    return isFloat() && getFPInfo() == Variant;
  }
  constexpr bool isVariantFloat(unsigned Size, FPInfo Variant) const {
    return isVariantFloat() && getScalarSizeInBits() == Size;
  }
  constexpr bool isFloatVector() const {
    return isVector() && Info == Kind::VECTOR_FLOAT;
  }
  constexpr bool isBFloat() const { return isVariantFloat(16, FPInfo::VARIANT_FLOAT_1); }
  constexpr bool isX86FP80() const { return isVariantFloat(80, FPInfo::VARIANT_FLOAT_1); }
  constexpr bool isPPCF128() const { return isVariantFloat(128, FPInfo::VARIANT_FLOAT_1); }
  constexpr bool isToken() const {
    return Info == Kind::INTEGER && RawData == 0;
  }
  constexpr bool isInteger() const {
    return isValid() && Info == Kind::INTEGER;
  }
  constexpr bool isInteger(unsigned Size) const {
    return isInteger() && getScalarSizeInBits() == Size;
  }
  constexpr bool isIntegerVector() const {
    return isVector() && Info == Kind::VECTOR_INTEGER;
  }
  constexpr bool isVector() const {
    return isValid() &&
           (Info == Kind::VECTOR_INTEGER || Info == Kind::VECTOR_FLOAT ||
            Info == Kind::VECTOR_POINTER);
  }
  constexpr bool isPointer() const {
    return isValid() && Info == Kind::POINTER;
  }
  constexpr bool isPointerVector() const {
    return isVector() && Info == Kind::VECTOR_POINTER;
  }
  constexpr bool isPointerOrPointerVector() const {
    return isPointer() || isPointerVector();
  }

  /// Returns the number of elements in a vector LLT. Must only be called on
  /// vector types.
  constexpr uint16_t getNumElements() const {
    if (isScalable())
      llvm::reportInvalidSizeRequest(
          "Possible incorrect use of LLT::getNumElements() for "
          "scalable vector. Scalable flag may be dropped, use "
          "LLT::getElementCount() instead");
    return getElementCount().getKnownMinValue();
  }

  /// Returns true if the LLT is a scalable vector. Must only be called on
  /// vector types.
  constexpr bool isScalable() const {
    assert(isVector() && "Expected a vector type");
    return getFieldValue(VectorScalableFieldInfo);
  }

  /// Returns true if the LLT is a fixed vector. Returns false otherwise, even
  /// if the LLT is not a vector type.
  constexpr bool isFixedVector() const { return isVector() && !isScalable(); }

  constexpr bool isFixedVector(unsigned NumElements,
                               unsigned ScalarSize) const {
    return isFixedVector() && getNumElements() == NumElements &&
           getScalarSizeInBits() == ScalarSize;
  }

  /// Returns true if the LLT is a scalable vector. Returns false otherwise,
  /// even if the LLT is not a vector type.
  constexpr bool isScalableVector() const { return isVector() && isScalable(); }

  constexpr ElementCount getElementCount() const {
    assert(isVector() && "cannot get number of elements on scalar/aggregate");
    return ElementCount::get(getFieldValue(VectorElementsFieldInfo),
                             isScalable());
  }

  /// Returns the total size of the type. Must only be called on sized types.
  constexpr TypeSize getSizeInBits() const {
    if (isPointer() || isScalar())
      return TypeSize::getFixed(getScalarSizeInBits());
    auto EC = getElementCount();
    return TypeSize(getScalarSizeInBits() * EC.getKnownMinValue(),
                    EC.isScalable());
  }

  /// Returns the total size of the type in bytes, i.e. number of whole bytes
  /// needed to represent the size in bits. Must only be called on sized types.
  constexpr TypeSize getSizeInBytes() const {
    TypeSize BaseSize = getSizeInBits();
    return {(BaseSize.getKnownMinValue() + 7) / 8, BaseSize.isScalable()};
  }

  constexpr LLT getScalarType() const {
    return isVector() ? getElementType() : *this;
  }

  constexpr FPInfo getFPInfo() const {
    assert((isFloat() || isFloatVector()) &&
           "cannot get FP info for non float type");

    return FPInfo(getFieldValue(ScalarFPFieldInfo));
  }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element type. Otherwise, return the new element type.
  constexpr LLT changeElementType(LLT NewEltTy) const {
    return isVector() ? LLT::vector(getElementCount(), NewEltTy) : NewEltTy;
  }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element size. Otherwise, return the new element type. Invalid
  /// for pointer types. For pointer types, use changeElementType.
  constexpr LLT changeElementSize(unsigned NewEltSize) const {
    assert(!isPointerOrPointerVector() && !(isFloat() || isFloatVector()) &&
           "invalid to directly change element size for pointers");
    return isVector() ? LLT::vector(getElementCount(), LLT::integer(NewEltSize))
                      : LLT::integer(NewEltSize);
  }

  /// Return a vector or scalar with the same element type and the new element
  /// count.
  constexpr LLT changeElementCount(ElementCount EC) const {
    return LLT::scalarOrVector(EC, getScalarType());
  }

  constexpr LLT changeElementCount(unsigned NumElements) const {
    return changeElementCount(ElementCount::getFixed(NumElements));
  }

  constexpr LLT changeFPInfo(FPInfo FP) const {
    assert(isFloat() ||
           isFloatVector() &&
               "cannot change FPInfo for non floating point types");
    if (isFloatVector())
      LLT::vector(getElementCount(), getElementType().changeFPInfo(FP));

    return LLT::floatingPoint(getSizeInBits(), FP);
  }

  /// Return a type that is \p Factor times smaller. Reduces the number of
  /// elements if this is a vector, or the bitwidth for scalar/pointers. Does
  /// not attempt to handle cases that aren't evenly divisible.
  constexpr LLT divide(int Factor) const {
    assert(Factor != 1);
    assert((!isScalar() || getScalarSizeInBits() != 0) &&
           "cannot divide scalar of size zero");
    if (isVector()) {
      assert(getElementCount().isKnownMultipleOf(Factor));
      return scalarOrVector(getElementCount().divideCoefficientBy(Factor),
                            getElementType());
    }

    assert(getScalarSizeInBits() % Factor == 0);
    return integer(getScalarSizeInBits() / Factor);
  }

  /// Produce a vector type that is \p Factor times bigger, preserving the
  /// element type. For a scalar or pointer, this will produce a new vector with
  /// \p Factor elements.
  constexpr LLT multiplyElements(int Factor) const {
    if (isVector()) {
      return scalarOrVector(getElementCount().multiplyCoefficientBy(Factor),
                            getElementType());
    }

    return fixed_vector(Factor, *this);
  }

  constexpr bool isByteSized() const {
    return getSizeInBits().isKnownMultipleOf(8);
  }

  constexpr unsigned getScalarSizeInBits() const {
    if (isPointerOrPointerVector())
      return getFieldValue(PointerSizeFieldInfo);
    return getFieldValue(ScalarSizeFieldInfo);
  }

  constexpr unsigned getAddressSpace() const {
    assert(isPointerOrPointerVector() &&
           "cannot get address space of non-pointer type");
    return getFieldValue(PointerAddressSpaceFieldInfo);
  }

  /// Returns the vector's element type. Only valid for vector types.
  constexpr LLT getElementType() const {
    assert(isVector() && "cannot get element type of scalar/aggregate");
    if (isPointerVector())
      return pointer(getAddressSpace(), getScalarSizeInBits());

    if (isFloatVector())
      return floatingPoint(getScalarSizeInBits(), getFPInfo());

    return integer(getScalarSizeInBits());
  }

  constexpr LLT dropType() const {
    if (isPointer() || isPointerVector())
      return *this;

    if (isVector())
      return vector(getElementCount(), LLT::integer(getScalarSizeInBits()));

    return integer(getSizeInBits());
  }

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  constexpr bool operator==(const LLT &RHS) const {
    return Info == RHS.Info && RawData == RHS.RawData;
  }

  constexpr bool operator!=(const LLT &RHS) const { return !(*this == RHS); }

  friend struct DenseMapInfo<LLT>;
  friend class GISelInstProfileBuilder;

private:
  /// LLT is packed into 64 bits as follows:
  /// Info : 3
  /// RawData : 61
  /// with 61 bits of RawData remaining for Kind-specific data, packed in
  /// bitfields as described below. As there isn't a simple portable way to pack
  /// bits into bitfields, here the different fields in the packed structure is
  /// described in static const *Field variables. Each of these variables
  /// is a 2-element array, with the first element describing the bitfield size
  /// and the second element describing the bitfield offset.
  ///
  /*
                                --- LLT ---

   63       56       47       39       31       23       15       7      0
   |        |        |        |        |        |        |        |      |
  |xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|
   ...................................                                      (1)
   *****************                                                        (2)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~                             (3)
                                                ^^^^^^^^^^^^^^^^^           (4)
                                                                      @     (5)
                                            ###                             (6)
                                                                       %%%  (7)

  (1) ScalarSize  (2) PointerSize  (3) PointerAddressSpace 
  (4) VectorElements  (5) VectorScalable  (6) FPInfo  (7) Kind
  
  */
  typedef int BitFieldInfo[2];
  ///
  /// This is how the bitfields are packed per Kind:
  /// * Invalid:
  ///   gets encoded as RawData == 0, as that is an invalid encoding, since for
  ///   valid encodings, SizeInBits/SizeOfElement must be larger than 0.
  /// * Non-pointer scalar (isPointer == 0 && isVector == 0):
  ///   SizeInBits: 32;
  static const constexpr BitFieldInfo ScalarSizeFieldInfo{32, 29};
  static const constexpr BitFieldInfo ScalarFPFieldInfo{2, 21};
  /// * Pointer (isPointer == 1 && isVector == 0):
  ///   SizeInBits: 16;
  ///   AddressSpace: 24;
  static const constexpr BitFieldInfo PointerSizeFieldInfo{16, 45};
  static const constexpr BitFieldInfo PointerAddressSpaceFieldInfo{24, 21};
  /// * Vector-of-non-pointer (isPointer == 0 && isVector == 1):
  ///   NumElements: 16;
  ///   SizeOfElement: 32;
  ///   Scalable: 1;
  static const constexpr BitFieldInfo VectorElementsFieldInfo{16, 5};
  static const constexpr BitFieldInfo VectorScalableFieldInfo{1, 0};
  /// * Vector-of-pointer (isPointer == 1 && isVector == 1):
  ///   NumElements: 16;
  ///   SizeOfElement: 16;
  ///   AddressSpace: 24;
  ///   Scalable: 1;

  Kind Info : 3;
  uint64_t RawData : 61;

  static constexpr uint64_t getMask(const BitFieldInfo FieldInfo) {
    const int FieldSizeInBits = FieldInfo[0];
    return (((uint64_t)1) << FieldSizeInBits) - 1;
  }

  static constexpr uint64_t maskAndShift(uint64_t Val, uint64_t Mask,
                                         uint8_t Shift) {
    assert(Val <= Mask && "Value too large for field");
    return (Val & Mask) << Shift;
  }

  static constexpr uint64_t maskAndShift(uint64_t Val,
                                         const BitFieldInfo FieldInfo) {
    return maskAndShift(Val, getMask(FieldInfo), FieldInfo[1]);
  }

  constexpr uint64_t getFieldValue(const BitFieldInfo FieldInfo) const {
    return getMask(FieldInfo) & (RawData >> FieldInfo[1]);
  }

  constexpr void init(Kind Info, ElementCount EC, uint64_t SizeInBits,
                      unsigned AddressSpace, FPInfo FP) {
    assert(SizeInBits <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    this->Info = Info;
    if (Info == Kind::POINTER || Info == Kind::VECTOR_POINTER) {
      RawData = maskAndShift(SizeInBits, PointerSizeFieldInfo) |
                maskAndShift(AddressSpace, PointerAddressSpaceFieldInfo);
    } else {
      RawData = maskAndShift(SizeInBits, ScalarSizeFieldInfo) | 
                maskAndShift((uint64_t) FP, ScalarFPFieldInfo);
    }

    if (Info == Kind::VECTOR_INTEGER || Info == Kind::VECTOR_FLOAT || Info == Kind::VECTOR_POINTER) {
      RawData |= maskAndShift(EC.getKnownMinValue(), VectorElementsFieldInfo) |
                 maskAndShift(EC.isScalable() ? 1 : 0, VectorScalableFieldInfo);
    }
  }

public:
  constexpr uint64_t getUniqueRAWLLTData() const {
    return ((uint64_t)RawData) << 3 | ((uint64_t)Info);
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const LLT &Ty) {
  Ty.print(OS);
  return OS;
}

template <> struct DenseMapInfo<LLT> {
  static inline LLT getEmptyKey() {
    LLT Invalid;
    Invalid.Info = static_cast<LLT::Kind>(2);
    return Invalid;
  }
  static inline LLT getTombstoneKey() {
    LLT Invalid;
    Invalid.Info = static_cast<LLT::Kind>(3);
    return Invalid;
  }
  static inline unsigned getHashValue(const LLT &Ty) {
    uint64_t Val = Ty.getUniqueRAWLLTData();
    return DenseMapInfo<uint64_t>::getHashValue(Val);
  }
  static bool isEqual(const LLT &LHS, const LLT &RHS) {
    return LHS.getUniqueRAWLLTData() == RHS.getUniqueRAWLLTData();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_LOWLEVELTYPE_H
