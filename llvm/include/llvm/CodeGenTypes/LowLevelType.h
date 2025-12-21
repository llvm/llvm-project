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
/// For a type attached to a MachineInstr, we care about total
/// size, the number of vector lanes (if any)
/// and the kind of the type (anyscalar, integer, float and etc).
/// Floating point are filled with APFloat::Semantics to make them
/// distinguishable.
///
/// Earlier other information required for correct selection was expected to be
/// carried only by the opcode, or non-type flags. For example the distinction
/// between G_ADD and G_FADD for int/float or fast-math flags.
///
/// Now we also able to rely on the kind of the type.
/// This may be useful to distinguish different types of the same size used at
/// the same opcode, for example, G_FADD with half vs G_FADD with bfloat16.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LOWLEVELTYPE_H
#define LLVM_CODEGEN_LOWLEVELTYPE_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {

class Type;
class raw_ostream;

class LLT {
public:
  using FpSemantics = APFloat::Semantics;

  enum class Kind : uint8_t {
    INVALID,
    ANY_SCALAR,
    INTEGER,
    FLOAT,
    POINTER,
    VECTOR_ANY,
    VECTOR_INTEGER,
    VECTOR_FLOAT,
    VECTOR_POINTER,
  };

  constexpr static Kind toVector(Kind Ty) {
    if (Ty == Kind::POINTER)
      return Kind::VECTOR_POINTER;

    if (Ty == Kind::INTEGER)
      return Kind::VECTOR_INTEGER;

    if (Ty == Kind::FLOAT)
      return Kind::VECTOR_FLOAT;

    return Kind::VECTOR_ANY;
  }

  constexpr static Kind toScalar(Kind Ty) {
    if (Ty == Kind::VECTOR_POINTER)
      return Kind::POINTER;

    if (Ty == Kind::VECTOR_INTEGER)
      return Kind::INTEGER;

    if (Ty == Kind::VECTOR_FLOAT)
      return Kind::FLOAT;

    return Kind::ANY_SCALAR;
  }

  /// Get a low-level scalar or aggregate "bag of bits".
  static constexpr LLT scalar(unsigned SizeInBits) {
    return LLT{Kind::ANY_SCALAR, ElementCount::getFixed(0), SizeInBits};
  }

  static constexpr LLT integer(unsigned SizeInBits) {
    return LLT{Kind::INTEGER, ElementCount::getFixed(0), SizeInBits};
  }

  static LLT floatingPoint(const FpSemantics &Sem) {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0),
               APFloat::getSizeInBits(APFloatBase::EnumToSemantics(Sem)), Sem};
  }

  /// Get a low-level token; just a scalar with zero bits (or no size).
  static constexpr LLT token() {
    return LLT{Kind::ANY_SCALAR, ElementCount::getFixed(0),
               /*SizeInBits=*/0};
  }

  /// Get a low-level pointer in the given address space.
  static constexpr LLT pointer(unsigned AddressSpace, unsigned SizeInBits) {
    assert(SizeInBits > 0 && "invalid pointer size");
    return LLT{Kind::POINTER, ElementCount::getFixed(0), SizeInBits,
               AddressSpace};
  }

  /// Get a low-level vector of some number of elements and element width.
  static constexpr LLT vector(ElementCount EC, unsigned ScalarSizeInBits) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    return LLT{Kind::VECTOR_ANY, EC, ScalarSizeInBits};
  }

  /// Get a low-level vector of some number of elements and element type.
  static constexpr LLT vector(ElementCount EC, LLT ScalarTy) {
    assert(!EC.isScalar() && "invalid number of vector elements");
    assert(!ScalarTy.isVector() && "invalid vector element type");

    Kind Info = toVector(ScalarTy.Info);
    if (ScalarTy.isPointer())
      return LLT{Info, EC, ScalarTy.getSizeInBits().getFixedValue(),
                 ScalarTy.getAddressSpace()};
    if (ScalarTy.isFloat())
      return LLT{Info, EC, ScalarTy.getSizeInBits().getFixedValue(),
                 ScalarTy.getFpSemantics()};

    return LLT{Info, EC, ScalarTy.getSizeInBits().getFixedValue()};
  }

  static constexpr LLT floatIEEE(unsigned SizeInBits) {
    switch (SizeInBits) {
    default:
      llvm_unreachable("Wrong SizeInBits for IEEE Floating point!");
    case 16:
      return float16();
    case 32:
      return float32();
    case 64:
      return float64();
    case 128:
      return float128();
    }
  }

  // Get a bfloat16 value.
  static constexpr LLT bfloat16() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 16,
               FpSemantics::S_BFloat};
  }
  /// Get a 16-bit IEEE half value.
  static constexpr LLT float16() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 16,
               FpSemantics::S_IEEEhalf};
  }
  /// Get a 32-bit IEEE float value.
  static constexpr LLT float32() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 32,
               FpSemantics::S_IEEEsingle};
  }
  /// Get a 64-bit IEEE double value.
  static constexpr LLT float64() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 64,
               FpSemantics::S_IEEEdouble};
  }

  /// Get a 80-bit X86 floating point value.
  static constexpr LLT x86fp80() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 80,
               FpSemantics::S_x87DoubleExtended};
  }

  /// Get a 128-bit IEEE quad value.
  static constexpr LLT float128() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 128,
               FpSemantics::S_IEEEquad};
  }

  /// Get a 128-bit PowerPC double double value.
  static constexpr LLT ppcf128() {
    return LLT{Kind::FLOAT, ElementCount::getFixed(0), 128,
               FpSemantics::S_PPCDoubleDouble};
  }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// width.
  static constexpr LLT fixed_vector(unsigned NumElements,
                                    unsigned ScalarSizeInBits) {
    return vector(ElementCount::getFixed(NumElements),
                  LLT::scalar(ScalarSizeInBits));
  }

  /// Get a low-level fixed-width vector of some number of elements and element
  /// type.
  static constexpr LLT fixed_vector(unsigned NumElements, LLT ScalarTy) {
    return vector(ElementCount::getFixed(NumElements), ScalarTy);
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// width.
  static constexpr LLT scalable_vector(unsigned MinNumElements,
                                       unsigned ScalarSizeInBits) {
    return vector(ElementCount::getScalable(MinNumElements),
                  LLT::scalar(ScalarSizeInBits));
  }

  /// Get a low-level scalable vector of some number of elements and element
  /// type.
  static constexpr LLT scalable_vector(unsigned MinNumElements, LLT ScalarTy) {
    return vector(ElementCount::getScalable(MinNumElements), ScalarTy);
  }

  static constexpr LLT scalarOrVector(ElementCount EC, LLT ScalarTy) {
    return EC.isScalar() ? ScalarTy : LLT::vector(EC, ScalarTy);
  }

  static constexpr LLT scalarOrVector(ElementCount EC, uint64_t ScalarSize) {
    assert(ScalarSize <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    return scalarOrVector(EC, LLT::scalar(static_cast<unsigned>(ScalarSize)));
  }

  explicit constexpr LLT(Kind Info, ElementCount EC, uint64_t SizeInBits)
      : LLT() {
    init(Info, EC, SizeInBits);
  }

  explicit constexpr LLT(Kind Info, ElementCount EC, uint64_t SizeInBits,
                         unsigned AddressSpace)
      : LLT() {
    init(Info, EC, SizeInBits, AddressSpace);
  }

  explicit constexpr LLT(Kind Info, ElementCount EC, uint64_t SizeInBits,
                         FpSemantics Sem)
      : LLT() {
    init(Info, EC, SizeInBits, Sem);
  }

  LLVM_ABI explicit LLT(MVT VT);
  explicit constexpr LLT() : RawData(0), Info(static_cast<Kind>(0)) {}

  constexpr bool isToken() const {
    return Info == Kind::ANY_SCALAR && RawData == 0;
  }
  constexpr bool isValid() const { return isToken() || RawData != 0; }
  constexpr bool isAnyScalar() const { return Info == Kind::ANY_SCALAR; }
  constexpr bool isInteger() const { return Info == Kind::INTEGER; }
  constexpr bool isFloat() const { return Info == Kind::FLOAT; }
  constexpr bool isPointer() const { return Info == Kind::POINTER; }
  constexpr bool isAnyVector() const { return Info == Kind::VECTOR_ANY; }
  constexpr bool isIntegerVector() const {
    return Info == Kind::VECTOR_INTEGER;
  }
  constexpr bool isFloatVector() const { return Info == Kind::VECTOR_FLOAT; }
  constexpr bool isPointerVector() const {
    return Info == Kind::VECTOR_POINTER;
  }
  constexpr bool isPointerOrPointerVector() const {
    return isPointer() || isPointerVector();
  }

  constexpr bool isScalar() const {
    return Info == Kind::ANY_SCALAR || Info == Kind::INTEGER ||
           Info == Kind::FLOAT;
  }
  constexpr bool isScalar(unsigned Size) const {
    return isScalar() && getScalarSizeInBits() == Size;
  }
  constexpr bool isVector() const {
    return Info == Kind::VECTOR_ANY || Info == Kind::VECTOR_INTEGER ||
           Info == Kind::VECTOR_FLOAT || Info == Kind::VECTOR_POINTER;
  }

  constexpr bool isInteger(unsigned Size) const {
    return isInteger() && getScalarSizeInBits() == Size;
  }

  constexpr bool isFloat(unsigned Size) const {
    return isFloat() && getScalarSizeInBits() == Size;
  }
  constexpr bool isFloat(FpSemantics Sem) const {
    return isFloat() && getFpSemantics() == Sem;
  }
  constexpr bool isFloatIEEE() const {
    return isFloat(APFloatBase::S_IEEEhalf) ||
           isFloat(APFloatBase::S_IEEEsingle) ||
           isFloat(APFloatBase::S_IEEEdouble) ||
           isFloat(APFloatBase::S_IEEEquad);
  }
  constexpr bool isBFloat16() const { return isFloat(FpSemantics::S_BFloat); }
  constexpr bool isX86FP80() const {
    return isFloat(FpSemantics::S_x87DoubleExtended);
  }
  constexpr bool isPPCF128() const {
    return isFloat(FpSemantics::S_PPCDoubleDouble);
  }

  /// Returns the number of elements in a vector LLT. Must only be called on
  /// vector types.
  constexpr uint16_t getNumElements() const {
    if (isScalable())
      llvm::reportFatalInternalError(
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

  constexpr FpSemantics getFpSemantics() const {
    assert((isFloat() || isFloatVector()) &&
           "cannot get FP info for non float type");
    return FpSemantics(getFieldValue(FpSemanticFieldInfo));
  }

  constexpr Kind getKind() const { return Info; }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element type. Otherwise, return the new element type.
  constexpr LLT changeElementType(LLT NewEltTy) const {
    return isVector() ? LLT::vector(getElementCount(), NewEltTy) : NewEltTy;
  }

  /// If this type is a vector, return a vector with the same number of elements
  /// but the new element size. Otherwise, return the new element type. Invalid
  /// for pointer and floating point types. For these, use changeElementType.
  constexpr LLT changeElementSize(unsigned NewEltSize) const {
    assert(!isPointerOrPointerVector() && !(isFloat() || isFloatVector()) &&
           "invalid to directly change element size for pointers and floats");
    return isVector()
               ? LLT::vector(getElementCount(), getElementType().isInteger()
                                                    ? LLT::integer(NewEltSize)
                                                    : LLT::scalar(NewEltSize))
           : isInteger() ? LLT::integer(NewEltSize)
                         : LLT::scalar(NewEltSize);
  }

  /// Return a vector or scalar with the same element type and the new element
  /// count.
  constexpr LLT changeElementCount(ElementCount EC) const {
    return LLT::scalarOrVector(EC, getScalarType());
  }

  constexpr LLT changeElementCount(unsigned NumElements) const {
    return changeElementCount(ElementCount::getFixed(NumElements));
  }

  /// Return a type that is \p Factor times smaller. Reduces the number of
  /// elements if this is a vector, or the bitwidth for scalar/pointers. Does
  /// not attempt to handle cases that aren't evenly divisible.
  constexpr LLT divide(int Factor) const {
    assert(Factor != 1);
    assert((!isScalar() || getScalarSizeInBits() != 0) && !isFloat() &&
           "cannot divide scalar of size zero and floats");
    if (isVector()) {
      assert(getElementCount().isKnownMultipleOf(Factor));
      return scalarOrVector(getElementCount().divideCoefficientBy(Factor),
                            getElementType());
    }

    assert(getScalarSizeInBits() % Factor == 0);
    if (isInteger())
      return integer(getScalarSizeInBits() / Factor);

    return scalar(getScalarSizeInBits() / Factor);
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
      return floatingPoint(getFpSemantics());

    if (isIntegerVector())
      return integer(getScalarSizeInBits());

    return scalar(getScalarSizeInBits());
  }

  constexpr LLT changeToInteger() const {
    if (isPointer() || isPointerVector())
      return *this;

    if (isVector())
      return vector(getElementCount(), LLT::integer(getScalarSizeInBits()));

    return integer(getSizeInBits());
  }

  LLVM_ABI void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  constexpr bool operator==(const LLT &RHS) const {
    if (isAnyScalar() || RHS.isAnyScalar())
      return isScalar() == RHS.isScalar() &&
             getScalarSizeInBits() == RHS.getScalarSizeInBits();

    if (isVector() && RHS.isVector())
      return getElementType() == RHS.getElementType() &&
             getElementCount() == RHS.getElementCount();

    return Info == RHS.Info && RawData == RHS.RawData;
  }

  constexpr bool operator!=(const LLT &RHS) const { return !(*this == RHS); }

  friend struct DenseMapInfo<LLT>;
  friend class GISelInstProfileBuilder;

private:
  /// LLT is packed into 64 bits as follows:
  /// RawData : 60
  /// Info : 4
  /// RawData remaining for Kind-specific data, packed in
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
   %%%%                                                                     (1)
       .... ........ ........ ........ ....                                 (2)
       **** ******** ****                                                   (3)
                         ~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~                        (4)
                                           #### ####                        (5)
                                                    ^^^^ ^^^^^^^^ ^^^^      (6)
                                                                         @  (7)

  (1) Kind:                [63:60]
  (2) ScalarSize:          [59:28]
  (3) PointerSize:         [59:44]
  (4) PointerAddressSpace: [43:20]
  (5) FpSemantics:         [27:20]
  (6) VectorElements:      [19:4]
  (7) VectorScalable:      [0:0]

  */

  /// This is how the LLT are packed per Kind:
  /// * Invalid:
  ///   Info: [63:60] = 0
  ///   RawData: [59:0] = 0;
  ///
  /// * Non-pointer scalar (isPointer == 0 && isVector == 0):
  ///   Info: [63:60];
  ///   SizeOfElement: [59:28];
  ///   FpSemantics: [27:20];
  ///
  /// * Pointer (isPointer == 1 && isVector == 0):
  ///   Info: [63:60];
  ///   SizeInBits: [59:44];
  ///   AddressSpace: [43:20];
  ///
  /// * Vector-of-non-pointer (isPointer == 0 && isVector == 1):
  ///   Info: [63:60]
  ///   SizeOfElement: [59:28];
  ///   FpSemantics: [27:20];
  ///   VectorElements: [19:4];
  ///   Scalable: [0:0];
  ///
  /// * Vector-of-pointer (isPointer == 1 && isVector == 1):
  ///   Info: [63:60];
  ///   SizeInBits: [59:44];
  ///   AddressSpace: [43:20];
  ///   VectorElements: [19:4];
  ///   Scalable: [0:0];

  /// BitFieldInfo: {Size, Offset}
  typedef int BitFieldInfo[2];
  static_assert(bit_width_constexpr((uint32_t)APFloat::S_MaxSemantics) <= 8);
  static constexpr BitFieldInfo VectorScalableFieldInfo{1, 0};
  static constexpr BitFieldInfo VectorElementsFieldInfo{16, 4};
  static constexpr BitFieldInfo FpSemanticFieldInfo{8, 20};
  static constexpr BitFieldInfo PointerAddressSpaceFieldInfo{24, 20};
  static constexpr BitFieldInfo ScalarSizeFieldInfo{32, 28};
  static constexpr BitFieldInfo PointerSizeFieldInfo{16, 44};

  uint64_t RawData : 60;
  Kind Info : 4;

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

  // Init for scalar and integer single or vector types
  constexpr void init(Kind Info, ElementCount EC, uint64_t SizeInBits) {
    assert(SizeInBits <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    assert((Info == Kind::ANY_SCALAR || Info == Kind::INTEGER ||
            Info == Kind::VECTOR_ANY || Info == Kind::VECTOR_INTEGER) &&
           "Called initializer for wrong LLT Kind");
    this->Info = Info;
    RawData = maskAndShift(SizeInBits, ScalarSizeFieldInfo);

    if (Info == Kind::VECTOR_ANY || Info == Kind::VECTOR_INTEGER) {
      RawData = maskAndShift(SizeInBits, ScalarSizeFieldInfo) |
                maskAndShift(EC.getKnownMinValue(), VectorElementsFieldInfo) |
                maskAndShift(EC.isScalable() ? 1 : 0, VectorScalableFieldInfo);
    }
  }

  // Init pointer or pointer vector
  constexpr void init(Kind Info, ElementCount EC, uint64_t SizeInBits,
                      unsigned AddressSpace) {
    assert(SizeInBits <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    assert((Info == Kind::POINTER || Info == Kind::VECTOR_POINTER) &&
           "Called initializer for wrong LLT Kind");
    this->Info = Info;
    RawData = maskAndShift(SizeInBits, PointerSizeFieldInfo) |
              maskAndShift(AddressSpace, PointerAddressSpaceFieldInfo);

    if (Info == Kind::VECTOR_POINTER) {
      RawData |= maskAndShift(EC.getKnownMinValue(), VectorElementsFieldInfo) |
                 maskAndShift(EC.isScalable() ? 1 : 0, VectorScalableFieldInfo);
    }
  }

  constexpr void init(Kind Info, ElementCount EC, uint64_t SizeInBits,
                      FpSemantics Sem) {
    assert(SizeInBits <= std::numeric_limits<unsigned>::max() &&
           "Not enough bits in LLT to represent size");
    assert((Info == Kind::FLOAT || Info == Kind::VECTOR_FLOAT) &&
           "Called initializer for wrong LLT Kind");
    this->Info = Info;
    RawData = maskAndShift(SizeInBits, ScalarSizeFieldInfo) |
              maskAndShift((uint64_t)Sem, FpSemanticFieldInfo);

    if (Info == Kind::VECTOR_FLOAT) {
      RawData |= maskAndShift(EC.getKnownMinValue(), VectorElementsFieldInfo) |
                 maskAndShift(EC.isScalable() ? 1 : 0, VectorScalableFieldInfo);
    }
  }

public:
  constexpr uint64_t getUniqueRAWLLTData() const {
    return ((uint64_t)RawData) | ((uint64_t)Info) << 60;
  }

  static bool getUseExtended() { return ExtendedLLT; }
  static void setUseExtended(bool Enable) { ExtendedLLT = Enable; }

private:
  static bool ExtendedLLT;
};

inline raw_ostream &operator<<(raw_ostream &OS, const LLT &Ty) {
  Ty.print(OS);
  return OS;
}

template <> struct DenseMapInfo<LLT> {
  static inline LLT getEmptyKey() {
    LLT Invalid;
    Invalid.Info = LLT::Kind::POINTER;
    return Invalid;
  }
  static inline LLT getTombstoneKey() {
    LLT Invalid;
    Invalid.Info = LLT::Kind::VECTOR_ANY;
    return Invalid;
  }
  static inline unsigned getHashValue(const LLT &Ty) {
    uint64_t Val = Ty.getUniqueRAWLLTData();
    return DenseMapInfo<uint64_t>::getHashValue(Val);
  }
  static bool isEqual(const LLT &LHS, const LLT &RHS) { return LHS == RHS; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_LOWLEVELTYPE_H
