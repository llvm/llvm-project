//===- ABI/Types.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the Types and related helper methods concerned to the
/// LLVMABI library which mirrors ABI related type information from
/// the LLVM frontend.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_ABI_TYPES_H
#define LLVM_ABI_TYPES_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstdint>

namespace llvm {
namespace abi {

enum class TypeKind {
  Void,
  MemberPointer,
  Complex,
  Integer,
  Float,
  Pointer,
  Array,
  Vector,
  Record,
};

class Type {
private:
  TypeSize getTypeStoreSize() const {
    TypeSize StoreSizeInBits = getTypeStoreSizeInBits();
    return {StoreSizeInBits.getKnownMinValue() / 8,
            StoreSizeInBits.isScalable()};
  }
  TypeSize getTypeStoreSizeInBits() const {
    TypeSize BaseSize = getSizeInBits();
    uint64_t AlignedSizeInBits =
        alignToPowerOf2(BaseSize.getKnownMinValue(), 8);
    return {AlignedSizeInBits, BaseSize.isScalable()};
  }

protected:
  TypeKind Kind;
  TypeSize SizeInBits;
  Align ABIAlignment;

  Type(TypeKind K, TypeSize SizeInBits, Align ABIAlign)
      : Kind(K), SizeInBits(SizeInBits), ABIAlignment(ABIAlign) {}

public:
  TypeKind getKind() const { return Kind; }
  TypeSize getSizeInBits() const { return SizeInBits; }
  Align getAlignment() const { return ABIAlignment; }

  TypeSize getTypeAllocSize() const {
    return alignTo(getTypeStoreSize(), getAlignment().value());
  }

  bool isVoid() const { return Kind == TypeKind::Void; }
  bool isInteger() const { return Kind == TypeKind::Integer; }
  bool isFloat() const { return Kind == TypeKind::Float; }
  bool isPointer() const { return Kind == TypeKind::Pointer; }
  bool isArray() const { return Kind == TypeKind::Array; }
  bool isVector() const { return Kind == TypeKind::Vector; }
  bool isRecord() const { return Kind == TypeKind::Record; }
  bool isMemberPointer() const { return Kind == TypeKind::MemberPointer; }
  bool isComplex() const { return Kind == TypeKind::Complex; }
};

class VoidType : public Type {
public:
  VoidType() : Type(TypeKind::Void, TypeSize::getFixed(0), Align(1)) {}

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Void; }
};

class ComplexType : public Type {
public:
  ComplexType(const Type *ElementType, uint64_t SizeInBits, Align Alignment)
      : Type(TypeKind::Complex, TypeSize::getFixed(SizeInBits), Alignment),
        ElementType(ElementType) {}

  const Type *getElementType() const { return ElementType; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Complex;
  }

private:
  const Type *ElementType;
};

class IntegerType : public Type {
private:
  bool IsSigned;
  bool IsBitInt;

public:
  IntegerType(uint64_t BitWidth, Align ABIAlign, bool IsSigned,
              bool IsBitInt = false)
      : Type(TypeKind::Integer, TypeSize::getFixed(BitWidth), ABIAlign),
        IsSigned(IsSigned), IsBitInt(IsBitInt) {}

  bool isSigned() const { return IsSigned; }
  bool isBitInt() const { return IsBitInt; }
  bool isBool() const {
    return getSizeInBits().getFixedValue() == 1 && !IsBitInt;
  }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Integer;
  }
};

class FloatType : public Type {
private:
  const fltSemantics *Semantics;

public:
  FloatType(const fltSemantics &FloatSemantics, Align ABIAlign)
      : Type(TypeKind::Float,
             TypeSize::getFixed(APFloat::getSizeInBits(FloatSemantics)),
             ABIAlign),
        Semantics(&FloatSemantics) {}

  const fltSemantics *getSemantics() const { return Semantics; }
  static bool classof(const Type *T) { return T->getKind() == TypeKind::Float; }
};

class PointerLikeType : public Type {
protected:
  unsigned AddrSpace;
  PointerLikeType(TypeKind K, TypeSize SizeInBits, Align ABIAlign, unsigned AS)
      : Type(K, SizeInBits, ABIAlign), AddrSpace(AS) {}

public:
  unsigned getAddrSpace() const { return AddrSpace; }
  bool isMemberPointer() const { return getKind() == TypeKind::MemberPointer; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer ||
           T->getKind() == TypeKind::MemberPointer;
  }
};

class PointerType : public PointerLikeType {
public:
  PointerType(uint64_t Size, Align ABIAlign, unsigned AddressSpace = 0)
      : PointerLikeType(TypeKind::Pointer, TypeSize::getFixed(Size), ABIAlign,
                        AddressSpace) {}

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer;
  }
};

class MemberPointerType : public PointerLikeType {
private:
  bool IsFunctionPointer;

public:
  MemberPointerType(bool IsFunctionPointer, uint64_t SizeInBits, Align ABIAlign,
                    unsigned AddressSpace = 0)
      : PointerLikeType(TypeKind::MemberPointer, TypeSize::getFixed(SizeInBits),
                        ABIAlign, AddressSpace),
        IsFunctionPointer(IsFunctionPointer) {}
  bool isFunctionPointer() const { return IsFunctionPointer; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::MemberPointer;
  }
};

class ArrayType : public Type {
private:
  const Type *ElementType;
  uint64_t NumElements;
  bool IsMatrix;

public:
  ArrayType(const Type *ElementType, uint64_t NumElements, uint64_t SizeInBits,
            bool IsMatrixType = false)
      : Type(TypeKind::Array, TypeSize::getFixed(SizeInBits),
             ElementType->getAlignment()),
        ElementType(ElementType), NumElements(NumElements),
        IsMatrix(IsMatrixType) {}

  const Type *getElementType() const { return ElementType; }
  uint64_t getNumElements() const { return NumElements; }
  bool isMatrixType() const { return IsMatrix; }

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Array; }
};

class VectorType : public Type {
private:
  const Type *ElementType;
  ElementCount NumElements;

public:
  VectorType(const Type *ElementType, ElementCount NumElements, Align ABIAlign)
      : Type(TypeKind::Vector,
             TypeSize(ElementType->getSizeInBits().getFixedValue() *
                          NumElements.getKnownMinValue(),
                      NumElements.isScalable()),
             ABIAlign),
        ElementType(ElementType), NumElements(NumElements) {}

  const Type *getElementType() const { return ElementType; }
  ElementCount getNumElements() const { return NumElements; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Vector;
  }
};

struct FieldInfo {
  const Type *FieldType;
  uint64_t OffsetInBits;
  uint64_t BitFieldWidth;
  bool IsBitField;
  bool IsUnnamedBitfield;

  FieldInfo(const Type *FieldType, uint64_t OffsetInBits = 0,
            bool IsBitField = false, uint64_t BitFieldWidth = 0,
            bool IsUnnamedBitField = false)
      : FieldType(FieldType), OffsetInBits(OffsetInBits),
        BitFieldWidth(BitFieldWidth), IsBitField(IsBitField),
        IsUnnamedBitfield(IsUnnamedBitField) {}
};

enum class StructPacking { Default, Packed, ExplicitPacking };

class RecordType : public Type {
private:
  ArrayRef<FieldInfo> Fields;
  ArrayRef<FieldInfo> BaseClasses;
  ArrayRef<FieldInfo> VirtualBaseClasses;
  StructPacking Packing;
  bool CanPassInRegisters;
  bool IsCoercedRecord;
  bool IsUnion;
  bool IsTransparent;

  bool IsCXXRecord;
  bool IsPolymorphic;
  bool HasNonTrivialCopyConstructor;
  bool HasNonTrivialDestructor;
  bool HasFlexibleArrayMember;
  bool HasUnalignedFields;

public:
  RecordType(ArrayRef<FieldInfo> StructFields, ArrayRef<FieldInfo> Bases,
             ArrayRef<FieldInfo> VBases, TypeSize Size, Align Align,
             StructPacking Pack = StructPacking::Default, bool Union = false,
             bool CXXRecord = false, bool Polymorphic = false,
             bool NonTrivialCopy = false, bool NonTrivialDtor = false,
             bool FlexibleArray = false, bool UnalignedFields = false,
             bool CanPassInRegs = false, bool IsCoercedRec = false,
             bool Transparent = false)
      : Type(TypeKind::Record, Size, Align), Fields(StructFields),
        BaseClasses(Bases), VirtualBaseClasses(VBases), Packing(Pack),
        CanPassInRegisters(CanPassInRegs), IsCoercedRecord(IsCoercedRec),
        IsUnion(Union), IsTransparent(Transparent), IsCXXRecord(CXXRecord),
        IsPolymorphic(Polymorphic),
        HasNonTrivialCopyConstructor(NonTrivialCopy),
        HasNonTrivialDestructor(NonTrivialDtor),
        HasFlexibleArrayMember(FlexibleArray),
        HasUnalignedFields(UnalignedFields) {}

  uint32_t getNumFields() const { return Fields.size(); }
  StructPacking getPacking() const { return Packing; }

  bool isUnion() const { return IsUnion; }
  bool isCXXRecord() const { return IsCXXRecord; }
  bool isPolymorphic() const { return IsPolymorphic; }
  bool hasNonTrivialCopyConstructor() const {
    return HasNonTrivialCopyConstructor;
  }
  bool isCoercedRecord() const { return IsCoercedRecord; }
  bool canPassInRegisters() const { return CanPassInRegisters; }
  bool hasNonTrivialDestructor() const { return HasNonTrivialDestructor; }
  bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
  bool hasUnalignedFields() const { return HasUnalignedFields; }

  uint32_t getNumBaseClasses() const { return BaseClasses.size(); }
  uint32_t getNumVirtualBaseClasses() const {
    return VirtualBaseClasses.size();
  }
  bool isTransparentUnion() const { return IsTransparent; }
  ArrayRef<FieldInfo> getFields() const { return Fields; }
  ArrayRef<FieldInfo> getBaseClasses() const { return BaseClasses; }
  ArrayRef<FieldInfo> getVirtualBaseClasses() const {
    return VirtualBaseClasses;
  }

  static bool isEmptyForABI(const llvm::abi::Type *Ty) {
    const auto *RT = dyn_cast<RecordType>(Ty);
    if (!RT)
      return false;

    for (const auto &Field : RT->getFields()) {
      if (!Field.IsUnnamedBitfield)
        return false;
    }

    if (RT->isCXXRecord()) {
      for (const auto &Base : RT->getBaseClasses()) {
        if (!isEmptyForABI(Base.FieldType))
          return false;
      }

      for (const auto &VBase : RT->getVirtualBaseClasses()) {
        if (!isEmptyForABI(VBase.FieldType))
          return false;
      }
    }

    return true;
  }

  const FieldInfo *getElementContainingOffset(unsigned OffsetInBits) const {
    SmallVector<std::pair<unsigned, const FieldInfo *>> AllElements;

    for (const auto &Base : BaseClasses) {
      if (!isEmptyForABI(Base.FieldType))
        AllElements.emplace_back(Base.OffsetInBits, &Base);
    }

    for (const auto &VBase : VirtualBaseClasses) {
      if (!isEmptyForABI(VBase.FieldType))
        AllElements.emplace_back(VBase.OffsetInBits, &VBase);
    }

    for (const auto &Field : Fields) {
      if (Field.IsUnnamedBitfield)
        continue;
      AllElements.emplace_back(Field.OffsetInBits, &Field);
    }

    llvm::stable_sort(AllElements, [](const auto &A, const auto &B) {
      return A.first < B.first;
    });

    auto *It = llvm::upper_bound(AllElements, OffsetInBits,
                                 [](unsigned Offset, const auto &Element) {
                                   return Offset < Element.first;
                                 });

    if (It == AllElements.begin())
      return nullptr;

    --It;

    const FieldInfo *Candidate = It->second;
    unsigned ElementStart = It->first;
    unsigned ElementSize =
        Candidate->FieldType->getSizeInBits().getFixedValue();

    if (OffsetInBits >= ElementStart &&
        OffsetInBits < ElementStart + ElementSize)
      return Candidate;

    return nullptr;
  }
  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Record;
  }
};

/// API for creating ABI Types
class TypeBuilder {
private:
  BumpPtrAllocator &Allocator;

public:
  explicit TypeBuilder(BumpPtrAllocator &Alloc) : Allocator(Alloc) {}

  const VoidType *getVoidType() {
    return new (Allocator.Allocate<VoidType>()) VoidType();
  }

  const IntegerType *getIntegerType(uint64_t BitWidth, Align Align, bool Signed,
                                    bool IsBitInt = false) {
    return new (Allocator.Allocate<IntegerType>())
        IntegerType(BitWidth, Align, Signed, IsBitInt);
  }

  const FloatType *getFloatType(const fltSemantics &Semantics, Align Align) {
    return new (Allocator.Allocate<FloatType>()) FloatType(Semantics, Align);
  }

  const PointerType *getPointerType(uint64_t Size, Align Align,
                                    unsigned Addrspace = 0) {
    return new (Allocator.Allocate<PointerType>())
        PointerType(Size, Align, Addrspace);
  }

  const ArrayType *getArrayType(const Type *ElementType, uint64_t NumElements,
                                uint64_t SizeInBits,
                                bool IsMatrixType = false) {
    return new (Allocator.Allocate<ArrayType>())
        ArrayType(ElementType, NumElements, SizeInBits, IsMatrixType);
  }

  const VectorType *getVectorType(const Type *ElementType,
                                  ElementCount NumElements, Align Align) {
    return new (Allocator.Allocate<VectorType>())
        VectorType(ElementType, NumElements, Align);
  }

  // TODO: clean up this function
  const RecordType *
  getRecordType(ArrayRef<FieldInfo> Fields, TypeSize Size, Align Align,
                StructPacking Pack = StructPacking::Default,
                ArrayRef<FieldInfo> BaseClasses = {},
                ArrayRef<FieldInfo> VirtualBaseClasses = {},
                bool CXXRecord = false, bool Polymorphic = false,
                bool NonTrivialCopy = false, bool NonTrivialDtor = false,
                bool FlexibleArray = false, bool UnalignedFields = false,
                bool CanPassInRegister = false) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());
    std::copy(Fields.begin(), Fields.end(), FieldArray);

    FieldInfo *BaseArray = nullptr;
    if (!BaseClasses.empty()) {
      BaseArray = Allocator.Allocate<FieldInfo>(BaseClasses.size());
      std::copy(BaseClasses.begin(), BaseClasses.end(), BaseArray);
    }

    FieldInfo *VBaseArray = nullptr;
    if (!VirtualBaseClasses.empty()) {
      VBaseArray = Allocator.Allocate<FieldInfo>(VirtualBaseClasses.size());
      std::copy(VirtualBaseClasses.begin(), VirtualBaseClasses.end(),
                VBaseArray);
    }

    ArrayRef<FieldInfo> FieldsRef(FieldArray, Fields.size());
    ArrayRef<FieldInfo> BasesRef(BaseArray, BaseClasses.size());
    ArrayRef<FieldInfo> VBasesRef(VBaseArray, VirtualBaseClasses.size());

    return new (Allocator.Allocate<RecordType>())
        RecordType(FieldsRef, BasesRef, VBasesRef, Size, Align, Pack, false,
                   CXXRecord, Polymorphic, NonTrivialCopy, NonTrivialDtor,
                   FlexibleArray, UnalignedFields, CanPassInRegister);
  }

  /// Creates a coerced record type for ABI purposes.
  ///
  /// Coerced record types are artificial struct representations used internally
  /// by the ABI layer to represent non-aggregate types in a convenient way.
  /// For example, a function argument that needs to be passed in two registers
  /// might be coerced into a struct with two fields: {i64, i32}.
  ///
  /// @param Fields The fields of the coerced struct
  /// @param Size Total size in bits
  /// @param Align Alignment requirements
  /// @param Pack Struct packing mode (usually Default)
  /// @return A RecordType marked as coerced for ABI purposes
  const RecordType *
  getCoercedRecordType(ArrayRef<FieldInfo> Fields, TypeSize Size, Align Align,
                       StructPacking Pack = StructPacking::Default) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());
    std::copy(Fields.begin(), Fields.end(), FieldArray);

    ArrayRef<FieldInfo> FieldsRef(FieldArray, Fields.size());

    return new (Allocator.Allocate<RecordType>()) RecordType(
        FieldsRef, ArrayRef<FieldInfo>(), ArrayRef<FieldInfo>(), Size, Align,
        Pack, false, false, false, false, false, false, false, true, true);
  }

  const RecordType *getUnionType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                 Align Align,
                                 StructPacking Pack = StructPacking::Default,
                                 bool IsTransparent = false,
                                 bool CanPassInRegs = false,
                                 bool CXXRecord = false) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      const FieldInfo &Field = Fields[I];
      new (&FieldArray[I])
          FieldInfo(Field.FieldType, 0, Field.IsBitField, Field.BitFieldWidth,
                    Field.IsUnnamedBitfield);
    }

    ArrayRef<FieldInfo> FieldsRef(FieldArray, Fields.size());

    return new (Allocator.Allocate<RecordType>())
        RecordType(FieldsRef, ArrayRef<FieldInfo>(), ArrayRef<FieldInfo>(),
                   Size, Align, Pack, true, CXXRecord, false, false, false,
                   false, false, CanPassInRegs, false, IsTransparent);
  }

  const ComplexType *getComplexType(const Type *ElementType, Align Align) {
    // Complex types have two elements (real and imaginary parts)
    uint64_t ElementSize = ElementType->getSizeInBits().getFixedValue();
    uint64_t ComplexSize = ElementSize * 2;

    return new (Allocator.Allocate<ComplexType>())
        ComplexType(ElementType, ComplexSize, Align);
  }

  const MemberPointerType *getMemberPointerType(bool IsFunctionPointer,
                                                uint64_t SizeInBits,
                                                Align Align) {
    return new (Allocator.Allocate<MemberPointerType>())
        MemberPointerType(IsFunctionPointer, SizeInBits, Align);
  }
};

} // namespace abi
} // namespace llvm

#endif
