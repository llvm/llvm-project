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
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
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
  Struct,
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
  Align PreferredAlignment;
  bool IsExplicitlyAligned;

  Type(TypeKind K, TypeSize Size, Align Align, bool ExplicitAlign = false)
      : Kind(K), SizeInBits(Size), ABIAlignment(Align),
        IsExplicitlyAligned(ExplicitAlign) {}

public:
  TypeKind getKind() const { return Kind; }
  TypeSize getSizeInBits() const { return SizeInBits; }
  Align getAlignment() const { return ABIAlignment; }
  Align getPrefferedAlign() const { return PreferredAlignment; }
  bool hasExplicitAlignment() const { return IsExplicitlyAligned; }

  void setExplicitAlignment(Align Align) {
    ABIAlignment = Align;
    IsExplicitlyAligned = true;
  }
  TypeSize getTypeAllocSize() const {
    return alignTo(getTypeStoreSize(), getAlignment().value());
  }

  bool isVoid() const { return Kind == TypeKind::Void; }
  bool isInteger() const { return Kind == TypeKind::Integer; }
  bool isFloat() const { return Kind == TypeKind::Float; }
  bool isPointer() const { return Kind == TypeKind::Pointer; }
  bool isArray() const { return Kind == TypeKind::Array; }
  bool isVector() const { return Kind == TypeKind::Vector; }
  bool isStruct() const { return Kind == TypeKind::Struct; }
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
  bool IsBoolean;
  bool IsBitInt;
  uint64_t ABISizeInBits;

public:
  IntegerType(uint64_t BitWidth, uint64_t ABIWidth, Align Align, bool Signed,
              bool IsBool = false, bool BitInt = false)
      : Type(TypeKind::Integer, TypeSize::getFixed(BitWidth), Align),
        IsSigned(Signed), IsBoolean(IsBool), IsBitInt(BitInt),
        ABISizeInBits(ABIWidth) {}

  bool isSigned() const { return IsSigned; }
  bool isBool() const { return IsBoolean; }
  bool isBitInt() const { return IsBitInt; }
  uint64_t getLogicalBitWidth() const {
    return getSizeInBits().getFixedValue();
  }
  uint64_t getABISizeInBits() const { return ABISizeInBits; }
  bool isABIWidened() const { return getLogicalBitWidth() != ABISizeInBits; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Integer;
  }
};

class FloatType : public Type {
private:
  const fltSemantics *Semantics;

public:
  FloatType(const fltSemantics &FloatSemantics, Align Align)
      : Type(TypeKind::Float,
             TypeSize::getFixed(APFloat::getSizeInBits(FloatSemantics)), Align),
        Semantics(&FloatSemantics) {}

  const fltSemantics *getSemantics() const { return Semantics; }
  static bool classof(const Type *T) { return T->getKind() == TypeKind::Float; }
};

class PointerLikeType : public Type {
protected:
  unsigned AddrSpace;
  PointerLikeType(TypeKind K, TypeSize Size, Align Align, unsigned AS = 0)
      : Type(K, Size, Align), AddrSpace(AS) {}

public:
  virtual ~PointerLikeType() = default;
  unsigned getAddrSpace() const { return AddrSpace; }
  virtual bool isMemberPointer() const = 0;

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer ||
           T->getKind() == TypeKind::MemberPointer;
  }
};

class PointerType : public PointerLikeType {
public:
  PointerType(uint64_t Size, Align Align, unsigned AddressSpace = 0)
      : PointerLikeType(TypeKind::Pointer, TypeSize::getFixed(Size), Align,
                        AddressSpace) {}

  bool isMemberPointer() const override { return false; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer;
  }
};

class MemberPointerType : public PointerLikeType {
private:
  bool IsFunctionPointer;
  bool Has64BitPointers;

public:
  MemberPointerType(bool IsFunctionPointer, bool Has64BitPointers,
                    uint64_t SizeInBits, Align Alignment,
                    unsigned AddressSpace = 0)
      : PointerLikeType(TypeKind::MemberPointer, TypeSize::getFixed(SizeInBits),
                        Alignment, AddressSpace),
        IsFunctionPointer(IsFunctionPointer),
        Has64BitPointers(Has64BitPointers) {}

  bool isMemberPointer() const override { return true; }
  bool isFunctionPointer() const { return IsFunctionPointer; }
  bool has64BitPointers() const { return Has64BitPointers; }

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
  ArrayType(const Type *ElemType, uint64_t NumElems, bool IsMatrixType = false)
      : Type(TypeKind::Array, ElemType->getSizeInBits() * NumElems,
             ElemType->getAlignment()),
        ElementType(ElemType), NumElements(NumElems), IsMatrix(IsMatrixType) {}

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
  VectorType(const Type *ElemType, ElementCount NumElems, Align Align)
      : Type(TypeKind::Vector,
             TypeSize(ElemType->getSizeInBits().getFixedValue() *
                          NumElems.getKnownMinValue(),
                      NumElems.isScalable()),
             Align),
        ElementType(ElemType), NumElements(NumElems) {}

  const Type *getElementType() const { return ElementType; }
  ElementCount getNumElements() const { return NumElements; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Vector;
  }
};

struct FieldInfo {
  const Type *FieldType;
  uint64_t OffsetInBits;
  bool IsBitField;
  bool IsUnnamedBitfield;
  uint64_t BitFieldWidth;

  FieldInfo(const Type *Type, uint64_t Offset = 0, bool BitField = false,
            uint64_t BFWidth = 0, bool IsUnnamedBF = false)
      : FieldType(Type), OffsetInBits(Offset), IsBitField(BitField),
        IsUnnamedBitfield(IsUnnamedBF), BitFieldWidth(BFWidth) {}
};

enum class StructPacking { Default, Packed, ExplicitPacking };

class StructType : public Type {
private:
  uint32_t NumFields;
  ArrayRef<FieldInfo> Fields;
  ArrayRef<FieldInfo> BaseClasses;
  ArrayRef<FieldInfo> VirtualBaseClasses;
  StructPacking Packing;
  bool CanPassInRegisters;
  bool IsCoercedStruct;
  bool IsUnion;
  bool IsTransparent;

  bool IsCXXRecord;
  bool IsPolymorphic;
  bool HasNonTrivialCopyConstructor;
  bool HasNonTrivialDestructor;
  bool HasFlexibleArrayMember;
  bool HasUnalignedFields;
  uint32_t NumBaseClasses;
  uint32_t NumVirtualBaseClasses;

public:
  StructType(ArrayRef<FieldInfo> StructFields, uint32_t FieldCount,
             ArrayRef<FieldInfo> Bases, uint32_t BaseCount,
             ArrayRef<FieldInfo> VBases, uint32_t VBaseCount, TypeSize Size,
             Align Align, StructPacking Pack = StructPacking::Default,
             bool Union = false, bool CXXRecord = false,
             bool Polymorphic = false, bool NonTrivialCopy = false,
             bool NonTrivialDtor = false, bool FlexibleArray = false,
             bool UnalignedFields = false, bool CanPassInRegs = false,
             bool IsCoercedStr = false, bool Transparent = false)
      : Type(TypeKind::Struct, Size, Align), NumFields(FieldCount),
        Fields(StructFields), BaseClasses(Bases), VirtualBaseClasses(VBases),
        Packing(Pack), CanPassInRegisters(CanPassInRegs),
        IsCoercedStruct(IsCoercedStr), IsUnion(Union),
        IsTransparent(Transparent), IsCXXRecord(CXXRecord),
        IsPolymorphic(Polymorphic),
        HasNonTrivialCopyConstructor(NonTrivialCopy),
        HasNonTrivialDestructor(NonTrivialDtor),
        HasFlexibleArrayMember(FlexibleArray),
        HasUnalignedFields(UnalignedFields), NumBaseClasses(BaseCount),
        NumVirtualBaseClasses(VBaseCount) {}

  uint32_t getNumFields() const { return NumFields; }
  StructPacking getPacking() const { return Packing; }

  bool isUnion() const { return IsUnion; }
  bool isCXXRecord() const { return IsCXXRecord; }
  bool isPolymorphic() const { return IsPolymorphic; }
  bool hasNonTrivialCopyConstructor() const {
    return HasNonTrivialCopyConstructor;
  }
  bool isCoercedStruct() const { return IsCoercedStruct; }
  bool canPassInRegisters() const { return CanPassInRegisters; }
  bool hasNonTrivialDestructor() const { return HasNonTrivialDestructor; }
  bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
  bool hasUnalignedFields() const { return HasUnalignedFields; }

  uint32_t getNumBaseClasses() const { return NumBaseClasses; }
  uint32_t getNumVirtualBaseClasses() const { return NumVirtualBaseClasses; }
  bool isTransparentUnion() const { return IsTransparent; }
  ArrayRef<FieldInfo> getFields() const { return Fields; }
  ArrayRef<FieldInfo> getBaseClasses() const { return BaseClasses; }
  ArrayRef<FieldInfo> getVirtualBaseClasses() const {
    return VirtualBaseClasses;
  }

  const FieldInfo *getElementContainingOffset(unsigned OffsetInBits) const {
    std::vector<std::pair<unsigned, const FieldInfo *>> AllElements;

    for (const auto &Field : Fields) {
      if (Field.IsUnnamedBitfield)
        continue;
      AllElements.emplace_back(Field.OffsetInBits, &Field);
    }

    for (const auto &Base : BaseClasses)
      AllElements.emplace_back(Base.OffsetInBits, &Base);

    for (const auto &VBase : VirtualBaseClasses)
      AllElements.emplace_back(VBase.OffsetInBits, &VBase);

    std::sort(AllElements.begin(), AllElements.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    auto it =
        std::upper_bound(AllElements.begin(), AllElements.end(), OffsetInBits,
                         [](unsigned offset, const auto &element) {
                           return offset < element.first;
                         });

    if (it == AllElements.begin())
      return nullptr;

    --it;

    const FieldInfo *candidate = it->second;
    unsigned elementStart = it->first;
    unsigned elementSize =
        candidate->FieldType->getSizeInBits().getFixedValue();

    if (OffsetInBits >= elementStart &&
        OffsetInBits < elementStart + elementSize)
      return candidate;

    return nullptr;
  }
  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Struct;
  }
};

// API for creating ABI Types
class TypeBuilder {
private:
  BumpPtrAllocator &Allocator;

public:
  explicit TypeBuilder(BumpPtrAllocator &Alloc) : Allocator(Alloc) {}

  const VoidType *getVoidType() {
    return new (Allocator.Allocate<VoidType>()) VoidType();
  }

  const IntegerType *getIntegerType(uint64_t BitWidth, Align Align, bool Signed,
                                    bool IsBoolean = false,
                                    bool IsBitInt = false,
                                    uint64_t ABISizeInBits = 0) {
    uint64_t ABISize = ABISizeInBits ? ABISizeInBits : BitWidth;
    return new (Allocator.Allocate<IntegerType>())
        IntegerType(BitWidth, ABISize, Align, Signed, IsBoolean, IsBitInt);
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
                                bool IsMatrixType = false) {
    return new (Allocator.Allocate<ArrayType>())
        ArrayType(ElementType, NumElements, IsMatrixType);
  }

  const VectorType *getVectorType(const Type *ElementType,
                                  ElementCount NumElements, Align Align) {
    return new (Allocator.Allocate<VectorType>())
        VectorType(ElementType, NumElements, Align);
  }

  // TODO: clean up this function
  const StructType *
  getStructType(ArrayRef<FieldInfo> Fields, TypeSize Size, Align Align,
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

    return new (Allocator.Allocate<StructType>()) StructType(
        FieldsRef, static_cast<uint32_t>(Fields.size()), BasesRef,
        static_cast<uint32_t>(BaseClasses.size()), VBasesRef,
        static_cast<uint32_t>(VirtualBaseClasses.size()), Size, Align, Pack,
        false, CXXRecord, Polymorphic, NonTrivialCopy, NonTrivialDtor,
        FlexibleArray, UnalignedFields, CanPassInRegister);
  }
  const StructType *
  getCoercedStructType(ArrayRef<FieldInfo> Fields, TypeSize Size, Align Align,
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
    return new (Allocator.Allocate<StructType>()) StructType(
        FieldsRef, static_cast<uint32_t>(Fields.size()), BasesRef,
        static_cast<uint32_t>(BaseClasses.size()), VBasesRef,
        static_cast<uint32_t>(VirtualBaseClasses.size()), Size, Align, Pack,
        false, CXXRecord, Polymorphic, NonTrivialCopy, NonTrivialDtor,
        FlexibleArray, UnalignedFields, CanPassInRegister, true);
  }
  const StructType *getUnionType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                 Align Align,
                                 StructPacking Pack = StructPacking::Default,
                                 bool IsTransparent = false,
                                 bool CanPassInRegs = false) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      const FieldInfo &field = Fields[I];
      new (&FieldArray[I])
          FieldInfo(field.FieldType, 0, field.IsBitField, field.BitFieldWidth,
                    field.IsUnnamedBitfield);
    }

    ArrayRef<FieldInfo> FieldsRef(FieldArray, Fields.size());

    return new (Allocator.Allocate<StructType>()) StructType(
        FieldsRef, static_cast<uint32_t>(Fields.size()), ArrayRef<FieldInfo>(),
        0, ArrayRef<FieldInfo>(), 0, Size, Align, Pack, true, false, false,
        false, false, false, false, CanPassInRegs, false, IsTransparent);
  }

  const ComplexType *getComplexType(const Type *ElementType, Align Align) {
    // Complex types have two elements (real and imaginary parts)
    uint64_t ElementSize = ElementType->getSizeInBits().getFixedValue();
    uint64_t ComplexSize = ElementSize * 2;

    return new (Allocator.Allocate<ComplexType>())
        ComplexType(ElementType, ComplexSize, Align);
  }

  const MemberPointerType *getMemberPointerType(bool IsFunctionPointer,
                                                bool Has64BitPointers,
                                                uint64_t SizeInBits,
                                                Align Align) {
    return new (Allocator.Allocate<MemberPointerType>()) MemberPointerType(
        IsFunctionPointer, Has64BitPointers, SizeInBits, Align);
  }
};

} // namespace abi
} // namespace llvm

#endif
