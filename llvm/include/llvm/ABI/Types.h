//===- ABI/Types.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the type system for the LLVMABI library, which mirrors
/// ABI-relevant aspects of frontend types.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_ABI_TYPES_H
#define LLVM_ABI_TYPES_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"

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

/// Represents the ABI-specific view of a type in LLVM.
///
/// This abstracts platform and language-specific ABI details from the
/// frontend, providing a consistent interface for the ABI Library.
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

enum RecordFlags : unsigned {
  None = 0,
  CanPassInRegisters = 1 << 0,
  IsUnion = 1 << 1,
  IsTransparent = 1 << 2,
  IsCXXRecord = 1 << 3,
  IsPolymorphic = 1 << 4,
  HasFlexibleArrayMember = 1 << 5,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ HasFlexibleArrayMember),
};

class RecordType : public Type {
private:
  ArrayRef<FieldInfo> Fields;
  ArrayRef<FieldInfo> BaseClasses;
  ArrayRef<FieldInfo> VirtualBaseClasses;
  StructPacking Packing;
  RecordFlags Flags;

public:
  RecordType(ArrayRef<FieldInfo> StructFields, ArrayRef<FieldInfo> Bases,
             ArrayRef<FieldInfo> VBases, TypeSize Size, Align Align,
             StructPacking Pack = StructPacking::Default,
             RecordFlags RecFlags = RecordFlags::None)
      : Type(TypeKind::Record, Size, Align), Fields(StructFields),
        BaseClasses(Bases), VirtualBaseClasses(VBases), Packing(Pack),
        Flags(RecFlags) {}
  uint32_t getNumFields() const { return Fields.size(); }
  StructPacking getPacking() const { return Packing; }

  bool isUnion() const {
    return static_cast<unsigned>(Flags & RecordFlags::IsUnion) != 0;
  }
  bool isCXXRecord() const {
    return static_cast<unsigned>(Flags & RecordFlags::IsCXXRecord) != 0;
  }
  bool isPolymorphic() const {
    return static_cast<unsigned>(Flags & RecordFlags::IsPolymorphic) != 0;
  }
  bool canPassInRegisters() const {
    return static_cast<unsigned>(Flags & RecordFlags::CanPassInRegisters) != 0;
  }
  bool hasFlexibleArrayMember() const {
    return static_cast<unsigned>(Flags & RecordFlags::HasFlexibleArrayMember) !=
           0;
  }
  uint32_t getNumBaseClasses() const { return BaseClasses.size(); }
  uint32_t getNumVirtualBaseClasses() const {
    return VirtualBaseClasses.size();
  }
  bool isTransparentUnion() const {
    return static_cast<unsigned>(Flags & RecordFlags::IsTransparent) != 0;
  }
  ArrayRef<FieldInfo> getFields() const { return Fields; }
  ArrayRef<FieldInfo> getBaseClasses() const { return BaseClasses; }
  ArrayRef<FieldInfo> getVirtualBaseClasses() const {
    return VirtualBaseClasses;
  }
};

/// TypeBuilder manages the lifecycle of ABI types using bump pointer
/// allocation. Types created by a TypeBuilder are valid for the lifetime of the
/// allocator.
///
/// Example usage:
/// \code
///   BumpPtrAllocator Alloc;
///   TypeBuilder Builder(Alloc);
///   const auto *IntTy = Builder.getIntegerType(32, Align(4), true);
/// \endcode
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

  const RecordType *getRecordType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                  Align Align,
                                  StructPacking Pack = StructPacking::Default,
                                  ArrayRef<FieldInfo> BaseClasses = {},
                                  ArrayRef<FieldInfo> VirtualBaseClasses = {},
                                  RecordFlags RecFlags = RecordFlags::None) {
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
        RecordType(FieldsRef, BasesRef, VBasesRef, Size, Align, Pack, RecFlags);
  }

  const RecordType *getUnionType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                 Align Align,
                                 StructPacking Pack = StructPacking::Default,
                                 RecordFlags RecFlags = RecordFlags::None) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0, E = Fields.size(); I != E; ++I) {
      const FieldInfo &Field = Fields[I];
      new (&FieldArray[I])
          FieldInfo(Field.FieldType, 0, Field.IsBitField, Field.BitFieldWidth,
                    Field.IsUnnamedBitfield);
    }

    ArrayRef<FieldInfo> FieldsRef(FieldArray, Fields.size());

    return new (Allocator.Allocate<RecordType>())
        RecordType(FieldsRef, ArrayRef<FieldInfo>(), ArrayRef<FieldInfo>(),
                   Size, Align, Pack, RecFlags | RecordFlags::IsUnion);
  }

  const ComplexType *getComplexType(const Type *ElementType, Align Align) {
    // Complex types have two elements (real and imaginary parts)
    uint64_t ElementSizeInBits = ElementType->getSizeInBits().getFixedValue();
    uint64_t ComplexSizeInBits = ElementSizeInBits * 2;

    return new (Allocator.Allocate<ComplexType>())
        ComplexType(ElementType, ComplexSizeInBits, Align);
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
