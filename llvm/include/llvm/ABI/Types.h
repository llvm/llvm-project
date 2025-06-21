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
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/TypeSize.h"
#include <cstdint>

namespace llvm {
namespace abi {

enum class TypeKind {
  Void,
  Integer,
  Float,
  Pointer,
  Array,
  Vector,
  Struct,
  Union,
};

class Type {
protected:
  TypeKind Kind;
  TypeSize SizeInBits;
  Align Alignment;
  bool IsExplicitlyAligned;

  Type(TypeKind K, TypeSize Size, Align Align, bool ExplicitAlign = false)
      : Kind(K), SizeInBits(Size), Alignment(Align),
        IsExplicitlyAligned(ExplicitAlign) {}

public:
  TypeKind getKind() const { return Kind; }
  TypeSize getSizeInBits() const { return SizeInBits; }
  Align getAlignment() const { return Alignment; }
  bool hasExplicitAlignment() const { return IsExplicitlyAligned; }

  void setExplicitAlignment(Align Align) {
    Alignment = Align;
    IsExplicitlyAligned = true;
  }

  bool isVoid() const { return Kind == TypeKind::Void; }
  bool isInteger() const { return Kind == TypeKind::Integer; }
  bool isFloat() const { return Kind == TypeKind::Float; }
  bool isPointer() const { return Kind == TypeKind::Pointer; }
  bool isArray() const { return Kind == TypeKind::Array; }
  bool isVector() const { return Kind == TypeKind::Vector; }
  bool isStruct() const { return Kind == TypeKind::Struct; }
  bool isUnion() const { return Kind == TypeKind::Union; }
};

class VoidType : public Type {
public:
  VoidType() : Type(TypeKind::Void, TypeSize::getFixed(0), Align(1)) {}

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Void; }
};

class IntegerType : public Type {
private:
  bool IsSigned;

public:
  IntegerType(uint64_t BitWidth, Align Align, bool Signed)
      : Type(TypeKind::Integer, TypeSize::getFixed(BitWidth), Align),
        IsSigned(Signed) {}

  bool isSigned() const { return IsSigned; }

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

  const fltSemantics *getSemantics() { return Semantics; }
  static bool classof(const Type *T) { return T->getKind() == TypeKind::Float; }
};

class PointerType : public Type {
  unsigned AddrSpace;

public:
  PointerType(uint64_t Size, Align Align, unsigned AddressSpace = 0)
      : Type(TypeKind::Pointer, TypeSize::getFixed(Size), Align),
        AddrSpace(AddressSpace) {}

  unsigned getAddrSpace() const { return AddrSpace; }
  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer;
  }
};

class ArrayType : public Type {
private:
  const Type *ElementType;
  uint64_t NumElements;

public:
  ArrayType(const Type *ElemType, uint64_t NumElems)
      : Type(TypeKind::Array, ElemType->getSizeInBits() * NumElems,
             ElemType->getAlignment()),
        ElementType(ElemType), NumElements(NumElems) {}

  const Type *getElementType() const { return ElementType; }
  uint64_t getNumElements() const { return NumElements; }

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
  uint64_t BitFieldWidth;

  FieldInfo(const Type *Type, uint64_t Offset = 0, bool BitField = false,
            uint64_t BFWidth = 0)
      : FieldType(Type), OffsetInBits(Offset), IsBitField(BitField),
        BitFieldWidth(BFWidth) {}
};

enum class StructPacking { Default, Packed, ExplicitPacking };

class StructType : public Type {
private:
  const FieldInfo *Fields;
  uint32_t NumFields;
  StructPacking Packing;

public:
  StructType(const FieldInfo *StructFields, uint32_t FieldCount, TypeSize Size,
             Align Align, StructPacking Pack = StructPacking::Default)
      : Type(TypeKind::Struct, Size, Align), Fields(StructFields),
        NumFields(FieldCount), Packing(Pack) {}

  const FieldInfo *getFields() const { return Fields; }
  uint32_t getNumFields() const { return NumFields; }
  StructPacking getPacking() const { return Packing; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Struct;
  }
};

class UnionType : public Type {
private:
  const FieldInfo *Fields;
  uint32_t NumFields;
  StructPacking Packing;

public:
  UnionType(const FieldInfo *UnionFields, uint32_t FieldCount, TypeSize Size,
            Align Align, StructPacking Pack = StructPacking::Default)
      : Type(TypeKind::Union, Size, Align), Fields(UnionFields),
        NumFields(FieldCount), Packing(Pack) {}

  const FieldInfo *getFields() const { return Fields; }
  uint32_t getNumFields() const { return NumFields; }
  StructPacking getPacking() const { return Packing; }

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Union; }
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

  const IntegerType *getIntegerType(uint64_t BitWidth, Align Align,
                                    bool Signed) {
    return new (Allocator.Allocate<IntegerType>())
        IntegerType(BitWidth, Align, Signed);
  }

  const FloatType *getFloatType(const fltSemantics &Semantics, Align Align) {
    return new (Allocator.Allocate<FloatType>()) FloatType(Semantics, Align);
  }

  const PointerType *getPointerType(uint64_t Size, Align Align) {
    return new (Allocator.Allocate<PointerType>()) PointerType(Size, Align);
  }

  const ArrayType *getArrayType(const Type *ElementType, uint64_t NumElements) {
    return new (Allocator.Allocate<ArrayType>())
        ArrayType(ElementType, NumElements);
  }

  const VectorType *getVectorType(const Type *ElementType,
                                  ElementCount NumElements, Align Align) {
    return new (Allocator.Allocate<VectorType>())
        VectorType(ElementType, NumElements, Align);
  }

  const StructType *getStructType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                  Align Align,
                                  StructPacking Pack = StructPacking::Default) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      new (&FieldArray[I]) FieldInfo(Fields[I]);
    }

    return new (Allocator.Allocate<StructType>()) StructType(
        FieldArray, static_cast<uint32_t>(Fields.size()), Size, Align, Pack);
  }

  const UnionType *getUnionType(ArrayRef<FieldInfo> Fields, TypeSize Size,
                                Align Align,
                                StructPacking Pack = StructPacking::Default) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      new (&FieldArray[I]) FieldInfo(Fields[I]);
    }

    return new (Allocator.Allocate<UnionType>()) UnionType(
        FieldArray, static_cast<uint32_t>(Fields.size()), Size, Align, Pack);
  }
};

} // namespace abi
} // namespace llvm

#endif
