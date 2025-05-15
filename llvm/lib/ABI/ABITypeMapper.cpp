//===---- ABITypeMapper.cpp - Maps LLVM ABI Types to LLVM IR Types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps LLVM ABI type representations to corresponding LLVM IR types.
/// This reverse mapper translates low-level ABI-specific types back into
/// LLVM IR types suitable for code generation and optimization passes.
///
//===----------------------------------------------------------------------===//

#include "llvm/ABI/ABITypeMapper.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace llvm::abi;

llvm::Type *ABITypeMapper::convertType(const abi::Type *ABIType) {
  if (!ABIType)
    return nullptr;

  auto It = TypeCache.find(ABIType);
  if (It != TypeCache.end())
    return It->second;

  llvm::Type *Result = nullptr;

  switch (ABIType->getKind()) {
  case abi::TypeKind::Integer: {
    const auto *IT = cast<abi::IntegerType>(ABIType);
    unsigned Bitwidth = IT->getSizeInBits().getFixedValue();
    Result = llvm::IntegerType::get(Context, Bitwidth);
    break;
  }
  case abi::TypeKind::Float: {
    const fltSemantics *Semantics =
        cast<abi::FloatType>(ABIType)->getSemantics();
    Result = llvm::Type::getFloatingPointTy(Context, *Semantics);
    break;
  }
  case abi::TypeKind::Pointer:
    Result = llvm::PointerType::get(
        Context, cast<abi::PointerType>(ABIType)->getAddrSpace());
    break;
  case abi::TypeKind::Array:
    Result = convertArrayType(cast<abi::ArrayType>(ABIType));
    break;
  case abi::TypeKind::Vector:
    Result = convertVectorType(cast<abi::VectorType>(ABIType));
    break;
  case abi::TypeKind::Record:
    Result = convertRecordType(cast<abi::RecordType>(ABIType));
    break;
  case abi::TypeKind::Void:
    Result = llvm::Type::getVoidTy(Context);
    break;
  case abi::TypeKind::Complex:
    Result = convertComplexType(cast<abi::ComplexType>(ABIType));
    break;
  case abi::TypeKind::MemberPointer:
    Result = convertMemberPointerType(cast<abi::MemberPointerType>(ABIType));
    break;
  }

  if (Result)
    TypeCache[ABIType] = Result;

  return Result;
}

llvm::Type *ABITypeMapper::convertArrayType(const abi::ArrayType *AT) {
  llvm::Type *ElementType = convertType(AT->getElementType());
  if (!ElementType)
    return nullptr;

  uint64_t NumElements = AT->getNumElements();
  if (AT->isMatrixType())
    return llvm::VectorType::get(ElementType,
                                 ElementCount::getFixed(NumElements));

  return llvm::ArrayType::get(ElementType, NumElements);
}

llvm::Type *ABITypeMapper::convertVectorType(const abi::VectorType *VT) {
  llvm::Type *ElementType = convertType(VT->getElementType());
  if (!ElementType)
    return nullptr;

  ElementCount EC = VT->getNumElements();
  return llvm::VectorType::get(ElementType, EC);
}

llvm::Type *ABITypeMapper::convertRecordType(const abi::RecordType *RT) {
  ArrayRef<abi::FieldInfo> FieldsArray = RT->getFields();
  return createStructFromFields(FieldsArray, RT->getSizeInBits(),
                                RT->getAlignment(), RT->isUnion(),
                                RT->isCoercedRecord());
}

llvm::Type *ABITypeMapper::convertComplexType(const abi::ComplexType *CT) {
  // Complex types are represented as structs with two elements: {real, imag}
  llvm::Type *ElementType = convertType(CT->getElementType());
  if (!ElementType)
    return nullptr;

  SmallVector<llvm::Type *, 2> Fields = {ElementType, ElementType};
  return llvm::StructType::get(Context, Fields, /*isPacked=*/false);
}

llvm::Type *
ABITypeMapper::convertMemberPointerType(const abi::MemberPointerType *MPT) {
  if (MPT->isFunctionPointer()) {
    llvm::Type *IntPtrTy = DL.getIntPtrType(Context);
    SmallVector<llvm::Type *, 2> Fields = {IntPtrTy, IntPtrTy};
    return llvm::StructType::get(Context, Fields, /*isPacked=*/false);
  }
  return DL.getIntPtrType(Context);
}

llvm::Type *ABITypeMapper::createPaddingType(uint64_t PaddingBits) {
  if (PaddingBits == 0)
    return nullptr;

  if (PaddingBits % 8 == 0) {
    llvm::Type *ByteType = llvm::IntegerType::get(Context, 8);
    return llvm::ArrayType::get(ByteType, PaddingBits / 8);
  }
  return llvm::IntegerType::get(Context, PaddingBits);
}

llvm::StructType *
ABITypeMapper::createStructFromFields(ArrayRef<abi::FieldInfo> Fields,
                                      TypeSize Size, Align Alignment,
                                      bool IsUnion, bool IsCoercedStr) {
  SmallVector<llvm::Type *, 16> FieldTypes;

  if (IsUnion) {
    llvm::Type *LargestFieldType = nullptr;
    uint64_t LargestFieldSize = 0;

    for (const auto &Field : Fields) {
      llvm::Type *FieldType = convertType(Field.FieldType);
      if (!FieldType)
        continue;

      uint64_t FieldSize = DL.getTypeSizeInBits(FieldType);
      if (FieldSize > LargestFieldSize) {
        LargestFieldSize = FieldSize;
        LargestFieldType = FieldType;
      }
    }

    if (LargestFieldType) {
      FieldTypes.push_back(LargestFieldType);

      uint64_t UnionSizeBits = Size.getFixedValue();
      if (LargestFieldSize < UnionSizeBits) {
        llvm::Type *PaddingType =
            createPaddingType(UnionSizeBits - LargestFieldSize);
        if (PaddingType)
          FieldTypes.push_back(PaddingType);
      }
    }
  } else {
    uint64_t CurrentOffset = 0;

    for (const auto &Field : Fields) {
      if (!IsCoercedStr && Field.OffsetInBits > CurrentOffset) {
        llvm::Type *PaddingType =
            createPaddingType(Field.OffsetInBits - CurrentOffset);
        if (PaddingType)
          FieldTypes.push_back(PaddingType);
        CurrentOffset = Field.OffsetInBits;
      }

      llvm::Type *FieldType = convertType(Field.FieldType);
      if (!FieldType)
        continue;

      if (Field.IsBitField && Field.BitFieldWidth > 0) {
        FieldType = llvm::IntegerType::get(Context, Field.BitFieldWidth);
        CurrentOffset += Field.BitFieldWidth;
      } else {
        FieldTypes.push_back(FieldType);
        CurrentOffset += DL.getTypeSizeInBits(FieldType);
      }
    }

    if (!IsCoercedStr) {
      uint64_t TotalSizeBits = Size.getFixedValue();
      if (CurrentOffset < TotalSizeBits) {
        llvm::Type *PaddingType =
            createPaddingType(TotalSizeBits - CurrentOffset);
        if (PaddingType)
          FieldTypes.push_back(PaddingType);
      }
    }
  }

  return StructType::get(Context, FieldTypes, /*isPacked=*/false);
}
