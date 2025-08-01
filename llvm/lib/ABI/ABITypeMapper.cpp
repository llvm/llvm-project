//===---- ABITypeMapper.cpp - Maps LLVM ABI Types to LLVM IR Types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps LLVM ABI type representations back to corresponding LLVM IR types.
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Type *ABITypeMapper::convertType(const abi::Type *ABIType) {
  if (!ABIType)
    return nullptr;

  auto It = TypeCache.find(ABIType);
  if (It != TypeCache.end())
    return It->second;

  Type *Result = nullptr;

  switch (ABIType->getKind()) {
  case abi::TypeKind::Integer:
    Result = IntegerType::get(Context,
                              cast<abi::IntegerType>(ABIType)->getSizeInBits());
    break;
  case abi::TypeKind::Float:
    Result = convertFloatType(cast<abi::FloatType>(ABIType));
    break;
  case abi::TypeKind::Pointer:
    Result = PointerType::get(Context,
                              cast<abi::PointerType>(ABIType)->getAddrSpace());
    break;
  case abi::TypeKind::Array:
    Result = convertArrayType(cast<abi::ArrayType>(ABIType));
    break;
  case abi::TypeKind::Vector:
    Result = convertVectorType(cast<abi::VectorType>(ABIType));
    break;
  case abi::TypeKind::Struct:
    Result = convertStructType(cast<abi::StructType>(ABIType));
    break;
  case abi::TypeKind::Union:
    Result = convertUnionType(cast<abi::UnionType>(ABIType));
    break;
  case abi::TypeKind::Void:
    Result = Type::getVoidTy(Context);
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

Type *ABITypeMapper::convertFloatType(const abi::FloatType *FT) {
  const fltSemantics *Semantics =
      const_cast<abi::FloatType *>(FT)->getSemantics();
  return Type::getFloatingPointTy(Context, *Semantics);
}

Type *ABITypeMapper::convertArrayType(const abi::ArrayType *AT) {
  Type *ElementType = convertType(AT->getElementType());
  if (!ElementType)
    return nullptr;

  uint64_t NumElements = AT->getNumElements();

  return ArrayType::get(ElementType, NumElements);
}

Type *ABITypeMapper::convertVectorType(const abi::VectorType *VT) {
  Type *ElementType = convertType(VT->getElementType());
  if (!ElementType)
    return nullptr;

  ElementCount EC = VT->getNumElements();

  if (EC.isScalable())
    return ScalableVectorType::get(ElementType, EC.getKnownMinValue());
  return VectorType::get(ElementType, EC);
}

Type *ABITypeMapper::convertStructType(const abi::StructType *ST) {
  ArrayRef<abi::FieldInfo> FieldsArray(ST->getFields(), ST->getNumFields());
  return createStructFromFields(FieldsArray, ST->getNumFields(),
                                ST->getSizeInBits(), ST->getAlignment(), false,
                                ST->isCoercedStruct());
}

Type *ABITypeMapper::convertUnionType(const abi::UnionType *UT) {
  return createStructFromFields(*UT->getFields(), UT->getNumFields(),
                                UT->getSizeInBits(), UT->getAlignment(), true);
}

Type *ABITypeMapper::convertComplexType(const abi::ComplexType *CT) {
  // Complex types are represented as structs with two elements: {real, imag}
  Type *ElementType = convertType(CT->getElementType());
  if (!ElementType)
    return nullptr;
  //   // Check if the element type is a float type and is 16 bits (half
  //   precision)
  // if (ElementType->isFloatingPointTy() &&
  // ElementType->getPrimitiveSizeInBits() == 2 /* bytes */ * 8) {
  //   // Return <2 x half> vector type for complex half
  //   return VectorType::get(ElementType, ElementCount::getFixed(2));
  // }

  SmallVector<Type *, 2> Fields = {ElementType, ElementType};
  return StructType::get(Context, Fields, /*isPacked=*/false);
}

Type *
ABITypeMapper::convertMemberPointerType(const abi::MemberPointerType *MPT) {

  if (MPT->isFunctionPointer()) {
    if (MPT->has64BitPointers()) {
      // {i64, i64} for function pointer + adjustment
      Type *I64 = IntegerType::get(Context, 64);
      SmallVector<Type *, 2> Fields = {I64, I64};
      return StructType::get(Context, Fields, /*isPacked=*/false);
    } else {
      // {i32, i32} for 32-bit systems
      Type *I32 = IntegerType::get(Context, 32);
      SmallVector<Type *, 2> Fields = {I32, I32};
      return StructType::get(Context, Fields, /*isPacked=*/false);
    }
  } else {
    // Data member pointer - single offset value
    if (MPT->has64BitPointers()) {
      return IntegerType::get(Context, 64);
    } else {
      return IntegerType::get(Context, 32);
    }
  }
}

StructType *ABITypeMapper::createStructFromFields(
    ArrayRef<abi::FieldInfo> Fields, uint32_t NumFields, TypeSize Size,
    Align Alignment, bool IsUnion, bool IsCoercedStr) {
  SmallVector<Type *, 16> FieldTypes;

  if (IsUnion) {
    Type *LargestFieldType = nullptr;
    uint64_t LargestFieldSize = 0;

    for (const auto &Field : Fields) {
      Type *FieldType = convertType(Field.FieldType);
      if (!FieldType)
        continue;

      uint64_t FieldSize = 0;
      if (auto *IntTy = dyn_cast<IntegerType>(FieldType)) {
        FieldSize = IntTy->getBitWidth();
      } else if (FieldType->isFloatingPointTy()) {
        FieldSize = FieldType->getPrimitiveSizeInBits();
      } else if (FieldType->isPointerTy()) {
        FieldSize = 64; // Assume 64-bit pointers
      }

      if (FieldSize > LargestFieldSize) {
        LargestFieldSize = FieldSize;
        LargestFieldType = FieldType;
      }
    }

    if (LargestFieldType) {
      FieldTypes.push_back(LargestFieldType);

      uint64_t UnionSizeBits = Size.getFixedValue();
      if (LargestFieldSize < UnionSizeBits) {
        uint64_t PaddingBits = UnionSizeBits - LargestFieldSize;
        if (PaddingBits % 8 == 0) {
          Type *ByteType = IntegerType::get(Context, 8);
          Type *PaddingType = ArrayType::get(ByteType, PaddingBits / 8);
          FieldTypes.push_back(PaddingType);
        } else {
          Type *PaddingType = IntegerType::get(Context, PaddingBits);
          FieldTypes.push_back(PaddingType);
        }
      }
    }
  } else {
    uint64_t CurrentOffset = 0;

    for (const auto &Field : Fields) {
      if (!IsCoercedStr && Field.OffsetInBits > CurrentOffset) {
        uint64_t PaddingBits = Field.OffsetInBits - CurrentOffset;
        if (PaddingBits % 8 == 0 && PaddingBits >= 8) {
          Type *ByteType = IntegerType::get(Context, 8);
          Type *PaddingType = ArrayType::get(ByteType, PaddingBits / 8);
          FieldTypes.push_back(PaddingType);
        } else if (PaddingBits > 0) {
          Type *PaddingType = IntegerType::get(Context, PaddingBits);
          FieldTypes.push_back(PaddingType);
        }
        CurrentOffset = Field.OffsetInBits;
      }

      Type *FieldType = convertType(Field.FieldType);
      if (!FieldType)
        continue;

      if (Field.IsBitField && Field.BitFieldWidth > 0) {
        FieldType = IntegerType::get(Context, Field.BitFieldWidth);
        CurrentOffset += Field.BitFieldWidth;
      } else {
        FieldTypes.push_back(FieldType);
        if (auto *IntTy = dyn_cast<IntegerType>(FieldType)) {
          CurrentOffset += IntTy->getBitWidth();
        } else if (FieldType->isFloatingPointTy()) {
          CurrentOffset += FieldType->getPrimitiveSizeInBits();
        } else if (FieldType->isPointerTy()) {
          CurrentOffset += 64; // Assume 64-bit pointers
        } else {
          CurrentOffset += 64; // Conservative estimate
        }
      }
    }

    if (!IsCoercedStr) {
      uint64_t TotalSizeBits = Size.getFixedValue();
      if (CurrentOffset < TotalSizeBits) {
        uint64_t PaddingBits = TotalSizeBits - CurrentOffset;
        if (PaddingBits % 8 == 0 && PaddingBits >= 8) {
          Type *ByteType = IntegerType::get(Context, 8);
          Type *PaddingType = ArrayType::get(ByteType, PaddingBits / 8);
          FieldTypes.push_back(PaddingType);
        } else if (PaddingBits > 0) {
          Type *PaddingType = IntegerType::get(Context, PaddingBits);
          FieldTypes.push_back(PaddingType);
        }
      }
    }
  }

  return StructType::get(Context, FieldTypes, /*isPacked=*/false);
}
