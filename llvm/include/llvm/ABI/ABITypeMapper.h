//===---- ABITypeMapper.h - Maps LLVM ABI Types to LLVM IR Types --------===//
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

#ifndef LLVM_CODEGEN_ABITYPEMAPPER_H
#define LLVM_CODEGEN_ABITYPEMAPPER_H

#include "llvm/ABI/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {

class ABITypeMapper {
public:
  explicit ABITypeMapper(LLVMContext &Ctx, const DataLayout &DataLayout)
      : Context(Ctx), DL(DataLayout) {}

  Type *convertType(const abi::Type *ABIType);

  void clearCache() { TypeCache.clear(); }

private:
  LLVMContext &Context;
  const DataLayout &DL;

  DenseMap<const abi::Type *, Type *> TypeCache;

  Type *convertIntegerType(const abi::IntegerType *IT);

  Type *convertFieldType(const abi::Type *FieldType);

  Type *convertFloatType(const abi::FloatType *FT);

  Type *convertPointerType(const abi::PointerType *PT);

  Type *convertArrayType(const abi::ArrayType *AT);

  Type *convertMatrixType(const abi::ArrayType *MT);

  Type *convertVectorType(const abi::VectorType *VT);

  Type *convertStructType(const abi::StructType *ST);

  Type *convertVoidType(const abi::VoidType *VT);

  Type *getFloatTypeForSemantics(const fltSemantics &Semantics);

  StructType *createStructFromFields(ArrayRef<abi::FieldInfo> Fields,
                                     uint32_t NumFields, TypeSize Size,
                                     Align Alignment, bool IsUnion = false,
                                     bool IsCoercedStr = false);
  Type *convertComplexType(const abi::ComplexType *CT);

  Type *convertMemberPointerType(const abi::MemberPointerType *MPT);
};

} // namespace llvm

#endif // LLVM_CODEGEN_ABITYPEMAPPER_H
