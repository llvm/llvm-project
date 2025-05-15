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

namespace llvm::abi {

class ABITypeMapper {
public:
  explicit ABITypeMapper(LLVMContext &Ctx, const DataLayout &DL)
      : Context(Ctx), DL(DL) {}

  llvm::Type *convertType(const abi::Type *ABIType);

  void clearCache() { TypeCache.clear(); }

private:
  LLVMContext &Context;
  const DataLayout &DL;

  llvm::DenseMap<const abi::Type *, llvm::Type *> TypeCache;

  llvm::Type *convertArrayType(const abi::ArrayType *AT);

  llvm::Type *convertMatrixType(const abi::ArrayType *MT);

  llvm::Type *convertVectorType(const abi::VectorType *VT);

  llvm::Type *convertRecordType(const abi::RecordType *RT);

  llvm::Type *getFloatTypeForSemantics(const fltSemantics &Semantics);

  llvm::StructType *createStructFromFields(ArrayRef<abi::FieldInfo> Fields,
                                           TypeSize Size, Align Alignment,
                                           bool IsUnion = false,
                                           bool IsCoercedStr = false);
  llvm::Type *createPaddingType(uint64_t PaddingBits);
  llvm::Type *convertComplexType(const abi::ComplexType *CT);

  llvm::Type *convertMemberPointerType(const abi::MemberPointerType *MPT);
};

} // namespace llvm::abi

#endif // LLVM_CODEGEN_ABITYPEMAPPER_H
