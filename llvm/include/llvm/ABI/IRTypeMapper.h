//===---- IRTypeMapper.h - Maps LLVM ABI Types to LLVM IR Types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps LLVM ABI type representations back to corresponding LLVM IR types.
/// Used by frontends after the ABI library has computed argument/return
/// classification: coerce-to types in the ABI representation must be
/// translated to llvm::Type before being handed back to the IR builder.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_IRTYPEMAPPER_H
#define LLVM_ABI_IRTYPEMAPPER_H

#include "llvm/ABI/Types.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class LLVMContext;
class Type;
class StructType;
class DataLayout;

namespace abi {

class IRTypeMapper {
public:
  IRTypeMapper(LLVMContext &Ctx, const DataLayout &DL) : Context(Ctx), DL(DL) {}

  llvm::Type *convertType(const abi::Type *ABIType);

  void clearCache() { TypeCache.clear(); }

private:
  LLVMContext &Context;
  const DataLayout &DL;

  llvm::DenseMap<const abi::Type *, llvm::Type *> TypeCache;

  llvm::Type *convertArrayType(const abi::ArrayType *AT);
  llvm::Type *convertVectorType(const abi::VectorType *VT);
  llvm::Type *convertRecordType(const abi::RecordType *RT);
  llvm::Type *convertComplexType(const abi::ComplexType *CT);
  llvm::Type *convertMemberPointerType(const abi::MemberPointerType *MPT);

  llvm::StructType *createStructFromFields(ArrayRef<abi::FieldInfo> Fields,
                                           TypeSize Size, Align Alignment,
                                           bool IsUnion);
  llvm::Type *createPaddingType(uint64_t PaddingBits);
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABITYPEMAPPER_H
