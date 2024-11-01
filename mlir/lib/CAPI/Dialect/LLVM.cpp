//===- LLVM.cpp - C Interface for LLVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/LLVM.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::LLVM;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLVM, llvm, LLVMDialect)

MlirType mlirLLVMPointerTypeGet(MlirContext ctx, unsigned addressSpace) {
  return wrap(LLVMPointerType::get(unwrap(ctx), addressSpace));
}

MlirType mlirLLVMVoidTypeGet(MlirContext ctx) {
  return wrap(LLVMVoidType::get(unwrap(ctx)));
}

MlirType mlirLLVMArrayTypeGet(MlirType elementType, unsigned numElements) {
  return wrap(LLVMArrayType::get(unwrap(elementType), numElements));
}

MlirType mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                                 MlirType const *argumentTypes, bool isVarArg) {
  SmallVector<Type, 2> argumentStorage;
  return wrap(LLVMFunctionType::get(
      unwrap(resultType),
      unwrapList(nArgumentTypes, argumentTypes, argumentStorage), isVarArg));
}

MlirType mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                                      MlirType const *fieldTypes,
                                      bool isPacked) {
  SmallVector<Type, 2> fieldStorage;
  return wrap(LLVMStructType::getLiteral(
      unwrap(ctx), unwrapList(nFieldTypes, fieldTypes, fieldStorage),
      isPacked));
}
