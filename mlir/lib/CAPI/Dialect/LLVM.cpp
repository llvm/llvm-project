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

bool mlirTypeIsALLVMStructType(MlirType type) {
  return isa<LLVM::LLVMStructType>(unwrap(type));
}

bool mlirLLVMStructTypeIsLiteral(MlirType type) {
  return !cast<LLVM::LLVMStructType>(unwrap(type)).isIdentified();
}

intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).getBody().size();
}

MlirType mlirLLVMStructTypeGetElementType(MlirType type, intptr_t position) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getBody()[position]);
}

bool mlirLLVMStructTypeIsPacked(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isPacked();
}

MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getName());
}

bool mlirLLVMStructTypeIsOpaque(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isOpaque();
}

MlirType mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                                      MlirType const *fieldTypes,
                                      bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteral(
      unwrap(ctx), unwrapList(nFieldTypes, fieldTypes, fieldStorage),
      isPacked));
}

MlirType mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc,
                                             intptr_t nFieldTypes,
                                             MlirType const *fieldTypes,
                                             bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteralChecked(
      [loc]() { return emitError(unwrap(loc)); }, unwrap(loc)->getContext(),
      unwrapList(nFieldTypes, fieldTypes, fieldStorage), isPacked));
}

MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getOpaque(unwrap(name), unwrap(ctx)));
}

MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getIdentified(unwrap(ctx), unwrap(name)));
}

MlirType mlirLLVMStructTypeIdentifiedNewGet(MlirContext ctx, MlirStringRef name,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(LLVMStructType::getNewIdentified(
      unwrap(ctx), unwrap(name), unwrapList(nFieldTypes, fieldTypes, fields),
      isPacked));
}

MlirLogicalResult mlirLLVMStructTypeSetBody(MlirType structType,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(
      cast<LLVM::LLVMStructType>(unwrap(structType))
          .setBody(unwrapList(nFieldTypes, fieldTypes, fields), isPacked));
}
