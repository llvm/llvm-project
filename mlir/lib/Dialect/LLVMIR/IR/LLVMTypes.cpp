//===- LLVMTypes.cpp - MLIR LLVM Dialect types ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the types for the LLVM dialect in MLIR. These MLIR types
// correspond to the LLVM IR type system.
//
//===----------------------------------------------------------------------===//

#include "TypeDetail.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Array type.

LLVMArrayType LLVMArrayType::get(LLVMTypeNew elementType,
                                 unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMTypeNew::ArrayType,
                   elementType, numElements);
}

LLVMTypeNew LLVMArrayType::getElementType() { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() { return getImpl()->numElements; }

//===----------------------------------------------------------------------===//
// Function type.

LLVMFunctionType LLVMFunctionType::get(LLVMTypeNew result,
                                       ArrayRef<LLVMTypeNew> arguments,
                                       bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::get(result.getContext(), LLVMTypeNew::FunctionType, result,
                   arguments, isVarArg);
}

LLVMTypeNew LLVMFunctionType::getReturnType() {
  return getImpl()->getReturnType();
}

unsigned LLVMFunctionType::getNumParams() {
  return getImpl()->getArgumentTypes().size();
}

LLVMTypeNew LLVMFunctionType::getParamType(unsigned i) {
  return getImpl()->getArgumentTypes()[i];
}

bool LLVMFunctionType::isVarArg() { return getImpl()->isVariadic(); }

ArrayRef<LLVMTypeNew> LLVMFunctionType::getParams() {
  return getImpl()->getArgumentTypes();
}

//===----------------------------------------------------------------------===//
// Integer type.

LLVMIntegerType LLVMIntegerType::get(MLIRContext *ctx, unsigned bitwidth) {
  return Base::get(ctx, LLVMTypeNew::IntegerType, bitwidth);
}

unsigned LLVMIntegerType::getBitWidth() { return getImpl()->bitwidth; }

//===----------------------------------------------------------------------===//
// Pointer type.

LLVMPointerType LLVMPointerType::get(LLVMTypeNew pointee,
                                     unsigned addressSpace) {
  assert(pointee && "expected non-null subtype");
  return Base::get(pointee.getContext(), LLVMTypeNew::PointerType, pointee,
                   addressSpace);
}

LLVMTypeNew LLVMPointerType::getElementType() { return getImpl()->pointeeType; }

unsigned LLVMPointerType::getAddressSpace() { return getImpl()->addressSpace; }

//===----------------------------------------------------------------------===//
// Struct type.

LLVMStructType LLVMStructType::getIdentified(MLIRContext *context,
                                             StringRef name) {
  return Base::get(context, LLVMTypeNew::StructType, name, /*opaque=*/false);
}

LLVMStructType LLVMStructType::getLiteral(MLIRContext *context,
                                          ArrayRef<LLVMTypeNew> types,
                                          bool isPacked) {
  return Base::get(context, LLVMTypeNew::StructType, types, isPacked);
}

LLVMStructType LLVMStructType::getOpaque(StringRef name, MLIRContext *context) {
  return Base::get(context, LLVMTypeNew::StructType, name, /*opaque=*/true);
}

LogicalResult LLVMStructType::setBody(ArrayRef<LLVMTypeNew> types,
                                      bool isPacked) {
  assert(isIdentified() && "can only set bodies of identified structs");
  return Base::mutate(types, isPacked);
}

bool LLVMStructType::isPacked() { return getImpl()->isPacked(); }
bool LLVMStructType::isIdentified() { return getImpl()->isIdentified(); }
bool LLVMStructType::isOpaque() {
  return getImpl()->isOpaque() || !getImpl()->isInitialized();
}
StringRef LLVMStructType::getName() { return getImpl()->getIdentifier(); }
ArrayRef<LLVMTypeNew> LLVMStructType::getBody() {
  return isIdentified() ? getImpl()->getIdentifiedStructBody()
                        : getImpl()->getTypeList();
}

//===----------------------------------------------------------------------===//
// Vector types.

LLVMTypeNew LLVMVectorType::getElementType() {
  // Both derived classes share the implementation type.
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

llvm::ElementCount LLVMVectorType::getElementCount() {
  // Both derived classes share the implementation type.
  return llvm::ElementCount(
      static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->numElements,
      this->isa<LLVMScalableVectorType>());
}

LLVMFixedVectorType LLVMFixedVectorType::get(LLVMTypeNew elementType,
                                             unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMTypeNew::FixedVectorType,
                   elementType, numElements)
      .cast<LLVMFixedVectorType>();
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

LLVMScalableVectorType LLVMScalableVectorType::get(LLVMTypeNew elementType,
                                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMTypeNew::ScalableVectorType,
                   elementType, minNumElements)
      .cast<LLVMScalableVectorType>();
}

unsigned LLVMScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}
