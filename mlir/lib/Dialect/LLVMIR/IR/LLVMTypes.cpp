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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// LLVMType.
//===----------------------------------------------------------------------===//

bool LLVMType::classof(Type type) {
  return llvm::isa<LLVMDialect>(type.getDialect());
}

LLVMDialect &LLVMType::getDialect() {
  return static_cast<LLVMDialect &>(Type::getDialect());
}

//----------------------------------------------------------------------------//
// Integer type utilities.

bool LLVMType::isIntegerTy(unsigned bitwidth) {
  if (auto intType = dyn_cast<LLVMIntegerType>())
    return intType.getBitWidth() == bitwidth;
  return false;
}
unsigned LLVMType::getIntegerBitWidth() {
  return cast<LLVMIntegerType>().getBitWidth();
}

LLVMType LLVMType::getArrayElementType() {
  return cast<LLVMArrayType>().getElementType();
}

//----------------------------------------------------------------------------//
// Array type utilities.

unsigned LLVMType::getArrayNumElements() {
  return cast<LLVMArrayType>().getNumElements();
}

bool LLVMType::isArrayTy() { return isa<LLVMArrayType>(); }

//----------------------------------------------------------------------------//
// Vector type utilities.

LLVMType LLVMType::getVectorElementType() {
  return cast<LLVMVectorType>().getElementType();
}

unsigned LLVMType::getVectorNumElements() {
  return cast<LLVMFixedVectorType>().getNumElements();
}
llvm::ElementCount LLVMType::getVectorElementCount() {
  return cast<LLVMVectorType>().getElementCount();
}

bool LLVMType::isVectorTy() { return isa<LLVMVectorType>(); }

//----------------------------------------------------------------------------//
// Function type utilities.

LLVMType LLVMType::getFunctionParamType(unsigned argIdx) {
  return cast<LLVMFunctionType>().getParamType(argIdx);
}

unsigned LLVMType::getFunctionNumParams() {
  return cast<LLVMFunctionType>().getNumParams();
}

LLVMType LLVMType::getFunctionResultType() {
  return cast<LLVMFunctionType>().getReturnType();
}

bool LLVMType::isFunctionTy() { return isa<LLVMFunctionType>(); }

bool LLVMType::isFunctionVarArg() {
  return cast<LLVMFunctionType>().isVarArg();
}

//----------------------------------------------------------------------------//
// Pointer type utilities.

LLVMType LLVMType::getPointerTo(unsigned addrSpace) {
  return LLVMPointerType::get(*this, addrSpace);
}

LLVMType LLVMType::getPointerElementTy() {
  return cast<LLVMPointerType>().getElementType();
}

bool LLVMType::isPointerTy() { return isa<LLVMPointerType>(); }

bool LLVMType::isValidPointerElementType(LLVMType type) {
  return !type.isa<LLVMVoidType>() && !type.isa<LLVMTokenType>() &&
         !type.isa<LLVMMetadataType>() && !type.isa<LLVMLabelType>();
}

//----------------------------------------------------------------------------//
// Struct type utilities.

LLVMType LLVMType::getStructElementType(unsigned i) {
  return cast<LLVMStructType>().getBody()[i];
}

unsigned LLVMType::getStructNumElements() {
  return cast<LLVMStructType>().getBody().size();
}

bool LLVMType::isStructTy() { return isa<LLVMStructType>(); }

//----------------------------------------------------------------------------//
// Utilities used to generate floating point types.

LLVMType LLVMType::getDoubleTy(MLIRContext *context) {
  return LLVMDoubleType::get(context);
}

LLVMType LLVMType::getFloatTy(MLIRContext *context) {
  return LLVMFloatType::get(context);
}

LLVMType LLVMType::getBFloatTy(MLIRContext *context) {
  return LLVMBFloatType::get(context);
}

LLVMType LLVMType::getHalfTy(MLIRContext *context) {
  return LLVMHalfType::get(context);
}

LLVMType LLVMType::getFP128Ty(MLIRContext *context) {
  return LLVMFP128Type::get(context);
}

LLVMType LLVMType::getX86_FP80Ty(MLIRContext *context) {
  return LLVMX86FP80Type::get(context);
}

//----------------------------------------------------------------------------//
// Utilities used to generate integer types.

LLVMType LLVMType::getIntNTy(MLIRContext *context, unsigned numBits) {
  return LLVMIntegerType::get(context, numBits);
}

//----------------------------------------------------------------------------//
// Utilities used to generate other miscellaneous types.

LLVMType LLVMType::getArrayTy(LLVMType elementType, uint64_t numElements) {
  return LLVMArrayType::get(elementType, numElements);
}

LLVMType LLVMType::getFunctionTy(LLVMType result, ArrayRef<LLVMType> params,
                                 bool isVarArg) {
  return LLVMFunctionType::get(result, params, isVarArg);
}

LLVMType LLVMType::getStructTy(MLIRContext *context,
                               ArrayRef<LLVMType> elements, bool isPacked) {
  return LLVMStructType::getLiteral(context, elements, isPacked);
}

LLVMType LLVMType::getVectorTy(LLVMType elementType, unsigned numElements) {
  return LLVMFixedVectorType::get(elementType, numElements);
}

//----------------------------------------------------------------------------//
// Void type utilities.

LLVMType LLVMType::getVoidTy(MLIRContext *context) {
  return LLVMVoidType::get(context);
}

bool LLVMType::isVoidTy() { return isa<LLVMVoidType>(); }

//----------------------------------------------------------------------------//
// Creation and setting of LLVM's identified struct types

LLVMType LLVMType::createStructTy(MLIRContext *context,
                                  ArrayRef<LLVMType> elements,
                                  Optional<StringRef> name, bool isPacked) {
  assert(name.hasValue() &&
         "identified structs with no identifier not supported");
  StringRef stringNameBase = name.getValueOr("");
  std::string stringName = stringNameBase.str();
  unsigned counter = 0;
  do {
    auto type = LLVMStructType::getIdentified(context, stringName);
    if (type.isInitialized() || failed(type.setBody(elements, isPacked))) {
      counter += 1;
      stringName =
          (Twine(stringNameBase) + "." + std::to_string(counter)).str();
      continue;
    }
    return type;
  } while (true);
}

LLVMType LLVMType::setStructTyBody(LLVMType structType,
                                   ArrayRef<LLVMType> elements, bool isPacked) {
  LogicalResult couldSet =
      structType.cast<LLVMStructType>().setBody(elements, isPacked);
  assert(succeeded(couldSet) && "failed to set the body");
  (void)couldSet;
  return structType;
}

//===----------------------------------------------------------------------===//
// Array type.

LLVMArrayType LLVMArrayType::get(LLVMType elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMType::ArrayType, elementType,
                   numElements);
}

LLVMType LLVMArrayType::getElementType() { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() { return getImpl()->numElements; }

//===----------------------------------------------------------------------===//
// Function type.

LLVMFunctionType LLVMFunctionType::get(LLVMType result,
                                       ArrayRef<LLVMType> arguments,
                                       bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::get(result.getContext(), LLVMType::FunctionType, result,
                   arguments, isVarArg);
}

LLVMType LLVMFunctionType::getReturnType() {
  return getImpl()->getReturnType();
}

unsigned LLVMFunctionType::getNumParams() {
  return getImpl()->getArgumentTypes().size();
}

LLVMType LLVMFunctionType::getParamType(unsigned i) {
  return getImpl()->getArgumentTypes()[i];
}

bool LLVMFunctionType::isVarArg() { return getImpl()->isVariadic(); }

ArrayRef<LLVMType> LLVMFunctionType::getParams() {
  return getImpl()->getArgumentTypes();
}

//===----------------------------------------------------------------------===//
// Integer type.

LLVMIntegerType LLVMIntegerType::get(MLIRContext *ctx, unsigned bitwidth) {
  return Base::get(ctx, LLVMType::IntegerType, bitwidth);
}

unsigned LLVMIntegerType::getBitWidth() { return getImpl()->bitwidth; }

//===----------------------------------------------------------------------===//
// Pointer type.

LLVMPointerType LLVMPointerType::get(LLVMType pointee, unsigned addressSpace) {
  assert(pointee && "expected non-null subtype");
  return Base::get(pointee.getContext(), LLVMType::PointerType, pointee,
                   addressSpace);
}

LLVMType LLVMPointerType::getElementType() { return getImpl()->pointeeType; }

unsigned LLVMPointerType::getAddressSpace() { return getImpl()->addressSpace; }

//===----------------------------------------------------------------------===//
// Struct type.

LLVMStructType LLVMStructType::getIdentified(MLIRContext *context,
                                             StringRef name) {
  return Base::get(context, LLVMType::StructType, name, /*opaque=*/false);
}

LLVMStructType LLVMStructType::getLiteral(MLIRContext *context,
                                          ArrayRef<LLVMType> types,
                                          bool isPacked) {
  return Base::get(context, LLVMType::StructType, types, isPacked);
}

LLVMStructType LLVMStructType::getOpaque(StringRef name, MLIRContext *context) {
  return Base::get(context, LLVMType::StructType, name, /*opaque=*/true);
}

LogicalResult LLVMStructType::setBody(ArrayRef<LLVMType> types, bool isPacked) {
  assert(isIdentified() && "can only set bodies of identified structs");
  return Base::mutate(types, isPacked);
}

bool LLVMStructType::isPacked() { return getImpl()->isPacked(); }
bool LLVMStructType::isIdentified() { return getImpl()->isIdentified(); }
bool LLVMStructType::isOpaque() {
  return getImpl()->isOpaque() || !getImpl()->isInitialized();
}
bool LLVMStructType::isInitialized() { return getImpl()->isInitialized(); }
StringRef LLVMStructType::getName() { return getImpl()->getIdentifier(); }
ArrayRef<LLVMType> LLVMStructType::getBody() {
  return isIdentified() ? getImpl()->getIdentifiedStructBody()
                        : getImpl()->getTypeList();
}

//===----------------------------------------------------------------------===//
// Vector types.

/// Support type casting functionality.
bool LLVMVectorType::classof(Type type) {
  return type.isa<LLVMFixedVectorType, LLVMScalableVectorType>();
}

LLVMType LLVMVectorType::getElementType() {
  // Both derived classes share the implementation type.
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

llvm::ElementCount LLVMVectorType::getElementCount() {
  // Both derived classes share the implementation type.
  return llvm::ElementCount(
      static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->numElements,
      isa<LLVMScalableVectorType>());
}

LLVMFixedVectorType LLVMFixedVectorType::get(LLVMType elementType,
                                             unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMType::FixedVectorType,
                   elementType, numElements)
      .cast<LLVMFixedVectorType>();
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

LLVMScalableVectorType LLVMScalableVectorType::get(LLVMType elementType,
                                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), LLVMType::ScalableVectorType,
                   elementType, minNumElements)
      .cast<LLVMScalableVectorType>();
}

unsigned LLVMScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}
