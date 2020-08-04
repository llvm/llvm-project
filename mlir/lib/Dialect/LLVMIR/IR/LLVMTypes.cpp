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
// LLVMTypeNew.
//===----------------------------------------------------------------------===//

// TODO: when these types are registered with the LLVMDialect, this method
// should be removed and the regular Type::getDialect should just work.
LLVMDialect &LLVMTypeNew::getDialect() {
  return *getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
}

//----------------------------------------------------------------------------//
// Integer type utilities.

bool LLVMTypeNew::isIntegerTy(unsigned bitwidth) {
  if (auto intType = dyn_cast<LLVMIntegerType>())
    return intType.getBitWidth() == bitwidth;
  return false;
}

unsigned LLVMTypeNew::getIntegerBitWidth() {
  return cast<LLVMIntegerType>().getBitWidth();
}

LLVMTypeNew LLVMTypeNew::getArrayElementType() {
  return cast<LLVMArrayType>().getElementType();
}

//----------------------------------------------------------------------------//
// Array type utilities.

unsigned LLVMTypeNew::getArrayNumElements() {
  return cast<LLVMArrayType>().getNumElements();
}

bool LLVMTypeNew::isArrayTy() { return isa<LLVMArrayType>(); }

//----------------------------------------------------------------------------//
// Vector type utilities.

LLVMTypeNew LLVMTypeNew::getVectorElementType() {
  return cast<LLVMVectorType>().getElementType();
}

unsigned LLVMTypeNew::getVectorNumElements() {
  return cast<LLVMFixedVectorType>().getNumElements();
}
llvm::ElementCount LLVMTypeNew::getVectorElementCount() {
  return cast<LLVMVectorType>().getElementCount();
}

bool LLVMTypeNew::isVectorTy() { return isa<LLVMVectorType>(); }

//----------------------------------------------------------------------------//
// Function type utilities.

LLVMTypeNew LLVMTypeNew::getFunctionParamType(unsigned argIdx) {
  return cast<LLVMFunctionType>().getParamType(argIdx);
}

unsigned LLVMTypeNew::getFunctionNumParams() {
  return cast<LLVMFunctionType>().getNumParams();
}

LLVMTypeNew LLVMTypeNew::getFunctionResultType() {
  return cast<LLVMFunctionType>().getReturnType();
}

bool LLVMTypeNew::isFunctionTy() { return isa<LLVMFunctionType>(); }

bool LLVMTypeNew::isFunctionVarArg() {
  return cast<LLVMFunctionType>().isVarArg();
}

//----------------------------------------------------------------------------//
// Pointer type utilities.

LLVMTypeNew LLVMTypeNew::getPointerTo(unsigned addrSpace) {
  return LLVMPointerType::get(*this, addrSpace);
}

LLVMTypeNew LLVMTypeNew::getPointerElementTy() {
  return cast<LLVMPointerType>().getElementType();
}

bool LLVMTypeNew::isPointerTy() { return isa<LLVMPointerType>(); }

bool LLVMTypeNew::isValidPointerElementType(LLVMTypeNew type) {
  return !type.isa<LLVMVoidType>() && !type.isa<LLVMTokenType>() &&
         !type.isa<LLVMMetadataType>() && !type.isa<LLVMLabelType>();
}

//----------------------------------------------------------------------------//
// Struct type utilities.

LLVMTypeNew LLVMTypeNew::getStructElementType(unsigned i) {
  return cast<LLVMStructType>().getBody()[i];
}

unsigned LLVMTypeNew::getStructNumElements() {
  return cast<LLVMStructType>().getBody().size();
}

bool LLVMTypeNew::isStructTy() { return isa<LLVMStructType>(); }

//----------------------------------------------------------------------------//
// Utilities used to generate floating point types.

LLVMTypeNew LLVMTypeNew::getDoubleTy(LLVMDialect *dialect) {
  return LLVMDoubleType::get(dialect->getContext());
}

LLVMTypeNew LLVMTypeNew::getFloatTy(LLVMDialect *dialect) {
  return LLVMFloatType::get(dialect->getContext());
}

LLVMTypeNew LLVMTypeNew::getBFloatTy(LLVMDialect *dialect) {
  return LLVMBFloatType::get(dialect->getContext());
}

LLVMTypeNew LLVMTypeNew::getHalfTy(LLVMDialect *dialect) {
  return LLVMHalfType::get(dialect->getContext());
}

LLVMTypeNew LLVMTypeNew::getFP128Ty(LLVMDialect *dialect) {
  return LLVMFP128Type::get(dialect->getContext());
}

LLVMTypeNew LLVMTypeNew::getX86_FP80Ty(LLVMDialect *dialect) {
  return LLVMX86FP80Type::get(dialect->getContext());
}

//----------------------------------------------------------------------------//
// Utilities used to generate integer types.

LLVMTypeNew LLVMTypeNew::getIntNTy(LLVMDialect *dialect, unsigned numBits) {
  return LLVMIntegerType::get(dialect->getContext(), numBits);
}

//----------------------------------------------------------------------------//
// Utilities used to generate other miscellaneous types.

LLVMTypeNew LLVMTypeNew::getArrayTy(LLVMTypeNew elementType,
                                    uint64_t numElements) {
  return LLVMArrayType::get(elementType, numElements);
}

LLVMTypeNew LLVMTypeNew::getFunctionTy(LLVMTypeNew result,
                                       ArrayRef<LLVMTypeNew> params,
                                       bool isVarArg) {
  return LLVMFunctionType::get(result, params, isVarArg);
}

LLVMTypeNew LLVMTypeNew::getStructTy(LLVMDialect *dialect,
                                     ArrayRef<LLVMTypeNew> elements,
                                     bool isPacked) {
  return LLVMStructType::getLiteral(dialect->getContext(), elements, isPacked);
}

LLVMTypeNew LLVMTypeNew::getVectorTy(LLVMTypeNew elementType,
                                     unsigned numElements) {
  return LLVMFixedVectorType::get(elementType, numElements);
}

//----------------------------------------------------------------------------//
// Void type utilities.

LLVMTypeNew LLVMTypeNew::getVoidTy(LLVMDialect *dialect) {
  return LLVMVoidType::get(dialect->getContext());
}

bool LLVMTypeNew::isVoidTy() { return isa<LLVMVoidType>(); }

//----------------------------------------------------------------------------//
// Creation and setting of LLVM's identified struct types

LLVMTypeNew LLVMTypeNew::createStructTy(LLVMDialect *dialect,
                                        ArrayRef<LLVMTypeNew> elements,
                                        Optional<StringRef> name,
                                        bool isPacked) {
  assert(name.hasValue() &&
         "identified structs with no identifier not supported");
  StringRef stringNameBase = name.getValueOr("");
  std::string stringName = stringNameBase.str();
  unsigned counter = 0;
  do {
    auto type =
        LLVMStructType::getIdentified(dialect->getContext(), stringName);
    if (type.isInitialized() || failed(type.setBody(elements, isPacked))) {
      counter += 1;
      stringName =
          (Twine(stringNameBase) + "." + std::to_string(counter)).str();
      continue;
    }
    return type;
  } while (true);
}

LLVMTypeNew LLVMTypeNew::setStructTyBody(LLVMTypeNew structType,
                                         ArrayRef<LLVMTypeNew> elements,
                                         bool isPacked) {
  LogicalResult couldSet =
      structType.cast<LLVMStructType>().setBody(elements, isPacked);
  assert(succeeded(couldSet) && "failed to set the body");
  (void)couldSet;
  return structType;
}

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
bool LLVMStructType::isInitialized() { return getImpl()->isInitialized(); }
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
