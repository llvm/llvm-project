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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Array type.
//===----------------------------------------------------------------------===//

bool LLVMArrayType::isValidElementType(Type type) {
  return !type.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                   LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>();
}

LLVMArrayType LLVMArrayType::get(Type elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

LLVMArrayType LLVMArrayType::getChecked(Location loc, Type elementType,
                                        unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, numElements);
}

Type LLVMArrayType::getElementType() { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() { return getImpl()->numElements; }

LogicalResult
LLVMArrayType::verifyConstructionInvariants(Location loc, Type elementType,
                                            unsigned numElements) {
  if (!isValidElementType(elementType))
    return emitError(loc, "invalid array element type: ") << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

bool LLVMFunctionType::isValidArgumentType(Type type) {
  return !type.isa<LLVMVoidType, LLVMFunctionType>();
}

bool LLVMFunctionType::isValidResultType(Type type) {
  return !type.isa<LLVMFunctionType, LLVMMetadataType, LLVMLabelType>();
}

LLVMFunctionType LLVMFunctionType::get(Type result, ArrayRef<Type> arguments,
                                       bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::get(result.getContext(), result, arguments, isVarArg);
}

LLVMFunctionType LLVMFunctionType::getChecked(Location loc, Type result,
                                              ArrayRef<Type> arguments,
                                              bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::getChecked(loc, result, arguments, isVarArg);
}

Type LLVMFunctionType::getReturnType() { return getImpl()->getReturnType(); }

unsigned LLVMFunctionType::getNumParams() {
  return getImpl()->getArgumentTypes().size();
}

Type LLVMFunctionType::getParamType(unsigned i) {
  return getImpl()->getArgumentTypes()[i];
}

bool LLVMFunctionType::isVarArg() { return getImpl()->isVariadic(); }

ArrayRef<Type> LLVMFunctionType::getParams() {
  return getImpl()->getArgumentTypes();
}

LogicalResult
LLVMFunctionType::verifyConstructionInvariants(Location loc, Type result,
                                               ArrayRef<Type> arguments, bool) {
  if (!isValidResultType(result))
    return emitError(loc, "invalid function result type: ") << result;

  for (Type arg : arguments)
    if (!isValidArgumentType(arg))
      return emitError(loc, "invalid function argument type: ") << arg;

  return success();
}

//===----------------------------------------------------------------------===//
// Pointer type.
//===----------------------------------------------------------------------===//

bool LLVMPointerType::isValidElementType(Type type) {
  return !type.isa<LLVMVoidType, LLVMTokenType, LLVMMetadataType,
                   LLVMLabelType>();
}

LLVMPointerType LLVMPointerType::get(Type pointee, unsigned addressSpace) {
  assert(pointee && "expected non-null subtype");
  return Base::get(pointee.getContext(), pointee, addressSpace);
}

LLVMPointerType LLVMPointerType::getChecked(Location loc, Type pointee,
                                            unsigned addressSpace) {
  return Base::getChecked(loc, pointee, addressSpace);
}

Type LLVMPointerType::getElementType() { return getImpl()->pointeeType; }

unsigned LLVMPointerType::getAddressSpace() { return getImpl()->addressSpace; }

LogicalResult LLVMPointerType::verifyConstructionInvariants(Location loc,
                                                            Type pointee,
                                                            unsigned) {
  if (!isValidElementType(pointee))
    return emitError(loc, "invalid pointer element type: ") << pointee;
  return success();
}

//===----------------------------------------------------------------------===//
// Struct type.
//===----------------------------------------------------------------------===//

bool LLVMStructType::isValidElementType(Type type) {
  return !type.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                   LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>();
}

LLVMStructType LLVMStructType::getIdentified(MLIRContext *context,
                                             StringRef name) {
  return Base::get(context, name, /*opaque=*/false);
}

LLVMStructType LLVMStructType::getIdentifiedChecked(Location loc,
                                                    StringRef name) {
  return Base::getChecked(loc, name, /*opaque=*/false);
}

LLVMStructType LLVMStructType::getNewIdentified(MLIRContext *context,
                                                StringRef name,
                                                ArrayRef<Type> elements,
                                                bool isPacked) {
  std::string stringName = name.str();
  unsigned counter = 0;
  do {
    auto type = LLVMStructType::getIdentified(context, stringName);
    if (type.isInitialized() || failed(type.setBody(elements, isPacked))) {
      counter += 1;
      stringName = (Twine(name) + "." + std::to_string(counter)).str();
      continue;
    }
    return type;
  } while (true);
}

LLVMStructType LLVMStructType::getLiteral(MLIRContext *context,
                                          ArrayRef<Type> types, bool isPacked) {
  return Base::get(context, types, isPacked);
}

LLVMStructType LLVMStructType::getLiteralChecked(Location loc,
                                                 ArrayRef<Type> types,
                                                 bool isPacked) {
  return Base::getChecked(loc, types, isPacked);
}

LLVMStructType LLVMStructType::getOpaque(StringRef name, MLIRContext *context) {
  return Base::get(context, name, /*opaque=*/true);
}

LLVMStructType LLVMStructType::getOpaqueChecked(Location loc, StringRef name) {
  return Base::getChecked(loc, name, /*opaque=*/true);
}

LogicalResult LLVMStructType::setBody(ArrayRef<Type> types, bool isPacked) {
  assert(isIdentified() && "can only set bodies of identified structs");
  assert(llvm::all_of(types, LLVMStructType::isValidElementType) &&
         "expected valid body types");
  return Base::mutate(types, isPacked);
}

bool LLVMStructType::isPacked() { return getImpl()->isPacked(); }
bool LLVMStructType::isIdentified() { return getImpl()->isIdentified(); }
bool LLVMStructType::isOpaque() {
  return getImpl()->isIdentified() &&
         (getImpl()->isOpaque() || !getImpl()->isInitialized());
}
bool LLVMStructType::isInitialized() { return getImpl()->isInitialized(); }
StringRef LLVMStructType::getName() { return getImpl()->getIdentifier(); }
ArrayRef<Type> LLVMStructType::getBody() {
  return isIdentified() ? getImpl()->getIdentifiedStructBody()
                        : getImpl()->getTypeList();
}

LogicalResult LLVMStructType::verifyConstructionInvariants(Location, StringRef,
                                                           bool) {
  return success();
}

LogicalResult LLVMStructType::verifyConstructionInvariants(Location loc,
                                                           ArrayRef<Type> types,
                                                           bool) {
  for (Type t : types)
    if (!isValidElementType(t))
      return emitError(loc, "invalid LLVM structure element type: ") << t;

  return success();
}

//===----------------------------------------------------------------------===//
// Vector types.
//===----------------------------------------------------------------------===//

bool LLVMVectorType::isValidElementType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();
  return type.isa<LLVMPointerType>() ||
         mlir::LLVM::isCompatibleFloatingPointType(type);
}

/// Support type casting functionality.
bool LLVMVectorType::classof(Type type) {
  return type.isa<LLVMFixedVectorType, LLVMScalableVectorType>();
}

Type LLVMVectorType::getElementType() {
  // Both derived classes share the implementation type.
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

llvm::ElementCount LLVMVectorType::getElementCount() {
  // Both derived classes share the implementation type.
  return llvm::ElementCount::get(
      static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->numElements,
      isa<LLVMScalableVectorType>());
}

/// Verifies that the type about to be constructed is well-formed.
LogicalResult
LLVMVectorType::verifyConstructionInvariants(Location loc, Type elementType,
                                             unsigned numElements) {
  if (numElements == 0)
    return emitError(loc, "the number of vector elements must be positive");

  if (!isValidElementType(elementType))
    return emitError(loc, "invalid vector element type");

  return success();
}

LLVMFixedVectorType LLVMFixedVectorType::get(Type elementType,
                                             unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

LLVMFixedVectorType LLVMFixedVectorType::getChecked(Location loc,
                                                    Type elementType,
                                                    unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, numElements);
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

LLVMScalableVectorType LLVMScalableVectorType::get(Type elementType,
                                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, minNumElements);
}

LLVMScalableVectorType
LLVMScalableVectorType::getChecked(Location loc, Type elementType,
                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, minNumElements);
}

unsigned LLVMScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool mlir::LLVM::isCompatibleType(Type type) {
  // Only signless integers are compatible.
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();

  // clang-format off
  return type.isa<
      LLVMArrayType,
      LLVMBFloatType,
      LLVMDoubleType,
      LLVMFP128Type,
      LLVMFloatType,
      LLVMFunctionType,
      LLVMHalfType,
      LLVMLabelType,
      LLVMMetadataType,
      LLVMPPCFP128Type,
      LLVMPointerType,
      LLVMStructType,
      LLVMTokenType,
      LLVMVectorType,
      LLVMVoidType,
      LLVMX86FP80Type,
      LLVMX86MMXType
  >();
  // clang-format on
}

llvm::TypeSize mlir::LLVM::getPrimitiveTypeSizeInBits(Type type) {
  assert(isCompatibleType(type) &&
         "expected a type compatible with the LLVM dialect");

  return llvm::TypeSwitch<Type, llvm::TypeSize>(type)
      .Case<LLVMHalfType, LLVMBFloatType>(
          [](Type) { return llvm::TypeSize::Fixed(16); })
      .Case<LLVMFloatType>([](Type) { return llvm::TypeSize::Fixed(32); })
      .Case<LLVMDoubleType, LLVMX86MMXType>(
          [](Type) { return llvm::TypeSize::Fixed(64); })
      .Case<IntegerType>([](IntegerType intTy) {
        return llvm::TypeSize::Fixed(intTy.getWidth());
      })
      .Case<LLVMX86FP80Type>([](Type) { return llvm::TypeSize::Fixed(80); })
      .Case<LLVMPPCFP128Type, LLVMFP128Type>(
          [](Type) { return llvm::TypeSize::Fixed(128); })
      .Case<LLVMVectorType>([](LLVMVectorType t) {
        llvm::TypeSize elementSize =
            getPrimitiveTypeSizeInBits(t.getElementType());
        llvm::ElementCount elementCount = t.getElementCount();
        assert(!elementSize.isScalable() &&
               "vector type should have fixed-width elements");
        return llvm::TypeSize(elementSize.getFixedSize() *
                                  elementCount.getKnownMinValue(),
                              elementCount.isScalable());
      })
      .Default([](Type ty) {
        assert((ty.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                       LLVMTokenType, LLVMStructType, LLVMArrayType,
                       LLVMPointerType, LLVMFunctionType>()) &&
               "unexpected missing support for primitive type");
        return llvm::TypeSize::Fixed(0);
      });
}
