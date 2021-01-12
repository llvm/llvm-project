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

/// Verifies that the type about to be constructed is well-formed.
template <typename VecTy>
static LogicalResult verifyVectorConstructionInvariants(Location loc,
                                                        Type elementType,
                                                        unsigned numElements) {
  if (numElements == 0)
    return emitError(loc, "the number of vector elements must be positive");

  if (!VecTy::isValidElementType(elementType))
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

Type LLVMFixedVectorType::getElementType() {
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

bool LLVMFixedVectorType::isValidElementType(Type type) {
  return type
      .isa<LLVMPointerType, LLVMX86FP80Type, LLVMFP128Type, LLVMPPCFP128Type>();
}

LogicalResult LLVMFixedVectorType::verifyConstructionInvariants(
    Location loc, Type elementType, unsigned numElements) {
  return verifyVectorConstructionInvariants<LLVMFixedVectorType>(
      loc, elementType, numElements);
}

//===----------------------------------------------------------------------===//
// LLVMScalableVectorType.
//===----------------------------------------------------------------------===//

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

Type LLVMScalableVectorType::getElementType() {
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

unsigned LLVMScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}

bool LLVMScalableVectorType::isValidElementType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();

  return isCompatibleFloatingPointType(type) || type.isa<LLVMPointerType>();
}

LogicalResult LLVMScalableVectorType::verifyConstructionInvariants(
    Location loc, Type elementType, unsigned numElements) {
  return verifyVectorConstructionInvariants<LLVMScalableVectorType>(
      loc, elementType, numElements);
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool mlir::LLVM::isCompatibleType(Type type) {
  // Only signless integers are compatible.
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();

  // 1D vector types are compatible if their element types are.
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getRank() == 1 && isCompatibleType(vecType.getElementType());

  // clang-format off
  return type.isa<
      BFloat16Type,
      Float16Type,
      Float32Type,
      Float64Type,
      LLVMArrayType,
      LLVMFP128Type,
      LLVMFunctionType,
      LLVMLabelType,
      LLVMMetadataType,
      LLVMPPCFP128Type,
      LLVMPointerType,
      LLVMStructType,
      LLVMTokenType,
      LLVMFixedVectorType,
      LLVMScalableVectorType,
      LLVMVoidType,
      LLVMX86FP80Type,
      LLVMX86MMXType
  >();
  // clang-format on
}

bool mlir::LLVM::isCompatibleFloatingPointType(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                  LLVMFP128Type, LLVMPPCFP128Type, LLVMX86FP80Type>();
}

bool mlir::LLVM::isCompatibleVectorType(Type type) {
  if (type.isa<LLVMFixedVectorType, LLVMScalableVectorType>())
    return true;

  if (auto vecType = type.dyn_cast<VectorType>()) {
    if (vecType.getRank() != 1)
      return false;
    Type elementType = vecType.getElementType();
    if (auto intType = elementType.dyn_cast<IntegerType>())
      return intType.isSignless();
    return elementType
        .isa<BFloat16Type, Float16Type, Float32Type, Float64Type>();
  }
  return false;
}

Type mlir::LLVM::getVectorElementType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<LLVMFixedVectorType, LLVMScalableVectorType, VectorType>(
          [](auto ty) { return ty.getElementType(); })
      .Default([](Type) -> Type {
        llvm_unreachable("incompatible with LLVM vector type");
      });
}

llvm::ElementCount mlir::LLVM::getVectorNumElements(Type type) {
  return llvm::TypeSwitch<Type, llvm::ElementCount>(type)
      .Case<LLVMFixedVectorType, VectorType>([](auto ty) {
        return llvm::ElementCount::getFixed(ty.getNumElements());
      })
      .Case([](LLVMScalableVectorType ty) {
        return llvm::ElementCount::getScalable(ty.getMinNumElements());
      })
      .Default([](Type) -> llvm::ElementCount {
        llvm_unreachable("incompatible with LLVM vector type");
      });
}

Type mlir::LLVM::getFixedVectorType(Type elementType, unsigned numElements) {
  bool useLLVM = LLVMFixedVectorType::isValidElementType(elementType);
  bool useBuiltIn = VectorType::isValidElementType(elementType);
  (void)useBuiltIn;
  assert((useLLVM ^ useBuiltIn) && "expected LLVM-compatible fixed-vector type "
                                   "to be either builtin or LLVM dialect type");
  if (useLLVM)
    return LLVMFixedVectorType::get(elementType, numElements);
  return VectorType::get(numElements, elementType);
}

llvm::TypeSize mlir::LLVM::getPrimitiveTypeSizeInBits(Type type) {
  assert(isCompatibleType(type) &&
         "expected a type compatible with the LLVM dialect");

  return llvm::TypeSwitch<Type, llvm::TypeSize>(type)
      .Case<BFloat16Type, Float16Type>(
          [](Type) { return llvm::TypeSize::Fixed(16); })
      .Case<Float32Type>([](Type) { return llvm::TypeSize::Fixed(32); })
      .Case<Float64Type, LLVMX86MMXType>(
          [](Type) { return llvm::TypeSize::Fixed(64); })
      .Case<IntegerType>([](IntegerType intTy) {
        return llvm::TypeSize::Fixed(intTy.getWidth());
      })
      .Case<LLVMX86FP80Type>([](Type) { return llvm::TypeSize::Fixed(80); })
      .Case<LLVMPPCFP128Type, LLVMFP128Type>(
          [](Type) { return llvm::TypeSize::Fixed(128); })
      .Case<LLVMFixedVectorType>([](LLVMFixedVectorType t) {
        llvm::TypeSize elementSize =
            getPrimitiveTypeSizeInBits(t.getElementType());
        return llvm::TypeSize(elementSize.getFixedSize() * t.getNumElements(),
                              elementSize.isScalable());
      })
      .Case<VectorType>([](VectorType t) {
        assert(isCompatibleVectorType(t) &&
               "unexpected incompatible with LLVM vector type");
        llvm::TypeSize elementSize =
            getPrimitiveTypeSizeInBits(t.getElementType());
        return llvm::TypeSize(elementSize.getFixedSize() * t.getNumElements(),
                              elementSize.isScalable());
      })
      .Default([](Type ty) {
        assert((ty.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                       LLVMTokenType, LLVMStructType, LLVMArrayType,
                       LLVMPointerType, LLVMFunctionType>()) &&
               "unexpected missing support for primitive type");
        return llvm::TypeSize::Fixed(0);
      });
}
