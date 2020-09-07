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

#include "llvm/ADT/TypeSwitch.h"
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
// Misc type utilities.

llvm::TypeSize LLVMType::getPrimitiveSizeInBits() {
  return llvm::TypeSwitch<LLVMType, llvm::TypeSize>(*this)
      .Case<LLVMHalfType, LLVMBFloatType>(
          [](LLVMType) { return llvm::TypeSize::Fixed(16); })
      .Case<LLVMFloatType>([](LLVMType) { return llvm::TypeSize::Fixed(32); })
      .Case<LLVMDoubleType, LLVMX86MMXType>(
          [](LLVMType) { return llvm::TypeSize::Fixed(64); })
      .Case<LLVMIntegerType>([](LLVMIntegerType intTy) {
        return llvm::TypeSize::Fixed(intTy.getBitWidth());
      })
      .Case<LLVMX86FP80Type>([](LLVMType) { return llvm::TypeSize::Fixed(80); })
      .Case<LLVMPPCFP128Type, LLVMFP128Type>(
          [](LLVMType) { return llvm::TypeSize::Fixed(128); })
      .Case<LLVMVectorType>([](LLVMVectorType t) {
        llvm::TypeSize elementSize =
            t.getElementType().getPrimitiveSizeInBits();
        llvm::ElementCount elementCount = t.getElementCount();
        assert(!elementSize.isScalable() &&
               "vector type should have fixed-width elements");
        return llvm::TypeSize(elementSize.getFixedSize() *
                                  elementCount.getKnownMinValue(),
                              elementCount.isScalable());
      })
      .Default([](LLVMType ty) {
        assert((ty.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                       LLVMTokenType, LLVMStructType, LLVMArrayType,
                       LLVMPointerType, LLVMFunctionType>()) &&
               "unexpected missing support for primitive type");
        return llvm::TypeSize::Fixed(0);
      });
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

bool LLVMArrayType::isValidElementType(LLVMType type) {
  return !type.isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                   LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>();
}

LLVMArrayType LLVMArrayType::get(LLVMType elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

LLVMArrayType LLVMArrayType::getChecked(Location loc, LLVMType elementType,
                                        unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, numElements);
}

LLVMType LLVMArrayType::getElementType() { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() { return getImpl()->numElements; }

LogicalResult
LLVMArrayType::verifyConstructionInvariants(Location loc, LLVMType elementType,
                                            unsigned numElements) {
  if (!isValidElementType(elementType))
    return emitError(loc, "invalid array element type: ") << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// Function type.

bool LLVMFunctionType::isValidArgumentType(LLVMType type) {
  return !type.isa<LLVMVoidType, LLVMFunctionType>();
}

bool LLVMFunctionType::isValidResultType(LLVMType type) {
  return !type.isa<LLVMFunctionType, LLVMMetadataType, LLVMLabelType>();
}

LLVMFunctionType LLVMFunctionType::get(LLVMType result,
                                       ArrayRef<LLVMType> arguments,
                                       bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::get(result.getContext(), result, arguments, isVarArg);
}

LLVMFunctionType LLVMFunctionType::getChecked(Location loc, LLVMType result,
                                              ArrayRef<LLVMType> arguments,
                                              bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::getChecked(loc, result, arguments, isVarArg);
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

LogicalResult LLVMFunctionType::verifyConstructionInvariants(
    Location loc, LLVMType result, ArrayRef<LLVMType> arguments, bool) {
  if (!isValidResultType(result))
    return emitError(loc, "invalid function result type: ") << result;

  for (LLVMType arg : arguments)
    if (!isValidArgumentType(arg))
      return emitError(loc, "invalid function argument type: ") << arg;

  return success();
}

//===----------------------------------------------------------------------===//
// Integer type.

LLVMIntegerType LLVMIntegerType::get(MLIRContext *ctx, unsigned bitwidth) {
  return Base::get(ctx, bitwidth);
}

LLVMIntegerType LLVMIntegerType::getChecked(Location loc, unsigned bitwidth) {
  return Base::getChecked(loc, bitwidth);
}

unsigned LLVMIntegerType::getBitWidth() { return getImpl()->bitwidth; }

LogicalResult LLVMIntegerType::verifyConstructionInvariants(Location loc,
                                                            unsigned bitwidth) {
  constexpr int maxSupportedBitwidth = (1 << 24);
  if (bitwidth >= maxSupportedBitwidth)
    return emitError(loc, "integer type too wide");
  return success();
}

//===----------------------------------------------------------------------===//
// Pointer type.

bool LLVMPointerType::isValidElementType(LLVMType type) {
  return !type.isa<LLVMVoidType, LLVMTokenType, LLVMMetadataType,
                   LLVMLabelType>();
}

LLVMPointerType LLVMPointerType::get(LLVMType pointee, unsigned addressSpace) {
  assert(pointee && "expected non-null subtype");
  return Base::get(pointee.getContext(), pointee, addressSpace);
}

LLVMPointerType LLVMPointerType::getChecked(Location loc, LLVMType pointee,
                                            unsigned addressSpace) {
  return Base::getChecked(loc, pointee, addressSpace);
}

LLVMType LLVMPointerType::getElementType() { return getImpl()->pointeeType; }

unsigned LLVMPointerType::getAddressSpace() { return getImpl()->addressSpace; }

LogicalResult LLVMPointerType::verifyConstructionInvariants(Location loc,
                                                            LLVMType pointee,
                                                            unsigned) {
  if (!isValidElementType(pointee))
    return emitError(loc, "invalid pointer element type: ") << pointee;
  return success();
}

//===----------------------------------------------------------------------===//
// Struct type.

bool LLVMStructType::isValidElementType(LLVMType type) {
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

LLVMStructType LLVMStructType::getLiteral(MLIRContext *context,
                                          ArrayRef<LLVMType> types,
                                          bool isPacked) {
  return Base::get(context, types, isPacked);
}

LLVMStructType LLVMStructType::getLiteralChecked(Location loc,
                                                 ArrayRef<LLVMType> types,
                                                 bool isPacked) {
  return Base::getChecked(loc, types, isPacked);
}

LLVMStructType LLVMStructType::getOpaque(StringRef name, MLIRContext *context) {
  return Base::get(context, name, /*opaque=*/true);
}

LLVMStructType LLVMStructType::getOpaqueChecked(Location loc, StringRef name) {
  return Base::getChecked(loc, name, /*opaque=*/true);
}

LogicalResult LLVMStructType::setBody(ArrayRef<LLVMType> types, bool isPacked) {
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
ArrayRef<LLVMType> LLVMStructType::getBody() {
  return isIdentified() ? getImpl()->getIdentifiedStructBody()
                        : getImpl()->getTypeList();
}

LogicalResult LLVMStructType::verifyConstructionInvariants(Location, StringRef,
                                                           bool) {
  return success();
}

LogicalResult
LLVMStructType::verifyConstructionInvariants(Location loc,
                                             ArrayRef<LLVMType> types, bool) {
  for (LLVMType t : types)
    if (!isValidElementType(t))
      return emitError(loc, "invalid LLVM structure element type: ") << t;

  return success();
}

//===----------------------------------------------------------------------===//
// Vector types.

bool LLVMVectorType::isValidElementType(LLVMType type) {
  return type.isa<LLVMIntegerType, LLVMPointerType>() ||
         type.isFloatingPointTy();
}

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
  return llvm::ElementCount::get(
      static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->numElements,
      isa<LLVMScalableVectorType>());
}

/// Verifies that the type about to be constructed is well-formed.
LogicalResult
LLVMVectorType::verifyConstructionInvariants(Location loc, LLVMType elementType,
                                             unsigned numElements) {
  if (numElements == 0)
    return emitError(loc, "the number of vector elements must be positive");

  if (!isValidElementType(elementType))
    return emitError(loc, "invalid vector element type");

  return success();
}

LLVMFixedVectorType LLVMFixedVectorType::get(LLVMType elementType,
                                             unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

LLVMFixedVectorType LLVMFixedVectorType::getChecked(Location loc,
                                                    LLVMType elementType,
                                                    unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, numElements);
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

LLVMScalableVectorType LLVMScalableVectorType::get(LLVMType elementType,
                                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, minNumElements);
}

LLVMScalableVectorType
LLVMScalableVectorType::getChecked(Location loc, LLVMType elementType,
                                   unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(loc, elementType, minNumElements);
}

unsigned LLVMScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}
