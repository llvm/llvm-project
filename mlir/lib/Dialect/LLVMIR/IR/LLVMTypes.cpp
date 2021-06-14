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

LLVMArrayType
LLVMArrayType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          Type elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(emitError, elementType.getContext(), elementType,
                          numElements);
}

Type LLVMArrayType::getElementType() { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() { return getImpl()->numElements; }

LogicalResult
LLVMArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                      Type elementType, unsigned numElements) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid array element type: " << elementType;
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

LLVMFunctionType
LLVMFunctionType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             Type result, ArrayRef<Type> arguments,
                             bool isVarArg) {
  assert(result && "expected non-null result");
  return Base::getChecked(emitError, result.getContext(), result, arguments,
                          isVarArg);
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
LLVMFunctionType::verify(function_ref<InFlightDiagnostic()> emitError,
                         Type result, ArrayRef<Type> arguments, bool) {
  if (!isValidResultType(result))
    return emitError() << "invalid function result type: " << result;

  for (Type arg : arguments)
    if (!isValidArgumentType(arg))
      return emitError() << "invalid function argument type: " << arg;

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

LLVMPointerType
LLVMPointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type pointee, unsigned addressSpace) {
  return Base::getChecked(emitError, pointee.getContext(), pointee,
                          addressSpace);
}

Type LLVMPointerType::getElementType() const { return getImpl()->pointeeType; }

unsigned LLVMPointerType::getAddressSpace() const {
  return getImpl()->addressSpace;
}

LogicalResult
LLVMPointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                        Type pointee, unsigned) {
  if (!isValidElementType(pointee))
    return emitError() << "invalid pointer element type: " << pointee;
  return success();
}

namespace {
/// The positions of different values in the data layout entry.
enum class DLEntryPos { Size = 0, Abi = 1, Preferred = 2, Address = 3 };
} // namespace

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kDefaultPointerAlignment = 8;
constexpr const static unsigned kBitsInByte = 8;

/// Returns the value that corresponds to named position `pos` from the
/// attribute `attr` assuming it's a dense integer elements attribute.
static unsigned extractPointerSpecValue(Attribute attr, DLEntryPos pos) {
  return attr.cast<DenseIntElementsAttr>().getValue<unsigned>(
      static_cast<unsigned>(pos));
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns None.
static Optional<unsigned>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, LLVMPointerType type,
                          DLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (entry.getKey().get<Type>().cast<LLVMPointerType>().getAddressSpace() ==
        type.getAddressSpace()) {
      currentEntry = entry.getValue();
      break;
    }
  }
  if (currentEntry) {
    return extractPointerSpecValue(currentEntry, pos) /
           (pos == DLEntryPos::Size ? 1 : kBitsInByte);
  }

  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getAddressSpace() == 0) {
    return pos == DLEntryPos::Size ? kDefaultPointerSizeBits
                                   : kDefaultPointerAlignment;
  }

  return llvm::None;
}

unsigned
LLVMPointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  if (Optional<unsigned> size =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Size))
    return *size;

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(get(getElementType()));
}

unsigned LLVMPointerType::getABIAlignment(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(get(getElementType()));
}

unsigned
LLVMPointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, DLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(get(getElementType()));
}

bool LLVMPointerType::areCompatible(DataLayoutEntryListRef oldLayout,
                                    DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    unsigned size = kDefaultPointerSizeBits;
    unsigned abi = kDefaultPointerAlignment;
    auto newType = newEntry.getKey().get<Type>().cast<LLVMPointerType>();
    auto it = llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
      if (auto type = entry.getKey().dyn_cast<Type>()) {
        return type.cast<LLVMPointerType>().getAddressSpace() ==
               newType.getAddressSpace();
      }
      return false;
    });
    if (it == oldLayout.end()) {
      llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = entry.getKey().dyn_cast<Type>()) {
          return type.cast<LLVMPointerType>().getAddressSpace() == 0;
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      size = extractPointerSpecValue(*it, DLEntryPos::Size);
      abi = extractPointerSpecValue(*it, DLEntryPos::Abi);
    }

    Attribute newSpec = newEntry.getValue().cast<DenseIntElementsAttr>();
    unsigned newSize = extractPointerSpecValue(newSpec, DLEntryPos::Size);
    unsigned newAbi = extractPointerSpecValue(newSpec, DLEntryPos::Abi);
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult LLVMPointerType::verifyEntries(DataLayoutEntryListRef entries,
                                             Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto key = entry.getKey().get<Type>().cast<LLVMPointerType>();
    auto values = entry.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (!key.getElementType().isInteger(8)) {
      return emitError(loc) << "unexpected layout attribute for pointer to "
                            << key.getElementType();
    }
    if (extractPointerSpecValue(values, DLEntryPos::Abi) >
        extractPointerSpecValue(values, DLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
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

LLVMStructType LLVMStructType::getIdentifiedChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    StringRef name) {
  return Base::getChecked(emitError, context, name, /*opaque=*/false);
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

LLVMStructType
LLVMStructType::getLiteralChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<Type> types,
                                  bool isPacked) {
  return Base::getChecked(emitError, context, types, isPacked);
}

LLVMStructType LLVMStructType::getOpaque(StringRef name, MLIRContext *context) {
  return Base::get(context, name, /*opaque=*/true);
}

LLVMStructType
LLVMStructType::getOpaqueChecked(function_ref<InFlightDiagnostic()> emitError,
                                 MLIRContext *context, StringRef name) {
  return Base::getChecked(emitError, context, name, /*opaque=*/true);
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

LogicalResult LLVMStructType::verify(function_ref<InFlightDiagnostic()>,
                                     StringRef, bool) {
  return success();
}

LogicalResult
LLVMStructType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<Type> types, bool) {
  for (Type t : types)
    if (!isValidElementType(t))
      return emitError() << "invalid LLVM structure element type: " << t;

  return success();
}

//===----------------------------------------------------------------------===//
// Vector types.
//===----------------------------------------------------------------------===//

/// Verifies that the type about to be constructed is well-formed.
template <typename VecTy>
static LogicalResult
verifyVectorConstructionInvariants(function_ref<InFlightDiagnostic()> emitError,
                                   Type elementType, unsigned numElements) {
  if (numElements == 0)
    return emitError() << "the number of vector elements must be positive";

  if (!VecTy::isValidElementType(elementType))
    return emitError() << "invalid vector element type";

  return success();
}

LLVMFixedVectorType LLVMFixedVectorType::get(Type elementType,
                                             unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

LLVMFixedVectorType
LLVMFixedVectorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                Type elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(emitError, elementType.getContext(), elementType,
                          numElements);
}

Type LLVMFixedVectorType::getElementType() {
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

unsigned LLVMFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

bool LLVMFixedVectorType::isValidElementType(Type type) {
  return type.isa<LLVMPointerType, LLVMPPCFP128Type>();
}

LogicalResult
LLVMFixedVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                            Type elementType, unsigned numElements) {
  return verifyVectorConstructionInvariants<LLVMFixedVectorType>(
      emitError, elementType, numElements);
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
LLVMScalableVectorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                   Type elementType, unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(emitError, elementType.getContext(), elementType,
                          minNumElements);
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

LogicalResult
LLVMScalableVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, unsigned numElements) {
  return verifyVectorConstructionInvariants<LLVMScalableVectorType>(
      emitError, elementType, numElements);
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
      Float80Type,
      Float128Type,
      LLVMArrayType,
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
      LLVMX86MMXType
  >();
  // clang-format on
}

bool mlir::LLVM::isCompatibleFloatingPointType(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                  Float80Type, Float128Type, LLVMPPCFP128Type>();
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
    return elementType.isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                           Float80Type, Float128Type>();
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
      .Case<Float80Type>([](Type) { return llvm::TypeSize::Fixed(80); })
      .Case<Float128Type>([](Type) { return llvm::TypeSize::Fixed(128); })
      .Case<IntegerType>([](IntegerType intTy) {
        return llvm::TypeSize::Fixed(intTy.getWidth());
      })
      .Case<LLVMPPCFP128Type>([](Type) { return llvm::TypeSize::Fixed(128); })
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
