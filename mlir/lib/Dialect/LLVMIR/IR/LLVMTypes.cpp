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

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::LLVM;

constexpr const static unsigned kBitsInByte = 8;

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

Type LLVMArrayType::getElementType() const { return getImpl()->elementType; }

unsigned LLVMArrayType::getNumElements() const {
  return getImpl()->numElements;
}

LogicalResult
LLVMArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                      Type elementType, unsigned numElements) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid array element type: " << elementType;
  return success();
}

unsigned LLVMArrayType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  return kBitsInByte * getTypeSize(dataLayout, params);
}

unsigned LLVMArrayType::getTypeSize(const DataLayout &dataLayout,
                                    DataLayoutEntryListRef params) const {
  return llvm::alignTo(dataLayout.getTypeSize(getElementType()),
                       dataLayout.getTypeABIAlignment(getElementType())) *
         getNumElements();
}

unsigned LLVMArrayType::getABIAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(getElementType());
}

unsigned
LLVMArrayType::getPreferredAlignment(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(getElementType());
}

void LLVMArrayType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

Type LLVMArrayType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replTypes.front(), getNumElements());
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

LLVMFunctionType LLVMFunctionType::clone(TypeRange inputs,
                                         TypeRange results) const {
  assert(results.size() == 1 && "expected a single result type");
  return get(results[0], llvm::to_vector(inputs), isVarArg());
}

Type LLVMFunctionType::getReturnType() const {
  return getImpl()->getReturnType();
}
ArrayRef<Type> LLVMFunctionType::getReturnTypes() const {
  return getImpl()->getReturnType();
}

unsigned LLVMFunctionType::getNumParams() {
  return getImpl()->getArgumentTypes().size();
}

Type LLVMFunctionType::getParamType(unsigned i) {
  return getImpl()->getArgumentTypes()[i];
}

bool LLVMFunctionType::isVarArg() const { return getImpl()->isVariadic(); }

ArrayRef<Type> LLVMFunctionType::getParams() const {
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

void LLVMFunctionType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (Type type : llvm::concat<const Type>(getReturnTypes(), getParams()))
    walkTypesFn(type);
}

Type LLVMFunctionType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replTypes.front(), replTypes.drop_front(), isVarArg());
}

//===----------------------------------------------------------------------===//
// Pointer type.
//===----------------------------------------------------------------------===//

bool LLVMPointerType::isValidElementType(Type type) {
  if (!type)
    return true;
  return isCompatibleOuterType(type)
             ? !type.isa<LLVMVoidType, LLVMTokenType, LLVMMetadataType,
                         LLVMLabelType>()
             : type.isa<PointerElementTypeInterface>();
}

LLVMPointerType LLVMPointerType::get(Type pointee, unsigned addressSpace) {
  assert(pointee && "expected non-null subtype, pass the context instead if "
                    "the opaque pointer type is desired");
  return Base::get(pointee.getContext(), pointee, addressSpace);
}

LLVMPointerType LLVMPointerType::get(MLIRContext *context,
                                     unsigned addressSpace) {
  return Base::get(context, Type(), addressSpace);
}

LLVMPointerType
LLVMPointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type pointee, unsigned addressSpace) {
  return Base::getChecked(emitError, pointee.getContext(), pointee,
                          addressSpace);
}

LLVMPointerType
LLVMPointerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            MLIRContext *context, unsigned addressSpace) {
  return Base::getChecked(emitError, context, Type(), addressSpace);
}

Type LLVMPointerType::getElementType() const { return getImpl()->pointeeType; }

bool LLVMPointerType::isOpaque() const { return !getImpl()->pointeeType; }

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

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kDefaultPointerAlignment = 8;

Optional<unsigned> mlir::LLVM::extractPointerSpecValue(Attribute attr,
                                                       PtrDLEntryPos pos) {
  auto spec = attr.cast<DenseIntElementsAttr>();
  auto idx = static_cast<unsigned>(pos);
  if (idx >= spec.size())
    return None;
  return spec.getValues<unsigned>()[idx];
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns None.
static Optional<unsigned>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, LLVMPointerType type,
                          PtrDLEntryPos pos) {
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
    return *extractPointerSpecValue(currentEntry, pos) /
           (pos == PtrDLEntryPos::Size ? 1 : kBitsInByte);
  }

  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getAddressSpace() == 0) {
    return pos == PtrDLEntryPos::Size ? kDefaultPointerSizeBits
                                      : kDefaultPointerAlignment;
  }

  return llvm::None;
}

unsigned
LLVMPointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  if (Optional<unsigned> size =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Size))
    return *size;

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  if (isOpaque())
    return dataLayout.getTypeSizeInBits(get(getContext()));
  return dataLayout.getTypeSizeInBits(get(getElementType()));
}

unsigned LLVMPointerType::getABIAlignment(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  if (isOpaque())
    return dataLayout.getTypeABIAlignment(get(getContext()));
  return dataLayout.getTypeABIAlignment(get(getElementType()));
}

unsigned
LLVMPointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const {
  if (Optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Preferred))
    return *alignment;

  if (isOpaque())
    return dataLayout.getTypePreferredAlignment(get(getContext()));
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
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
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
      size = *extractPointerSpecValue(*it, PtrDLEntryPos::Size);
      abi = *extractPointerSpecValue(*it, PtrDLEntryPos::Abi);
    }

    Attribute newSpec = newEntry.getValue().cast<DenseIntElementsAttr>();
    unsigned newSize = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Size);
    unsigned newAbi = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Abi);
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
    if (key.getElementType() && !key.getElementType().isInteger(8)) {
      return emitError(loc) << "unexpected layout attribute for pointer to "
                            << key.getElementType();
    }
    if (extractPointerSpecValue(values, PtrDLEntryPos::Abi) >
        extractPointerSpecValue(values, PtrDLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return success();
}

void LLVMPointerType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

Type LLVMPointerType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replTypes.front(), getAddressSpace());
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

bool LLVMStructType::isPacked() const { return getImpl()->isPacked(); }
bool LLVMStructType::isIdentified() const { return getImpl()->isIdentified(); }
bool LLVMStructType::isOpaque() {
  return getImpl()->isIdentified() &&
         (getImpl()->isOpaque() || !getImpl()->isInitialized());
}
bool LLVMStructType::isInitialized() { return getImpl()->isInitialized(); }
StringRef LLVMStructType::getName() { return getImpl()->getIdentifier(); }
ArrayRef<Type> LLVMStructType::getBody() const {
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

unsigned
LLVMStructType::getTypeSizeInBits(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  unsigned structSize = 0;
  unsigned structAlignment = 1;
  for (Type element : getBody()) {
    unsigned elementAlignment =
        isPacked() ? 1 : dataLayout.getTypeABIAlignment(element);
    // Add padding to the struct size to align it to the abi alignment of the
    // element type before than adding the size of the element
    structSize = llvm::alignTo(structSize, elementAlignment);
    structSize += dataLayout.getTypeSize(element);

    // The alignment requirement of a struct is equal to the strictest alignment
    // requirement of its elements.
    structAlignment = std::max(elementAlignment, structAlignment);
  }
  // At the end, add padding to the struct to satisfy its own alignment
  // requirement. Otherwise structs inside of arrays would be misaligned.
  structSize = llvm::alignTo(structSize, structAlignment);
  return structSize * kBitsInByte;
}

namespace {
enum class StructDLEntryPos { Abi = 0, Preferred = 1 };
} // namespace

static Optional<unsigned>
getStructDataLayoutEntry(DataLayoutEntryListRef params, LLVMStructType type,
                         StructDLEntryPos pos) {
  const auto *currentEntry =
      llvm::find_if(params, [](DataLayoutEntryInterface entry) {
        return entry.isTypeEntry();
      });
  if (currentEntry == params.end())
    return llvm::None;

  auto attr = currentEntry->getValue().cast<DenseIntElementsAttr>();
  if (pos == StructDLEntryPos::Preferred &&
      attr.size() <= static_cast<unsigned>(StructDLEntryPos::Preferred))
    // If no preferred was specified, fall back to abi alignment
    pos = StructDLEntryPos::Abi;

  return attr.getValues<unsigned>()[static_cast<unsigned>(pos)];
}

static unsigned calculateStructAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params,
                                         LLVMStructType type,
                                         StructDLEntryPos pos) {
  // Packed structs always have an abi alignment of 1
  if (pos == StructDLEntryPos::Abi && type.isPacked()) {
    return 1;
  }

  // The alignment requirement of a struct is equal to the strictest alignment
  // requirement of its elements.
  unsigned structAlignment = 1;
  for (Type iter : type.getBody()) {
    structAlignment =
        std::max(dataLayout.getTypeABIAlignment(iter), structAlignment);
  }

  // Entries are only allowed to be stricter than the required alignment
  if (Optional<unsigned> entryResult =
          getStructDataLayoutEntry(params, type, pos))
    return std::max(*entryResult / kBitsInByte, structAlignment);

  return structAlignment;
}

unsigned LLVMStructType::getABIAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Abi);
}

unsigned
LLVMStructType::getPreferredAlignment(const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Preferred);
}

static unsigned extractStructSpecValue(Attribute attr, StructDLEntryPos pos) {
  return attr.cast<DenseIntElementsAttr>()
      .getValues<unsigned>()[static_cast<unsigned>(pos)];
}

bool LLVMStructType::areCompatible(DataLayoutEntryListRef oldLayout,
                                   DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;

    const auto *previousEntry =
        llvm::find_if(oldLayout, [](DataLayoutEntryInterface entry) {
          return entry.isTypeEntry();
        });
    if (previousEntry == oldLayout.end())
      continue;

    unsigned abi = extractStructSpecValue(previousEntry->getValue(),
                                          StructDLEntryPos::Abi);
    unsigned newAbi =
        extractStructSpecValue(newEntry.getValue(), StructDLEntryPos::Abi);
    if (abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult LLVMStructType::verifyEntries(DataLayoutEntryListRef entries,
                                            Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;

    auto key = entry.getKey().get<Type>().cast<LLVMStructType>();
    auto values = entry.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!values || (values.size() != 2 && values.size() != 1)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute of 1 or 2 elements";
    }

    if (key.isIdentified() || !key.getBody().empty()) {
      return emitError(loc) << "unexpected layout attribute for struct " << key;
    }

    if (values.size() == 1)
      continue;

    if (extractStructSpecValue(values, StructDLEntryPos::Abi) >
        extractStructSpecValue(values, StructDLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return mlir::success();
}

void LLVMStructType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (Type type : getBody())
    walkTypesFn(type);
}

Type LLVMStructType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  // TODO: It's not clear how we support replacing sub-elements of mutable
  // types.
  return nullptr;
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

Type LLVMFixedVectorType::getElementType() const {
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

unsigned LLVMFixedVectorType::getNumElements() const {
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

void LLVMFixedVectorType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

Type LLVMFixedVectorType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replTypes[0], getNumElements());
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

Type LLVMScalableVectorType::getElementType() const {
  return static_cast<detail::LLVMTypeAndSizeStorage *>(impl)->elementType;
}

unsigned LLVMScalableVectorType::getMinNumElements() const {
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

void LLVMScalableVectorType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

Type LLVMScalableVectorType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(replTypes[0], getMinNumElements());
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool mlir::LLVM::isCompatibleOuterType(Type type) {
  // clang-format off
  if (type.isa<
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
    >()) {
    // clang-format on
    return true;
  }

  // Only signless integers are compatible.
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.isSignless();

  // 1D vector types are compatible.
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getRank() == 1;

  return false;
}

static bool isCompatibleImpl(Type type, DenseSet<Type> &compatibleTypes) {
  if (!compatibleTypes.insert(type).second)
    return true;

  auto isCompatible = [&](Type type) {
    return isCompatibleImpl(type, compatibleTypes);
  };

  bool result =
      llvm::TypeSwitch<Type, bool>(type)
          .Case<LLVMStructType>([&](auto structType) {
            return llvm::all_of(structType.getBody(), isCompatible);
          })
          .Case<LLVMFunctionType>([&](auto funcType) {
            return isCompatible(funcType.getReturnType()) &&
                   llvm::all_of(funcType.getParams(), isCompatible);
          })
          .Case<IntegerType>([](auto intType) { return intType.isSignless(); })
          .Case<VectorType>([&](auto vecType) {
            return vecType.getRank() == 1 &&
                   isCompatible(vecType.getElementType());
          })
          .Case<LLVMPointerType>([&](auto pointerType) {
            if (pointerType.isOpaque())
              return true;
            return isCompatible(pointerType.getElementType());
          })
          // clang-format off
          .Case<
              LLVMFixedVectorType,
              LLVMScalableVectorType,
              LLVMArrayType
          >([&](auto containerType) {
            return isCompatible(containerType.getElementType());
          })
          .Case<
            BFloat16Type,
            Float16Type,
            Float32Type,
            Float64Type,
            Float80Type,
            Float128Type,
            LLVMLabelType,
            LLVMMetadataType,
            LLVMPPCFP128Type,
            LLVMTokenType,
            LLVMVoidType,
            LLVMX86MMXType
          >([](Type) { return true; })
          // clang-format on
          .Default([](Type) { return false; });

  if (!result)
    compatibleTypes.erase(type);

  return result;
}

bool LLVMDialect::isCompatibleType(Type type) {
  if (auto *llvmDialect =
          type.getContext()->getLoadedDialect<LLVM::LLVMDialect>())
    return isCompatibleImpl(type, llvmDialect->compatibleTypes.get());

  DenseSet<Type> localCompatibleTypes;
  return isCompatibleImpl(type, localCompatibleTypes);
}

bool mlir::LLVM::isCompatibleType(Type type) {
  return LLVMDialect::isCompatibleType(type);
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
      .Case([](VectorType ty) {
        if (ty.isScalable())
          return llvm::ElementCount::getScalable(ty.getNumElements());
        return llvm::ElementCount::getFixed(ty.getNumElements());
      })
      .Case([](LLVMFixedVectorType ty) {
        return llvm::ElementCount::getFixed(ty.getNumElements());
      })
      .Case([](LLVMScalableVectorType ty) {
        return llvm::ElementCount::getScalable(ty.getMinNumElements());
      })
      .Default([](Type) -> llvm::ElementCount {
        llvm_unreachable("incompatible with LLVM vector type");
      });
}

bool mlir::LLVM::isScalableVectorType(Type vectorType) {
  assert(
      (vectorType
           .isa<LLVMFixedVectorType, LLVMScalableVectorType, VectorType>()) &&
      "expected LLVM-compatible vector type");
  return !vectorType.isa<LLVMFixedVectorType>() &&
         (vectorType.isa<LLVMScalableVectorType>() ||
          vectorType.cast<VectorType>().isScalable());
}

Type mlir::LLVM::getVectorType(Type elementType, unsigned numElements,
                               bool isScalable) {
  bool useLLVM = LLVMFixedVectorType::isValidElementType(elementType);
  bool useBuiltIn = VectorType::isValidElementType(elementType);
  (void)useBuiltIn;
  assert((useLLVM ^ useBuiltIn) && "expected LLVM-compatible fixed-vector type "
                                   "to be either builtin or LLVM dialect type");
  if (useLLVM) {
    if (isScalable)
      return LLVMScalableVectorType::get(elementType, numElements);
    return LLVMFixedVectorType::get(elementType, numElements);
  }
  return VectorType::get(numElements, elementType, (unsigned)isScalable);
}

Type mlir::LLVM::getVectorType(Type elementType,
                               const llvm::ElementCount &numElements) {
  if (numElements.isScalable())
    return getVectorType(elementType, numElements.getKnownMinValue(),
                         /*isScalable=*/true);
  return getVectorType(elementType, numElements.getFixedValue(),
                       /*isScalable=*/false);
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

Type mlir::LLVM::getScalableVectorType(Type elementType, unsigned numElements) {
  bool useLLVM = LLVMScalableVectorType::isValidElementType(elementType);
  bool useBuiltIn = VectorType::isValidElementType(elementType);
  (void)useBuiltIn;
  assert((useLLVM ^ useBuiltIn) && "expected LLVM-compatible scalable-vector "
                                   "type to be either builtin or LLVM dialect "
                                   "type");
  if (useLLVM)
    return LLVMScalableVectorType::get(elementType, numElements);
  return VectorType::get(numElements, elementType, /*numScalableDims=*/1);
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

#include "mlir/Dialect/LLVMIR/LLVMTypeInterfaces.cpp.inc"
