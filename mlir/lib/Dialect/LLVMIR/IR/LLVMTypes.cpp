//===- LLVMTypes.cpp - MLIR LLVM dialect types ------------------*- C++ -*-===//
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
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

constexpr const static uint64_t kBitsInByte = 8;

//===----------------------------------------------------------------------===//
// custom<FunctionTypes>
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionTypes(AsmParser &p, SmallVector<Type> &params,
                                      bool &isVarArg) {
  isVarArg = false;
  // `(` `)`
  if (succeeded(p.parseOptionalRParen()))
    return success();

  // `(` `...` `)`
  if (succeeded(p.parseOptionalEllipsis())) {
    isVarArg = true;
    return p.parseRParen();
  }

  // type (`,` type)* (`,` `...`)?
  Type type;
  if (parsePrettyLLVMType(p, type))
    return failure();
  params.push_back(type);
  while (succeeded(p.parseOptionalComma())) {
    if (succeeded(p.parseOptionalEllipsis())) {
      isVarArg = true;
      return p.parseRParen();
    }
    if (parsePrettyLLVMType(p, type))
      return failure();
    params.push_back(type);
  }
  return p.parseRParen();
}

static void printFunctionTypes(AsmPrinter &p, ArrayRef<Type> params,
                               bool isVarArg) {
  llvm::interleaveComma(params, p,
                        [&](Type type) { printPrettyLLVMType(p, type); });
  if (isVarArg) {
    if (!params.empty())
      p << ", ";
    p << "...";
  }
  p << ')';
}

//===----------------------------------------------------------------------===//
// custom<ExtTypeParams>
//===----------------------------------------------------------------------===//

/// Parses the parameter list for a target extension type. The parameter list
/// contains an optional list of type parameters, followed by an optional list
/// of integer parameters. Type and integer parameters cannot be interleaved in
/// the list.
/// extTypeParams ::= typeList? | intList? | (typeList "," intList)
/// typeList      ::= type ("," type)*
/// intList       ::= integer ("," integer)*
static ParseResult
parseExtTypeParams(AsmParser &p, SmallVectorImpl<Type> &typeParams,
                   SmallVectorImpl<unsigned int> &intParams) {
  bool parseType = true;
  auto typeOrIntParser = [&]() -> ParseResult {
    unsigned int i;
    auto intResult = p.parseOptionalInteger(i);
    if (intResult.has_value() && !failed(*intResult)) {
      // Successfully parsed an integer.
      intParams.push_back(i);
      // After the first integer was successfully parsed, no
      // more types can be parsed.
      parseType = false;
      return success();
    }
    if (parseType) {
      Type t;
      if (!parsePrettyLLVMType(p, t)) {
        // Successfully parsed a type.
        typeParams.push_back(t);
        return success();
      }
    }
    return failure();
  };
  if (p.parseCommaSeparatedList(typeOrIntParser)) {
    p.emitError(p.getCurrentLocation(),
                "failed to parse parameter list for target extension type");
    return failure();
  }
  return success();
}

static void printExtTypeParams(AsmPrinter &p, ArrayRef<Type> typeParams,
                               ArrayRef<unsigned int> intParams) {
  p << typeParams;
  if (!typeParams.empty() && !intParams.empty())
    p << ", ";

  p << intParams;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

/// These are unused for now.
/// TODO: Move over to these once more types have been migrated to TypeDef.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult
generatedTypeParser(AsmParser &parser, StringRef *mnemonic, Type &value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult
generatedTypePrinter(Type def, AsmPrinter &printer);

#include "mlir/Dialect/LLVMIR/LLVMTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMArrayType
//===----------------------------------------------------------------------===//

bool LLVMArrayType::isValidElementType(Type type) {
  return !llvm::isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                    LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>(
      type);
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

LogicalResult
LLVMArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                      Type elementType, unsigned numElements) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid array element type: " << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// DataLayoutTypeInterface

llvm::TypeSize
LLVMArrayType::getTypeSizeInBits(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(kBitsInByte *
                                  getTypeSize(dataLayout, params));
}

llvm::TypeSize LLVMArrayType::getTypeSize(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  return llvm::alignTo(dataLayout.getTypeSize(getElementType()),
                       dataLayout.getTypeABIAlignment(getElementType())) *
         getNumElements();
}

uint64_t LLVMArrayType::getABIAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(getElementType());
}

uint64_t
LLVMArrayType::getPreferredAlignment(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(getElementType());
}

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

bool LLVMFunctionType::isValidArgumentType(Type type) {
  return !llvm::isa<LLVMVoidType, LLVMFunctionType>(type);
}

bool LLVMFunctionType::isValidResultType(Type type) {
  return !llvm::isa<LLVMFunctionType, LLVMMetadataType, LLVMLabelType>(type);
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

ArrayRef<Type> LLVMFunctionType::getReturnTypes() const {
  return static_cast<detail::LLVMFunctionTypeStorage *>(getImpl())->returnType;
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
// DataLayoutTypeInterface

constexpr const static uint64_t kDefaultPointerSizeBits = 64;
constexpr const static uint64_t kDefaultPointerAlignment = 8;

std::optional<uint64_t> mlir::LLVM::extractPointerSpecValue(Attribute attr,
                                                            PtrDLEntryPos pos) {
  auto spec = cast<DenseIntElementsAttr>(attr);
  auto idx = static_cast<int64_t>(pos);
  if (idx >= spec.size())
    return std::nullopt;
  return spec.getValues<uint64_t>()[idx];
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns std::nullopt.
static std::optional<uint64_t>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, LLVMPointerType type,
                          PtrDLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (cast<LLVMPointerType>(entry.getKey().get<Type>()).getAddressSpace() ==
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

  return std::nullopt;
}

llvm::TypeSize
LLVMPointerType::getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> size =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Size))
    return llvm::TypeSize::getFixed(*size);

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(get(getContext()));
}

uint64_t LLVMPointerType::getABIAlignment(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(get(getContext()));
}

uint64_t
LLVMPointerType::getPreferredAlignment(const DataLayout &dataLayout,
                                       DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(get(getContext()));
}

bool LLVMPointerType::areCompatible(DataLayoutEntryListRef oldLayout,
                                    DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    uint64_t size = kDefaultPointerSizeBits;
    uint64_t abi = kDefaultPointerAlignment;
    auto newType = llvm::cast<LLVMPointerType>(newEntry.getKey().get<Type>());
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
          if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
            return llvm::cast<LLVMPointerType>(type).getAddressSpace() ==
                   newType.getAddressSpace();
          }
          return false;
        });
    if (it == oldLayout.end()) {
      llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
          return llvm::cast<LLVMPointerType>(type).getAddressSpace() == 0;
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      size = *extractPointerSpecValue(*it, PtrDLEntryPos::Size);
      abi = *extractPointerSpecValue(*it, PtrDLEntryPos::Abi);
    }

    Attribute newSpec = llvm::cast<DenseIntElementsAttr>(newEntry.getValue());
    uint64_t newSize = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Size);
    uint64_t newAbi = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Abi);
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
    auto key = entry.getKey().get<Type>();
    auto values = llvm::dyn_cast<DenseIntElementsAttr>(entry.getValue());
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << key
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (!values.getElementType().isInteger(64))
      return emitError(loc) << "expected i64 parameters for " << key;

    if (extractPointerSpecValue(values, PtrDLEntryPos::Abi) >
        extractPointerSpecValue(values, PtrDLEntryPos::Preferred)) {
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
  return !llvm::isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                    LLVMFunctionType, LLVMTokenType, LLVMScalableVectorType>(
      type);
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

llvm::TypeSize
LLVMStructType::getTypeSizeInBits(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  auto structSize = llvm::TypeSize::getFixed(0);
  uint64_t structAlignment = 1;
  for (Type element : getBody()) {
    uint64_t elementAlignment =
        isPacked() ? 1 : dataLayout.getTypeABIAlignment(element);
    // Add padding to the struct size to align it to the abi alignment of the
    // element type before than adding the size of the element.
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

static std::optional<uint64_t>
getStructDataLayoutEntry(DataLayoutEntryListRef params, LLVMStructType type,
                         StructDLEntryPos pos) {
  const auto *currentEntry =
      llvm::find_if(params, [](DataLayoutEntryInterface entry) {
        return entry.isTypeEntry();
      });
  if (currentEntry == params.end())
    return std::nullopt;

  auto attr = llvm::cast<DenseIntElementsAttr>(currentEntry->getValue());
  if (pos == StructDLEntryPos::Preferred &&
      attr.size() <= static_cast<int64_t>(StructDLEntryPos::Preferred))
    // If no preferred was specified, fall back to abi alignment
    pos = StructDLEntryPos::Abi;

  return attr.getValues<uint64_t>()[static_cast<size_t>(pos)];
}

static uint64_t calculateStructAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params,
                                         LLVMStructType type,
                                         StructDLEntryPos pos) {
  // Packed structs always have an abi alignment of 1
  if (pos == StructDLEntryPos::Abi && type.isPacked()) {
    return 1;
  }

  // The alignment requirement of a struct is equal to the strictest alignment
  // requirement of its elements.
  uint64_t structAlignment = 1;
  for (Type iter : type.getBody()) {
    structAlignment =
        std::max(dataLayout.getTypeABIAlignment(iter), structAlignment);
  }

  // Entries are only allowed to be stricter than the required alignment
  if (std::optional<uint64_t> entryResult =
          getStructDataLayoutEntry(params, type, pos))
    return std::max(*entryResult / kBitsInByte, structAlignment);

  return structAlignment;
}

uint64_t LLVMStructType::getABIAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Abi);
}

uint64_t
LLVMStructType::getPreferredAlignment(const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) const {
  return calculateStructAlignment(dataLayout, params, *this,
                                  StructDLEntryPos::Preferred);
}

static uint64_t extractStructSpecValue(Attribute attr, StructDLEntryPos pos) {
  return llvm::cast<DenseIntElementsAttr>(attr)
      .getValues<uint64_t>()[static_cast<size_t>(pos)];
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

    uint64_t abi = extractStructSpecValue(previousEntry->getValue(),
                                          StructDLEntryPos::Abi);
    uint64_t newAbi =
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

    auto key = llvm::cast<LLVMStructType>(entry.getKey().get<Type>());
    auto values = llvm::dyn_cast<DenseIntElementsAttr>(entry.getValue());
    if (!values || (values.size() != 2 && values.size() != 1)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute of 1 or 2 elements";
    }
    if (!values.getElementType().isInteger(64))
      return emitError(loc) << "expected i64 entries for " << key;

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

bool LLVMFixedVectorType::isValidElementType(Type type) {
  return llvm::isa<LLVMPointerType, LLVMPPCFP128Type>(type);
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

bool LLVMScalableVectorType::isValidElementType(Type type) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type))
    return intType.isSignless();

  return isCompatibleFloatingPointType(type) ||
         llvm::isa<LLVMPointerType>(type);
}

LogicalResult
LLVMScalableVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, unsigned numElements) {
  return verifyVectorConstructionInvariants<LLVMScalableVectorType>(
      emitError, elementType, numElements);
}

//===----------------------------------------------------------------------===//
// LLVMTargetExtType.
//===----------------------------------------------------------------------===//

static constexpr llvm::StringRef kSpirvPrefix = "spirv.";
static constexpr llvm::StringRef kArmSVCount = "aarch64.svcount";

bool LLVM::LLVMTargetExtType::hasProperty(Property prop) const {
  // See llvm/lib/IR/Type.cpp for reference.
  uint64_t properties = 0;

  if (getExtTypeName().starts_with(kSpirvPrefix))
    properties |=
        (LLVMTargetExtType::HasZeroInit | LLVM::LLVMTargetExtType::CanBeGlobal);

  return (properties & prop) == prop;
}

bool LLVM::LLVMTargetExtType::supportsMemOps() const {
  // See llvm/lib/IR/Type.cpp for reference.
  if (getExtTypeName().starts_with(kSpirvPrefix))
    return true;

  if (getExtTypeName() == kArmSVCount)
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool mlir::LLVM::isCompatibleOuterType(Type type) {
  // clang-format off
  if (llvm::isa<
      BFloat16Type,
      Float16Type,
      Float32Type,
      Float64Type,
      Float80Type,
      Float128Type,
      Float8E4M3FNType,
      Float8E5M2Type,
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
      LLVMTargetExtType,
      LLVMVoidType,
      LLVMX86MMXType
    >(type)) {
    // clang-format on
    return true;
  }

  // Only signless integers are compatible.
  if (auto intType = llvm::dyn_cast<IntegerType>(type))
    return intType.isSignless();

  // 1D vector types are compatible.
  if (auto vecType = llvm::dyn_cast<VectorType>(type))
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
          .Case<LLVMPointerType>([&](auto pointerType) { return true; })
          .Case<LLVMTargetExtType>([&](auto extType) {
            return llvm::all_of(extType.getTypeParams(), isCompatible);
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
            Float8E4M3FNType,
            Float8E5M2Type,
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
  return llvm::isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                   Float80Type, Float128Type, LLVMPPCFP128Type>(type);
}

bool mlir::LLVM::isCompatibleVectorType(Type type) {
  if (llvm::isa<LLVMFixedVectorType, LLVMScalableVectorType>(type))
    return true;

  if (auto vecType = llvm::dyn_cast<VectorType>(type)) {
    if (vecType.getRank() != 1)
      return false;
    Type elementType = vecType.getElementType();
    if (auto intType = llvm::dyn_cast<IntegerType>(elementType))
      return intType.isSignless();
    return llvm::isa<BFloat16Type, Float16Type, Float32Type, Float64Type,
                     Float80Type, Float128Type>(elementType);
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
  assert((llvm::isa<LLVMFixedVectorType, LLVMScalableVectorType, VectorType>(
             vectorType)) &&
         "expected LLVM-compatible vector type");
  return !llvm::isa<LLVMFixedVectorType>(vectorType) &&
         (llvm::isa<LLVMScalableVectorType>(vectorType) ||
          llvm::cast<VectorType>(vectorType).isScalable());
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

  // LLVM vectors are always 1-D, hence only 1 bool is required to mark it as
  // scalable/non-scalable.
  return VectorType::get(numElements, elementType, {isScalable});
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

  // LLVM vectors are always 1-D, hence only 1 bool is required to mark it as
  // scalable/non-scalable.
  return VectorType::get(numElements, elementType, /*scalableDims=*/true);
}

llvm::TypeSize mlir::LLVM::getPrimitiveTypeSizeInBits(Type type) {
  assert(isCompatibleType(type) &&
         "expected a type compatible with the LLVM dialect");

  return llvm::TypeSwitch<Type, llvm::TypeSize>(type)
      .Case<BFloat16Type, Float16Type>(
          [](Type) { return llvm::TypeSize::getFixed(16); })
      .Case<Float32Type>([](Type) { return llvm::TypeSize::getFixed(32); })
      .Case<Float64Type, LLVMX86MMXType>(
          [](Type) { return llvm::TypeSize::getFixed(64); })
      .Case<Float80Type>([](Type) { return llvm::TypeSize::getFixed(80); })
      .Case<Float128Type>([](Type) { return llvm::TypeSize::getFixed(128); })
      .Case<IntegerType>([](IntegerType intTy) {
        return llvm::TypeSize::getFixed(intTy.getWidth());
      })
      .Case<LLVMPPCFP128Type>(
          [](Type) { return llvm::TypeSize::getFixed(128); })
      .Case<LLVMFixedVectorType>([](LLVMFixedVectorType t) {
        llvm::TypeSize elementSize =
            getPrimitiveTypeSizeInBits(t.getElementType());
        return llvm::TypeSize(elementSize.getFixedValue() * t.getNumElements(),
                              elementSize.isScalable());
      })
      .Case<VectorType>([](VectorType t) {
        assert(isCompatibleVectorType(t) &&
               "unexpected incompatible with LLVM vector type");
        llvm::TypeSize elementSize =
            getPrimitiveTypeSizeInBits(t.getElementType());
        return llvm::TypeSize(elementSize.getFixedValue() * t.getNumElements(),
                              elementSize.isScalable());
      })
      .Default([](Type ty) {
        assert((llvm::isa<LLVMVoidType, LLVMLabelType, LLVMMetadataType,
                          LLVMTokenType, LLVMStructType, LLVMArrayType,
                          LLVMPointerType, LLVMFunctionType, LLVMTargetExtType>(
                   ty)) &&
               "unexpected missing support for primitive type");
        return llvm::TypeSize::getFixed(0);
      });
}

//===----------------------------------------------------------------------===//
// LLVMDialect
//===----------------------------------------------------------------------===//

void LLVMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMTypes.cpp.inc"
      >();
}

Type LLVMDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}

void LLVMDialect::printType(Type type, DialectAsmPrinter &os) const {
  return detail::printType(type, os);
}
