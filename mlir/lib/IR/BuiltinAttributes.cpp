//===- BuiltinAttributes.cpp - MLIR Builtin Attribute Classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "AttributeDetail.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DecodeAttributesInterfaces.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

AffineMapAttr AffineMapAttr::get(AffineMap value) {
  return Base::get(value.getContext(), value);
}

AffineMap AffineMapAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

ArrayAttr ArrayAttr::get(MLIRContext *context, ArrayRef<Attribute> value) {
  return Base::get(context, value);
}

ArrayRef<Attribute> ArrayAttr::getValue() const { return getImpl()->value; }

Attribute ArrayAttr::operator[](unsigned idx) const {
  assert(idx < size() && "index out of bounds");
  return getValue()[idx];
}

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

/// Helper function that does either an in place sort or sorts from source array
/// into destination. If inPlace then storage is both the source and the
/// destination, else value is the source and storage destination. Returns
/// whether source was sorted.
template <bool inPlace>
static bool dictionaryAttrSort(ArrayRef<NamedAttribute> value,
                               SmallVectorImpl<NamedAttribute> &storage) {
  // Specialize for the common case.
  switch (value.size()) {
  case 0:
    // Zero already sorted.
    break;
  case 1:
    // One already sorted but may need to be copied.
    if (!inPlace)
      storage.assign({value[0]});
    break;
  case 2: {
    bool isSorted = value[0] < value[1];
    if (inPlace) {
      if (!isSorted)
        std::swap(storage[0], storage[1]);
    } else if (isSorted) {
      storage.assign({value[0], value[1]});
    } else {
      storage.assign({value[1], value[0]});
    }
    return !isSorted;
  }
  default:
    if (!inPlace)
      storage.assign(value.begin(), value.end());
    // Check to see they are sorted already.
    bool isSorted = llvm::is_sorted(value);
    if (!isSorted) {
      // If not, do a general sort.
      llvm::array_pod_sort(storage.begin(), storage.end());
      value = storage;
    }
    return !isSorted;
  }
  return false;
}

/// Returns an entry with a duplicate name from the given sorted array of named
/// attributes. Returns llvm::None if all elements have unique names.
static Optional<NamedAttribute>
findDuplicateElement(ArrayRef<NamedAttribute> value) {
  const Optional<NamedAttribute> none{llvm::None};
  if (value.size() < 2)
    return none;

  if (value.size() == 2)
    return value[0].first == value[1].first ? value[0] : none;

  auto it = std::adjacent_find(
      value.begin(), value.end(),
      [](NamedAttribute l, NamedAttribute r) { return l.first == r.first; });
  return it != value.end() ? *it : none;
}

bool DictionaryAttr::sort(ArrayRef<NamedAttribute> value,
                          SmallVectorImpl<NamedAttribute> &storage) {
  bool isSorted = dictionaryAttrSort</*inPlace=*/false>(value, storage);
  assert(!findDuplicateElement(storage) &&
         "DictionaryAttr element names must be unique");
  return isSorted;
}

bool DictionaryAttr::sortInPlace(SmallVectorImpl<NamedAttribute> &array) {
  bool isSorted = dictionaryAttrSort</*inPlace=*/true>(array, array);
  assert(!findDuplicateElement(array) &&
         "DictionaryAttr element names must be unique");
  return isSorted;
}

Optional<NamedAttribute>
DictionaryAttr::findDuplicate(SmallVectorImpl<NamedAttribute> &array,
                              bool isSorted) {
  if (!isSorted)
    dictionaryAttrSort</*inPlace=*/true>(array, array);
  return findDuplicateElement(array);
}

DictionaryAttr DictionaryAttr::get(MLIRContext *context,
                                   ArrayRef<NamedAttribute> value) {
  if (value.empty())
    return DictionaryAttr::getEmpty(context);
  assert(llvm::all_of(value,
                      [](const NamedAttribute &attr) { return attr.second; }) &&
         "value cannot have null entries");

  // We need to sort the element list to canonicalize it.
  SmallVector<NamedAttribute, 8> storage;
  if (dictionaryAttrSort</*inPlace=*/false>(value, storage))
    value = storage;
  assert(!findDuplicateElement(value) &&
         "DictionaryAttr element names must be unique");
  return Base::get(context, value);
}
/// Construct a dictionary with an array of values that is known to already be
/// sorted by name and uniqued.
DictionaryAttr DictionaryAttr::getWithSorted(ArrayRef<NamedAttribute> value,
                                             MLIRContext *context) {
  if (value.empty())
    return DictionaryAttr::getEmpty(context);
  // Ensure that the attribute elements are unique and sorted.
  assert(llvm::is_sorted(value,
                         [](NamedAttribute l, NamedAttribute r) {
                           return l.first.strref() < r.first.strref();
                         }) &&
         "expected attribute values to be sorted");
  assert(!findDuplicateElement(value) &&
         "DictionaryAttr element names must be unique");
  return Base::get(context, value);
}

ArrayRef<NamedAttribute> DictionaryAttr::getValue() const {
  return getImpl()->getElements();
}

/// Return the specified attribute if present, null otherwise.
Attribute DictionaryAttr::get(StringRef name) const {
  Optional<NamedAttribute> attr = getNamed(name);
  return attr ? attr->second : nullptr;
}
Attribute DictionaryAttr::get(Identifier name) const {
  Optional<NamedAttribute> attr = getNamed(name);
  return attr ? attr->second : nullptr;
}

/// Return the specified named attribute if present, None otherwise.
Optional<NamedAttribute> DictionaryAttr::getNamed(StringRef name) const {
  ArrayRef<NamedAttribute> values = getValue();
  const auto *it = llvm::lower_bound(values, name);
  return it != values.end() && it->first == name ? *it
                                                 : Optional<NamedAttribute>();
}
Optional<NamedAttribute> DictionaryAttr::getNamed(Identifier name) const {
  for (auto elt : getValue())
    if (elt.first == name)
      return elt;
  return llvm::None;
}

DictionaryAttr::iterator DictionaryAttr::begin() const {
  return getValue().begin();
}
DictionaryAttr::iterator DictionaryAttr::end() const {
  return getValue().end();
}
size_t DictionaryAttr::size() const { return getValue().size(); }

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

FloatAttr FloatAttr::get(Type type, double value) {
  return Base::get(type.getContext(), type, value);
}

FloatAttr FloatAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                Type type, double value) {
  return Base::getChecked(emitError, type.getContext(), type, value);
}

FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), type, value);
}

FloatAttr FloatAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                Type type, const APFloat &value) {
  return Base::getChecked(emitError, type.getContext(), type, value);
}

APFloat FloatAttr::getValue() const { return getImpl()->getValue(); }

double FloatAttr::getValueAsDouble() const {
  return getValueAsDouble(getValue());
}
double FloatAttr::getValueAsDouble(APFloat value) {
  if (&value.getSemantics() != &APFloat::IEEEdouble()) {
    bool losesInfo = false;
    value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  }
  return value.convertToDouble();
}

/// Verify construction invariants.
static LogicalResult
verifyFloatTypeInvariants(function_ref<InFlightDiagnostic()> emitError,
                          Type type) {
  if (!type.isa<FloatType>())
    return emitError() << "expected floating point type";
  return success();
}

LogicalResult FloatAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type type, double value) {
  return verifyFloatTypeInvariants(emitError, type);
}

LogicalResult FloatAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type type, const APFloat &value) {
  // Verify that the type is correct.
  if (failed(verifyFloatTypeInvariants(emitError, type)))
    return failure();

  // Verify that the type semantics match that of the value.
  if (&type.cast<FloatType>().getFloatSemantics() != &value.getSemantics()) {
    return emitError()
           << "FloatAttr type doesn't match the type implied by its value";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

FlatSymbolRefAttr SymbolRefAttr::get(MLIRContext *ctx, StringRef value) {
  return Base::get(ctx, value, llvm::None).cast<FlatSymbolRefAttr>();
}

SymbolRefAttr SymbolRefAttr::get(MLIRContext *ctx, StringRef value,
                                 ArrayRef<FlatSymbolRefAttr> nestedReferences) {
  return Base::get(ctx, value, nestedReferences);
}

StringRef SymbolRefAttr::getRootReference() const { return getImpl()->value; }

StringRef SymbolRefAttr::getLeafReference() const {
  ArrayRef<FlatSymbolRefAttr> nestedRefs = getNestedReferences();
  return nestedRefs.empty() ? getRootReference() : nestedRefs.back().getValue();
}

ArrayRef<FlatSymbolRefAttr> SymbolRefAttr::getNestedReferences() const {
  return getImpl()->getNestedRefs();
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  if (type.isSignlessInteger(1))
    return BoolAttr::get(type.getContext(), value.getBoolValue());
  return Base::get(type.getContext(), type, value);
}

IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  // This uses 64 bit APInts by default for index type.
  if (type.isIndex())
    return get(type, APInt(IndexType::kInternalStorageBitWidth, value));

  auto intType = type.cast<IntegerType>();
  return get(type, APInt(intType.getWidth(), value, intType.isSignedInteger()));
}

APInt IntegerAttr::getValue() const { return getImpl()->getValue(); }

int64_t IntegerAttr::getInt() const {
  assert((getImpl()->getType().isIndex() ||
          getImpl()->getType().isSignlessInteger()) &&
         "must be signless integer");
  return getValue().getSExtValue();
}

int64_t IntegerAttr::getSInt() const {
  assert(getImpl()->getType().isSignedInteger() && "must be signed integer");
  return getValue().getSExtValue();
}

uint64_t IntegerAttr::getUInt() const {
  assert(getImpl()->getType().isUnsignedInteger() &&
         "must be unsigned integer");
  return getValue().getZExtValue();
}

static LogicalResult
verifyIntegerTypeInvariants(function_ref<InFlightDiagnostic()> emitError,
                            Type type) {
  if (type.isa<IntegerType, IndexType>())
    return success();
  return emitError() << "expected integer or index type";
}

LogicalResult IntegerAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, int64_t value) {
  return verifyIntegerTypeInvariants(emitError, type);
}

LogicalResult IntegerAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, const APInt &value) {
  if (failed(verifyIntegerTypeInvariants(emitError, type)))
    return failure();
  if (auto integerType = type.dyn_cast<IntegerType>())
    if (integerType.getWidth() != value.getBitWidth())
      return emitError() << "integer type bit width (" << integerType.getWidth()
                         << ") doesn't match value bit width ("
                         << value.getBitWidth() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// BoolAttr

bool BoolAttr::getValue() const {
  auto *storage = reinterpret_cast<IntegerAttributeStorage *>(impl);
  return storage->getValue().getBoolValue();
}

bool BoolAttr::classof(Attribute attr) {
  IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>();
  return intAttr && intAttr.getType().isSignlessInteger(1);
}

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  return Base::get(value.getConstraint(0).getContext(), value);
}

IntegerSet IntegerSetAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

OpaqueAttr OpaqueAttr::get(Identifier dialect, StringRef attrData, Type type) {
  return Base::get(dialect.getContext(), dialect, attrData, type);
}

OpaqueAttr OpaqueAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  Identifier dialect, StringRef attrData,
                                  Type type) {
  return Base::getChecked(emitError, dialect.getContext(), dialect, attrData,
                          type);
}

/// Returns the dialect namespace of the opaque attribute.
Identifier OpaqueAttr::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

/// Returns the raw attribute data of the opaque attribute.
StringRef OpaqueAttr::getAttrData() const { return getImpl()->attrData; }

/// Verify the construction of an opaque attribute.
LogicalResult OpaqueAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Identifier dialect, StringRef attrData,
                                 Type type) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitError() << "invalid dialect namespace '" << dialect << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

StringAttr StringAttr::get(MLIRContext *context, StringRef bytes) {
  return get(bytes, NoneType::get(context));
}

/// Get an instance of a StringAttr with the given string and Type.
StringAttr StringAttr::get(StringRef bytes, Type type) {
  return Base::get(type.getContext(), bytes, type);
}

StringRef StringAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

TypeAttr TypeAttr::get(Type value) {
  return Base::get(value.getContext(), value);
}

Type TypeAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

ShapedType ElementsAttr::getType() const {
  return Attribute::getType().cast<ShapedType>();
}

/// Returns the number of elements held by this attribute.
int64_t ElementsAttr::getNumElements() const {
  return getType().getNumElements();
}

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute ElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  if (auto denseAttr = dyn_cast<DenseElementsAttr>())
    return denseAttr.getValue(index);
  if (auto opaqueAttr = dyn_cast<OpaqueElementsAttr>())
    return opaqueAttr.getValue(index);
  return cast<SparseElementsAttr>().getValue(index);
}

/// Return if the given 'index' refers to a valid element in this attribute.
bool ElementsAttr::isValidIndex(ArrayRef<uint64_t> index) const {
  auto type = getType();

  // Verify that the rank of the indices matches the held type.
  auto rank = type.getRank();
  if (rank == 0 && index.size() == 1 && index[0] == 0)
    return true;
  if (rank != static_cast<int64_t>(index.size()))
    return false;

  // Verify that all of the indices are within the shape dimensions.
  auto shape = type.getShape();
  return llvm::all_of(llvm::seq<int>(0, rank), [&](int i) {
    return static_cast<int64_t>(index[i]) < shape[i];
  });
}

ElementsAttr
ElementsAttr::mapValues(Type newElementType,
                        function_ref<APInt(const APInt &)> mapping) const {
  if (auto intOrFpAttr = dyn_cast<DenseElementsAttr>())
    return intOrFpAttr.mapValues(newElementType, mapping);
  llvm_unreachable("unsupported ElementsAttr subtype");
}

ElementsAttr
ElementsAttr::mapValues(Type newElementType,
                        function_ref<APInt(const APFloat &)> mapping) const {
  if (auto intOrFpAttr = dyn_cast<DenseElementsAttr>())
    return intOrFpAttr.mapValues(newElementType, mapping);
  llvm_unreachable("unsupported ElementsAttr subtype");
}

/// Method for support type inquiry through isa, cast and dyn_cast.
bool ElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseIntOrFPElementsAttr, DenseStringElementsAttr,
                  OpaqueElementsAttr, SparseElementsAttr>();
}

/// Returns the 1 dimensional flattened row-major index from the given
/// multi-dimensional index.
uint64_t ElementsAttr::getFlattenedIndex(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  auto type = getType();

  // Reduce the provided multidimensional index into a flattended 1D row-major
  // index.
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

//===----------------------------------------------------------------------===//
// DenseElementsAttr Utilities
//===----------------------------------------------------------------------===//

/// Get the bitwidth of a dense element type within the buffer.
/// DenseElementsAttr requires bitwidths greater than 1 to be aligned by 8.
static size_t getDenseElementStorageWidth(size_t origWidth) {
  return origWidth == 1 ? origWidth : llvm::alignTo<8>(origWidth);
}
static size_t getDenseElementStorageWidth(Type elementType) {
  return getDenseElementStorageWidth(getDenseElementBitWidth(elementType));
}

/// Set a bit to a specific value.
static void setBit(char *rawData, size_t bitPos, bool value) {
  if (value)
    rawData[bitPos / CHAR_BIT] |= (1 << (bitPos % CHAR_BIT));
  else
    rawData[bitPos / CHAR_BIT] &= ~(1 << (bitPos % CHAR_BIT));
}

/// Return the value of the specified bit.
static bool getBit(const char *rawData, size_t bitPos) {
  return (rawData[bitPos / CHAR_BIT] & (1 << (bitPos % CHAR_BIT))) != 0;
}

/// Copy actual `numBytes` data from `value` (APInt) to char array(`result`) for
/// BE format.
static void copyAPIntToArrayForBEmachine(APInt value, size_t numBytes,
                                         char *result) {
  assert(llvm::support::endian::system_endianness() == // NOLINT
         llvm::support::endianness::big);              // NOLINT
  assert(value.getNumWords() * APInt::APINT_WORD_SIZE >= numBytes);

  // Copy the words filled with data.
  // For example, when `value` has 2 words, the first word is filled with data.
  // `value` (10 bytes, BE):|abcdefgh|------ij| ==> `result` (BE):|abcdefgh|--|
  size_t numFilledWords = (value.getNumWords() - 1) * APInt::APINT_WORD_SIZE;
  std::copy_n(reinterpret_cast<const char *>(value.getRawData()),
              numFilledWords, result);
  // Convert last word of APInt to LE format and store it in char
  // array(`valueLE`).
  // ex. last word of `value` (BE): |------ij|  ==> `valueLE` (LE): |ji------|
  size_t lastWordPos = numFilledWords;
  SmallVector<char, 8> valueLE(APInt::APINT_WORD_SIZE);
  DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
      reinterpret_cast<const char *>(value.getRawData()) + lastWordPos,
      valueLE.begin(), APInt::APINT_BITS_PER_WORD, 1);
  // Extract actual APInt data from `valueLE`, convert endianness to BE format,
  // and store it in `result`.
  // ex. `valueLE` (LE): |ji------|  ==> `result` (BE): |abcdefgh|ij|
  DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
      valueLE.begin(), result + lastWordPos,
      (numBytes - lastWordPos) * CHAR_BIT, 1);
}

/// Copy `numBytes` data from `inArray`(char array) to `result`(APINT) for BE
/// format.
static void copyArrayToAPIntForBEmachine(const char *inArray, size_t numBytes,
                                         APInt &result) {
  assert(llvm::support::endian::system_endianness() == // NOLINT
         llvm::support::endianness::big);              // NOLINT
  assert(result.getNumWords() * APInt::APINT_WORD_SIZE >= numBytes);

  // Copy the data that fills the word of `result` from `inArray`.
  // For example, when `result` has 2 words, the first word will be filled with
  // data. So, the first 8 bytes are copied from `inArray` here.
  // `inArray` (10 bytes, BE): |abcdefgh|ij|
  //                     ==> `result` (2 words, BE): |abcdefgh|--------|
  size_t numFilledWords = (result.getNumWords() - 1) * APInt::APINT_WORD_SIZE;
  std::copy_n(
      inArray, numFilledWords,
      const_cast<char *>(reinterpret_cast<const char *>(result.getRawData())));

  // Convert array data which will be last word of `result` to LE format, and
  // store it in char array(`inArrayLE`).
  // ex. `inArray` (last two bytes, BE): |ij|  ==> `inArrayLE` (LE): |ji------|
  size_t lastWordPos = numFilledWords;
  SmallVector<char, 8> inArrayLE(APInt::APINT_WORD_SIZE);
  DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
      inArray + lastWordPos, inArrayLE.begin(),
      (numBytes - lastWordPos) * CHAR_BIT, 1);

  // Convert `inArrayLE` to BE format, and store it in last word of `result`.
  // ex. `inArrayLE` (LE): |ji------|  ==> `result` (BE): |abcdefgh|------ij|
  DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
      inArrayLE.begin(),
      const_cast<char *>(reinterpret_cast<const char *>(result.getRawData())) +
          lastWordPos,
      APInt::APINT_BITS_PER_WORD, 1);
}

/// Writes value to the bit position `bitPos` in array `rawData`.
static void writeBits(char *rawData, size_t bitPos, APInt value) {
  size_t bitWidth = value.getBitWidth();

  // If the bitwidth is 1 we just toggle the specific bit.
  if (bitWidth == 1)
    return setBit(rawData, bitPos, value.isOneValue());

  // Otherwise, the bit position is guaranteed to be byte aligned.
  assert((bitPos % CHAR_BIT) == 0 && "expected bitPos to be 8-bit aligned");
  if (llvm::support::endian::system_endianness() ==
      llvm::support::endianness::big) {
    // Copy from `value` to `rawData + (bitPos / CHAR_BIT)`.
    // Copying the first `llvm::divideCeil(bitWidth, CHAR_BIT)` bytes doesn't
    // work correctly in BE format.
    // ex. `value` (2 words including 10 bytes)
    // ==> BE: |abcdefgh|------ij|,  LE: |hgfedcba|ji------|
    copyAPIntToArrayForBEmachine(value, llvm::divideCeil(bitWidth, CHAR_BIT),
                                 rawData + (bitPos / CHAR_BIT));
  } else {
    std::copy_n(reinterpret_cast<const char *>(value.getRawData()),
                llvm::divideCeil(bitWidth, CHAR_BIT),
                rawData + (bitPos / CHAR_BIT));
  }
}

/// Reads the next `bitWidth` bits from the bit position `bitPos` in array
/// `rawData`.
static APInt readBits(const char *rawData, size_t bitPos, size_t bitWidth) {
  // Handle a boolean bit position.
  if (bitWidth == 1)
    return APInt(1, getBit(rawData, bitPos) ? 1 : 0);

  // Otherwise, the bit position must be 8-bit aligned.
  assert((bitPos % CHAR_BIT) == 0 && "expected bitPos to be 8-bit aligned");
  APInt result(bitWidth, 0);
  if (llvm::support::endian::system_endianness() ==
      llvm::support::endianness::big) {
    // Copy from `rawData + (bitPos / CHAR_BIT)` to `result`.
    // Copying the first `llvm::divideCeil(bitWidth, CHAR_BIT)` bytes doesn't
    // work correctly in BE format.
    // ex. `result` (2 words including 10 bytes)
    // ==> BE: |abcdefgh|------ij|,  LE: |hgfedcba|ji------| This function
    copyArrayToAPIntForBEmachine(rawData + (bitPos / CHAR_BIT),
                                 llvm::divideCeil(bitWidth, CHAR_BIT), result);
  } else {
    std::copy_n(rawData + (bitPos / CHAR_BIT),
                llvm::divideCeil(bitWidth, CHAR_BIT),
                const_cast<char *>(
                    reinterpret_cast<const char *>(result.getRawData())));
  }
  return result;
}

/// Returns true if 'values' corresponds to a splat, i.e. one element, or has
/// the same element count as 'type'.
template <typename Values>
static bool hasSameElementsOrSplat(ShapedType type, const Values &values) {
  return (values.size() == 1) ||
         (type.getNumElements() == static_cast<int64_t>(values.size()));
}

//===----------------------------------------------------------------------===//
// DenseElementsAttr Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttributeElementIterator

DenseElementsAttr::AttributeElementIterator::AttributeElementIterator(
    DenseElementsAttr attr, size_t index)
    : llvm::indexed_accessor_iterator<AttributeElementIterator, const void *,
                                      Attribute, Attribute, Attribute>(
          attr.getAsOpaquePointer(), index) {}

Attribute DenseElementsAttr::AttributeElementIterator::operator*() const {
  auto owner = getFromOpaquePointer(base).cast<DenseElementsAttr>();
  Type eltTy = owner.getType().getElementType();
  if (auto intEltTy = eltTy.dyn_cast<IntegerType>())
    return IntegerAttr::get(eltTy, *IntElementIterator(owner, index));
  if (eltTy.isa<IndexType>())
    return IntegerAttr::get(eltTy, *IntElementIterator(owner, index));
  if (auto floatEltTy = eltTy.dyn_cast<FloatType>()) {
    IntElementIterator intIt(owner, index);
    FloatElementIterator floatIt(floatEltTy.getFloatSemantics(), intIt);
    return FloatAttr::get(eltTy, *floatIt);
  }
  if (owner.isa<DenseStringElementsAttr>()) {
    ArrayRef<StringRef> vals = owner.getRawStringData();
    return StringAttr::get(owner.isSplat() ? vals.front() : vals[index], eltTy);
  }
  llvm_unreachable("unexpected element type");
}

//===----------------------------------------------------------------------===//
// BoolElementIterator

DenseElementsAttr::BoolElementIterator::BoolElementIterator(
    DenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<BoolElementIterator, bool, bool, bool>(
          attr.getRawData().data(), attr.isSplat(), dataIndex) {}

bool DenseElementsAttr::BoolElementIterator::operator*() const {
  return getBit(getData(), getDataIndex());
}

//===----------------------------------------------------------------------===//
// IntElementIterator

DenseElementsAttr::IntElementIterator::IntElementIterator(
    DenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<IntElementIterator, APInt, APInt, APInt>(
          attr.getRawData().data(), attr.isSplat(), dataIndex),
      bitWidth(getDenseElementBitWidth(attr.getType().getElementType())) {}

APInt DenseElementsAttr::IntElementIterator::operator*() const {
  return readBits(getData(),
                  getDataIndex() * getDenseElementStorageWidth(bitWidth),
                  bitWidth);
}

//===----------------------------------------------------------------------===//
// ComplexIntElementIterator

DenseElementsAttr::ComplexIntElementIterator::ComplexIntElementIterator(
    DenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<ComplexIntElementIterator,
                                      std::complex<APInt>, std::complex<APInt>,
                                      std::complex<APInt>>(
          attr.getRawData().data(), attr.isSplat(), dataIndex) {
  auto complexType = attr.getType().getElementType().cast<ComplexType>();
  bitWidth = getDenseElementBitWidth(complexType.getElementType());
}

std::complex<APInt>
DenseElementsAttr::ComplexIntElementIterator::operator*() const {
  size_t storageWidth = getDenseElementStorageWidth(bitWidth);
  size_t offset = getDataIndex() * storageWidth * 2;
  return {readBits(getData(), offset, bitWidth),
          readBits(getData(), offset + storageWidth, bitWidth)};
}

//===----------------------------------------------------------------------===//
// FloatElementIterator

DenseElementsAttr::FloatElementIterator::FloatElementIterator(
    const llvm::fltSemantics &smt, IntElementIterator it)
    : llvm::mapped_iterator<IntElementIterator,
                            std::function<APFloat(const APInt &)>>(
          it, [&](const APInt &val) { return APFloat(smt, val); }) {}

//===----------------------------------------------------------------------===//
// ComplexFloatElementIterator

DenseElementsAttr::ComplexFloatElementIterator::ComplexFloatElementIterator(
    const llvm::fltSemantics &smt, ComplexIntElementIterator it)
    : llvm::mapped_iterator<
          ComplexIntElementIterator,
          std::function<std::complex<APFloat>(const std::complex<APInt> &)>>(
          it, [&](const std::complex<APInt> &val) -> std::complex<APFloat> {
            return {APFloat(smt, val.real()), APFloat(smt, val.imag())};
          }) {}

//===----------------------------------------------------------------------===//
// DenseElementsAttr
//===----------------------------------------------------------------------===//

/// Method for support type inquiry through isa, cast and dyn_cast.
bool DenseElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseIntOrFPElementsAttr, DenseStringElementsAttr>();
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<Attribute> values) {
  assert(hasSameElementsOrSplat(type, values));

  // If the element type is not based on int/float/index, assume it is a string
  // type.
  auto eltType = type.getElementType();
  if (!type.getElementType().isIntOrIndexOrFloat()) {
    SmallVector<StringRef, 8> stringValues;
    stringValues.reserve(values.size());
    for (Attribute attr : values) {
      assert(attr.isa<StringAttr>() &&
             "expected string value for non integer/index/float element");
      stringValues.push_back(attr.cast<StringAttr>().getValue());
    }
    return get(type, stringValues);
  }

  // Otherwise, get the raw storage width to use for the allocation.
  size_t bitWidth = getDenseElementBitWidth(eltType);
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

  // Compress the attribute values into a character buffer.
  SmallVector<char, 8> data(llvm::divideCeil(storageBitWidth, CHAR_BIT) *
                            values.size());
  APInt intVal;
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    assert(eltType == values[i].getType() &&
           "expected attribute value to have element type");
    if (eltType.isa<FloatType>())
      intVal = values[i].cast<FloatAttr>().getValue().bitcastToAPInt();
    else if (eltType.isa<IntegerType>())
      intVal = values[i].cast<IntegerAttr>().getValue();
    else
      llvm_unreachable("unexpected element type");

    assert(intVal.getBitWidth() == bitWidth &&
           "expected value to have same bitwidth as element type");
    writeBits(data.data(), i * storageBitWidth, intVal);
  }
  return DenseIntOrFPElementsAttr::getRaw(type, data,
                                          /*isSplat=*/(values.size() == 1));
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<bool> values) {
  assert(hasSameElementsOrSplat(type, values));
  assert(type.getElementType().isInteger(1));

  std::vector<char> buff(llvm::divideCeil(values.size(), CHAR_BIT));
  for (int i = 0, e = values.size(); i != e; ++i)
    setBit(buff.data(), i, values[i]);
  return DenseIntOrFPElementsAttr::getRaw(type, buff,
                                          /*isSplat=*/(values.size() == 1));
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<StringRef> values) {
  assert(!type.getElementType().isIntOrFloat());
  return DenseStringElementsAttr::get(type, values);
}

/// Constructs a dense integer elements attribute from an array of APInt
/// values. Each APInt value is expected to have the same bitwidth as the
/// element type of 'type'.
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<APInt> values) {
  assert(type.getElementType().isIntOrIndex());
  assert(hasSameElementsOrSplat(type, values));
  size_t storageBitWidth = getDenseElementStorageWidth(type.getElementType());
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, values,
                                          /*isSplat=*/(values.size() == 1));
}
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<std::complex<APInt>> values) {
  ComplexType complex = type.getElementType().cast<ComplexType>();
  assert(complex.getElementType().isa<IntegerType>());
  assert(hasSameElementsOrSplat(type, values));
  size_t storageBitWidth = getDenseElementStorageWidth(complex) / 2;
  ArrayRef<APInt> intVals(reinterpret_cast<const APInt *>(values.data()),
                          values.size() * 2);
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, intVals,
                                          /*isSplat=*/(values.size() == 1));
}

// Constructs a dense float elements attribute from an array of APFloat
// values. Each APFloat value is expected to have the same bitwidth as the
// element type of 'type'.
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<APFloat> values) {
  assert(type.getElementType().isa<FloatType>());
  assert(hasSameElementsOrSplat(type, values));
  size_t storageBitWidth = getDenseElementStorageWidth(type.getElementType());
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, values,
                                          /*isSplat=*/(values.size() == 1));
}
DenseElementsAttr
DenseElementsAttr::get(ShapedType type,
                       ArrayRef<std::complex<APFloat>> values) {
  ComplexType complex = type.getElementType().cast<ComplexType>();
  assert(complex.getElementType().isa<FloatType>());
  assert(hasSameElementsOrSplat(type, values));
  ArrayRef<APFloat> apVals(reinterpret_cast<const APFloat *>(values.data()),
                           values.size() * 2);
  size_t storageBitWidth = getDenseElementStorageWidth(complex) / 2;
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, apVals,
                                          /*isSplat=*/(values.size() == 1));
}

/// Construct a dense elements attribute from a raw buffer representing the
/// data for this attribute. Users should generally not use this methods as
/// the expected buffer format may not be a form the user expects.
DenseElementsAttr DenseElementsAttr::getFromRawBuffer(ShapedType type,
                                                      ArrayRef<char> rawBuffer,
                                                      bool isSplatBuffer) {
  return DenseIntOrFPElementsAttr::getRaw(type, rawBuffer, isSplatBuffer);
}

/// Returns true if the given buffer is a valid raw buffer for the given type.
bool DenseElementsAttr::isValidRawBuffer(ShapedType type,
                                         ArrayRef<char> rawBuffer,
                                         bool &detectedSplat) {
  size_t storageWidth = getDenseElementStorageWidth(type.getElementType());
  size_t rawBufferWidth = rawBuffer.size() * CHAR_BIT;

  // Storage width of 1 is special as it is packed by the bit.
  if (storageWidth == 1) {
    // Check for a splat, or a buffer equal to the number of elements.
    if ((detectedSplat = rawBuffer.size() == 1))
      return true;
    return rawBufferWidth == llvm::alignTo<8>(type.getNumElements());
  }
  // All other types are 8-bit aligned.
  if ((detectedSplat = rawBufferWidth == storageWidth))
    return true;
  return rawBufferWidth == (storageWidth * type.getNumElements());
}

/// Check the information for a C++ data type, check if this type is valid for
/// the current attribute. This method is used to verify specific type
/// invariants that the templatized 'getValues' method cannot.
static bool isValidIntOrFloat(Type type, int64_t dataEltSize, bool isInt,
                              bool isSigned) {
  // Make sure that the data element size is the same as the type element width.
  if (getDenseElementBitWidth(type) !=
      static_cast<size_t>(dataEltSize * CHAR_BIT))
    return false;

  // Check that the element type is either float or integer or index.
  if (!isInt)
    return type.isa<FloatType>();
  if (type.isIndex())
    return true;

  auto intType = type.dyn_cast<IntegerType>();
  if (!intType)
    return false;

  // Make sure signedness semantics is consistent.
  if (intType.isSignless())
    return true;
  return intType.isSigned() ? isSigned : !isSigned;
}

/// Defaults down the subclass implementation.
DenseElementsAttr DenseElementsAttr::getRawComplex(ShapedType type,
                                                   ArrayRef<char> data,
                                                   int64_t dataEltSize,
                                                   bool isInt, bool isSigned) {
  return DenseIntOrFPElementsAttr::getRawComplex(type, data, dataEltSize, isInt,
                                                 isSigned);
}
DenseElementsAttr DenseElementsAttr::getRawIntOrFloat(ShapedType type,
                                                      ArrayRef<char> data,
                                                      int64_t dataEltSize,
                                                      bool isInt,
                                                      bool isSigned) {
  return DenseIntOrFPElementsAttr::getRawIntOrFloat(type, data, dataEltSize,
                                                    isInt, isSigned);
}

/// A method used to verify specific type invariants that the templatized 'get'
/// method cannot.
bool DenseElementsAttr::isValidIntOrFloat(int64_t dataEltSize, bool isInt,
                                          bool isSigned) const {
  return ::isValidIntOrFloat(getType().getElementType(), dataEltSize, isInt,
                             isSigned);
}

/// Check the information for a C++ data type, check if this type is valid for
/// the current attribute.
bool DenseElementsAttr::isValidComplex(int64_t dataEltSize, bool isInt,
                                       bool isSigned) const {
  return ::isValidIntOrFloat(
      getType().getElementType().cast<ComplexType>().getElementType(),
      dataEltSize / 2, isInt, isSigned);
}

/// Returns true if this attribute corresponds to a splat, i.e. if all element
/// values are the same.
bool DenseElementsAttr::isSplat() const {
  return static_cast<DenseElementsAttributeStorage *>(impl)->isSplat;
}

/// Return the held element values as a range of Attributes.
auto DenseElementsAttr::getAttributeValues() const
    -> llvm::iterator_range<AttributeElementIterator> {
  return {attr_value_begin(), attr_value_end()};
}
auto DenseElementsAttr::attr_value_begin() const -> AttributeElementIterator {
  return AttributeElementIterator(*this, 0);
}
auto DenseElementsAttr::attr_value_end() const -> AttributeElementIterator {
  return AttributeElementIterator(*this, getNumElements());
}

/// Return the held element values as a range of bool. The element type of
/// this attribute must be of integer type of bitwidth 1.
auto DenseElementsAttr::getBoolValues() const
    -> llvm::iterator_range<BoolElementIterator> {
  auto eltType = getType().getElementType().dyn_cast<IntegerType>();
  assert(eltType && eltType.getWidth() == 1 && "expected i1 integer type");
  (void)eltType;
  return {BoolElementIterator(*this, 0),
          BoolElementIterator(*this, getNumElements())};
}

/// Return the held element values as a range of APInts. The element type of
/// this attribute must be of integer type.
auto DenseElementsAttr::getIntValues() const
    -> llvm::iterator_range<IntElementIterator> {
  assert(getType().getElementType().isIntOrIndex() && "expected integral type");
  return {raw_int_begin(), raw_int_end()};
}
auto DenseElementsAttr::int_value_begin() const -> IntElementIterator {
  assert(getType().getElementType().isIntOrIndex() && "expected integral type");
  return raw_int_begin();
}
auto DenseElementsAttr::int_value_end() const -> IntElementIterator {
  assert(getType().getElementType().isIntOrIndex() && "expected integral type");
  return raw_int_end();
}
auto DenseElementsAttr::getComplexIntValues() const
    -> llvm::iterator_range<ComplexIntElementIterator> {
  Type eltTy = getType().getElementType().cast<ComplexType>().getElementType();
  (void)eltTy;
  assert(eltTy.isa<IntegerType>() && "expected complex integral type");
  return {ComplexIntElementIterator(*this, 0),
          ComplexIntElementIterator(*this, getNumElements())};
}

/// Return the held element values as a range of APFloat. The element type of
/// this attribute must be of float type.
auto DenseElementsAttr::getFloatValues() const
    -> llvm::iterator_range<FloatElementIterator> {
  auto elementType = getType().getElementType().cast<FloatType>();
  const auto &elementSemantics = elementType.getFloatSemantics();
  return {FloatElementIterator(elementSemantics, raw_int_begin()),
          FloatElementIterator(elementSemantics, raw_int_end())};
}
auto DenseElementsAttr::float_value_begin() const -> FloatElementIterator {
  return getFloatValues().begin();
}
auto DenseElementsAttr::float_value_end() const -> FloatElementIterator {
  return getFloatValues().end();
}
auto DenseElementsAttr::getComplexFloatValues() const
    -> llvm::iterator_range<ComplexFloatElementIterator> {
  Type eltTy = getType().getElementType().cast<ComplexType>().getElementType();
  assert(eltTy.isa<FloatType>() && "expected complex float type");
  const auto &semantics = eltTy.cast<FloatType>().getFloatSemantics();
  return {{semantics, {*this, 0}},
          {semantics, {*this, static_cast<size_t>(getNumElements())}}};
}

/// Return the raw storage data held by this attribute.
ArrayRef<char> DenseElementsAttr::getRawData() const {
  return static_cast<DenseIntOrFPElementsAttributeStorage *>(impl)->data;
}

ArrayRef<StringRef> DenseElementsAttr::getRawStringData() const {
  return static_cast<DenseStringElementsAttributeStorage *>(impl)->data;
}

/// Return a new DenseElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newType'. The new type must have the
/// same total number of elements as well as element type.
DenseElementsAttr DenseElementsAttr::reshape(ShapedType newType) {
  ShapedType curType = getType();
  if (curType == newType)
    return *this;

  (void)curType;
  assert(newType.getElementType() == curType.getElementType() &&
         "expected the same element type");
  assert(newType.getNumElements() == curType.getNumElements() &&
         "expected the same number of elements");
  return DenseIntOrFPElementsAttr::getRaw(newType, getRawData(), isSplat());
}

DenseElementsAttr
DenseElementsAttr::mapValues(Type newElementType,
                             function_ref<APInt(const APInt &)> mapping) const {
  return cast<DenseIntElementsAttr>().mapValues(newElementType, mapping);
}

DenseElementsAttr DenseElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APFloat &)> mapping) const {
  return cast<DenseFPElementsAttr>().mapValues(newElementType, mapping);
}

//===----------------------------------------------------------------------===//
// DenseStringElementsAttr
//===----------------------------------------------------------------------===//

DenseStringElementsAttr
DenseStringElementsAttr::get(ShapedType type, ArrayRef<StringRef> values) {
  return Base::get(type.getContext(), type, values, (values.size() == 1));
}

//===----------------------------------------------------------------------===//
// DenseIntOrFPElementsAttr
//===----------------------------------------------------------------------===//

/// Utility method to write a range of APInt values to a buffer.
template <typename APRangeT>
static void writeAPIntsToBuffer(size_t storageWidth, std::vector<char> &data,
                                APRangeT &&values) {
  data.resize(llvm::divideCeil(storageWidth, CHAR_BIT) * llvm::size(values));
  size_t offset = 0;
  for (auto it = values.begin(), e = values.end(); it != e;
       ++it, offset += storageWidth) {
    assert((*it).getBitWidth() <= storageWidth);
    writeBits(data.data(), offset, *it);
  }
}

/// Constructs a dense elements attribute from an array of raw APFloat values.
/// Each APFloat value is expected to have the same bitwidth as the element
/// type of 'type'. 'type' must be a vector or tensor with static shape.
DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   size_t storageWidth,
                                                   ArrayRef<APFloat> values,
                                                   bool isSplat) {
  std::vector<char> data;
  auto unwrapFloat = [](const APFloat &val) { return val.bitcastToAPInt(); };
  writeAPIntsToBuffer(storageWidth, data, llvm::map_range(values, unwrapFloat));
  return DenseIntOrFPElementsAttr::getRaw(type, data, isSplat);
}

/// Constructs a dense elements attribute from an array of raw APInt values.
/// Each APInt value is expected to have the same bitwidth as the element type
/// of 'type'.
DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   size_t storageWidth,
                                                   ArrayRef<APInt> values,
                                                   bool isSplat) {
  std::vector<char> data;
  writeAPIntsToBuffer(storageWidth, data, values);
  return DenseIntOrFPElementsAttr::getRaw(type, data, isSplat);
}

DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   ArrayRef<char> data,
                                                   bool isSplat) {
  assert((type.isa<RankedTensorType, VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), type, data, isSplat);
}

/// Overload of the raw 'get' method that asserts that the given type is of
/// complex type. This method is used to verify type invariants that the
/// templatized 'get' method cannot.
DenseElementsAttr DenseIntOrFPElementsAttr::getRawComplex(ShapedType type,
                                                          ArrayRef<char> data,
                                                          int64_t dataEltSize,
                                                          bool isInt,
                                                          bool isSigned) {
  assert(::isValidIntOrFloat(
      type.getElementType().cast<ComplexType>().getElementType(),
      dataEltSize / 2, isInt, isSigned));

  int64_t numElements = data.size() / dataEltSize;
  assert(numElements == 1 || numElements == type.getNumElements());
  return getRaw(type, data, /*isSplat=*/numElements == 1);
}

/// Overload of the 'getRaw' method that asserts that the given type is of
/// integer type. This method is used to verify type invariants that the
/// templatized 'get' method cannot.
DenseElementsAttr
DenseIntOrFPElementsAttr::getRawIntOrFloat(ShapedType type, ArrayRef<char> data,
                                           int64_t dataEltSize, bool isInt,
                                           bool isSigned) {
  assert(
      ::isValidIntOrFloat(type.getElementType(), dataEltSize, isInt, isSigned));

  int64_t numElements = data.size() / dataEltSize;
  assert(numElements == 1 || numElements == type.getNumElements());
  return getRaw(type, data, /*isSplat=*/numElements == 1);
}

void DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
    const char *inRawData, char *outRawData, size_t elementBitWidth,
    size_t numElements) {
  using llvm::support::ulittle16_t;
  using llvm::support::ulittle32_t;
  using llvm::support::ulittle64_t;

  assert(llvm::support::endian::system_endianness() == // NOLINT
         llvm::support::endianness::big);              // NOLINT
  // NOLINT to avoid warning message about replacing by static_assert()

  // Following std::copy_n always converts endianness on BE machine.
  switch (elementBitWidth) {
  case 16: {
    const ulittle16_t *inRawDataPos =
        reinterpret_cast<const ulittle16_t *>(inRawData);
    uint16_t *outDataPos = reinterpret_cast<uint16_t *>(outRawData);
    std::copy_n(inRawDataPos, numElements, outDataPos);
    break;
  }
  case 32: {
    const ulittle32_t *inRawDataPos =
        reinterpret_cast<const ulittle32_t *>(inRawData);
    uint32_t *outDataPos = reinterpret_cast<uint32_t *>(outRawData);
    std::copy_n(inRawDataPos, numElements, outDataPos);
    break;
  }
  case 64: {
    const ulittle64_t *inRawDataPos =
        reinterpret_cast<const ulittle64_t *>(inRawData);
    uint64_t *outDataPos = reinterpret_cast<uint64_t *>(outRawData);
    std::copy_n(inRawDataPos, numElements, outDataPos);
    break;
  }
  default: {
    size_t nBytes = elementBitWidth / CHAR_BIT;
    for (size_t i = 0; i < nBytes; i++)
      std::copy_n(inRawData + (nBytes - 1 - i), 1, outRawData + i);
    break;
  }
  }
}

void DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine(
    ArrayRef<char> inRawData, MutableArrayRef<char> outRawData,
    ShapedType type) {
  size_t numElements = type.getNumElements();
  Type elementType = type.getElementType();
  if (ComplexType complexTy = elementType.dyn_cast<ComplexType>()) {
    elementType = complexTy.getElementType();
    numElements = numElements * 2;
  }
  size_t elementBitWidth = getDenseElementStorageWidth(elementType);
  assert(numElements * elementBitWidth == inRawData.size() * CHAR_BIT &&
         inRawData.size() <= outRawData.size());
  convertEndianOfCharForBEmachine(inRawData.begin(), outRawData.begin(),
                                  elementBitWidth, numElements);
}

//===----------------------------------------------------------------------===//
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

template <typename Fn, typename Attr>
static ShapedType mappingHelper(Fn mapping, Attr &attr, ShapedType inType,
                                Type newElementType,
                                llvm::SmallVectorImpl<char> &data) {
  size_t bitWidth = getDenseElementBitWidth(newElementType);
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

  ShapedType newArrayType;
  if (inType.isa<RankedTensorType>())
    newArrayType = RankedTensorType::get(inType.getShape(), newElementType);
  else if (inType.isa<UnrankedTensorType>())
    newArrayType = RankedTensorType::get(inType.getShape(), newElementType);
  else if (inType.isa<VectorType>())
    newArrayType = VectorType::get(inType.getShape(), newElementType);
  else
    assert(newArrayType && "Unhandled tensor type");

  size_t numRawElements = attr.isSplat() ? 1 : newArrayType.getNumElements();
  data.resize(llvm::divideCeil(storageBitWidth, CHAR_BIT) * numRawElements);

  // Functor used to process a single element value of the attribute.
  auto processElt = [&](decltype(*attr.begin()) value, size_t index) {
    auto newInt = mapping(value);
    assert(newInt.getBitWidth() == bitWidth);
    writeBits(data.data(), index * storageBitWidth, newInt);
  };

  // Check for the splat case.
  if (attr.isSplat()) {
    processElt(*attr.begin(), /*index=*/0);
    return newArrayType;
  }

  // Otherwise, process all of the element values.
  uint64_t elementIdx = 0;
  for (auto value : attr)
    processElt(value, elementIdx++);
  return newArrayType;
}

DenseElementsAttr DenseFPElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APFloat &)> mapping) const {
  llvm::SmallVector<char, 8> elementData;
  auto newArrayType =
      mappingHelper(mapping, *this, getType(), newElementType, elementData);

  return getRaw(newArrayType, elementData, isSplat());
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseFPElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseElementsAttr>() &&
         attr.getType().cast<ShapedType>().getElementType().isa<FloatType>();
}

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

DenseElementsAttr DenseIntElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APInt &)> mapping) const {
  llvm::SmallVector<char, 8> elementData;
  auto newArrayType =
      mappingHelper(mapping, *this, getType(), newElementType, elementData);

  return getRaw(newArrayType, elementData, isSplat());
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseIntElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseElementsAttr>() &&
         attr.getType().cast<ShapedType>().getElementType().isIntOrIndex();
}

//===----------------------------------------------------------------------===//
// OpaqueElementsAttr
//===----------------------------------------------------------------------===//

OpaqueElementsAttr OpaqueElementsAttr::get(Dialect *dialect, ShapedType type,
                                           StringRef bytes) {
  assert(TensorType::isValidElementType(type.getElementType()) &&
         "Input element type should be a valid tensor element type");
  return Base::get(type.getContext(), type, dialect, bytes);
}

StringRef OpaqueElementsAttr::getValue() const { return getImpl()->bytes; }

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute OpaqueElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  return Attribute();
}

Dialect *OpaqueElementsAttr::getDialect() const { return getImpl()->dialect; }

bool OpaqueElementsAttr::decode(ElementsAttr &result) {
  auto *d = getDialect();
  if (!d)
    return true;
  auto *interface =
      d->getRegisteredInterface<DialectDecodeAttributesInterface>();
  if (!interface)
    return true;
  return failed(interface->decode(*this, result));
}

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

SparseElementsAttr SparseElementsAttr::get(ShapedType type,
                                           DenseElementsAttr indices,
                                           DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((type.isa<RankedTensorType, VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), type,
                   indices.cast<DenseIntElementsAttr>(), values);
}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return getImpl()->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return getImpl()->values;
}

/// Return the value of the element at the given index.
Attribute SparseElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  auto type = getType();

  // The sparse indices are 64-bit integers, so we can reinterpret the raw data
  // as a 1-D index array.
  auto sparseIndices = getIndices();
  auto sparseIndexValues = sparseIndices.getValues<uint64_t>();

  // Check to see if the indices are a splat.
  if (sparseIndices.isSplat()) {
    // If the index is also not a splat of the index value, we know that the
    // value is zero.
    auto splatIndex = *sparseIndexValues.begin();
    if (llvm::any_of(index, [=](uint64_t i) { return i != splatIndex; }))
      return getZeroAttr();

    // If the indices are a splat, we also expect the values to be a splat.
    assert(getValues().isSplat() && "expected splat values");
    return getValues().getSplatValue();
  }

  // Build a mapping between known indices and the offset of the stored element.
  llvm::SmallDenseMap<llvm::ArrayRef<uint64_t>, size_t> mappedIndices;
  auto numSparseIndices = sparseIndices.getType().getDimSize(0);
  size_t rank = type.getRank();
  for (size_t i = 0, e = numSparseIndices; i != e; ++i)
    mappedIndices.try_emplace(
        {&*std::next(sparseIndexValues.begin(), i * rank), rank}, i);

  // Look for the provided index key within the mapped indices. If the provided
  // index is not found, then return a zero attribute.
  auto it = mappedIndices.find(index);
  if (it == mappedIndices.end())
    return getZeroAttr();

  // Otherwise, return the held sparse value element.
  return getValues().getValue(it->second);
}

/// Get a zero APFloat for the given sparse attribute.
APFloat SparseElementsAttr::getZeroAPFloat() const {
  auto eltType = getType().getElementType().cast<FloatType>();
  return APFloat(eltType.getFloatSemantics());
}

/// Get a zero APInt for the given sparse attribute.
APInt SparseElementsAttr::getZeroAPInt() const {
  auto eltType = getType().getElementType().cast<IntegerType>();
  return APInt::getNullValue(eltType.getWidth());
}

/// Get a zero attribute for the given attribute type.
Attribute SparseElementsAttr::getZeroAttr() const {
  auto eltType = getType().getElementType();

  // Handle floating point elements.
  if (eltType.isa<FloatType>())
    return FloatAttr::get(eltType, 0);

  // Otherwise, this is an integer.
  // TODO: Handle StringAttr here.
  return IntegerAttr::get(eltType, 0);
}

/// Flatten, and return, all of the sparse indices in this attribute in
/// row-major order.
std::vector<ptrdiff_t> SparseElementsAttr::getFlattenedSparseIndices() const {
  std::vector<ptrdiff_t> flatSparseIndices;

  // The sparse indices are 64-bit integers, so we can reinterpret the raw data
  // as a 1-D index array.
  auto sparseIndices = getIndices();
  auto sparseIndexValues = sparseIndices.getValues<uint64_t>();
  if (sparseIndices.isSplat()) {
    SmallVector<uint64_t, 8> indices(getType().getRank(),
                                     *sparseIndexValues.begin());
    flatSparseIndices.push_back(getFlattenedIndex(indices));
    return flatSparseIndices;
  }

  // Otherwise, reinterpret each index as an ArrayRef when flattening.
  auto numSparseIndices = sparseIndices.getType().getDimSize(0);
  size_t rank = getType().getRank();
  for (size_t i = 0, e = numSparseIndices; i != e; ++i)
    flatSparseIndices.push_back(getFlattenedIndex(
        {&*std::next(sparseIndexValues.begin(), i * rank), rank}));
  return flatSparseIndices;
}
