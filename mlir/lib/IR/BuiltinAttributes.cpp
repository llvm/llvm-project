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
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Endian.h"
#include <optional>

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

void BuiltinDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/IR/BuiltinAttributes.cpp.inc"
      >();
  addAttributes<DistinctAttr>();
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
    if (!inPlace)
      storage.clear();
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
    // If not, do a general sort.
    if (!isSorted)
      llvm::array_pod_sort(storage.begin(), storage.end());
    return !isSorted;
  }
  return false;
}

/// Returns an entry with a duplicate name from the given sorted array of named
/// attributes. Returns std::nullopt if all elements have unique names.
static std::optional<NamedAttribute>
findDuplicateElement(ArrayRef<NamedAttribute> value) {
  const std::optional<NamedAttribute> none{std::nullopt};
  if (value.size() < 2)
    return none;

  if (value.size() == 2)
    return value[0].getName() == value[1].getName() ? value[0] : none;

  const auto *it = std::adjacent_find(value.begin(), value.end(),
                                      [](NamedAttribute l, NamedAttribute r) {
                                        return l.getName() == r.getName();
                                      });
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

std::optional<NamedAttribute>
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
DictionaryAttr DictionaryAttr::getWithSorted(MLIRContext *context,
                                             ArrayRef<NamedAttribute> value) {
  if (value.empty())
    return DictionaryAttr::getEmpty(context);
  // Ensure that the attribute elements are unique and sorted.
  assert(llvm::is_sorted(
             value, [](NamedAttribute l, NamedAttribute r) { return l < r; }) &&
         "expected attribute values to be sorted");
  assert(!findDuplicateElement(value) &&
         "DictionaryAttr element names must be unique");
  return Base::get(context, value);
}

/// Return the specified attribute if present, null otherwise.
Attribute DictionaryAttr::get(StringRef name) const {
  auto it = impl::findAttrSorted(begin(), end(), name);
  return it.second ? it.first->getValue() : Attribute();
}
Attribute DictionaryAttr::get(StringAttr name) const {
  auto it = impl::findAttrSorted(begin(), end(), name);
  return it.second ? it.first->getValue() : Attribute();
}

/// Return the specified named attribute if present, std::nullopt otherwise.
std::optional<NamedAttribute> DictionaryAttr::getNamed(StringRef name) const {
  auto it = impl::findAttrSorted(begin(), end(), name);
  return it.second ? *it.first : std::optional<NamedAttribute>();
}
std::optional<NamedAttribute> DictionaryAttr::getNamed(StringAttr name) const {
  auto it = impl::findAttrSorted(begin(), end(), name);
  return it.second ? *it.first : std::optional<NamedAttribute>();
}

/// Return whether the specified attribute is present.
bool DictionaryAttr::contains(StringRef name) const {
  return impl::findAttrSorted(begin(), end(), name).second;
}
bool DictionaryAttr::contains(StringAttr name) const {
  return impl::findAttrSorted(begin(), end(), name).second;
}

DictionaryAttr::iterator DictionaryAttr::begin() const {
  return getValue().begin();
}
DictionaryAttr::iterator DictionaryAttr::end() const {
  return getValue().end();
}
size_t DictionaryAttr::size() const { return getValue().size(); }

DictionaryAttr DictionaryAttr::getEmptyUnchecked(MLIRContext *context) {
  return Base::get(context, ArrayRef<NamedAttribute>());
}

//===----------------------------------------------------------------------===//
// StridedLayoutAttr
//===----------------------------------------------------------------------===//

/// Prints a strided layout attribute.
void StridedLayoutAttr::print(llvm::raw_ostream &os) const {
  auto printIntOrQuestion = [&](int64_t value) {
    if (ShapedType::isDynamic(value))
      os << "?";
    else
      os << value;
  };

  os << "strided<[";
  llvm::interleaveComma(getStrides(), os, printIntOrQuestion);
  os << "]";

  if (getOffset() != 0) {
    os << ", offset: ";
    printIntOrQuestion(getOffset());
  }
  os << ">";
}

/// Returns the strided layout as an affine map.
AffineMap StridedLayoutAttr::getAffineMap() const {
  return makeStridedLinearLayoutMap(getStrides(), getOffset(), getContext());
}

/// Checks that the type-agnostic strided layout invariants are satisfied.
LogicalResult
StridedLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          int64_t offset, ArrayRef<int64_t> strides) {
  if (llvm::is_contained(strides, 0))
    return emitError() << "strides must not be zero";

  return success();
}

/// Checks that the type-specific strided layout invariants are satisfied.
LogicalResult StridedLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != getStrides().size())
    return emitError() << "expected the number of strides to match the rank";

  return success();
}

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

StringAttr StringAttr::getEmptyStringAttrUnchecked(MLIRContext *context) {
  return Base::get(context, "", NoneType::get(context));
}

/// Twine support for StringAttr.
StringAttr StringAttr::get(MLIRContext *context, const Twine &twine) {
  // Fast-path empty twine.
  if (twine.isTriviallyEmpty())
    return get(context);
  SmallVector<char, 32> tempStr;
  return Base::get(context, twine.toStringRef(tempStr), NoneType::get(context));
}

/// Twine support for StringAttr.
StringAttr StringAttr::get(const Twine &twine, Type type) {
  SmallVector<char, 32> tempStr;
  return Base::get(type.getContext(), twine.toStringRef(tempStr), type);
}

StringRef StringAttr::getValue() const { return getImpl()->value; }

Type StringAttr::getType() const { return getImpl()->type; }

Dialect *StringAttr::getReferencedDialect() const {
  return getImpl()->referencedDialect;
}

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

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

LogicalResult FloatAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type type, APFloat value) {
  // Verify that the type is correct.
  if (!llvm::isa<FloatType>(type))
    return emitError() << "expected floating point type";

  // Verify that the type semantics match that of the value.
  if (&llvm::cast<FloatType>(type).getFloatSemantics() !=
      &value.getSemantics()) {
    return emitError()
           << "FloatAttr type doesn't match the type implied by its value";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

SymbolRefAttr SymbolRefAttr::get(MLIRContext *ctx, StringRef value,
                                 ArrayRef<FlatSymbolRefAttr> nestedRefs) {
  return get(StringAttr::get(ctx, value), nestedRefs);
}

FlatSymbolRefAttr SymbolRefAttr::get(MLIRContext *ctx, StringRef value) {
  return llvm::cast<FlatSymbolRefAttr>(get(ctx, value, {}));
}

FlatSymbolRefAttr SymbolRefAttr::get(StringAttr value) {
  return llvm::cast<FlatSymbolRefAttr>(get(value, {}));
}

FlatSymbolRefAttr SymbolRefAttr::get(Operation *symbol) {
  auto symName =
      symbol->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  assert(symName && "value does not have a valid symbol name");
  return SymbolRefAttr::get(symName);
}

StringAttr SymbolRefAttr::getLeafReference() const {
  ArrayRef<FlatSymbolRefAttr> nestedRefs = getNestedReferences();
  return nestedRefs.empty() ? getRootReference() : nestedRefs.back().getAttr();
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

int64_t IntegerAttr::getInt() const {
  assert((getType().isIndex() || getType().isSignlessInteger()) &&
         "must be signless integer");
  return getValue().getSExtValue();
}

int64_t IntegerAttr::getSInt() const {
  assert(getType().isSignedInteger() && "must be signed integer");
  return getValue().getSExtValue();
}

uint64_t IntegerAttr::getUInt() const {
  assert(getType().isUnsignedInteger() && "must be unsigned integer");
  return getValue().getZExtValue();
}

/// Return the value as an APSInt which carries the signed from the type of
/// the attribute.  This traps on signless integers types!
APSInt IntegerAttr::getAPSInt() const {
  assert(!getType().isSignlessInteger() &&
         "Signless integers don't carry a sign for APSInt");
  return APSInt(getValue(), getType().isUnsignedInteger());
}

LogicalResult IntegerAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, APInt value) {
  if (IntegerType integerType = llvm::dyn_cast<IntegerType>(type)) {
    if (integerType.getWidth() != value.getBitWidth())
      return emitError() << "integer type bit width (" << integerType.getWidth()
                         << ") doesn't match value bit width ("
                         << value.getBitWidth() << ")";
    return success();
  }
  if (llvm::isa<IndexType>(type)) {
    if (value.getBitWidth() != IndexType::kInternalStorageBitWidth)
      return emitError()
             << "value bit width (" << value.getBitWidth()
             << ") doesn't match index type internal storage bit width ("
             << IndexType::kInternalStorageBitWidth << ")";
    return success();
  }
  return emitError() << "expected integer or index type";
}

BoolAttr IntegerAttr::getBoolAttrUnchecked(IntegerType type, bool value) {
  auto attr = Base::get(type.getContext(), type, APInt(/*numBits=*/1, value));
  return llvm::cast<BoolAttr>(attr);
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

bool BoolAttr::getValue() const {
  auto *storage = reinterpret_cast<IntegerAttrStorage *>(impl);
  return storage->value.getBoolValue();
}

bool BoolAttr::classof(Attribute attr) {
  IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(attr);
  return intAttr && intAttr.getType().isSignlessInteger(1);
}

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

LogicalResult OpaqueAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringAttr dialect, StringRef attrData,
                                 Type type) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitError() << "invalid dialect namespace '" << dialect << "'";

  // Check that the dialect is actually registered.
  MLIRContext *context = dialect.getContext();
  if (!context->allowsUnregisteredDialects() &&
      !context->getLoadedDialect(dialect.strref())) {
    return emitError()
           << "#" << dialect << "<\"" << attrData << "\"> : " << type
           << " attribute created with unregistered dialect. If this is "
              "intended, please call allowUnregisteredDialects() on the "
              "MLIRContext, or use -allow-unregistered-dialect with "
              "the MLIR opt tool used";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DenseElementsAttr Utilities
//===----------------------------------------------------------------------===//

const char DenseIntOrFPElementsAttrStorage::kSplatTrue = ~0;
const char DenseIntOrFPElementsAttrStorage::kSplatFalse = 0;

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
  assert(llvm::endianness::native == llvm::endianness::big);
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
  assert(llvm::endianness::native == llvm::endianness::big);
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
    return setBit(rawData, bitPos, value.isOne());

  // Otherwise, the bit position is guaranteed to be byte aligned.
  assert((bitPos % CHAR_BIT) == 0 && "expected bitPos to be 8-bit aligned");
  if (llvm::endianness::native == llvm::endianness::big) {
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
  if (llvm::endianness::native == llvm::endianness::big) {
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
  auto owner = llvm::cast<DenseElementsAttr>(getFromOpaquePointer(base));
  Type eltTy = owner.getElementType();
  if (llvm::dyn_cast<IntegerType>(eltTy))
    return IntegerAttr::get(eltTy, *IntElementIterator(owner, index));
  if (llvm::isa<IndexType>(eltTy))
    return IntegerAttr::get(eltTy, *IntElementIterator(owner, index));
  if (auto floatEltTy = llvm::dyn_cast<FloatType>(eltTy)) {
    IntElementIterator intIt(owner, index);
    FloatElementIterator floatIt(floatEltTy.getFloatSemantics(), intIt);
    return FloatAttr::get(eltTy, *floatIt);
  }
  if (auto complexTy = llvm::dyn_cast<ComplexType>(eltTy)) {
    auto complexEltTy = complexTy.getElementType();
    ComplexIntElementIterator complexIntIt(owner, index);
    if (llvm::isa<IntegerType>(complexEltTy)) {
      auto value = *complexIntIt;
      auto real = IntegerAttr::get(complexEltTy, value.real());
      auto imag = IntegerAttr::get(complexEltTy, value.imag());
      return ArrayAttr::get(complexTy.getContext(),
                            ArrayRef<Attribute>{real, imag});
    }

    ComplexFloatElementIterator complexFloatIt(
        llvm::cast<FloatType>(complexEltTy).getFloatSemantics(), complexIntIt);
    auto value = *complexFloatIt;
    auto real = FloatAttr::get(complexEltTy, value.real());
    auto imag = FloatAttr::get(complexEltTy, value.imag());
    return ArrayAttr::get(complexTy.getContext(),
                          ArrayRef<Attribute>{real, imag});
  }
  if (llvm::isa<DenseStringElementsAttr>(owner)) {
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
      bitWidth(getDenseElementBitWidth(attr.getElementType())) {}

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
  auto complexType = llvm::cast<ComplexType>(attr.getElementType());
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
// DenseArrayAttr
//===----------------------------------------------------------------------===//

LogicalResult
DenseArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       Type elementType, int64_t size, ArrayRef<char> rawData) {
  if (!elementType.isIntOrIndexOrFloat())
    return emitError() << "expected integer or floating point element type";
  int64_t dataSize = rawData.size();
  int64_t elementSize =
      llvm::divideCeil(elementType.getIntOrFloatBitWidth(), CHAR_BIT);
  if (size * elementSize != dataSize) {
    return emitError() << "expected data size (" << size << " elements, "
                       << elementSize
                       << " bytes each) does not match: " << dataSize
                       << " bytes";
  }
  return success();
}

namespace {
/// Instantiations of this class provide utilities for interacting with native
/// data types in the context of DenseArrayAttr.
template <size_t width,
          IntegerType::SignednessSemantics signedness = IntegerType::Signless>
struct DenseArrayAttrIntUtil {
  static bool checkElementType(Type eltType) {
    auto type = llvm::dyn_cast<IntegerType>(eltType);
    if (!type || type.getWidth() != width)
      return false;
    return type.getSignedness() == signedness;
  }

  static Type getElementType(MLIRContext *ctx) {
    return IntegerType::get(ctx, width, signedness);
  }

  template <typename T>
  static void printElement(raw_ostream &os, T value) {
    os << value;
  }

  template <typename T>
  static ParseResult parseElement(AsmParser &parser, T &value) {
    return parser.parseInteger(value);
  }
};
template <typename T>
struct DenseArrayAttrUtil;

/// Specialization for boolean elements to print 'true' and 'false' literals for
/// elements.
template <>
struct DenseArrayAttrUtil<bool> : public DenseArrayAttrIntUtil<1> {
  static void printElement(raw_ostream &os, bool value) {
    os << (value ? "true" : "false");
  }
};

/// Specialization for 8-bit integers to ensure values are printed as integers
/// and not characters.
template <>
struct DenseArrayAttrUtil<int8_t> : public DenseArrayAttrIntUtil<8> {
  static void printElement(raw_ostream &os, int8_t value) {
    os << static_cast<int>(value);
  }
};
template <>
struct DenseArrayAttrUtil<int16_t> : public DenseArrayAttrIntUtil<16> {};
template <>
struct DenseArrayAttrUtil<int32_t> : public DenseArrayAttrIntUtil<32> {};
template <>
struct DenseArrayAttrUtil<int64_t> : public DenseArrayAttrIntUtil<64> {};

/// Specialization for 32-bit floats.
template <>
struct DenseArrayAttrUtil<float> {
  static bool checkElementType(Type eltType) { return eltType.isF32(); }
  static Type getElementType(MLIRContext *ctx) { return Float32Type::get(ctx); }
  static void printElement(raw_ostream &os, float value) { os << value; }

  /// Parse a double and cast it to a float.
  static ParseResult parseElement(AsmParser &parser, float &value) {
    double doubleVal;
    if (parser.parseFloat(doubleVal))
      return failure();
    value = doubleVal;
    return success();
  }
};

/// Specialization for 64-bit floats.
template <>
struct DenseArrayAttrUtil<double> {
  static bool checkElementType(Type eltType) { return eltType.isF64(); }
  static Type getElementType(MLIRContext *ctx) { return Float64Type::get(ctx); }
  static void printElement(raw_ostream &os, float value) { os << value; }
  static ParseResult parseElement(AsmParser &parser, double &value) {
    return parser.parseFloat(value);
  }
};
} // namespace

template <typename T>
void DenseArrayAttrImpl<T>::print(AsmPrinter &printer) const {
  print(printer.getStream());
}

template <typename T>
void DenseArrayAttrImpl<T>::printWithoutBraces(raw_ostream &os) const {
  llvm::interleaveComma(asArrayRef(), os, [&](T value) {
    DenseArrayAttrUtil<T>::printElement(os, value);
  });
}

template <typename T>
void DenseArrayAttrImpl<T>::print(raw_ostream &os) const {
  os << "[";
  printWithoutBraces(os);
  os << "]";
}

/// Parse a DenseArrayAttr without the braces: `1, 2, 3`
template <typename T>
Attribute DenseArrayAttrImpl<T>::parseWithoutBraces(AsmParser &parser,
                                                    Type odsType) {
  SmallVector<T> data;
  if (failed(parser.parseCommaSeparatedList([&]() {
        T value;
        if (DenseArrayAttrUtil<T>::parseElement(parser, value))
          return failure();
        data.push_back(value);
        return success();
      })))
    return {};
  return get(parser.getContext(), data);
}

/// Parse a DenseArrayAttr: `[ 1, 2, 3 ]`
template <typename T>
Attribute DenseArrayAttrImpl<T>::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLSquare())
    return {};
  // Handle empty list case.
  if (succeeded(parser.parseOptionalRSquare()))
    return get(parser.getContext(), {});
  Attribute result = parseWithoutBraces(parser, odsType);
  if (parser.parseRSquare())
    return {};
  return result;
}

/// Conversion from DenseArrayAttr<T> to ArrayRef<T>.
template <typename T>
DenseArrayAttrImpl<T>::operator ArrayRef<T>() const {
  ArrayRef<char> raw = getRawData();
  assert((raw.size() % sizeof(T)) == 0);
  return ArrayRef<T>(reinterpret_cast<const T *>(raw.data()),
                     raw.size() / sizeof(T));
}

/// Builds a DenseArrayAttr<T> from an ArrayRef<T>.
template <typename T>
DenseArrayAttrImpl<T> DenseArrayAttrImpl<T>::get(MLIRContext *context,
                                                 ArrayRef<T> content) {
  Type elementType = DenseArrayAttrUtil<T>::getElementType(context);
  auto rawArray = ArrayRef<char>(reinterpret_cast<const char *>(content.data()),
                                 content.size() * sizeof(T));
  return llvm::cast<DenseArrayAttrImpl<T>>(
      Base::get(context, elementType, content.size(), rawArray));
}

template <typename T>
bool DenseArrayAttrImpl<T>::classof(Attribute attr) {
  if (auto denseArray = llvm::dyn_cast<DenseArrayAttr>(attr))
    return DenseArrayAttrUtil<T>::checkElementType(denseArray.getElementType());
  return false;
}

namespace mlir {
namespace detail {
// Explicit instantiation for all the supported DenseArrayAttr.
template class DenseArrayAttrImpl<bool>;
template class DenseArrayAttrImpl<int8_t>;
template class DenseArrayAttrImpl<int16_t>;
template class DenseArrayAttrImpl<int32_t>;
template class DenseArrayAttrImpl<int64_t>;
template class DenseArrayAttrImpl<float>;
template class DenseArrayAttrImpl<double>;
} // namespace detail
} // namespace mlir

//===----------------------------------------------------------------------===//
// DenseElementsAttr
//===----------------------------------------------------------------------===//

/// Method for support type inquiry through isa, cast and dyn_cast.
bool DenseElementsAttr::classof(Attribute attr) {
  return llvm::isa<DenseIntOrFPElementsAttr, DenseStringElementsAttr>(attr);
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<Attribute> values) {
  assert(hasSameElementsOrSplat(type, values));

  Type eltType = type.getElementType();

  // Take care complex type case first.
  if (auto complexType = llvm::dyn_cast<ComplexType>(eltType)) {
    if (complexType.getElementType().isIntOrIndex()) {
      SmallVector<std::complex<APInt>> complexValues;
      complexValues.reserve(values.size());
      for (Attribute attr : values) {
        assert(llvm::isa<ArrayAttr>(attr) && "expected ArrayAttr for complex");
        auto arrayAttr = llvm::cast<ArrayAttr>(attr);
        assert(arrayAttr.size() == 2 && "expected 2 element for complex");
        auto attr0 = arrayAttr[0];
        auto attr1 = arrayAttr[1];
        complexValues.push_back(
            std::complex<APInt>(llvm::cast<IntegerAttr>(attr0).getValue(),
                                llvm::cast<IntegerAttr>(attr1).getValue()));
      }
      return DenseElementsAttr::get(type, complexValues);
    }
    // Must be float.
    SmallVector<std::complex<APFloat>> complexValues;
    complexValues.reserve(values.size());
    for (Attribute attr : values) {
      assert(llvm::isa<ArrayAttr>(attr) && "expected ArrayAttr for complex");
      auto arrayAttr = llvm::cast<ArrayAttr>(attr);
      assert(arrayAttr.size() == 2 && "expected 2 element for complex");
      auto attr0 = arrayAttr[0];
      auto attr1 = arrayAttr[1];
      complexValues.push_back(
          std::complex<APFloat>(llvm::cast<FloatAttr>(attr0).getValue(),
                                llvm::cast<FloatAttr>(attr1).getValue()));
    }
    return DenseElementsAttr::get(type, complexValues);
  }

  // If the element type is not based on int/float/index, assume it is a string
  // type.
  if (!eltType.isIntOrIndexOrFloat()) {
    SmallVector<StringRef, 8> stringValues;
    stringValues.reserve(values.size());
    for (Attribute attr : values) {
      assert(llvm::isa<StringAttr>(attr) &&
             "expected string value for non integer/index/float element");
      stringValues.push_back(llvm::cast<StringAttr>(attr).getValue());
    }
    return get(type, stringValues);
  }

  // Otherwise, get the raw storage width to use for the allocation.
  size_t bitWidth = getDenseElementBitWidth(eltType);
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

  // Compress the attribute values into a character buffer.
  SmallVector<char, 8> data(
      llvm::divideCeil(storageBitWidth * values.size(), CHAR_BIT));
  APInt intVal;
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(values[i])) {
      assert(floatAttr.getType() == eltType &&
             "expected float attribute type to equal element type");
      intVal = floatAttr.getValue().bitcastToAPInt();
    } else {
      auto intAttr = llvm::cast<IntegerAttr>(values[i]);
      assert(intAttr.getType() == eltType &&
             "expected integer attribute type to equal element type");
      intVal = intAttr.getValue();
    }

    assert(intVal.getBitWidth() == bitWidth &&
           "expected value to have same bitwidth as element type");
    writeBits(data.data(), i * storageBitWidth, intVal);
  }

  // Handle the special encoding of splat of bool.
  if (values.size() == 1 && eltType.isInteger(1))
    data[0] = data[0] ? -1 : 0;

  return DenseIntOrFPElementsAttr::getRaw(type, data);
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<bool> values) {
  assert(hasSameElementsOrSplat(type, values));
  assert(type.getElementType().isInteger(1));

  std::vector<char> buff(llvm::divideCeil(values.size(), CHAR_BIT));

  if (!values.empty()) {
    bool isSplat = true;
    bool firstValue = values[0];
    for (int i = 0, e = values.size(); i != e; ++i) {
      isSplat &= values[i] == firstValue;
      setBit(buff.data(), i, values[i]);
    }

    // Splat of bool is encoded as a byte with all-ones in it.
    if (isSplat) {
      buff.resize(1);
      buff[0] = values[0] ? -1 : 0;
    }
  }

  return DenseIntOrFPElementsAttr::getRaw(type, buff);
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
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, values);
}
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<std::complex<APInt>> values) {
  ComplexType complex = llvm::cast<ComplexType>(type.getElementType());
  assert(llvm::isa<IntegerType>(complex.getElementType()));
  assert(hasSameElementsOrSplat(type, values));
  size_t storageBitWidth = getDenseElementStorageWidth(complex) / 2;
  ArrayRef<APInt> intVals(reinterpret_cast<const APInt *>(values.data()),
                          values.size() * 2);
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, intVals);
}

// Constructs a dense float elements attribute from an array of APFloat
// values. Each APFloat value is expected to have the same bitwidth as the
// element type of 'type'.
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<APFloat> values) {
  assert(llvm::isa<FloatType>(type.getElementType()));
  assert(hasSameElementsOrSplat(type, values));
  size_t storageBitWidth = getDenseElementStorageWidth(type.getElementType());
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, values);
}
DenseElementsAttr
DenseElementsAttr::get(ShapedType type,
                       ArrayRef<std::complex<APFloat>> values) {
  ComplexType complex = llvm::cast<ComplexType>(type.getElementType());
  assert(llvm::isa<FloatType>(complex.getElementType()));
  assert(hasSameElementsOrSplat(type, values));
  ArrayRef<APFloat> apVals(reinterpret_cast<const APFloat *>(values.data()),
                           values.size() * 2);
  size_t storageBitWidth = getDenseElementStorageWidth(complex) / 2;
  return DenseIntOrFPElementsAttr::getRaw(type, storageBitWidth, apVals);
}

/// Construct a dense elements attribute from a raw buffer representing the
/// data for this attribute. Users should generally not use this methods as
/// the expected buffer format may not be a form the user expects.
DenseElementsAttr
DenseElementsAttr::getFromRawBuffer(ShapedType type, ArrayRef<char> rawBuffer) {
  return DenseIntOrFPElementsAttr::getRaw(type, rawBuffer);
}

/// Returns true if the given buffer is a valid raw buffer for the given type.
bool DenseElementsAttr::isValidRawBuffer(ShapedType type,
                                         ArrayRef<char> rawBuffer,
                                         bool &detectedSplat) {
  size_t storageWidth = getDenseElementStorageWidth(type.getElementType());
  size_t rawBufferWidth = rawBuffer.size() * CHAR_BIT;
  int64_t numElements = type.getNumElements();

  // The initializer is always a splat if the result type has a single element.
  detectedSplat = numElements == 1;

  // Storage width of 1 is special as it is packed by the bit.
  if (storageWidth == 1) {
    // Check for a splat, or a buffer equal to the number of elements which
    // consists of either all 0's or all 1's.
    if (rawBuffer.size() == 1) {
      auto rawByte = static_cast<uint8_t>(rawBuffer[0]);
      if (rawByte == 0 || rawByte == 0xff) {
        detectedSplat = true;
        return true;
      }
    }

    // This is a valid non-splat buffer if it has the right size.
    return rawBufferWidth == llvm::alignTo<8>(numElements);
  }

  // All other types are 8-bit aligned, so we can just check the buffer width
  // to know if only a single initializer element was passed in.
  if (rawBufferWidth == storageWidth) {
    detectedSplat = true;
    return true;
  }

  // The raw buffer is valid if it has the right size.
  return rawBufferWidth == storageWidth * numElements;
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
    return llvm::isa<FloatType>(type);
  if (type.isIndex())
    return true;

  auto intType = llvm::dyn_cast<IntegerType>(type);
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

bool DenseElementsAttr::isValidIntOrFloat(int64_t dataEltSize, bool isInt,
                                          bool isSigned) const {
  return ::isValidIntOrFloat(getElementType(), dataEltSize, isInt, isSigned);
}
bool DenseElementsAttr::isValidComplex(int64_t dataEltSize, bool isInt,
                                       bool isSigned) const {
  return ::isValidIntOrFloat(
      llvm::cast<ComplexType>(getElementType()).getElementType(),
      dataEltSize / 2, isInt, isSigned);
}

/// Returns true if this attribute corresponds to a splat, i.e. if all element
/// values are the same.
bool DenseElementsAttr::isSplat() const {
  return static_cast<DenseElementsAttributeStorage *>(impl)->isSplat;
}

/// Return if the given complex type has an integer element type.
static bool isComplexOfIntType(Type type) {
  return llvm::isa<IntegerType>(llvm::cast<ComplexType>(type).getElementType());
}

auto DenseElementsAttr::tryGetComplexIntValues() const
    -> FailureOr<iterator_range_impl<ComplexIntElementIterator>> {
  if (!isComplexOfIntType(getElementType()))
    return failure();
  return iterator_range_impl<ComplexIntElementIterator>(
      getType(), ComplexIntElementIterator(*this, 0),
      ComplexIntElementIterator(*this, getNumElements()));
}

auto DenseElementsAttr::tryGetFloatValues() const
    -> FailureOr<iterator_range_impl<FloatElementIterator>> {
  auto eltTy = llvm::dyn_cast<FloatType>(getElementType());
  if (!eltTy)
    return failure();
  const auto &elementSemantics = eltTy.getFloatSemantics();
  return iterator_range_impl<FloatElementIterator>(
      getType(), FloatElementIterator(elementSemantics, raw_int_begin()),
      FloatElementIterator(elementSemantics, raw_int_end()));
}

auto DenseElementsAttr::tryGetComplexFloatValues() const
    -> FailureOr<iterator_range_impl<ComplexFloatElementIterator>> {
  auto complexTy = llvm::dyn_cast<ComplexType>(getElementType());
  if (!complexTy)
    return failure();
  auto eltTy = llvm::dyn_cast<FloatType>(complexTy.getElementType());
  if (!eltTy)
    return failure();
  const auto &semantics = eltTy.getFloatSemantics();
  return iterator_range_impl<ComplexFloatElementIterator>(
      getType(), {semantics, {*this, 0}},
      {semantics, {*this, static_cast<size_t>(getNumElements())}});
}

/// Return the raw storage data held by this attribute.
ArrayRef<char> DenseElementsAttr::getRawData() const {
  return static_cast<DenseIntOrFPElementsAttrStorage *>(impl)->data;
}

ArrayRef<StringRef> DenseElementsAttr::getRawStringData() const {
  return static_cast<DenseStringElementsAttrStorage *>(impl)->data;
}

/// Return a new DenseElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newType'. The new type must have the
/// same total number of elements as well as element type.
DenseElementsAttr DenseElementsAttr::reshape(ShapedType newType) {
  ShapedType curType = getType();
  if (curType == newType)
    return *this;

  assert(newType.getElementType() == curType.getElementType() &&
         "expected the same element type");
  assert(newType.getNumElements() == curType.getNumElements() &&
         "expected the same number of elements");
  return DenseIntOrFPElementsAttr::getRaw(newType, getRawData());
}

DenseElementsAttr DenseElementsAttr::resizeSplat(ShapedType newType) {
  assert(isSplat() && "expected a splat type");

  ShapedType curType = getType();
  if (curType == newType)
    return *this;

  assert(newType.getElementType() == curType.getElementType() &&
         "expected the same element type");
  return DenseIntOrFPElementsAttr::getRaw(newType, getRawData());
}

/// Return a new DenseElementsAttr that has the same data as the current
/// attribute, but has bitcast elements such that it is now 'newType'. The new
/// type must have the same shape and element types of the same bitwidth as the
/// current type.
DenseElementsAttr DenseElementsAttr::bitcast(Type newElType) {
  ShapedType curType = getType();
  Type curElType = curType.getElementType();
  if (curElType == newElType)
    return *this;

  assert(getDenseElementBitWidth(newElType) ==
             getDenseElementBitWidth(curElType) &&
         "expected element types with the same bitwidth");
  return DenseIntOrFPElementsAttr::getRaw(curType.clone(newElType),
                                          getRawData());
}

DenseElementsAttr
DenseElementsAttr::mapValues(Type newElementType,
                             function_ref<APInt(const APInt &)> mapping) const {
  return llvm::cast<DenseIntElementsAttr>(*this).mapValues(newElementType, mapping);
}

DenseElementsAttr DenseElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APFloat &)> mapping) const {
  return llvm::cast<DenseFPElementsAttr>(*this).mapValues(newElementType, mapping);
}

ShapedType DenseElementsAttr::getType() const {
  return static_cast<const DenseElementsAttributeStorage *>(impl)->type;
}

Type DenseElementsAttr::getElementType() const {
  return getType().getElementType();
}

int64_t DenseElementsAttr::getNumElements() const {
  return getType().getNumElements();
}

//===----------------------------------------------------------------------===//
// DenseIntOrFPElementsAttr
//===----------------------------------------------------------------------===//

/// Utility method to write a range of APInt values to a buffer.
template <typename APRangeT>
static void writeAPIntsToBuffer(size_t storageWidth, std::vector<char> &data,
                                APRangeT &&values) {
  size_t numValues = llvm::size(values);
  data.resize(llvm::divideCeil(storageWidth * numValues, CHAR_BIT));
  size_t offset = 0;
  for (auto it = values.begin(), e = values.end(); it != e;
       ++it, offset += storageWidth) {
    assert((*it).getBitWidth() <= storageWidth);
    writeBits(data.data(), offset, *it);
  }

  // Handle the special encoding of splat of a boolean.
  if (numValues == 1 && (*values.begin()).getBitWidth() == 1)
    data[0] = data[0] ? -1 : 0;
}

/// Constructs a dense elements attribute from an array of raw APFloat values.
/// Each APFloat value is expected to have the same bitwidth as the element
/// type of 'type'. 'type' must be a vector or tensor with static shape.
DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   size_t storageWidth,
                                                   ArrayRef<APFloat> values) {
  std::vector<char> data;
  auto unwrapFloat = [](const APFloat &val) { return val.bitcastToAPInt(); };
  writeAPIntsToBuffer(storageWidth, data, llvm::map_range(values, unwrapFloat));
  return DenseIntOrFPElementsAttr::getRaw(type, data);
}

/// Constructs a dense elements attribute from an array of raw APInt values.
/// Each APInt value is expected to have the same bitwidth as the element type
/// of 'type'.
DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   size_t storageWidth,
                                                   ArrayRef<APInt> values) {
  std::vector<char> data;
  writeAPIntsToBuffer(storageWidth, data, values);
  return DenseIntOrFPElementsAttr::getRaw(type, data);
}

DenseElementsAttr DenseIntOrFPElementsAttr::getRaw(ShapedType type,
                                                   ArrayRef<char> data) {
  assert(type.hasStaticShape() && "type must have static shape");
  bool isSplat = false;
  bool isValid = isValidRawBuffer(type, data, isSplat);
  assert(isValid);
  (void)isValid;
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
      llvm::cast<ComplexType>(type.getElementType()).getElementType(),
      dataEltSize / 2, isInt, isSigned));

  int64_t numElements = data.size() / dataEltSize;
  (void)numElements;
  assert(numElements == 1 || numElements == type.getNumElements());
  return getRaw(type, data);
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
  (void)numElements;
  return getRaw(type, data);
}

void DenseIntOrFPElementsAttr::convertEndianOfCharForBEmachine(
    const char *inRawData, char *outRawData, size_t elementBitWidth,
    size_t numElements) {
  using llvm::support::ulittle16_t;
  using llvm::support::ulittle32_t;
  using llvm::support::ulittle64_t;

  assert(llvm::endianness::native == llvm::endianness::big);
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
  if (ComplexType complexTy = llvm::dyn_cast<ComplexType>(elementType)) {
    elementType = complexTy.getElementType();
    numElements = numElements * 2;
  }
  size_t elementBitWidth = getDenseElementStorageWidth(elementType);
  assert(numElements * elementBitWidth == inRawData.size() * CHAR_BIT &&
         inRawData.size() <= outRawData.size());
  if (elementBitWidth <= CHAR_BIT)
    std::memcpy(outRawData.begin(), inRawData.begin(), inRawData.size());
  else
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

  ShapedType newArrayType = inType.cloneWith(inType.getShape(), newElementType);

  size_t numRawElements = attr.isSplat() ? 1 : newArrayType.getNumElements();
  data.resize(llvm::divideCeil(storageBitWidth * numRawElements, CHAR_BIT));

  // Functor used to process a single element value of the attribute.
  auto processElt = [&](decltype(*attr.begin()) value, size_t index) {
    auto newInt = mapping(value);
    assert(newInt.getBitWidth() == bitWidth);
    writeBits(data.data(), index * storageBitWidth, newInt);
  };

  // Check for the splat case.
  if (attr.isSplat()) {
    if (bitWidth == 1) {
      // Handle the special encoding of splat of bool.
      data[0] = mapping(*attr.begin()).isZero() ? 0 : -1;
    } else {
      processElt(*attr.begin(), /*index=*/0);
    }
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

  return getRaw(newArrayType, elementData);
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseFPElementsAttr::classof(Attribute attr) {
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr))
    return llvm::isa<FloatType>(denseAttr.getType().getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

DenseElementsAttr DenseIntElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APInt &)> mapping) const {
  llvm::SmallVector<char, 8> elementData;
  auto newArrayType =
      mappingHelper(mapping, *this, getType(), newElementType, elementData);
  return getRaw(newArrayType, elementData);
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseIntElementsAttr::classof(Attribute attr) {
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr))
    return denseAttr.getType().getElementType().isIntOrIndex();
  return false;
}

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

DenseResourceElementsAttr
DenseResourceElementsAttr::get(ShapedType type,
                               DenseResourceElementsHandle handle) {
  return Base::get(type.getContext(), type, handle);
}

DenseResourceElementsAttr DenseResourceElementsAttr::get(ShapedType type,
                                                         StringRef blobName,
                                                         AsmResourceBlob blob) {
  // Extract the builtin dialect resource manager from context and construct a
  // handle by inserting a new resource using the provided blob.
  auto &manager =
      DenseResourceElementsHandle::getManagerInterface(type.getContext());
  return get(type, manager.insert(blobName, std::move(blob)));
}

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttrBase

namespace {
/// Instantiations of this class provide utilities for interacting with native
/// data types in the context of DenseResourceElementsAttr.
template <typename T>
struct DenseResourceAttrUtil;
template <size_t width, bool isSigned>
struct DenseResourceElementsAttrIntUtil {
  static bool checkElementType(Type eltType) {
    IntegerType type = llvm::dyn_cast<IntegerType>(eltType);
    if (!type || type.getWidth() != width)
      return false;
    return isSigned ? !type.isUnsigned() : !type.isSigned();
  }
};
template <>
struct DenseResourceAttrUtil<bool> {
  static bool checkElementType(Type eltType) {
    return eltType.isSignlessInteger(1);
  }
};
template <>
struct DenseResourceAttrUtil<int8_t>
    : public DenseResourceElementsAttrIntUtil<8, true> {};
template <>
struct DenseResourceAttrUtil<uint8_t>
    : public DenseResourceElementsAttrIntUtil<8, false> {};
template <>
struct DenseResourceAttrUtil<int16_t>
    : public DenseResourceElementsAttrIntUtil<16, true> {};
template <>
struct DenseResourceAttrUtil<uint16_t>
    : public DenseResourceElementsAttrIntUtil<16, false> {};
template <>
struct DenseResourceAttrUtil<int32_t>
    : public DenseResourceElementsAttrIntUtil<32, true> {};
template <>
struct DenseResourceAttrUtil<uint32_t>
    : public DenseResourceElementsAttrIntUtil<32, false> {};
template <>
struct DenseResourceAttrUtil<int64_t>
    : public DenseResourceElementsAttrIntUtil<64, true> {};
template <>
struct DenseResourceAttrUtil<uint64_t>
    : public DenseResourceElementsAttrIntUtil<64, false> {};
template <>
struct DenseResourceAttrUtil<float> {
  static bool checkElementType(Type eltType) { return eltType.isF32(); }
};
template <>
struct DenseResourceAttrUtil<double> {
  static bool checkElementType(Type eltType) { return eltType.isF64(); }
};
} // namespace

template <typename T>
DenseResourceElementsAttrBase<T>
DenseResourceElementsAttrBase<T>::get(ShapedType type, StringRef blobName,
                                      AsmResourceBlob blob) {
  // Check that the blob is in the form we were expecting.
  assert(blob.getDataAlignment() == alignof(T) &&
         "alignment mismatch between expected alignment and blob alignment");
  assert(((blob.getData().size() % sizeof(T)) == 0) &&
         "size mismatch between expected element width and blob size");
  assert(DenseResourceAttrUtil<T>::checkElementType(type.getElementType()) &&
         "invalid shape element type for provided type `T`");
  return llvm::cast<DenseResourceElementsAttrBase<T>>(
      DenseResourceElementsAttr::get(type, blobName, std::move(blob)));
}

template <typename T>
std::optional<ArrayRef<T>>
DenseResourceElementsAttrBase<T>::tryGetAsArrayRef() const {
  if (AsmResourceBlob *blob = this->getRawHandle().getBlob())
    return blob->template getDataAs<T>();
  return std::nullopt;
}

template <typename T>
bool DenseResourceElementsAttrBase<T>::classof(Attribute attr) {
  auto resourceAttr = llvm::dyn_cast<DenseResourceElementsAttr>(attr);
  return resourceAttr && DenseResourceAttrUtil<T>::checkElementType(
                             resourceAttr.getElementType());
}

namespace mlir {
namespace detail {
// Explicit instantiation for all the supported DenseResourceElementsAttr.
template class DenseResourceElementsAttrBase<bool>;
template class DenseResourceElementsAttrBase<int8_t>;
template class DenseResourceElementsAttrBase<int16_t>;
template class DenseResourceElementsAttrBase<int32_t>;
template class DenseResourceElementsAttrBase<int64_t>;
template class DenseResourceElementsAttrBase<uint8_t>;
template class DenseResourceElementsAttrBase<uint16_t>;
template class DenseResourceElementsAttrBase<uint32_t>;
template class DenseResourceElementsAttrBase<uint64_t>;
template class DenseResourceElementsAttrBase<float>;
template class DenseResourceElementsAttrBase<double>;
} // namespace detail
} // namespace mlir

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

/// Get a zero APFloat for the given sparse attribute.
APFloat SparseElementsAttr::getZeroAPFloat() const {
  auto eltType = llvm::cast<FloatType>(getElementType());
  return APFloat(eltType.getFloatSemantics());
}

/// Get a zero APInt for the given sparse attribute.
APInt SparseElementsAttr::getZeroAPInt() const {
  auto eltType = llvm::cast<IntegerType>(getElementType());
  return APInt::getZero(eltType.getWidth());
}

/// Get a zero attribute for the given attribute type.
Attribute SparseElementsAttr::getZeroAttr() const {
  auto eltType = getElementType();

  // Handle floating point elements.
  if (llvm::isa<FloatType>(eltType))
    return FloatAttr::get(eltType, 0);

  // Handle complex elements.
  if (auto complexTy = llvm::dyn_cast<ComplexType>(eltType)) {
    auto eltType = complexTy.getElementType();
    Attribute zero;
    if (llvm::isa<FloatType>(eltType))
      zero = FloatAttr::get(eltType, 0);
    else // must be integer
      zero = IntegerAttr::get(eltType, 0);
    return ArrayAttr::get(complexTy.getContext(),
                          ArrayRef<Attribute>{zero, zero});
  }

  // Handle string type.
  if (llvm::isa<DenseStringElementsAttr>(getValues()))
    return StringAttr::get("", eltType);

  // Otherwise, this is an integer.
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

LogicalResult
SparseElementsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           ShapedType type, DenseIntElementsAttr sparseIndices,
                           DenseElementsAttr values) {
  ShapedType valuesType = values.getType();
  if (valuesType.getRank() != 1)
    return emitError() << "expected 1-d tensor for sparse element values";

  // Verify the indices and values shape.
  ShapedType indicesType = sparseIndices.getType();
  auto emitShapeError = [&]() {
    return emitError() << "expected shape ([" << type.getShape()
                       << "]); inferred shape of indices literal (["
                       << indicesType.getShape()
                       << "]); inferred shape of values literal (["
                       << valuesType.getShape() << "])";
  };
  // Verify indices shape.
  size_t rank = type.getRank(), indicesRank = indicesType.getRank();
  if (indicesRank == 2) {
    if (indicesType.getDimSize(1) != static_cast<int64_t>(rank))
      return emitShapeError();
  } else if (indicesRank != 1 || rank != 1) {
    return emitShapeError();
  }
  // Verify the values shape.
  int64_t numSparseIndices = indicesType.getDimSize(0);
  if (numSparseIndices != valuesType.getDimSize(0))
    return emitShapeError();

  // Verify that the sparse indices are within the value shape.
  auto emitIndexError = [&](unsigned indexNum, ArrayRef<uint64_t> index) {
    return emitError()
           << "sparse index #" << indexNum
           << " is not contained within the value shape, with index=[" << index
           << "], and type=" << type;
  };

  // Handle the case where the index values are a splat.
  auto sparseIndexValues = sparseIndices.getValues<uint64_t>();
  if (sparseIndices.isSplat()) {
    SmallVector<uint64_t> indices(rank, *sparseIndexValues.begin());
    if (!ElementsAttr::isValidIndex(type, indices))
      return emitIndexError(0, indices);
    return success();
  }

  // Otherwise, reinterpret each index as an ArrayRef.
  for (size_t i = 0, e = numSparseIndices; i != e; ++i) {
    ArrayRef<uint64_t> index(&*std::next(sparseIndexValues.begin(), i * rank),
                             rank);
    if (!ElementsAttr::isValidIndex(type, index))
      return emitIndexError(i, index);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DistinctAttr
//===----------------------------------------------------------------------===//

DistinctAttr DistinctAttr::create(Attribute referencedAttr) {
  return Base::get(referencedAttr.getContext(), referencedAttr);
}

Attribute DistinctAttr::getReferencedAttr() const {
  return getImpl()->referencedAttr;
}

//===----------------------------------------------------------------------===//
// Attribute Utilities
//===----------------------------------------------------------------------===//

AffineMap mlir::makeStridedLinearLayoutMap(ArrayRef<int64_t> strides,
                                           int64_t offset,
                                           MLIRContext *context) {
  AffineExpr expr;
  unsigned nSymbols = 0;

  // AffineExpr for offset.
  // Static case.
  if (!ShapedType::isDynamic(offset)) {
    auto cst = getAffineConstantExpr(offset, context);
    expr = cst;
  } else {
    // Dynamic case, new symbol for the offset.
    auto sym = getAffineSymbolExpr(nSymbols++, context);
    expr = sym;
  }

  // AffineExpr for strides.
  for (const auto &en : llvm::enumerate(strides)) {
    auto dim = en.index();
    auto stride = en.value();
    assert(stride != 0 && "Invalid stride specification");
    auto d = getAffineDimExpr(dim, context);
    AffineExpr mult;
    // Static case.
    if (!ShapedType::isDynamic(stride))
      mult = getAffineConstantExpr(stride, context);
    else
      // Dynamic case, new symbol for each new stride.
      mult = getAffineSymbolExpr(nSymbols++, context);
    expr = expr + d * mult;
  }

  return AffineMap::get(strides.size(), nSymbols, expr);
}
