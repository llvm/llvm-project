//===- BuiltinAttributes.cpp - C Interface to MLIR Builtin Attributes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

MlirAttribute mlirAttributeGetNull() { return {nullptr}; }

//===----------------------------------------------------------------------===//
// Location attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsALocation(MlirAttribute attr) {
  return llvm::isa<LocationAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAAffineMap(MlirAttribute attr) {
  return llvm::isa<AffineMapAttr>(unwrap(attr));
}

MlirAttribute mlirAffineMapAttrGet(MlirAffineMap map) {
  return wrap(AffineMapAttr::get(unwrap(map)));
}

MlirAffineMap mlirAffineMapAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<AffineMapAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirAffineMapAttrGetTypeID(void) {
  return wrap(AffineMapAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAArray(MlirAttribute attr) {
  return llvm::isa<ArrayAttr>(unwrap(attr));
}

MlirAttribute mlirArrayAttrGet(MlirContext ctx, intptr_t numElements,
                               MlirAttribute const *elements) {
  SmallVector<Attribute, 8> attrs;
  return wrap(
      ArrayAttr::get(unwrap(ctx), unwrapList(static_cast<size_t>(numElements),
                                             elements, attrs)));
}

intptr_t mlirArrayAttrGetNumElements(MlirAttribute attr) {
  return static_cast<intptr_t>(llvm::cast<ArrayAttr>(unwrap(attr)).size());
}

MlirAttribute mlirArrayAttrGetElement(MlirAttribute attr, intptr_t pos) {
  return wrap(llvm::cast<ArrayAttr>(unwrap(attr)).getValue()[pos]);
}

MlirTypeID mlirArrayAttrGetTypeID(void) { return wrap(ArrayAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsADictionary(MlirAttribute attr) {
  return llvm::isa<DictionaryAttr>(unwrap(attr));
}

MlirAttribute mlirDictionaryAttrGet(MlirContext ctx, intptr_t numElements,
                                    MlirNamedAttribute const *elements) {
  SmallVector<NamedAttribute, 8> attributes;
  attributes.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    attributes.emplace_back(unwrap(elements[i].name),
                            unwrap(elements[i].attribute));
  return wrap(DictionaryAttr::get(unwrap(ctx), attributes));
}

intptr_t mlirDictionaryAttrGetNumElements(MlirAttribute attr) {
  return static_cast<intptr_t>(llvm::cast<DictionaryAttr>(unwrap(attr)).size());
}

MlirNamedAttribute mlirDictionaryAttrGetElement(MlirAttribute attr,
                                                intptr_t pos) {
  NamedAttribute attribute =
      llvm::cast<DictionaryAttr>(unwrap(attr)).getValue()[pos];
  return {wrap(attribute.getName()), wrap(attribute.getValue())};
}

MlirAttribute mlirDictionaryAttrGetElementByName(MlirAttribute attr,
                                                 MlirStringRef name) {
  return wrap(llvm::cast<DictionaryAttr>(unwrap(attr)).get(unwrap(name)));
}

MlirTypeID mlirDictionaryAttrGetTypeID(void) {
  return wrap(DictionaryAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAFloat(MlirAttribute attr) {
  return llvm::isa<FloatAttr>(unwrap(attr));
}

MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx, MlirType type,
                                     double value) {
  return wrap(FloatAttr::get(unwrap(type), value));
}

MlirAttribute mlirFloatAttrDoubleGetChecked(MlirLocation loc, MlirType type,
                                            double value) {
  return wrap(FloatAttr::getChecked(unwrap(loc), unwrap(type), value));
}

double mlirFloatAttrGetValueDouble(MlirAttribute attr) {
  return llvm::cast<FloatAttr>(unwrap(attr)).getValueAsDouble();
}

MlirTypeID mlirFloatAttrGetTypeID(void) { return wrap(FloatAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAInteger(MlirAttribute attr) {
  return llvm::isa<IntegerAttr>(unwrap(attr));
}

MlirAttribute mlirIntegerAttrGet(MlirType type, int64_t value) {
  return wrap(IntegerAttr::get(unwrap(type), value));
}

int64_t mlirIntegerAttrGetValueInt(MlirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getInt();
}

int64_t mlirIntegerAttrGetValueSInt(MlirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getSInt();
}

uint64_t mlirIntegerAttrGetValueUInt(MlirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getUInt();
}

MlirTypeID mlirIntegerAttrGetTypeID(void) {
  return wrap(IntegerAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsABool(MlirAttribute attr) {
  return llvm::isa<BoolAttr>(unwrap(attr));
}

MlirAttribute mlirBoolAttrGet(MlirContext ctx, int value) {
  return wrap(BoolAttr::get(unwrap(ctx), value));
}

bool mlirBoolAttrGetValue(MlirAttribute attr) {
  return llvm::cast<BoolAttr>(unwrap(attr)).getValue();
}

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAIntegerSet(MlirAttribute attr) {
  return llvm::isa<IntegerSetAttr>(unwrap(attr));
}

MlirTypeID mlirIntegerSetAttrGetTypeID(void) {
  return wrap(IntegerSetAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAOpaque(MlirAttribute attr) {
  return llvm::isa<OpaqueAttr>(unwrap(attr));
}

MlirAttribute mlirOpaqueAttrGet(MlirContext ctx, MlirStringRef dialectNamespace,
                                intptr_t dataLength, const char *data,
                                MlirType type) {
  return wrap(
      OpaqueAttr::get(StringAttr::get(unwrap(ctx), unwrap(dialectNamespace)),
                      StringRef(data, dataLength), unwrap(type)));
}

MlirStringRef mlirOpaqueAttrGetDialectNamespace(MlirAttribute attr) {
  return wrap(
      llvm::cast<OpaqueAttr>(unwrap(attr)).getDialectNamespace().strref());
}

MlirStringRef mlirOpaqueAttrGetData(MlirAttribute attr) {
  return wrap(llvm::cast<OpaqueAttr>(unwrap(attr)).getAttrData());
}

MlirTypeID mlirOpaqueAttrGetTypeID(void) {
  return wrap(OpaqueAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAString(MlirAttribute attr) {
  return llvm::isa<StringAttr>(unwrap(attr));
}

MlirAttribute mlirStringAttrGet(MlirContext ctx, MlirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(ctx), unwrap(str)));
}

MlirAttribute mlirStringAttrTypedGet(MlirType type, MlirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(str), unwrap(type)));
}

MlirStringRef mlirStringAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<StringAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirStringAttrGetTypeID(void) {
  return wrap(StringAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsASymbolRef(MlirAttribute attr) {
  return llvm::isa<SymbolRefAttr>(unwrap(attr));
}

MlirAttribute mlirSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol,
                                   intptr_t numReferences,
                                   MlirAttribute const *references) {
  SmallVector<FlatSymbolRefAttr, 4> refs;
  refs.reserve(numReferences);
  for (intptr_t i = 0; i < numReferences; ++i)
    refs.push_back(llvm::cast<FlatSymbolRefAttr>(unwrap(references[i])));
  auto symbolAttr = StringAttr::get(unwrap(ctx), unwrap(symbol));
  return wrap(SymbolRefAttr::get(symbolAttr, refs));
}

MlirStringRef mlirSymbolRefAttrGetRootReference(MlirAttribute attr) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getRootReference().getValue());
}

MlirStringRef mlirSymbolRefAttrGetLeafReference(MlirAttribute attr) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getLeafReference().getValue());
}

intptr_t mlirSymbolRefAttrGetNumNestedReferences(MlirAttribute attr) {
  return static_cast<intptr_t>(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getNestedReferences().size());
}

MlirAttribute mlirSymbolRefAttrGetNestedReference(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getNestedReferences()[pos]);
}

MlirTypeID mlirSymbolRefAttrGetTypeID(void) {
  return wrap(SymbolRefAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAFlatSymbolRef(MlirAttribute attr) {
  return llvm::isa<FlatSymbolRefAttr>(unwrap(attr));
}

MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol) {
  return wrap(FlatSymbolRefAttr::get(unwrap(ctx), unwrap(symbol)));
}

MlirStringRef mlirFlatSymbolRefAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<FlatSymbolRefAttr>(unwrap(attr)).getValue());
}

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAType(MlirAttribute attr) {
  return llvm::isa<TypeAttr>(unwrap(attr));
}

MlirAttribute mlirTypeAttrGet(MlirType type) {
  return wrap(TypeAttr::get(unwrap(type)));
}

MlirType mlirTypeAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<TypeAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirTypeAttrGetTypeID(void) { return wrap(TypeAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAUnit(MlirAttribute attr) {
  return llvm::isa<UnitAttr>(unwrap(attr));
}

MlirAttribute mlirUnitAttrGet(MlirContext ctx) {
  return wrap(UnitAttr::get(unwrap(ctx)));
}

MlirTypeID mlirUnitAttrGetTypeID(void) { return wrap(UnitAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAElements(MlirAttribute attr) {
  return llvm::isa<ElementsAttr>(unwrap(attr));
}

MlirAttribute mlirElementsAttrGetValue(MlirAttribute attr, intptr_t rank,
                                       uint64_t *idxs) {
  return wrap(llvm::cast<ElementsAttr>(unwrap(attr))
                  .getValues<Attribute>()[llvm::ArrayRef(idxs, rank)]);
}

bool mlirElementsAttrIsValidIndex(MlirAttribute attr, intptr_t rank,
                                  uint64_t *idxs) {
  return llvm::cast<ElementsAttr>(unwrap(attr))
      .isValidIndex(llvm::ArrayRef(idxs, rank));
}

int64_t mlirElementsAttrGetNumElements(MlirAttribute attr) {
  return llvm::cast<ElementsAttr>(unwrap(attr)).getNumElements();
}

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//

MlirTypeID mlirDenseArrayAttrGetTypeID() {
  return wrap(DenseArrayAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// IsA support.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsADenseBoolArray(MlirAttribute attr) {
  return llvm::isa<DenseBoolArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseI8Array(MlirAttribute attr) {
  return llvm::isa<DenseI8ArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseI16Array(MlirAttribute attr) {
  return llvm::isa<DenseI16ArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseI32Array(MlirAttribute attr) {
  return llvm::isa<DenseI32ArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseI64Array(MlirAttribute attr) {
  return llvm::isa<DenseI64ArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseF32Array(MlirAttribute attr) {
  return llvm::isa<DenseF32ArrayAttr>(unwrap(attr));
}
bool mlirAttributeIsADenseF64Array(MlirAttribute attr) {
  return llvm::isa<DenseF64ArrayAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

MlirAttribute mlirDenseBoolArrayGet(MlirContext ctx, intptr_t size,
                                    int const *values) {
  SmallVector<bool, 4> elements(values, values + size);
  return wrap(DenseBoolArrayAttr::get(unwrap(ctx), elements));
}
MlirAttribute mlirDenseI8ArrayGet(MlirContext ctx, intptr_t size,
                                  int8_t const *values) {
  return wrap(
      DenseI8ArrayAttr::get(unwrap(ctx), ArrayRef<int8_t>(values, size)));
}
MlirAttribute mlirDenseI16ArrayGet(MlirContext ctx, intptr_t size,
                                   int16_t const *values) {
  return wrap(
      DenseI16ArrayAttr::get(unwrap(ctx), ArrayRef<int16_t>(values, size)));
}
MlirAttribute mlirDenseI32ArrayGet(MlirContext ctx, intptr_t size,
                                   int32_t const *values) {
  return wrap(
      DenseI32ArrayAttr::get(unwrap(ctx), ArrayRef<int32_t>(values, size)));
}
MlirAttribute mlirDenseI64ArrayGet(MlirContext ctx, intptr_t size,
                                   int64_t const *values) {
  return wrap(
      DenseI64ArrayAttr::get(unwrap(ctx), ArrayRef<int64_t>(values, size)));
}
MlirAttribute mlirDenseF32ArrayGet(MlirContext ctx, intptr_t size,
                                   float const *values) {
  return wrap(
      DenseF32ArrayAttr::get(unwrap(ctx), ArrayRef<float>(values, size)));
}
MlirAttribute mlirDenseF64ArrayGet(MlirContext ctx, intptr_t size,
                                   double const *values) {
  return wrap(
      DenseF64ArrayAttr::get(unwrap(ctx), ArrayRef<double>(values, size)));
}

//===----------------------------------------------------------------------===//
// Accessors.
//===----------------------------------------------------------------------===//

intptr_t mlirDenseArrayGetNumElements(MlirAttribute attr) {
  return llvm::cast<DenseArrayAttr>(unwrap(attr)).size();
}

//===----------------------------------------------------------------------===//
// Indexed accessors.
//===----------------------------------------------------------------------===//

bool mlirDenseBoolArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseBoolArrayAttr>(unwrap(attr))[pos];
}
int8_t mlirDenseI8ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI8ArrayAttr>(unwrap(attr))[pos];
}
int16_t mlirDenseI16ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI16ArrayAttr>(unwrap(attr))[pos];
}
int32_t mlirDenseI32ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI32ArrayAttr>(unwrap(attr))[pos];
}
int64_t mlirDenseI64ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI64ArrayAttr>(unwrap(attr))[pos];
}
float mlirDenseF32ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseF32ArrayAttr>(unwrap(attr))[pos];
}
double mlirDenseF64ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseF64ArrayAttr>(unwrap(attr))[pos];
}

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IsA support.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsADenseElements(MlirAttribute attr) {
  return llvm::isa<DenseElementsAttr>(unwrap(attr));
}

bool mlirAttributeIsADenseIntElements(MlirAttribute attr) {
  return llvm::isa<DenseIntElementsAttr>(unwrap(attr));
}

bool mlirAttributeIsADenseFPElements(MlirAttribute attr) {
  return llvm::isa<DenseFPElementsAttr>(unwrap(attr));
}

MlirTypeID mlirDenseIntOrFPElementsAttrGetTypeID(void) {
  return wrap(DenseIntOrFPElementsAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

MlirAttribute mlirDenseElementsAttrGet(MlirType shapedType,
                                       intptr_t numElements,
                                       MlirAttribute const *elements) {
  SmallVector<Attribute, 8> attributes;
  return wrap(
      DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                             unwrapList(numElements, elements, attributes)));
}

MlirAttribute mlirDenseElementsAttrRawBufferGet(MlirType shapedType,
                                                size_t rawBufferSize,
                                                const void *rawBuffer) {
  auto shapedTypeCpp = llvm::cast<ShapedType>(unwrap(shapedType));
  ArrayRef<char> rawBufferCpp(static_cast<const char *>(rawBuffer),
                              rawBufferSize);
  bool isSplat = false;
  if (!DenseElementsAttr::isValidRawBuffer(shapedTypeCpp, rawBufferCpp,
                                           isSplat))
    return mlirAttributeGetNull();
  return wrap(DenseElementsAttr::getFromRawBuffer(shapedTypeCpp, rawBufferCpp));
}

MlirAttribute mlirDenseElementsAttrSplatGet(MlirType shapedType,
                                            MlirAttribute element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     unwrap(element)));
}
MlirAttribute mlirDenseElementsAttrBoolSplatGet(MlirType shapedType,
                                                bool element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrUInt8SplatGet(MlirType shapedType,
                                                 uint8_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrInt8SplatGet(MlirType shapedType,
                                                int8_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrUInt32SplatGet(MlirType shapedType,
                                                  uint32_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrInt32SplatGet(MlirType shapedType,
                                                 int32_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrUInt64SplatGet(MlirType shapedType,
                                                  uint64_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrInt64SplatGet(MlirType shapedType,
                                                 int64_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrFloatSplatGet(MlirType shapedType,
                                                 float element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
MlirAttribute mlirDenseElementsAttrDoubleSplatGet(MlirType shapedType,
                                                  double element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}

MlirAttribute mlirDenseElementsAttrBoolGet(MlirType shapedType,
                                           intptr_t numElements,
                                           const int *elements) {
  SmallVector<bool, 8> values(elements, elements + numElements);
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     values));
}

/// Creates a dense attribute with elements of the type deduced by templates.
template <typename T>
static MlirAttribute getDenseAttribute(MlirType shapedType,
                                       intptr_t numElements,
                                       const T *elements) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     llvm::ArrayRef(elements, numElements)));
}

MlirAttribute mlirDenseElementsAttrUInt8Get(MlirType shapedType,
                                            intptr_t numElements,
                                            const uint8_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrInt8Get(MlirType shapedType,
                                           intptr_t numElements,
                                           const int8_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrUInt16Get(MlirType shapedType,
                                             intptr_t numElements,
                                             const uint16_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrInt16Get(MlirType shapedType,
                                            intptr_t numElements,
                                            const int16_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrUInt32Get(MlirType shapedType,
                                             intptr_t numElements,
                                             const uint32_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrInt32Get(MlirType shapedType,
                                            intptr_t numElements,
                                            const int32_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrUInt64Get(MlirType shapedType,
                                             intptr_t numElements,
                                             const uint64_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrInt64Get(MlirType shapedType,
                                            intptr_t numElements,
                                            const int64_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrFloatGet(MlirType shapedType,
                                            intptr_t numElements,
                                            const float *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrDoubleGet(MlirType shapedType,
                                             intptr_t numElements,
                                             const double *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
MlirAttribute mlirDenseElementsAttrBFloat16Get(MlirType shapedType,
                                               intptr_t numElements,
                                               const uint16_t *elements) {
  size_t bufferSize = numElements * 2;
  const void *buffer = static_cast<const void *>(elements);
  return mlirDenseElementsAttrRawBufferGet(shapedType, bufferSize, buffer);
}
MlirAttribute mlirDenseElementsAttrFloat16Get(MlirType shapedType,
                                              intptr_t numElements,
                                              const uint16_t *elements) {
  size_t bufferSize = numElements * 2;
  const void *buffer = static_cast<const void *>(elements);
  return mlirDenseElementsAttrRawBufferGet(shapedType, bufferSize, buffer);
}

MlirAttribute mlirDenseElementsAttrStringGet(MlirType shapedType,
                                             intptr_t numElements,
                                             MlirStringRef *strs) {
  SmallVector<StringRef, 8> values;
  values.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    values.push_back(unwrap(strs[i]));

  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     values));
}

MlirAttribute mlirDenseElementsAttrReshapeGet(MlirAttribute attr,
                                              MlirType shapedType) {
  return wrap(llvm::cast<DenseElementsAttr>(unwrap(attr))
                  .reshape(llvm::cast<ShapedType>(unwrap(shapedType))));
}

//===----------------------------------------------------------------------===//
// Splat accessors.
//===----------------------------------------------------------------------===//

bool mlirDenseElementsAttrIsSplat(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).isSplat();
}

MlirAttribute mlirDenseElementsAttrGetSplatValue(MlirAttribute attr) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<Attribute>());
}
int mlirDenseElementsAttrGetBoolSplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<bool>();
}
int8_t mlirDenseElementsAttrGetInt8SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int8_t>();
}
uint8_t mlirDenseElementsAttrGetUInt8SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint8_t>();
}
int32_t mlirDenseElementsAttrGetInt32SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int32_t>();
}
uint32_t mlirDenseElementsAttrGetUInt32SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint32_t>();
}
int64_t mlirDenseElementsAttrGetInt64SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int64_t>();
}
uint64_t mlirDenseElementsAttrGetUInt64SplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint64_t>();
}
float mlirDenseElementsAttrGetFloatSplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<float>();
}
double mlirDenseElementsAttrGetDoubleSplatValue(MlirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<double>();
}
MlirStringRef mlirDenseElementsAttrGetStringSplatValue(MlirAttribute attr) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<StringRef>());
}

//===----------------------------------------------------------------------===//
// Indexed accessors.
//===----------------------------------------------------------------------===//

bool mlirDenseElementsAttrGetBoolValue(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<bool>()[pos];
}
int8_t mlirDenseElementsAttrGetInt8Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int8_t>()[pos];
}
uint8_t mlirDenseElementsAttrGetUInt8Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint8_t>()[pos];
}
int16_t mlirDenseElementsAttrGetInt16Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int16_t>()[pos];
}
uint16_t mlirDenseElementsAttrGetUInt16Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint16_t>()[pos];
}
int32_t mlirDenseElementsAttrGetInt32Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int32_t>()[pos];
}
uint32_t mlirDenseElementsAttrGetUInt32Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint32_t>()[pos];
}
int64_t mlirDenseElementsAttrGetInt64Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int64_t>()[pos];
}
uint64_t mlirDenseElementsAttrGetUInt64Value(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint64_t>()[pos];
}
float mlirDenseElementsAttrGetFloatValue(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<float>()[pos];
}
double mlirDenseElementsAttrGetDoubleValue(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<double>()[pos];
}
MlirStringRef mlirDenseElementsAttrGetStringValue(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<StringRef>()[pos]);
}

//===----------------------------------------------------------------------===//
// Raw data accessors.
//===----------------------------------------------------------------------===//

const void *mlirDenseElementsAttrGetRawData(MlirAttribute attr) {
  return static_cast<const void *>(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getRawData().data());
}

//===----------------------------------------------------------------------===//
// Resource blob attributes.
//===----------------------------------------------------------------------===//

template <typename U, typename T>
static MlirAttribute getDenseResource(MlirType shapedType, MlirStringRef name,
                                      intptr_t numElements, const T *elements) {
  return wrap(U::get(llvm::cast<ShapedType>(unwrap(shapedType)), unwrap(name),
                     UnmanagedAsmResourceBlob::allocateInferAlign(
                         llvm::ArrayRef(elements, numElements))));
}

MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseBoolResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const int *elements) {
  return getDenseResource<DenseBoolResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const uint8_t *elements) {
  return getDenseResource<DenseUI8ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute
mlirUnmanagedDenseUInt16ResourceElementsAttrGet(MlirType shapedType,
                                                MlirStringRef name,
                                                intptr_t numElements,
                                                const uint16_t *elements) {
  return getDenseResource<DenseUI16ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute
mlirUnmanagedDenseUInt32ResourceElementsAttrGet(MlirType shapedType,
                                                MlirStringRef name,
                                                intptr_t numElements,
                                                const uint32_t *elements) {
  return getDenseResource<DenseUI32ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute
mlirUnmanagedDenseUInt64ResourceElementsAttrGet(MlirType shapedType,
                                                MlirStringRef name,
                                                intptr_t numElements,
                                                const uint64_t *elements) {
  return getDenseResource<DenseUI64ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseInt8ResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const int8_t *elements) {
  return getDenseResource<DenseUI8ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseInt16ResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const int16_t *elements) {
  return getDenseResource<DenseUI16ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseInt32ResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const int32_t *elements) {
  return getDenseResource<DenseUI32ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseInt64ResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const int64_t *elements) {
  return getDenseResource<DenseUI64ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute mlirUnmanagedDenseFloatResourceElementsAttrGet(
    MlirType shapedType, MlirStringRef name, intptr_t numElements,
    const float *elements) {
  return getDenseResource<DenseF32ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
MLIR_CAPI_EXPORTED MlirAttribute
mlirUnmanagedDenseDoubleResourceElementsAttrGet(MlirType shapedType,
                                                MlirStringRef name,
                                                intptr_t numElements,
                                                const double *elements) {
  return getDenseResource<DenseF64ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}

template <typename U, typename T>
static T getDenseResourceVal(MlirAttribute attr, intptr_t pos) {
  return (*llvm::cast<U>(unwrap(attr)).tryGetAsArrayRef())[pos];
}

MLIR_CAPI_EXPORTED bool
mlirDenseBoolResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseBoolResourceElementsAttr, uint8_t>(attr, pos);
}
MLIR_CAPI_EXPORTED uint8_t
mlirDenseUInt8ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI8ResourceElementsAttr, uint8_t>(attr, pos);
}
MLIR_CAPI_EXPORTED uint16_t
mlirDenseUInt16ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI16ResourceElementsAttr, uint16_t>(attr,
                                                                      pos);
}
MLIR_CAPI_EXPORTED uint32_t
mlirDenseUInt32ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI32ResourceElementsAttr, uint32_t>(attr,
                                                                      pos);
}
MLIR_CAPI_EXPORTED uint64_t
mlirDenseUInt64ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI64ResourceElementsAttr, uint64_t>(attr,
                                                                      pos);
}
MLIR_CAPI_EXPORTED int8_t
mlirDenseInt8ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI8ResourceElementsAttr, int8_t>(attr, pos);
}
MLIR_CAPI_EXPORTED int16_t
mlirDenseInt16ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI16ResourceElementsAttr, int16_t>(attr, pos);
}
MLIR_CAPI_EXPORTED int32_t
mlirDenseInt32ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI32ResourceElementsAttr, int32_t>(attr, pos);
}
MLIR_CAPI_EXPORTED int64_t
mlirDenseInt64ResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseUI64ResourceElementsAttr, int64_t>(attr, pos);
}
MLIR_CAPI_EXPORTED float
mlirDenseFloatResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseF32ResourceElementsAttr, float>(attr, pos);
}
MLIR_CAPI_EXPORTED double
mlirDenseDoubleResourceElementsAttrGetValue(MlirAttribute attr, intptr_t pos) {
  return getDenseResourceVal<DenseF64ResourceElementsAttr, double>(attr, pos);
}

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsASparseElements(MlirAttribute attr) {
  return llvm::isa<SparseElementsAttr>(unwrap(attr));
}

MlirAttribute mlirSparseElementsAttribute(MlirType shapedType,
                                          MlirAttribute denseIndices,
                                          MlirAttribute denseValues) {
  return wrap(SparseElementsAttr::get(
      llvm::cast<ShapedType>(unwrap(shapedType)),
      llvm::cast<DenseElementsAttr>(unwrap(denseIndices)),
      llvm::cast<DenseElementsAttr>(unwrap(denseValues))));
}

MlirAttribute mlirSparseElementsAttrGetIndices(MlirAttribute attr) {
  return wrap(llvm::cast<SparseElementsAttr>(unwrap(attr)).getIndices());
}

MlirAttribute mlirSparseElementsAttrGetValues(MlirAttribute attr) {
  return wrap(llvm::cast<SparseElementsAttr>(unwrap(attr)).getValues());
}

MlirTypeID mlirSparseElementsAttrGetTypeID(void) {
  return wrap(SparseElementsAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAStridedLayout(MlirAttribute attr) {
  return llvm::isa<StridedLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirStridedLayoutAttrGet(MlirContext ctx, int64_t offset,
                                       intptr_t numStrides,
                                       const int64_t *strides) {
  return wrap(StridedLayoutAttr::get(unwrap(ctx), offset,
                                     ArrayRef<int64_t>(strides, numStrides)));
}

int64_t mlirStridedLayoutAttrGetOffset(MlirAttribute attr) {
  return llvm::cast<StridedLayoutAttr>(unwrap(attr)).getOffset();
}

intptr_t mlirStridedLayoutAttrGetNumStrides(MlirAttribute attr) {
  return static_cast<intptr_t>(
      llvm::cast<StridedLayoutAttr>(unwrap(attr)).getStrides().size());
}

int64_t mlirStridedLayoutAttrGetStride(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<StridedLayoutAttr>(unwrap(attr)).getStrides()[pos];
}

MlirTypeID mlirStridedLayoutAttrGetTypeID(void) {
  return wrap(StridedLayoutAttr::getTypeID());
}
