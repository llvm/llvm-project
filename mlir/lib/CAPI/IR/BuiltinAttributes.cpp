//===- BuiltinAttributes.cpp - C Interface to MLIR Builtin Attributes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

MlirAttribute mlirAttributeGetNull() { return {nullptr}; }

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAAffineMap(MlirAttribute attr) {
  return unwrap(attr).isa<AffineMapAttr>();
}

MlirAttribute mlirAffineMapAttrGet(MlirAffineMap map) {
  return wrap(AffineMapAttr::get(unwrap(map)));
}

MlirAffineMap mlirAffineMapAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<AffineMapAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAArray(MlirAttribute attr) {
  return unwrap(attr).isa<ArrayAttr>();
}

MlirAttribute mlirArrayAttrGet(MlirContext ctx, intptr_t numElements,
                               MlirAttribute const *elements) {
  SmallVector<Attribute, 8> attrs;
  return wrap(
      ArrayAttr::get(unwrap(ctx), unwrapList(static_cast<size_t>(numElements),
                                             elements, attrs)));
}

intptr_t mlirArrayAttrGetNumElements(MlirAttribute attr) {
  return static_cast<intptr_t>(unwrap(attr).cast<ArrayAttr>().size());
}

MlirAttribute mlirArrayAttrGetElement(MlirAttribute attr, intptr_t pos) {
  return wrap(unwrap(attr).cast<ArrayAttr>().getValue()[pos]);
}

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsADictionary(MlirAttribute attr) {
  return unwrap(attr).isa<DictionaryAttr>();
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
  return static_cast<intptr_t>(unwrap(attr).cast<DictionaryAttr>().size());
}

MlirNamedAttribute mlirDictionaryAttrGetElement(MlirAttribute attr,
                                                intptr_t pos) {
  NamedAttribute attribute =
      unwrap(attr).cast<DictionaryAttr>().getValue()[pos];
  return {wrap(attribute.getName()), wrap(attribute.getValue())};
}

MlirAttribute mlirDictionaryAttrGetElementByName(MlirAttribute attr,
                                                 MlirStringRef name) {
  return wrap(unwrap(attr).cast<DictionaryAttr>().get(unwrap(name)));
}

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAFloat(MlirAttribute attr) {
  return unwrap(attr).isa<FloatAttr>();
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
  return unwrap(attr).cast<FloatAttr>().getValueAsDouble();
}

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAInteger(MlirAttribute attr) {
  return unwrap(attr).isa<IntegerAttr>();
}

MlirAttribute mlirIntegerAttrGet(MlirType type, int64_t value) {
  return wrap(IntegerAttr::get(unwrap(type), value));
}

int64_t mlirIntegerAttrGetValueInt(MlirAttribute attr) {
  return unwrap(attr).cast<IntegerAttr>().getInt();
}

int64_t mlirIntegerAttrGetValueSInt(MlirAttribute attr) {
  return unwrap(attr).cast<IntegerAttr>().getSInt();
}

uint64_t mlirIntegerAttrGetValueUInt(MlirAttribute attr) {
  return unwrap(attr).cast<IntegerAttr>().getUInt();
}

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsABool(MlirAttribute attr) {
  return unwrap(attr).isa<BoolAttr>();
}

MlirAttribute mlirBoolAttrGet(MlirContext ctx, int value) {
  return wrap(BoolAttr::get(unwrap(ctx), value));
}

bool mlirBoolAttrGetValue(MlirAttribute attr) {
  return unwrap(attr).cast<BoolAttr>().getValue();
}

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAIntegerSet(MlirAttribute attr) {
  return unwrap(attr).isa<IntegerSetAttr>();
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAOpaque(MlirAttribute attr) {
  return unwrap(attr).isa<OpaqueAttr>();
}

MlirAttribute mlirOpaqueAttrGet(MlirContext ctx, MlirStringRef dialectNamespace,
                                intptr_t dataLength, const char *data,
                                MlirType type) {
  return wrap(
      OpaqueAttr::get(StringAttr::get(unwrap(ctx), unwrap(dialectNamespace)),
                      StringRef(data, dataLength), unwrap(type)));
}

MlirStringRef mlirOpaqueAttrGetDialectNamespace(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<OpaqueAttr>().getDialectNamespace().strref());
}

MlirStringRef mlirOpaqueAttrGetData(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<OpaqueAttr>().getAttrData());
}

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAString(MlirAttribute attr) {
  return unwrap(attr).isa<StringAttr>();
}

MlirAttribute mlirStringAttrGet(MlirContext ctx, MlirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(ctx), unwrap(str)));
}

MlirAttribute mlirStringAttrTypedGet(MlirType type, MlirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(str), unwrap(type)));
}

MlirStringRef mlirStringAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<StringAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsASymbolRef(MlirAttribute attr) {
  return unwrap(attr).isa<SymbolRefAttr>();
}

MlirAttribute mlirSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol,
                                   intptr_t numReferences,
                                   MlirAttribute const *references) {
  SmallVector<FlatSymbolRefAttr, 4> refs;
  refs.reserve(numReferences);
  for (intptr_t i = 0; i < numReferences; ++i)
    refs.push_back(unwrap(references[i]).cast<FlatSymbolRefAttr>());
  auto symbolAttr = StringAttr::get(unwrap(ctx), unwrap(symbol));
  return wrap(SymbolRefAttr::get(symbolAttr, refs));
}

MlirStringRef mlirSymbolRefAttrGetRootReference(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SymbolRefAttr>().getRootReference().getValue());
}

MlirStringRef mlirSymbolRefAttrGetLeafReference(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SymbolRefAttr>().getLeafReference().getValue());
}

intptr_t mlirSymbolRefAttrGetNumNestedReferences(MlirAttribute attr) {
  return static_cast<intptr_t>(
      unwrap(attr).cast<SymbolRefAttr>().getNestedReferences().size());
}

MlirAttribute mlirSymbolRefAttrGetNestedReference(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(unwrap(attr).cast<SymbolRefAttr>().getNestedReferences()[pos]);
}

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAFlatSymbolRef(MlirAttribute attr) {
  return unwrap(attr).isa<FlatSymbolRefAttr>();
}

MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol) {
  return wrap(FlatSymbolRefAttr::get(unwrap(ctx), unwrap(symbol)));
}

MlirStringRef mlirFlatSymbolRefAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<FlatSymbolRefAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAType(MlirAttribute attr) {
  return unwrap(attr).isa<TypeAttr>();
}

MlirAttribute mlirTypeAttrGet(MlirType type) {
  return wrap(TypeAttr::get(unwrap(type)));
}

MlirType mlirTypeAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<TypeAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAUnit(MlirAttribute attr) {
  return unwrap(attr).isa<UnitAttr>();
}

MlirAttribute mlirUnitAttrGet(MlirContext ctx) {
  return wrap(UnitAttr::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAElements(MlirAttribute attr) {
  return unwrap(attr).isa<ElementsAttr>();
}

MlirAttribute mlirElementsAttrGetValue(MlirAttribute attr, intptr_t rank,
                                       uint64_t *idxs) {
  return wrap(unwrap(attr)
                  .cast<ElementsAttr>()
                  .getValues<Attribute>()[llvm::makeArrayRef(idxs, rank)]);
}

bool mlirElementsAttrIsValidIndex(MlirAttribute attr, intptr_t rank,
                                  uint64_t *idxs) {
  return unwrap(attr).cast<ElementsAttr>().isValidIndex(
      llvm::makeArrayRef(idxs, rank));
}

int64_t mlirElementsAttrGetNumElements(MlirAttribute attr) {
  return unwrap(attr).cast<ElementsAttr>().getNumElements();
}

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IsA support.

bool mlirAttributeIsADenseBoolArray(MlirAttribute attr) {
  return unwrap(attr).isa<DenseBoolArrayAttr>();
}
bool mlirAttributeIsADenseI8Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseI8ArrayAttr>();
}
bool mlirAttributeIsADenseI16Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseI16ArrayAttr>();
}
bool mlirAttributeIsADenseI32Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseI32ArrayAttr>();
}
bool mlirAttributeIsADenseI64Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseI64ArrayAttr>();
}
bool mlirAttributeIsADenseF32Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseF32ArrayAttr>();
}
bool mlirAttributeIsADenseF64Array(MlirAttribute attr) {
  return unwrap(attr).isa<DenseF64ArrayAttr>();
}

//===----------------------------------------------------------------------===//
// Constructors.

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

intptr_t mlirDenseArrayGetNumElements(MlirAttribute attr) {
  return unwrap(attr).cast<DenseArrayAttr>().size();
}

//===----------------------------------------------------------------------===//
// Indexed accessors.

bool mlirDenseBoolArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseBoolArrayAttr>()[pos];
}
int8_t mlirDenseI8ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseI8ArrayAttr>()[pos];
}
int16_t mlirDenseI16ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseI16ArrayAttr>()[pos];
}
int32_t mlirDenseI32ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseI32ArrayAttr>()[pos];
}
int64_t mlirDenseI64ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseI64ArrayAttr>()[pos];
}
float mlirDenseF32ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseF32ArrayAttr>()[pos];
}
double mlirDenseF64ArrayGetElement(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseF64ArrayAttr>()[pos];
}

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IsA support.

bool mlirAttributeIsADenseElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseElementsAttr>();
}
bool mlirAttributeIsADenseIntElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseIntElementsAttr>();
}
bool mlirAttributeIsADenseFPElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseFPElementsAttr>();
}

//===----------------------------------------------------------------------===//
// Constructors.

MlirAttribute mlirDenseElementsAttrGet(MlirType shapedType,
                                       intptr_t numElements,
                                       MlirAttribute const *elements) {
  SmallVector<Attribute, 8> attributes;
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                             unwrapList(numElements, elements, attributes)));
}

MlirAttribute mlirDenseElementsAttrRawBufferGet(MlirType shapedType,
                                                size_t rawBufferSize,
                                                const void *rawBuffer) {
  auto shapedTypeCpp = unwrap(shapedType).cast<ShapedType>();
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
  return wrap(DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                                     unwrap(element)));
}
MlirAttribute mlirDenseElementsAttrBoolSplatGet(MlirType shapedType,
                                                bool element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrUInt8SplatGet(MlirType shapedType,
                                                 uint8_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrInt8SplatGet(MlirType shapedType,
                                                int8_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrUInt32SplatGet(MlirType shapedType,
                                                  uint32_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrInt32SplatGet(MlirType shapedType,
                                                 int32_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrUInt64SplatGet(MlirType shapedType,
                                                  uint64_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrInt64SplatGet(MlirType shapedType,
                                                 int64_t element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrFloatSplatGet(MlirType shapedType,
                                                 float element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}
MlirAttribute mlirDenseElementsAttrDoubleSplatGet(MlirType shapedType,
                                                  double element) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), element));
}

MlirAttribute mlirDenseElementsAttrBoolGet(MlirType shapedType,
                                           intptr_t numElements,
                                           const int *elements) {
  SmallVector<bool, 8> values(elements, elements + numElements);
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), values));
}

/// Creates a dense attribute with elements of the type deduced by templates.
template <typename T>
static MlirAttribute getDenseAttribute(MlirType shapedType,
                                       intptr_t numElements,
                                       const T *elements) {
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                             llvm::makeArrayRef(elements, numElements)));
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

  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(), values));
}

MlirAttribute mlirDenseElementsAttrReshapeGet(MlirAttribute attr,
                                              MlirType shapedType) {
  return wrap(unwrap(attr).cast<DenseElementsAttr>().reshape(
      unwrap(shapedType).cast<ShapedType>()));
}

//===----------------------------------------------------------------------===//
// Splat accessors.

bool mlirDenseElementsAttrIsSplat(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().isSplat();
}

MlirAttribute mlirDenseElementsAttrGetSplatValue(MlirAttribute attr) {
  return wrap(
      unwrap(attr).cast<DenseElementsAttr>().getSplatValue<Attribute>());
}
int mlirDenseElementsAttrGetBoolSplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<bool>();
}
int8_t mlirDenseElementsAttrGetInt8SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<int8_t>();
}
uint8_t mlirDenseElementsAttrGetUInt8SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<uint8_t>();
}
int32_t mlirDenseElementsAttrGetInt32SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<int32_t>();
}
uint32_t mlirDenseElementsAttrGetUInt32SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<uint32_t>();
}
int64_t mlirDenseElementsAttrGetInt64SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<int64_t>();
}
uint64_t mlirDenseElementsAttrGetUInt64SplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<uint64_t>();
}
float mlirDenseElementsAttrGetFloatSplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<float>();
}
double mlirDenseElementsAttrGetDoubleSplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<double>();
}
MlirStringRef mlirDenseElementsAttrGetStringSplatValue(MlirAttribute attr) {
  return wrap(
      unwrap(attr).cast<DenseElementsAttr>().getSplatValue<StringRef>());
}

//===----------------------------------------------------------------------===//
// Indexed accessors.

bool mlirDenseElementsAttrGetBoolValue(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<bool>()[pos];
}
int8_t mlirDenseElementsAttrGetInt8Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<int8_t>()[pos];
}
uint8_t mlirDenseElementsAttrGetUInt8Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<uint8_t>()[pos];
}
int16_t mlirDenseElementsAttrGetInt16Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<int16_t>()[pos];
}
uint16_t mlirDenseElementsAttrGetUInt16Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<uint16_t>()[pos];
}
int32_t mlirDenseElementsAttrGetInt32Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<int32_t>()[pos];
}
uint32_t mlirDenseElementsAttrGetUInt32Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<uint32_t>()[pos];
}
int64_t mlirDenseElementsAttrGetInt64Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<int64_t>()[pos];
}
uint64_t mlirDenseElementsAttrGetUInt64Value(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<uint64_t>()[pos];
}
float mlirDenseElementsAttrGetFloatValue(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<float>()[pos];
}
double mlirDenseElementsAttrGetDoubleValue(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<DenseElementsAttr>().getValues<double>()[pos];
}
MlirStringRef mlirDenseElementsAttrGetStringValue(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      unwrap(attr).cast<DenseElementsAttr>().getValues<StringRef>()[pos]);
}

//===----------------------------------------------------------------------===//
// Raw data accessors.

const void *mlirDenseElementsAttrGetRawData(MlirAttribute attr) {
  return static_cast<const void *>(
      unwrap(attr).cast<DenseElementsAttr>().getRawData().data());
}

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsASparseElements(MlirAttribute attr) {
  return unwrap(attr).isa<SparseElementsAttr>();
}

MlirAttribute mlirSparseElementsAttribute(MlirType shapedType,
                                          MlirAttribute denseIndices,
                                          MlirAttribute denseValues) {
  return wrap(
      SparseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                              unwrap(denseIndices).cast<DenseElementsAttr>(),
                              unwrap(denseValues).cast<DenseElementsAttr>()));
}

MlirAttribute mlirSparseElementsAttrGetIndices(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SparseElementsAttr>().getIndices());
}

MlirAttribute mlirSparseElementsAttrGetValues(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SparseElementsAttr>().getValues());
}

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAStridedLayout(MlirAttribute attr) {
  return unwrap(attr).isa<StridedLayoutAttr>();
}

MlirAttribute mlirStridedLayoutAttrGet(MlirContext ctx, int64_t offset,
                                       intptr_t numStrides,
                                       const int64_t *strides) {
  return wrap(StridedLayoutAttr::get(unwrap(ctx), offset,
                                     ArrayRef<int64_t>(strides, numStrides)));
}

int64_t mlirStridedLayoutAttrGetOffset(MlirAttribute attr) {
  return unwrap(attr).cast<StridedLayoutAttr>().getOffset();
}

intptr_t mlirStridedLayoutAttrGetNumStrides(MlirAttribute attr) {
  return static_cast<intptr_t>(
      unwrap(attr).cast<StridedLayoutAttr>().getStrides().size());
}

int64_t mlirStridedLayoutAttrGetStride(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr).cast<StridedLayoutAttr>().getStrides()[pos];
}
