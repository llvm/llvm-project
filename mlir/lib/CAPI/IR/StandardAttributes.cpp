//===- StandardAttributes.cpp - C Interface to MLIR Standard Attributes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/StandardAttributes.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAAffineMap(MlirAttribute attr) {
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

int mlirAttributeIsAArray(MlirAttribute attr) {
  return unwrap(attr).isa<ArrayAttr>();
}

MlirAttribute mlirArrayAttrGet(MlirContext ctx, intptr_t numElements,
                               MlirAttribute *elements) {
  SmallVector<Attribute, 8> attrs;
  return wrap(ArrayAttr::get(
      unwrapList(static_cast<size_t>(numElements), elements, attrs),
      unwrap(ctx)));
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

int mlirAttributeIsADictionary(MlirAttribute attr) {
  return unwrap(attr).isa<DictionaryAttr>();
}

MlirAttribute mlirDictionaryAttrGet(MlirContext ctx, intptr_t numElements,
                                    MlirNamedAttribute *elements) {
  SmallVector<NamedAttribute, 8> attributes;
  attributes.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    attributes.emplace_back(Identifier::get(elements[i].name, unwrap(ctx)),
                            unwrap(elements[i].attribute));
  return wrap(DictionaryAttr::get(attributes, unwrap(ctx)));
}

intptr_t mlirDictionaryAttrGetNumElements(MlirAttribute attr) {
  return static_cast<intptr_t>(unwrap(attr).cast<DictionaryAttr>().size());
}

MlirNamedAttribute mlirDictionaryAttrGetElement(MlirAttribute attr,
                                                intptr_t pos) {
  NamedAttribute attribute =
      unwrap(attr).cast<DictionaryAttr>().getValue()[pos];
  return {attribute.first.c_str(), wrap(attribute.second)};
}

MlirAttribute mlirDictionaryAttrGetElementByName(MlirAttribute attr,
                                                 const char *name) {
  return wrap(unwrap(attr).cast<DictionaryAttr>().get(name));
}

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAFloat(MlirAttribute attr) {
  return unwrap(attr).isa<FloatAttr>();
}

MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx, MlirType type,
                                     double value) {
  return wrap(FloatAttr::get(unwrap(type), value));
}

MlirAttribute mlirFloatAttrDoubleGetChecked(MlirType type, double value,
                                            MlirLocation loc) {
  return wrap(FloatAttr::getChecked(unwrap(type), value, unwrap(loc)));
}

double mlirFloatAttrGetValueDouble(MlirAttribute attr) {
  return unwrap(attr).cast<FloatAttr>().getValueAsDouble();
}

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAInteger(MlirAttribute attr) {
  return unwrap(attr).isa<IntegerAttr>();
}

MlirAttribute mlirIntegerAttrGet(MlirType type, int64_t value) {
  return wrap(IntegerAttr::get(unwrap(type), value));
}

int64_t mlirIntegerAttrGetValueInt(MlirAttribute attr) {
  return unwrap(attr).cast<IntegerAttr>().getInt();
}

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsABool(MlirAttribute attr) {
  return unwrap(attr).isa<BoolAttr>();
}

MlirAttribute mlirBoolAttrGet(MlirContext ctx, int value) {
  return wrap(BoolAttr::get(value, unwrap(ctx)));
}

int mlirBoolAttrGetValue(MlirAttribute attr) {
  return unwrap(attr).cast<BoolAttr>().getValue();
}

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAIntegerSet(MlirAttribute attr) {
  return unwrap(attr).isa<IntegerSetAttr>();
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAOpaque(MlirAttribute attr) {
  return unwrap(attr).isa<OpaqueAttr>();
}

MlirAttribute mlirOpaqueAttrGet(MlirContext ctx, const char *dialectNamespace,
                                intptr_t dataLength, const char *data,
                                MlirType type) {
  return wrap(OpaqueAttr::get(Identifier::get(dialectNamespace, unwrap(ctx)),
                              StringRef(data, dataLength), unwrap(type),
                              unwrap(ctx)));
}

const char *mlirOpaqueAttrGetDialectNamespace(MlirAttribute attr) {
  return unwrap(attr).cast<OpaqueAttr>().getDialectNamespace().c_str();
}

MlirStringRef mlirOpaqueAttrGetData(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<OpaqueAttr>().getAttrData());
}

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAString(MlirAttribute attr) {
  return unwrap(attr).isa<StringAttr>();
}

MlirAttribute mlirStringAttrGet(MlirContext ctx, intptr_t length,
                                const char *data) {
  return wrap(StringAttr::get(StringRef(data, length), unwrap(ctx)));
}

MlirAttribute mlirStringAttrTypedGet(MlirType type, intptr_t length,
                                     const char *data) {
  return wrap(StringAttr::get(StringRef(data, length), unwrap(type)));
}

MlirStringRef mlirStringAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<StringAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsASymbolRef(MlirAttribute attr) {
  return unwrap(attr).isa<SymbolRefAttr>();
}

MlirAttribute mlirSymbolRefAttrGet(MlirContext ctx, intptr_t length,
                                   const char *symbol, intptr_t numReferences,
                                   MlirAttribute *references) {
  SmallVector<FlatSymbolRefAttr, 4> refs;
  refs.reserve(numReferences);
  for (intptr_t i = 0; i < numReferences; ++i)
    refs.push_back(unwrap(references[i]).cast<FlatSymbolRefAttr>());
  return wrap(SymbolRefAttr::get(StringRef(symbol, length), refs, unwrap(ctx)));
}

MlirStringRef mlirSymbolRefAttrGetRootReference(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SymbolRefAttr>().getRootReference());
}

MlirStringRef mlirSymbolRefAttrGetLeafReference(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SymbolRefAttr>().getLeafReference());
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

int mlirAttributeIsAFlatSymbolRef(MlirAttribute attr) {
  return unwrap(attr).isa<FlatSymbolRefAttr>();
}

MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, intptr_t length,
                                       const char *symbol) {
  return wrap(FlatSymbolRefAttr::get(StringRef(symbol, length), unwrap(ctx)));
}

MlirStringRef mlirFlatSymbolRefAttrGetValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<FlatSymbolRefAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAType(MlirAttribute attr) {
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

int mlirAttributeIsAUnit(MlirAttribute attr) {
  return unwrap(attr).isa<UnitAttr>();
}

MlirAttribute mlirUnitAttrGet(MlirContext ctx) {
  return wrap(UnitAttr::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAElements(MlirAttribute attr) {
  return unwrap(attr).isa<ElementsAttr>();
}

MlirAttribute mlirElementsAttrGetValue(MlirAttribute attr, intptr_t rank,
                                       uint64_t *idxs) {
  return wrap(unwrap(attr).cast<ElementsAttr>().getValue(
      llvm::makeArrayRef(idxs, rank)));
}

int mlirElementsAttrIsValidIndex(MlirAttribute attr, intptr_t rank,
                                 uint64_t *idxs) {
  return unwrap(attr).cast<ElementsAttr>().isValidIndex(
      llvm::makeArrayRef(idxs, rank));
}

int64_t mlirElementsAttrGetNumElements(MlirAttribute attr) {
  return unwrap(attr).cast<ElementsAttr>().getNumElements();
}

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IsA support.

int mlirAttributeIsADenseElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseElementsAttr>();
}
int mlirAttributeIsADenseIntElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseIntElementsAttr>();
}
int mlirAttributeIsADenseFPElements(MlirAttribute attr) {
  return unwrap(attr).isa<DenseFPElementsAttr>();
}

//===----------------------------------------------------------------------===//
// Constructors.

MlirAttribute mlirDenseElementsAttrGet(MlirType shapedType,
                                       intptr_t numElements,
                                       MlirAttribute *elements) {
  SmallVector<Attribute, 8> attributes;
  return wrap(
      DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                             unwrapList(numElements, elements, attributes)));
}

MlirAttribute mlirDenseElementsAttrSplatGet(MlirType shapedType,
                                            MlirAttribute element) {
  return wrap(DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                                     unwrap(element)));
}
MlirAttribute mlirDenseElementsAttrBoolSplatGet(MlirType shapedType,
                                                int element) {
  return wrap(DenseElementsAttr::get(unwrap(shapedType).cast<ShapedType>(),
                                     static_cast<bool>(element)));
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

MlirAttribute mlirDenseElementsAttrStringGet(MlirType shapedType,
                                             intptr_t numElements,
                                             intptr_t *strLengths,
                                             const char **strs) {
  SmallVector<StringRef, 8> values;
  values.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    values.push_back(StringRef(strs[i], strLengths[i]));

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

int mlirDenseElementsAttrIsSplat(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().isSplat();
}

MlirAttribute mlirDenseElementsAttrGetSplatValue(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<DenseElementsAttr>().getSplatValue());
}
int mlirDenseElementsAttrGetBoolSplatValue(MlirAttribute attr) {
  return unwrap(attr).cast<DenseElementsAttr>().getSplatValue<bool>();
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

int mlirDenseElementsAttrGetBoolValue(MlirAttribute attr, intptr_t pos) {
  return *(unwrap(attr).cast<DenseElementsAttr>().getValues<bool>().begin() +
           pos);
}
int32_t mlirDenseElementsAttrGetInt32Value(MlirAttribute attr, intptr_t pos) {
  return *(unwrap(attr).cast<DenseElementsAttr>().getValues<int32_t>().begin() +
           pos);
}
uint32_t mlirDenseElementsAttrGetUInt32Value(MlirAttribute attr, intptr_t pos) {
  return *(
      unwrap(attr).cast<DenseElementsAttr>().getValues<uint32_t>().begin() +
      pos);
}
int64_t mlirDenseElementsAttrGetInt64Value(MlirAttribute attr, intptr_t pos) {
  return *(unwrap(attr).cast<DenseElementsAttr>().getValues<int64_t>().begin() +
           pos);
}
uint64_t mlirDenseElementsAttrGetUInt64Value(MlirAttribute attr, intptr_t pos) {
  return *(
      unwrap(attr).cast<DenseElementsAttr>().getValues<uint64_t>().begin() +
      pos);
}
float mlirDenseElementsAttrGetFloatValue(MlirAttribute attr, intptr_t pos) {
  return *(unwrap(attr).cast<DenseElementsAttr>().getValues<float>().begin() +
           pos);
}
double mlirDenseElementsAttrGetDoubleValue(MlirAttribute attr, intptr_t pos) {
  return *(unwrap(attr).cast<DenseElementsAttr>().getValues<double>().begin() +
           pos);
}
MlirStringRef mlirDenseElementsAttrGetStringValue(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      *(unwrap(attr).cast<DenseElementsAttr>().getValues<StringRef>().begin() +
        pos));
}

//===----------------------------------------------------------------------===//
// Opaque elements attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsAOpaqueElements(MlirAttribute attr) {
  return unwrap(attr).isa<OpaqueElementsAttr>();
}

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

int mlirAttributeIsASparseElements(MlirAttribute attr) {
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
