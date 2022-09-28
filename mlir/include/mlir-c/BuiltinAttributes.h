//===-- mlir-c/BuiltinAttributes.h - C API for Builtin Attributes -*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR Builtin attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_BUILTINATTRIBUTES_H
#define MLIR_C_BUILTINATTRIBUTES_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns an empty attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirAttributeGetNull();

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an affine map attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAAffineMap(MlirAttribute attr);

/// Creates an affine map attribute wrapping the given map. The attribute
/// belongs to the same context as the affine map.
MLIR_CAPI_EXPORTED MlirAttribute mlirAffineMapAttrGet(MlirAffineMap map);

/// Returns the affine map wrapped in the given affine map attribute.
MLIR_CAPI_EXPORTED MlirAffineMap mlirAffineMapAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an array attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAArray(MlirAttribute attr);

/// Creates an array element containing the given list of elements in the given
/// context.
MLIR_CAPI_EXPORTED MlirAttribute mlirArrayAttrGet(
    MlirContext ctx, intptr_t numElements, MlirAttribute const *elements);

/// Returns the number of elements stored in the given array attribute.
MLIR_CAPI_EXPORTED intptr_t mlirArrayAttrGetNumElements(MlirAttribute attr);

/// Returns pos-th element stored in the given array attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirArrayAttrGetElement(MlirAttribute attr,
                                                         intptr_t pos);

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a dictionary attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsADictionary(MlirAttribute attr);

/// Creates a dictionary attribute containing the given list of elements in the
/// provided context.
MLIR_CAPI_EXPORTED MlirAttribute mlirDictionaryAttrGet(
    MlirContext ctx, intptr_t numElements, MlirNamedAttribute const *elements);

/// Returns the number of attributes contained in a dictionary attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirDictionaryAttrGetNumElements(MlirAttribute attr);

/// Returns pos-th element of the given dictionary attribute.
MLIR_CAPI_EXPORTED MlirNamedAttribute
mlirDictionaryAttrGetElement(MlirAttribute attr, intptr_t pos);

/// Returns the dictionary attribute element with the given name or NULL if the
/// given name does not exist in the dictionary.
MLIR_CAPI_EXPORTED MlirAttribute
mlirDictionaryAttrGetElementByName(MlirAttribute attr, MlirStringRef name);

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is a floating point attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAFloat(MlirAttribute attr);

/// Creates a floating point attribute in the given context with the given
/// double value and double-precision FP semantics.
MLIR_CAPI_EXPORTED MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx,
                                                        MlirType type,
                                                        double value);

/// Same as "mlirFloatAttrDoubleGet", but if the type is not valid for a
/// construction of a FloatAttr, returns a null MlirAttribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirFloatAttrDoubleGetChecked(MlirLocation loc,
                                                               MlirType type,
                                                               double value);

/// Returns the value stored in the given floating point attribute, interpreting
/// the value as double.
MLIR_CAPI_EXPORTED double mlirFloatAttrGetValueDouble(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is an integer attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAInteger(MlirAttribute attr);

/// Creates an integer attribute of the given type with the given integer
/// value.
MLIR_CAPI_EXPORTED MlirAttribute mlirIntegerAttrGet(MlirType type,
                                                    int64_t value);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of signless type and fits into a signed 64-bit integer.
MLIR_CAPI_EXPORTED int64_t mlirIntegerAttrGetValueInt(MlirAttribute attr);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of signed type and fits into a signed 64-bit integer.
MLIR_CAPI_EXPORTED int64_t mlirIntegerAttrGetValueSInt(MlirAttribute attr);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of unsigned type and fits into an unsigned 64-bit integer.
MLIR_CAPI_EXPORTED uint64_t mlirIntegerAttrGetValueUInt(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a bool attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsABool(MlirAttribute attr);

/// Creates a bool attribute in the given context with the given value.
MLIR_CAPI_EXPORTED MlirAttribute mlirBoolAttrGet(MlirContext ctx, int value);

/// Returns the value stored in the given bool attribute.
MLIR_CAPI_EXPORTED bool mlirBoolAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an integer set attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAIntegerSet(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an opaque attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAOpaque(MlirAttribute attr);

/// Creates an opaque attribute in the given context associated with the dialect
/// identified by its namespace. The attribute contains opaque byte data of the
/// specified length (data need not be null-terminated).
MLIR_CAPI_EXPORTED MlirAttribute
mlirOpaqueAttrGet(MlirContext ctx, MlirStringRef dialectNamespace,
                  intptr_t dataLength, const char *data, MlirType type);

/// Returns the namespace of the dialect with which the given opaque attribute
/// is associated. The namespace string is owned by the context.
MLIR_CAPI_EXPORTED MlirStringRef
mlirOpaqueAttrGetDialectNamespace(MlirAttribute attr);

/// Returns the raw data as a string reference. The data remains live as long as
/// the context in which the attribute lives.
MLIR_CAPI_EXPORTED MlirStringRef mlirOpaqueAttrGetData(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a string attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAString(MlirAttribute attr);

/// Creates a string attribute in the given context containing the given string.

MLIR_CAPI_EXPORTED MlirAttribute mlirStringAttrGet(MlirContext ctx,
                                                   MlirStringRef str);

/// Creates a string attribute in the given context containing the given string.
/// Additionally, the attribute has the given type.
MLIR_CAPI_EXPORTED MlirAttribute mlirStringAttrTypedGet(MlirType type,
                                                        MlirStringRef str);

/// Returns the attribute values as a string reference. The data remains live as
/// long as the context in which the attribute lives.
MLIR_CAPI_EXPORTED MlirStringRef mlirStringAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a symbol reference attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsASymbolRef(MlirAttribute attr);

/// Creates a symbol reference attribute in the given context referencing a
/// symbol identified by the given string inside a list of nested references.
/// Each of the references in the list must not be nested.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSymbolRefAttrGet(MlirContext ctx, MlirStringRef symbol,
                     intptr_t numReferences, MlirAttribute const *references);

/// Returns the string reference to the root referenced symbol. The data remains
/// live as long as the context in which the attribute lives.
MLIR_CAPI_EXPORTED MlirStringRef
mlirSymbolRefAttrGetRootReference(MlirAttribute attr);

/// Returns the string reference to the leaf referenced symbol. The data remains
/// live as long as the context in which the attribute lives.
MLIR_CAPI_EXPORTED MlirStringRef
mlirSymbolRefAttrGetLeafReference(MlirAttribute attr);

/// Returns the number of references nested in the given symbol reference
/// attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSymbolRefAttrGetNumNestedReferences(MlirAttribute attr);

/// Returns pos-th reference nested in the given symbol reference attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSymbolRefAttrGetNestedReference(MlirAttribute attr, intptr_t pos);

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a flat symbol reference attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlatSymbolRef(MlirAttribute attr);

/// Creates a flat symbol reference attribute in the given context referencing a
/// symbol identified by the given string.
MLIR_CAPI_EXPORTED MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx,
                                                          MlirStringRef symbol);

/// Returns the referenced symbol as a string reference. The data remains live
/// as long as the context in which the attribute lives.
MLIR_CAPI_EXPORTED MlirStringRef
mlirFlatSymbolRefAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a type attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAType(MlirAttribute attr);

/// Creates a type attribute wrapping the given type in the same context as the
/// type.
MLIR_CAPI_EXPORTED MlirAttribute mlirTypeAttrGet(MlirType type);

/// Returns the type stored in the given type attribute.
MLIR_CAPI_EXPORTED MlirType mlirTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a unit attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAUnit(MlirAttribute attr);

/// Creates a unit attribute in the given context.
MLIR_CAPI_EXPORTED MlirAttribute mlirUnitAttrGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an elements attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAElements(MlirAttribute attr);

/// Returns the element at the given rank-dimensional index.
MLIR_CAPI_EXPORTED MlirAttribute mlirElementsAttrGetValue(MlirAttribute attr,
                                                          intptr_t rank,
                                                          uint64_t *idxs);

/// Checks whether the given rank-dimensional index is valid in the given
/// elements attribute.
MLIR_CAPI_EXPORTED bool
mlirElementsAttrIsValidIndex(MlirAttribute attr, intptr_t rank, uint64_t *idxs);

/// Gets the total number of elements in the given elements attribute. In order
/// to iterate over the attribute, obtain its type, which must be a statically
/// shaped type and use its sizes to build a multi-dimensional index.
MLIR_CAPI_EXPORTED int64_t mlirElementsAttrGetNumElements(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a dense array attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseBoolArray(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseI8Array(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseI16Array(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseI32Array(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseI64Array(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseF32Array(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseF64Array(MlirAttribute attr);

/// Create a dense array attribute with the given elements.
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseBoolArrayGet(MlirContext ctx,
                                                       intptr_t size,
                                                       int const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseI8ArrayGet(MlirContext ctx,
                                                     intptr_t size,
                                                     int8_t const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseI16ArrayGet(MlirContext ctx,
                                                      intptr_t size,
                                                      int16_t const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseI32ArrayGet(MlirContext ctx,
                                                      intptr_t size,
                                                      int32_t const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseI64ArrayGet(MlirContext ctx,
                                                      intptr_t size,
                                                      int64_t const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseF32ArrayGet(MlirContext ctx,
                                                      intptr_t size,
                                                      float const *values);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseF64ArrayGet(MlirContext ctx,
                                                      intptr_t size,
                                                      double const *values);

/// Get the size of a dense array.
MLIR_CAPI_EXPORTED intptr_t mlirDenseArrayGetNumElements(MlirAttribute attr);

/// Get an element of a dense array.
MLIR_CAPI_EXPORTED bool mlirDenseBoolArrayGetElement(MlirAttribute attr,
                                                     intptr_t pos);
MLIR_CAPI_EXPORTED int8_t mlirDenseI8ArrayGetElement(MlirAttribute attr,
                                                     intptr_t pos);
MLIR_CAPI_EXPORTED int16_t mlirDenseI16ArrayGetElement(MlirAttribute attr,
                                                       intptr_t pos);
MLIR_CAPI_EXPORTED int32_t mlirDenseI32ArrayGetElement(MlirAttribute attr,
                                                       intptr_t pos);
MLIR_CAPI_EXPORTED int64_t mlirDenseI64ArrayGetElement(MlirAttribute attr,
                                                       intptr_t pos);
MLIR_CAPI_EXPORTED float mlirDenseF32ArrayGetElement(MlirAttribute attr,
                                                     intptr_t pos);
MLIR_CAPI_EXPORTED double mlirDenseF64ArrayGetElement(MlirAttribute attr,
                                                      intptr_t pos);

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

// TODO: decide on the interface and add support for complex elements.
// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is a dense elements attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseElements(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseIntElements(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirAttributeIsADenseFPElements(MlirAttribute attr);

/// Creates a dense elements attribute with the given Shaped type and elements
/// in the same context as the type.
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrGet(
    MlirType shapedType, intptr_t numElements, MlirAttribute const *elements);

/// Creates a dense elements attribute with the given Shaped type and elements
/// populated from a packed, row-major opaque buffer of contents.
///
/// The format of the raw buffer is a densely packed array of values that
/// can be bitcast to the storage format of the element type specified.
/// Types that are not byte aligned will be:
///   - For bitwidth > 1: Rounded up to the next byte.
///   - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to
///     the linear order of the shape type from MSB to LSB, padded to on the
///     right.
///
/// A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255)
/// will be interpreted as a splat. User code should be prepared for additional,
/// conformant patterns to be identified as splats in the future.
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrRawBufferGet(
    MlirType shapedType, size_t rawBufferSize, const void *rawBuffer);

/// Creates a dense elements attribute with the given Shaped type containing a
/// single replicated element (splat).
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrSplatGet(MlirType shapedType, MlirAttribute element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrBoolSplatGet(MlirType shapedType, bool element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrUInt8SplatGet(MlirType shapedType, uint8_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrInt8SplatGet(MlirType shapedType, int8_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrUInt32SplatGet(MlirType shapedType, uint32_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrInt32SplatGet(MlirType shapedType, int32_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrUInt64SplatGet(MlirType shapedType, uint64_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrInt64SplatGet(MlirType shapedType, int64_t element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrFloatSplatGet(MlirType shapedType, float element);
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrDoubleSplatGet(MlirType shapedType, double element);

/// Creates a dense elements attribute with the given shaped type from elements
/// of a specific type. Expects the element type of the shaped type to match the
/// data element type.
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrBoolGet(
    MlirType shapedType, intptr_t numElements, const int *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrUInt8Get(
    MlirType shapedType, intptr_t numElements, const uint8_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrInt8Get(
    MlirType shapedType, intptr_t numElements, const int8_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrUInt16Get(
    MlirType shapedType, intptr_t numElements, const uint16_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrInt16Get(
    MlirType shapedType, intptr_t numElements, const int16_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrUInt32Get(
    MlirType shapedType, intptr_t numElements, const uint32_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrInt32Get(
    MlirType shapedType, intptr_t numElements, const int32_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrUInt64Get(
    MlirType shapedType, intptr_t numElements, const uint64_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrInt64Get(
    MlirType shapedType, intptr_t numElements, const int64_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrFloatGet(
    MlirType shapedType, intptr_t numElements, const float *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrDoubleGet(
    MlirType shapedType, intptr_t numElements, const double *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrBFloat16Get(
    MlirType shapedType, intptr_t numElements, const uint16_t *elements);
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrFloat16Get(
    MlirType shapedType, intptr_t numElements, const uint16_t *elements);

/// Creates a dense elements attribute with the given shaped type from string
/// elements.
MLIR_CAPI_EXPORTED MlirAttribute mlirDenseElementsAttrStringGet(
    MlirType shapedType, intptr_t numElements, MlirStringRef *strs);

/// Creates a dense elements attribute that has the same data as the given dense
/// elements attribute and a different shaped type. The new type must have the
/// same total number of elements.
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrReshapeGet(MlirAttribute attr, MlirType shapedType);

/// Checks whether the given dense elements attribute contains a single
/// replicated value (splat).
MLIR_CAPI_EXPORTED bool mlirDenseElementsAttrIsSplat(MlirAttribute attr);

/// Returns the single replicated value (splat) of a specific type contained by
/// the given dense elements attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirDenseElementsAttrGetSplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED int
mlirDenseElementsAttrGetBoolSplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED int8_t
mlirDenseElementsAttrGetInt8SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint8_t
mlirDenseElementsAttrGetUInt8SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t
mlirDenseElementsAttrGetInt32SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t
mlirDenseElementsAttrGetUInt32SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
mlirDenseElementsAttrGetInt64SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirDenseElementsAttrGetUInt64SplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED float
mlirDenseElementsAttrGetFloatSplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED double
mlirDenseElementsAttrGetDoubleSplatValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirDenseElementsAttrGetStringSplatValue(MlirAttribute attr);

/// Returns the pos-th value (flat contiguous indexing) of a specific type
/// contained by the given dense elements attribute.
MLIR_CAPI_EXPORTED bool mlirDenseElementsAttrGetBoolValue(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED int8_t mlirDenseElementsAttrGetInt8Value(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED uint8_t
mlirDenseElementsAttrGetUInt8Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int16_t
mlirDenseElementsAttrGetInt16Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED uint16_t
mlirDenseElementsAttrGetUInt16Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int32_t
mlirDenseElementsAttrGetInt32Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED uint32_t
mlirDenseElementsAttrGetUInt32Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
mlirDenseElementsAttrGetInt64Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED uint64_t
mlirDenseElementsAttrGetUInt64Value(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED float mlirDenseElementsAttrGetFloatValue(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED double
mlirDenseElementsAttrGetDoubleValue(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED MlirStringRef
mlirDenseElementsAttrGetStringValue(MlirAttribute attr, intptr_t pos);

/// Returns the raw data of the given dense elements attribute.
MLIR_CAPI_EXPORTED const void *
mlirDenseElementsAttrGetRawData(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a sparse elements attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsASparseElements(MlirAttribute attr);

/// Creates a sparse elements attribute of the given shape from a list of
/// indices and a list of associated values. Both lists are expected to be dense
/// elements attributes with the same number of elements. The list of indices is
/// expected to contain 64-bit integers. The attribute is created in the same
/// context as the type.
MLIR_CAPI_EXPORTED MlirAttribute mlirSparseElementsAttribute(
    MlirType shapedType, MlirAttribute denseIndices, MlirAttribute denseValues);

/// Returns the dense elements attribute containing 64-bit integer indices of
/// non-null elements in the given sparse elements attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSparseElementsAttrGetIndices(MlirAttribute attr);

/// Returns the dense elements attribute containing the non-null elements in the
/// given sparse elements attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSparseElementsAttrGetValues(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//

// Checks wheather the given attribute is a strided layout attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAStridedLayout(MlirAttribute attr);

// Creates a strided layout attribute from given strides and offset.
MLIR_CAPI_EXPORTED MlirAttribute mlirStridedLayoutAttrGet(MlirContext ctx,
                                                          int64_t offset,
                                                          intptr_t numStrides,
                                                          int64_t *strides);

// Returns the offset in the given strided layout layout attribute.
MLIR_CAPI_EXPORTED int64_t mlirStridedLayoutAttrGetOffset(MlirAttribute attr);

// Returns the number of strides in the given strided layout attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirStridedLayoutAttrGetNumStrides(MlirAttribute attr);

// Returns the pos-th stride stored in the given strided layout attribute.
MLIR_CAPI_EXPORTED int64_t mlirStridedLayoutAttrGetStride(MlirAttribute attr,
                                                          intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BUILTINATTRIBUTES_H
