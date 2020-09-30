/*===-- mlir-c/StandardAttributes.h - C API for Std Attributes-----*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to MLIR Standard attributes.          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_STANDARDATTRIBUTES_H
#define MLIR_C_STANDARDATTRIBUTES_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/* Affine map attribute.                                                      */
/*============================================================================*/

/** Checks whether the given attribute is an affine map attribute. */
int mlirAttributeIsAAffineMap(MlirAttribute attr);

/** Creates an affine map attribute wrapping the given map. The attribute
 * belongs to the same context as the affine map. */
MlirAttribute mlirAffineMapAttrGet(MlirAffineMap map);

/** Returns the affine map wrapped in the given affine map attribute. */
MlirAffineMap mlirAffineMapAttrGetValue(MlirAttribute attr);

/*============================================================================*/
/* Array attribute.                                                           */
/*============================================================================*/

/** Checks whether the given attribute is an array attribute. */
int mlirAttributeIsAArray(MlirAttribute attr);

/** Creates an array element containing the given list of elements in the given
 * context. */
MlirAttribute mlirArrayAttrGet(MlirContext ctx, intptr_t numElements,
                               MlirAttribute *elements);

/** Returns the number of elements stored in the given array attribute. */
intptr_t mlirArrayAttrGetNumElements(MlirAttribute attr);

/** Returns pos-th element stored in the given array attribute. */
MlirAttribute mlirArrayAttrGetElement(MlirAttribute attr, intptr_t pos);

/*============================================================================*/
/* Dictionary attribute.                                                      */
/*============================================================================*/

/** Checks whether the given attribute is a dictionary attribute. */
int mlirAttributeIsADictionary(MlirAttribute attr);

/** Creates a dictionary attribute containing the given list of elements in the
 * provided context. */
MlirAttribute mlirDictionaryAttrGet(MlirContext ctx, intptr_t numElements,
                                    MlirNamedAttribute *elements);

/** Returns the number of attributes contained in a dictionary attribute. */
intptr_t mlirDictionaryAttrGetNumElements(MlirAttribute attr);

/** Returns pos-th element of the given dictionary attribute. */
MlirNamedAttribute mlirDictionaryAttrGetElement(MlirAttribute attr,
                                                intptr_t pos);

/** Returns the dictionary attribute element with the given name or NULL if the
 * given name does not exist in the dictionary. */
MlirAttribute mlirDictionaryAttrGetElementByName(MlirAttribute attr,
                                                 const char *name);

/*============================================================================*/
/* Floating point attribute.                                                  */
/*============================================================================*/

/* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
 * relevant functions here. */

/** Checks whether the given attribute is a floating point attribute. */
int mlirAttributeIsAFloat(MlirAttribute attr);

/** Creates a floating point attribute in the given context with the given
 * double value and double-precision FP semantics. */
MlirAttribute mlirFloatAttrDoubleGet(MlirContext ctx, MlirType type,
                                     double value);

/** Same as "mlirFloatAttrDoubleGet", but if the type is not valid for a
 * construction of a FloatAttr, returns a null MlirAttribute. */
MlirAttribute mlirFloatAttrDoubleGetChecked(MlirType type, double value,
                                            MlirLocation loc);

/** Returns the value stored in the given floating point attribute, interpreting
 * the value as double. */
double mlirFloatAttrGetValueDouble(MlirAttribute attr);

/*============================================================================*/
/* Integer attribute.                                                         */
/*============================================================================*/

/* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
 * relevant functions here. */

/** Checks whether the given attribute is an integer attribute. */
int mlirAttributeIsAInteger(MlirAttribute attr);

/** Creates an integer attribute of the given type with the given integer
 * value. */
MlirAttribute mlirIntegerAttrGet(MlirType type, int64_t value);

/** Returns the value stored in the given integer attribute, assuming the value
 * fits into a 64-bit integer. */
int64_t mlirIntegerAttrGetValueInt(MlirAttribute attr);

/*============================================================================*/
/* Bool attribute.                                                            */
/*============================================================================*/

/** Checks whether the given attribute is a bool attribute. */
int mlirAttributeIsABool(MlirAttribute attr);

/** Creates a bool attribute in the given context with the given value. */
MlirAttribute mlirBoolAttrGet(MlirContext ctx, int value);

/** Returns the value stored in the given bool attribute. */
int mlirBoolAttrGetValue(MlirAttribute attr);

/*============================================================================*/
/* Integer set attribute.                                                     */
/*============================================================================*/

/** Checks whether the given attribute is an integer set attribute. */
int mlirAttributeIsAIntegerSet(MlirAttribute attr);

/*============================================================================*/
/* Opaque attribute.                                                          */
/*============================================================================*/

/** Checks whether the given attribute is an opaque attribute. */
int mlirAttributeIsAOpaque(MlirAttribute attr);

/** Creates an opaque attribute in the given context associated with the dialect
 * identified by its namespace. The attribute contains opaque byte data of the
 * specified length (data need not be null-terminated). */
MlirAttribute mlirOpaqueAttrGet(MlirContext ctx, const char *dialectNamespace,
                                intptr_t dataLength, const char *data,
                                MlirType type);

/** Returns the namespace of the dialect with which the given opaque attribute
 * is associated. The namespace string is owned by the context. */
const char *mlirOpaqueAttrGetDialectNamespace(MlirAttribute attr);

/** Returns the raw data as a string reference. The data remains live as long as
 * the context in which the attribute lives. */
MlirStringRef mlirOpaqueAttrGetData(MlirAttribute attr);

/*============================================================================*/
/* String attribute.                                                          */
/*============================================================================*/

/** Checks whether the given attribute is a string attribute. */
int mlirAttributeIsAString(MlirAttribute attr);

/** Creates a string attribute in the given context containing the given string.
 * The string need not be null-terminated and its length must be specified. */
MlirAttribute mlirStringAttrGet(MlirContext ctx, intptr_t length,
                                const char *data);

/** Creates a string attribute in the given context containing the given string.
 * The string need not be null-terminated and its length must be specified.
 * Additionally, the attribute has the given type. */
MlirAttribute mlirStringAttrTypedGet(MlirType type, intptr_t length,
                                     const char *data);

/** Returns the attribute values as a string reference. The data remains live as
 * long as the context in which the attribute lives. */
MlirStringRef mlirStringAttrGetValue(MlirAttribute attr);

/*============================================================================*/
/* SymbolRef attribute.                                                       */
/*============================================================================*/

/** Checks whether the given attribute is a symbol reference attribute. */
int mlirAttributeIsASymbolRef(MlirAttribute attr);

/** Creates a symbol reference attribute in the given context referencing a
 * symbol identified by the given string inside a list of nested references.
 * Each of the references in the list must not be nested. The string need not be
 * null-terminated and its length must be specified. */
MlirAttribute mlirSymbolRefAttrGet(MlirContext ctx, intptr_t length,
                                   const char *symbol, intptr_t numReferences,
                                   MlirAttribute *references);

/** Returns the string reference to the root referenced symbol. The data remains
 * live as long as the context in which the attribute lives. */
MlirStringRef mlirSymbolRefAttrGetRootReference(MlirAttribute attr);

/** Returns the stirng reference to the leaf referenced symbol. The data remains
 * live as long as the context in which the attribute lives. */
MlirStringRef mlirSymbolRefAttrGetLeafReference(MlirAttribute attr);

/** Returns the number of references nested in the given symbol reference
 * attribute. */
intptr_t mlirSymbolRefAttrGetNumNestedReferences(MlirAttribute attr);

/** Returns pos-th reference nested in the given symbol reference attribute. */
MlirAttribute mlirSymbolRefAttrGetNestedReference(MlirAttribute attr,
                                                  intptr_t pos);

/*============================================================================*/
/* Flat SymbolRef attribute.                                                  */
/*============================================================================*/

/** Checks whether the given attribute is a flat symbol reference attribute. */
int mlirAttributeIsAFlatSymbolRef(MlirAttribute attr);

/** Creates a flat symbol reference attribute in the given context referencing a
 * symbol identified by the given string. The string need not be null-terminated
 * and its length must be specified. */
MlirAttribute mlirFlatSymbolRefAttrGet(MlirContext ctx, intptr_t length,
                                       const char *symbol);

/** Returns the referenced symbol as a string reference. The data remains live
 * as long as the context in which the attribute lives. */
MlirStringRef mlirFlatSymbolRefAttrGetValue(MlirAttribute attr);

/*============================================================================*/
/* Type attribute.                                                            */
/*============================================================================*/

/** Checks whether the given attribute is a type attribute. */
int mlirAttributeIsAType(MlirAttribute attr);

/** Creates a type attribute wrapping the given type in the same context as the
 * type. */
MlirAttribute mlirTypeAttrGet(MlirType type);

/** Returns the type stored in the given type attribute. */
MlirType mlirTypeAttrGetValue(MlirAttribute attr);

/*============================================================================*/
/* Unit attribute.                                                            */
/*============================================================================*/

/** Checks whether the given attribute is a unit attribute. */
int mlirAttributeIsAUnit(MlirAttribute attr);

/** Creates a unit attribute in the given context. */
MlirAttribute mlirUnitAttrGet(MlirContext ctx);

/*============================================================================*/
/* Elements attributes.                                                       */
/*============================================================================*/

/** Checks whether the given attribute is an elements attribute. */
int mlirAttributeIsAElements(MlirAttribute attr);

/** Returns the element at the given rank-dimensional index. */
MlirAttribute mlirElementsAttrGetValue(MlirAttribute attr, intptr_t rank,
                                       uint64_t *idxs);

/** Checks whether the given rank-dimensional index is valid in the given
 * elements attribute. */
int mlirElementsAttrIsValidIndex(MlirAttribute attr, intptr_t rank,
                                 uint64_t *idxs);

/** Gets the total number of elements in the given elements attribute. In order
 * to iterate over the attribute, obtain its type, which must be a statically
 * shaped type and use its sizes to build a multi-dimensional index. */
int64_t mlirElementsAttrGetNumElements(MlirAttribute attr);

/*============================================================================*/
/* Dense elements attribute.                                                  */
/*============================================================================*/

/* TODO: decide on the interface and add support for complex elements. */
/* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
 * relevant functions here. */

/** Checks whether the given attribute is a dense elements attribute. */
int mlirAttributeIsADenseElements(MlirAttribute attr);
int mlirAttributeIsADenseIntElements(MlirAttribute attr);
int mlirAttributeIsADenseFPElements(MlirAttribute attr);

/** Creates a dense elements attribute with the given Shaped type and elements
 * in the same context as the type. */
MlirAttribute mlirDenseElementsAttrGet(MlirType shapedType,
                                       intptr_t numElements,
                                       MlirAttribute *elements);

/** Creates a dense elements attribute with the given Shaped type containing a
 * single replicated element (splat). */
MlirAttribute mlirDenseElementsAttrSplatGet(MlirType shapedType,
                                            MlirAttribute element);
MlirAttribute mlirDenseElementsAttrBoolSplatGet(MlirType shapedType,
                                                int element);
MlirAttribute mlirDenseElementsAttrUInt32SplatGet(MlirType shapedType,
                                                  uint32_t element);
MlirAttribute mlirDenseElementsAttrInt32SplatGet(MlirType shapedType,
                                                 int32_t element);
MlirAttribute mlirDenseElementsAttrUInt64SplatGet(MlirType shapedType,
                                                  uint64_t element);
MlirAttribute mlirDenseElementsAttrInt64SplatGet(MlirType shapedType,
                                                 int64_t element);
MlirAttribute mlirDenseElementsAttrFloatSplatGet(MlirType shapedType,
                                                 float element);
MlirAttribute mlirDenseElementsAttrDoubleSplatGet(MlirType shapedType,
                                                  double element);

/** Creates a dense elements attribute with the given shaped type from elements
 * of a specific type. Expects the element type of the shaped type to match the
 * data element type. */
MlirAttribute mlirDenseElementsAttrBoolGet(MlirType shapedType,
                                           intptr_t numElements, int *elements);
MlirAttribute mlirDenseElementsAttrUInt32Get(MlirType shapedType,
                                             intptr_t numElements,
                                             uint32_t *elements);
MlirAttribute mlirDenseElementsAttrInt32Get(MlirType shapedType,
                                            intptr_t numElements,
                                            int32_t *elements);
MlirAttribute mlirDenseElementsAttrUInt64Get(MlirType shapedType,
                                             intptr_t numElements,
                                             uint64_t *elements);
MlirAttribute mlirDenseElementsAttrInt64Get(MlirType shapedType,
                                            intptr_t numElements,
                                            int64_t *elements);
MlirAttribute mlirDenseElementsAttrFloatGet(MlirType shapedType,
                                            intptr_t numElements,
                                            float *elements);
MlirAttribute mlirDenseElementsAttrDoubleGet(MlirType shapedType,
                                             intptr_t numElements,
                                             double *elements);

/** Creates a dense elements attribute with the given shaped type from string
 * elements. The strings need not be null-terminated and their lengths are
 * provided as a separate argument co-indexed with the strs argument. */
MlirAttribute mlirDenseElementsAttrStringGet(MlirType shapedType,
                                             intptr_t numElements,
                                             intptr_t *strLengths,
                                             const char **strs);
/** Creates a dense elements attribute that has the same data as the given dense
 * elements attribute and a different shaped type. The new type must have the
 * same total number of elements. */
MlirAttribute mlirDenseElementsAttrReshapeGet(MlirAttribute attr,
                                              MlirType shapedType);

/** Checks whether the given dense elements attribute contains a single
 * replicated value (splat). */
int mlirDenseElementsAttrIsSplat(MlirAttribute attr);

/** Returns the single replicated value (splat) of a specific type contained by
 * the given dense elements attribute. */
MlirAttribute mlirDenseElementsAttrGetSplatValue(MlirAttribute attr);
int mlirDenseElementsAttrGetBoolSplatValue(MlirAttribute attr);
int32_t mlirDenseElementsAttrGetInt32SplatValue(MlirAttribute attr);
uint32_t mlirDenseElementsAttrGetUInt32SplatValue(MlirAttribute attr);
int64_t mlirDenseElementsAttrGetInt64SplatValue(MlirAttribute attr);
uint64_t mlirDenseElementsAttrGetUInt64SplatValue(MlirAttribute attr);
float mlirDenseElementsAttrGetFloatSplatValue(MlirAttribute attr);
double mlirDenseElementsAttrGetDoubleSplatValue(MlirAttribute attr);
MlirStringRef mlirDenseElementsAttrGetStringSplatValue(MlirAttribute attr);

/** Returns the pos-th value (flat contiguous indexing) of a specific type
 * contained by the given dense elements attribute. */
int mlirDenseElementsAttrGetBoolValue(MlirAttribute attr, intptr_t pos);
int32_t mlirDenseElementsAttrGetInt32Value(MlirAttribute attr, intptr_t pos);
uint32_t mlirDenseElementsAttrGetUInt32Value(MlirAttribute attr, intptr_t pos);
int64_t mlirDenseElementsAttrGetInt64Value(MlirAttribute attr, intptr_t pos);
uint64_t mlirDenseElementsAttrGetUInt64Value(MlirAttribute attr, intptr_t pos);
float mlirDenseElementsAttrGetFloatValue(MlirAttribute attr, intptr_t pos);
double mlirDenseElementsAttrGetDoubleValue(MlirAttribute attr, intptr_t pos);
MlirStringRef mlirDenseElementsAttrGetStringValue(MlirAttribute attr,
                                                  intptr_t pos);

/*============================================================================*/
/* Opaque elements attribute.                                                 */
/*============================================================================*/

/* TODO: expose Dialect to the bindings and implement accessors here. */

/** Checks whether the given attribute is an opaque elements attribute. */
int mlirAttributeIsAOpaqueElements(MlirAttribute attr);

/*============================================================================*/
/* Sparse elements attribute.                                                 */
/*============================================================================*/

/** Checks whether the given attribute is a sparse elements attribute. */
int mlirAttributeIsASparseElements(MlirAttribute attr);

/** Creates a sparse elements attribute of the given shape from a list of
 * indices and a list of associated values. Both lists are expected to be dense
 * elements attributes with the same number of elements. The list of indices is
 * expected to contain 64-bit integers. The attribute is created in the same
 * context as the type. */
MlirAttribute mlirSparseElementsAttribute(MlirType shapedType,
                                          MlirAttribute denseIndices,
                                          MlirAttribute denseValues);

/** Returns the dense elements attribute containing 64-bit integer indices of
 * non-null elements in the given sparse elements attribute. */
MlirAttribute mlirSparseElementsAttrGetIndices(MlirAttribute attr);

/** Returns the dense elements attribute containing the non-null elements in the
 * given sparse elements attribute. */
MlirAttribute mlirSparseElementsAttrGetValues(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_STANDARDATTRIBUTES_H
