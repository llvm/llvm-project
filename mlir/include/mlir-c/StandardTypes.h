/*===-- mlir-c/StandardTypes.h - C API for MLIR Standard types ----*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_STANDARDTYPES_H
#define MLIR_C_STANDARDTYPES_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/* Integer types.                                                             */
/*============================================================================*/

/** Checks whether the given type is an integer type. */
int mlirTypeIsAInteger(MlirType type);

/** Creates a signless integer type of the given bitwidth in the context. The
 * type is owned by the context. */
MlirType mlirIntegerTypeGet(MlirContext ctx, unsigned bitwidth);

/** Creates a signed integer type of the given bitwidth in the context. The type
 * is owned by the context. */
MlirType mlirIntegerTypeSignedGet(MlirContext ctx, unsigned bitwidth);

/** Creates an unsigned integer type of the given bitwidth in the context. The
 * type is owned by the context. */
MlirType mlirIntegerTypeUnsignedGet(MlirContext ctx, unsigned bitwidth);

/** Returns the bitwidth of an integer type. */
unsigned mlirIntegerTypeGetWidth(MlirType type);

/** Checks whether the given integer type is signless. */
int mlirIntegerTypeIsSignless(MlirType type);

/** Checks whether the given integer type is signed. */
int mlirIntegerTypeIsSigned(MlirType type);

/** Checks whether the given integer type is unsigned. */
int mlirIntegerTypeIsUnsigned(MlirType type);

/*============================================================================*/
/* Index type.                                                                */
/*============================================================================*/

/** Checks whether the given type is an index type. */
int mlirTypeIsAIndex(MlirType type);

/** Creates an index type in the given context. The type is owned by the
 * context. */
MlirType mlirIndexTypeGet(MlirContext ctx);

/*============================================================================*/
/* Floating-point types.                                                      */
/*============================================================================*/

/** Checks whether the given type is a bf16 type. */
int mlirTypeIsABF16(MlirType type);

/** Creates a bf16 type in the given context. The type is owned by the
 * context. */
MlirType mlirBF16TypeGet(MlirContext ctx);

/** Checks whether the given type is an f16 type. */
int mlirTypeIsAF16(MlirType type);

/** Creates an f16 type in the given context. The type is owned by the
 * context. */
MlirType mlirF16TypeGet(MlirContext ctx);

/** Checks whether the given type is an f32 type. */
int mlirTypeIsAF32(MlirType type);

/** Creates an f32 type in the given context. The type is owned by the
 * context. */
MlirType mlirF32TypeGet(MlirContext ctx);

/** Checks whether the given type is an f64 type. */
int mlirTypeIsAF64(MlirType type);

/** Creates a f64 type in the given context. The type is owned by the
 * context. */
MlirType mlirF64TypeGet(MlirContext ctx);

/*============================================================================*/
/* None type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is a None type. */
int mlirTypeIsANone(MlirType type);

/** Creates a None type in the given context. The type is owned by the
 * context. */
MlirType mlirNoneTypeGet(MlirContext ctx);

/*============================================================================*/
/* Complex type.                                                              */
/*============================================================================*/

/** Checks whether the given type is a Complex type. */
int mlirTypeIsAComplex(MlirType type);

/** Creates a complex type with the given element type in the same context as
 * the element type. The type is owned by the context. */
MlirType mlirComplexTypeGet(MlirType elementType);

/** Returns the element type of the given complex type. */
MlirType mlirComplexTypeGetElementType(MlirType type);

/*============================================================================*/
/* Shaped type.                                                               */
/*============================================================================*/

/** Checks whether the given type is a Shaped type. */
int mlirTypeIsAShaped(MlirType type);

/** Returns the element type of the shaped type. */
MlirType mlirShapedTypeGetElementType(MlirType type);

/** Checks whether the given shaped type is ranked. */
int mlirShapedTypeHasRank(MlirType type);

/** Returns the rank of the given ranked shaped type. */
int64_t mlirShapedTypeGetRank(MlirType type);

/** Checks whether the given shaped type has a static shape. */
int mlirShapedTypeHasStaticShape(MlirType type);

/** Checks wither the dim-th dimension of the given shaped type is dynamic. */
int mlirShapedTypeIsDynamicDim(MlirType type, intptr_t dim);

/** Returns the dim-th dimension of the given ranked shaped type. */
int64_t mlirShapedTypeGetDimSize(MlirType type, intptr_t dim);

/** Checks whether the given value is used as a placeholder for dynamic sizes
 * in shaped types. */
int mlirShapedTypeIsDynamicSize(int64_t size);

/** Checks whether the given value is used as a placeholder for dynamic strides
 * and offsets in shaped types. */
int mlirShapedTypeIsDynamicStrideOrOffset(int64_t val);

/*============================================================================*/
/* Vector type.                                                               */
/*============================================================================*/

/** Checks whether the given type is a Vector type. */
int mlirTypeIsAVector(MlirType type);

/** Creates a vector type of the shape identified by its rank and dimensios,
 * with the given element type in the same context as the element type. The type
 * is owned by the context. */
MlirType mlirVectorTypeGet(intptr_t rank, int64_t *shape, MlirType elementType);

/** Same as "mlirVectorTypeGet" but returns a nullptr wrapping MlirType on
 * illegal arguments, emitting appropriate diagnostics. */
MlirType mlirVectorTypeGetChecked(intptr_t rank, int64_t *shape,
                                  MlirType elementType, MlirLocation loc);

/*============================================================================*/
/* Ranked / Unranked Tensor type.                                             */
/*============================================================================*/

/** Checks whether the given type is a Tensor type. */
int mlirTypeIsATensor(MlirType type);

/** Checks whether the given type is a ranked tensor type. */
int mlirTypeIsARankedTensor(MlirType type);

/** Checks whether the given type is an unranked tensor type. */
int mlirTypeIsAUnrankedTensor(MlirType type);

/** Creates a tensor type of a fixed rank with the given shape and element type
 * in the same context as the element type. The type is owned by the context. */
MlirType mlirRankedTensorTypeGet(intptr_t rank, int64_t *shape,
                                 MlirType elementType);

/** Same as "mlirRankedTensorTypeGet" but returns a nullptr wrapping MlirType on
 * illegal arguments, emitting appropriate diagnostics. */
MlirType mlirRankedTensorTypeGetChecked(intptr_t rank, int64_t *shape,
                                        MlirType elementType, MlirLocation loc);

/** Creates an unranked tensor type with the given element type in the same
 * context as the element type. The type is owned by the context. */
MlirType mlirUnrankedTensorTypeGet(MlirType elementType);

/** Same as "mlirUnrankedTensorTypeGet" but returns a nullptr wrapping MlirType
 * on illegal arguments, emitting appropriate diagnostics. */
MlirType mlirUnrankedTensorTypeGetChecked(MlirType elementType,
                                          MlirLocation loc);

/*============================================================================*/
/* Ranked / Unranked MemRef type.                                             */
/*============================================================================*/

/** Checks whether the given type is a MemRef type. */
int mlirTypeIsAMemRef(MlirType type);

/** Checks whether the given type is an UnrankedMemRef type. */
int mlirTypeIsAUnrankedMemRef(MlirType type);

/** Creates a MemRef type with the given rank and shape, a potentially empty
 * list of affine layout maps, the given memory space and element type, in the
 * same context as element type. The type is owned by the context. */
MlirType mlirMemRefTypeGet(MlirType elementType, intptr_t rank, int64_t *shape,
                           intptr_t numMaps, MlirAttribute *affineMaps,
                           unsigned memorySpace);

/** Creates a MemRef type with the given rank, shape, memory space and element
 * type in the same context as the element type. The type has no affine maps,
 * i.e. represents a default row-major contiguous memref. The type is owned by
 * the context. */
MlirType mlirMemRefTypeContiguousGet(MlirType elementType, intptr_t rank,
                                     int64_t *shape, unsigned memorySpace);

/** Same as "mlirMemRefTypeContiguousGet" but returns a nullptr wrapping
 * MlirType on illegal arguments, emitting appropriate diagnostics. */
MlirType mlirMemRefTypeContiguousGetChecked(MlirType elementType, intptr_t rank,
                                            int64_t *shape,
                                            unsigned memorySpace,
                                            MlirLocation loc);

/** Creates an Unranked MemRef type with the given element type and in the given
 * memory space. The type is owned by the context of element type. */
MlirType mlirUnrankedMemRefTypeGet(MlirType elementType, unsigned memorySpace);

/** Same as "mlirUnrankedMemRefTypeGet" but returns a nullptr wrapping
 * MlirType on illegal arguments, emitting appropriate diagnostics. */
MlirType mlirUnrankedMemRefTypeGetChecked(MlirType elementType,
                                          unsigned memorySpace,
                                          MlirLocation loc);

/** Returns the number of affine layout maps in the given MemRef type. */
intptr_t mlirMemRefTypeGetNumAffineMaps(MlirType type);

/** Returns the pos-th affine map of the given MemRef type. */
MlirAffineMap mlirMemRefTypeGetAffineMap(MlirType type, intptr_t pos);

/** Returns the memory space of the given MemRef type. */
unsigned mlirMemRefTypeGetMemorySpace(MlirType type);

/** Returns the memory spcae of the given Unranked MemRef type. */
unsigned mlirUnrankedMemrefGetMemorySpace(MlirType type);

/*============================================================================*/
/* Tuple type.                                                                */
/*============================================================================*/

/** Checks whether the given type is a tuple type. */
int mlirTypeIsATuple(MlirType type);

/** Creates a tuple type that consists of the given list of elemental types. The
 * type is owned by the context. */
MlirType mlirTupleTypeGet(MlirContext ctx, intptr_t numElements,
                          MlirType *elements);

/** Returns the number of types contained in a tuple. */
intptr_t mlirTupleTypeGetNumTypes(MlirType type);

/** Returns the pos-th type in the tuple type. */
MlirType mlirTupleTypeGetType(MlirType type, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_STANDARDTYPES_H
