//===-- mlir-c/BuiltinTypes.h - C API for MLIR Builtin types ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_BUILTINTYPES_H
#define MLIR_C_BUILTINTYPES_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is an integer type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAInteger(MlirType type);

/// Creates a signless integer type of the given bitwidth in the context. The
/// type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirIntegerTypeGet(MlirContext ctx,
                                               unsigned bitwidth);

/// Creates a signed integer type of the given bitwidth in the context. The type
/// is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirIntegerTypeSignedGet(MlirContext ctx,
                                                     unsigned bitwidth);

/// Creates an unsigned integer type of the given bitwidth in the context. The
/// type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirIntegerTypeUnsignedGet(MlirContext ctx,
                                                       unsigned bitwidth);

/// Returns the bitwidth of an integer type.
MLIR_CAPI_EXPORTED unsigned mlirIntegerTypeGetWidth(MlirType type);

/// Checks whether the given integer type is signless.
MLIR_CAPI_EXPORTED bool mlirIntegerTypeIsSignless(MlirType type);

/// Checks whether the given integer type is signed.
MLIR_CAPI_EXPORTED bool mlirIntegerTypeIsSigned(MlirType type);

/// Checks whether the given integer type is unsigned.
MLIR_CAPI_EXPORTED bool mlirIntegerTypeIsUnsigned(MlirType type);

//===----------------------------------------------------------------------===//
// Index type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is an index type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAIndex(MlirType type);

/// Creates an index type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirIndexTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a bf16 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsABF16(MlirType type);

/// Creates a bf16 type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirBF16TypeGet(MlirContext ctx);

/// Checks whether the given type is an f16 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAF16(MlirType type);

/// Creates an f16 type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirF16TypeGet(MlirContext ctx);

/// Checks whether the given type is an f32 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAF32(MlirType type);

/// Creates an f32 type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirF32TypeGet(MlirContext ctx);

/// Checks whether the given type is an f64 type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAF64(MlirType type);

/// Creates a f64 type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirF64TypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// None type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a None type.
MLIR_CAPI_EXPORTED bool mlirTypeIsANone(MlirType type);

/// Creates a None type in the given context. The type is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirNoneTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Complex type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Complex type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAComplex(MlirType type);

/// Creates a complex type with the given element type in the same context as
/// the element type. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirComplexTypeGet(MlirType elementType);

/// Returns the element type of the given complex type.
MLIR_CAPI_EXPORTED MlirType mlirComplexTypeGetElementType(MlirType type);

//===----------------------------------------------------------------------===//
// Shaped type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Shaped type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAShaped(MlirType type);

/// Returns the element type of the shaped type.
MLIR_CAPI_EXPORTED MlirType mlirShapedTypeGetElementType(MlirType type);

/// Checks whether the given shaped type is ranked.
MLIR_CAPI_EXPORTED bool mlirShapedTypeHasRank(MlirType type);

/// Returns the rank of the given ranked shaped type.
MLIR_CAPI_EXPORTED int64_t mlirShapedTypeGetRank(MlirType type);

/// Checks whether the given shaped type has a static shape.
MLIR_CAPI_EXPORTED bool mlirShapedTypeHasStaticShape(MlirType type);

/// Checks wither the dim-th dimension of the given shaped type is dynamic.
MLIR_CAPI_EXPORTED bool mlirShapedTypeIsDynamicDim(MlirType type, intptr_t dim);

/// Returns the dim-th dimension of the given ranked shaped type.
MLIR_CAPI_EXPORTED int64_t mlirShapedTypeGetDimSize(MlirType type,
                                                    intptr_t dim);

/// Checks whether the given value is used as a placeholder for dynamic sizes
/// in shaped types.
MLIR_CAPI_EXPORTED bool mlirShapedTypeIsDynamicSize(int64_t size);

/// Checks whether the given value is used as a placeholder for dynamic strides
/// and offsets in shaped types.
MLIR_CAPI_EXPORTED bool mlirShapedTypeIsDynamicStrideOrOffset(int64_t val);

//===----------------------------------------------------------------------===//
// Vector type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Vector type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAVector(MlirType type);

/// Creates a vector type of the shape identified by its rank and dimensions,
/// with the given element type in the same context as the element type. The
/// type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirVectorTypeGet(intptr_t rank,
                                              const int64_t *shape,
                                              MlirType elementType);

/// Same as "mlirVectorTypeGet" but returns a nullptr wrapping MlirType on
/// illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType mlirVectorTypeGetChecked(MlirLocation loc,
                                                     intptr_t rank,
                                                     const int64_t *shape,
                                                     MlirType elementType);

//===----------------------------------------------------------------------===//
// Ranked / Unranked Tensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Tensor type.
MLIR_CAPI_EXPORTED bool mlirTypeIsATensor(MlirType type);

/// Checks whether the given type is a ranked tensor type.
MLIR_CAPI_EXPORTED bool mlirTypeIsARankedTensor(MlirType type);

/// Checks whether the given type is an unranked tensor type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAUnrankedTensor(MlirType type);

/// Creates a tensor type of a fixed rank with the given shape and element type
/// in the same context as the element type. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirRankedTensorTypeGet(intptr_t rank,
                                                    const int64_t *shape,
                                                    MlirType elementType);

/// Same as "mlirRankedTensorTypeGet" but returns a nullptr wrapping MlirType on
/// illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType
mlirRankedTensorTypeGetChecked(MlirLocation loc, intptr_t rank,
                               const int64_t *shape, MlirType elementType);

/// Creates an unranked tensor type with the given element type in the same
/// context as the element type. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirUnrankedTensorTypeGet(MlirType elementType);

/// Same as "mlirUnrankedTensorTypeGet" but returns a nullptr wrapping MlirType
/// on illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType
mlirUnrankedTensorTypeGetChecked(MlirLocation loc, MlirType elementType);

//===----------------------------------------------------------------------===//
// Ranked / Unranked MemRef type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a MemRef type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAMemRef(MlirType type);

/// Checks whether the given type is an UnrankedMemRef type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAUnrankedMemRef(MlirType type);

/// Creates a MemRef type with the given rank and shape, a potentially empty
/// list of affine layout maps, the given memory space and element type, in the
/// same context as element type. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirMemRefTypeGet(
    MlirType elementType, intptr_t rank, const int64_t *shape, intptr_t numMaps,
    MlirAffineMap const *affineMaps, MlirAttribute memorySpace);

/// Same as "mlirMemRefTypeGet" but returns a nullptr-wrapping MlirType o
/// illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType mlirMemRefTypeGetChecked(
    MlirLocation loc, MlirType elementType, intptr_t rank, const int64_t *shape,
    intptr_t numMaps, MlirAffineMap const *affineMaps,
    MlirAttribute memorySpace);

/// Creates a MemRef type with the given rank, shape, memory space and element
/// type in the same context as the element type. The type has no affine maps,
/// i.e. represents a default row-major contiguous memref. The type is owned by
/// the context.
MLIR_CAPI_EXPORTED MlirType
mlirMemRefTypeContiguousGet(MlirType elementType, intptr_t rank,
                            const int64_t *shape, MlirAttribute memorySpace);

/// Same as "mlirMemRefTypeContiguousGet" but returns a nullptr wrapping
/// MlirType on illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType mlirMemRefTypeContiguousGetChecked(
    MlirLocation loc, MlirType elementType, intptr_t rank, const int64_t *shape,
    MlirAttribute memorySpace);

/// Creates an Unranked MemRef type with the given element type and in the given
/// memory space. The type is owned by the context of element type.
MLIR_CAPI_EXPORTED MlirType
mlirUnrankedMemRefTypeGet(MlirType elementType, MlirAttribute memorySpace);

/// Same as "mlirUnrankedMemRefTypeGet" but returns a nullptr wrapping
/// MlirType on illegal arguments, emitting appropriate diagnostics.
MLIR_CAPI_EXPORTED MlirType mlirUnrankedMemRefTypeGetChecked(
    MlirLocation loc, MlirType elementType, MlirAttribute memorySpace);

/// Returns the number of affine layout maps in the given MemRef type.
MLIR_CAPI_EXPORTED intptr_t mlirMemRefTypeGetNumAffineMaps(MlirType type);

/// Returns the pos-th affine map of the given MemRef type.
MLIR_CAPI_EXPORTED MlirAffineMap mlirMemRefTypeGetAffineMap(MlirType type,
                                                            intptr_t pos);

/// Returns the memory space of the given MemRef type.
MLIR_CAPI_EXPORTED MlirAttribute mlirMemRefTypeGetMemorySpace(MlirType type);

/// Returns the memory spcae of the given Unranked MemRef type.
MLIR_CAPI_EXPORTED MlirAttribute
mlirUnrankedMemrefGetMemorySpace(MlirType type);

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a tuple type.
MLIR_CAPI_EXPORTED bool mlirTypeIsATuple(MlirType type);

/// Creates a tuple type that consists of the given list of elemental types. The
/// type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirTupleTypeGet(MlirContext ctx,
                                             intptr_t numElements,
                                             MlirType const *elements);

/// Returns the number of types contained in a tuple.
MLIR_CAPI_EXPORTED intptr_t mlirTupleTypeGetNumTypes(MlirType type);

/// Returns the pos-th type in the tuple type.
MLIR_CAPI_EXPORTED MlirType mlirTupleTypeGetType(MlirType type, intptr_t pos);

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a function type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAFunction(MlirType type);

/// Creates a function type, mapping a list of input types to result types.
MLIR_CAPI_EXPORTED MlirType mlirFunctionTypeGet(MlirContext ctx,
                                                intptr_t numInputs,
                                                MlirType const *inputs,
                                                intptr_t numResults,
                                                MlirType const *results);

/// Returns the number of input types.
MLIR_CAPI_EXPORTED intptr_t mlirFunctionTypeGetNumInputs(MlirType type);

/// Returns the number of result types.
MLIR_CAPI_EXPORTED intptr_t mlirFunctionTypeGetNumResults(MlirType type);

/// Returns the pos-th input type.
MLIR_CAPI_EXPORTED MlirType mlirFunctionTypeGetInput(MlirType type,
                                                     intptr_t pos);

/// Returns the pos-th result type.
MLIR_CAPI_EXPORTED MlirType mlirFunctionTypeGetResult(MlirType type,
                                                      intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BUILTINTYPES_H
