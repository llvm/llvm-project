//===-- mlir-c/Dialect/Quant.h - C API for LLVM -------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_QUANT_H
#define MLIR_C_DIALECT_QUANT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(quant, quant);

//===---------------------------------------------------------------------===//
// QuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a quantization dialect type.
MLIR_CAPI_EXPORTED bool mlirTypeIsAQuantizedType(MlirType type);

/// Returns the bit flag used to indicate signedness of a quantized type.
MLIR_CAPI_EXPORTED unsigned mlirQuantizedTypeGetSignedFlag();

/// Returns the minimum possible value stored by a quantized type.
MLIR_CAPI_EXPORTED int64_t mlirQuantizedTypeGetDefaultMinimumForInteger(
    bool isSigned, unsigned integralWidth);

/// Returns the maximum possible value stored by a quantized type.
MLIR_CAPI_EXPORTED int64_t mlirQuantizedTypeGetDefaultMaximumForInteger(
    bool isSigned, unsigned integralWidth);

/// Gets the original type approximated by the given quantized type.
MLIR_CAPI_EXPORTED MlirType mlirQuantizedTypeGetExpressedType(MlirType type);

/// Gets the flags associated with the given quantized type.
MLIR_CAPI_EXPORTED unsigned mlirQuantizedTypeGetFlags(MlirType type);

/// Returns `true` if the given type is signed, `false` otherwise.
MLIR_CAPI_EXPORTED bool mlirQuantizedTypeIsSigned(MlirType type);

/// Returns the underlying type used to store the values.
MLIR_CAPI_EXPORTED MlirType mlirQuantizedTypeGetStorageType(MlirType type);

/// Returns the minimum value that the storage type of the given quantized type
/// can take.
MLIR_CAPI_EXPORTED int64_t mlirQuantizedTypeGetStorageTypeMin(MlirType type);

/// Returns the maximum value that the storage type of the given quantized type
/// can take.
MLIR_CAPI_EXPORTED int64_t mlirQuantizedTypeGetStorageTypeMax(MlirType type);

/// Returns the integral bitwidth that the storage type of the given quantized
/// type can represent exactly.
MLIR_CAPI_EXPORTED unsigned
mlirQuantizedTypeGetStorageTypeIntegralWidth(MlirType type);

/// Returns `true` if the `candidate` type is compatible with the given
/// quantized `type`.
MLIR_CAPI_EXPORTED bool
mlirQuantizedTypeIsCompatibleExpressedType(MlirType type, MlirType candidate);

/// Returns the element type of the given quantized type as another quantized
/// type.
MLIR_CAPI_EXPORTED MlirType
mlirQuantizedTypeGetQuantizedElementType(MlirType type);

/// Casts from a type based on the storage type of the given type to a
/// corresponding type based on the given type. Returns a null type if the cast
/// is not valid.
MLIR_CAPI_EXPORTED MlirType
mlirQuantizedTypeCastFromStorageType(MlirType type, MlirType candidate);

/// Casts from a type based on a quantized type to a corresponding typed based
/// on the storage type. Returns a null type if the cast is not valid.
MLIR_CAPI_EXPORTED MlirType mlirQuantizedTypeCastToStorageType(MlirType type);

/// Casts from a type based on the expressed type of the given type to a
/// corresponding type based on the given type. Returns a null type if the cast
/// is not valid.
MLIR_CAPI_EXPORTED MlirType
mlirQuantizedTypeCastFromExpressedType(MlirType type, MlirType candidate);

/// Casts from a type based on a quantized type to a corresponding typed based
/// on the expressed type. Returns a null type if the cast is not valid.
MLIR_CAPI_EXPORTED MlirType mlirQuantizedTypeCastToExpressedType(MlirType type);

/// Casts from a type based on the expressed type of the given quantized type to
/// equivalent type based on storage type of the same quantized type.
MLIR_CAPI_EXPORTED MlirType
mlirQuantizedTypeCastExpressedToStorageType(MlirType type, MlirType candidate);

//===---------------------------------------------------------------------===//
// AnyQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is an AnyQuantizedType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAAnyQuantizedType(MlirType type);

/// Creates an instance of AnyQuantizedType with the given parameters in the
/// same context as `storageType` and returns it. The instance is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirAnyQuantizedTypeGet(unsigned flags,
                                                    MlirType storageType,
                                                    MlirType expressedType,
                                                    int64_t storageTypeMin,
                                                    int64_t storageTypeMax);

//===---------------------------------------------------------------------===//
// UniformQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a UniformQuantizedType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAUniformQuantizedType(MlirType type);

/// Creates an instance of UniformQuantizedType with the given parameters in the
/// same context as `storageType` and returns it. The instance is owned by the
/// context.
MLIR_CAPI_EXPORTED MlirType mlirUniformQuantizedTypeGet(
    unsigned flags, MlirType storageType, MlirType expressedType, double scale,
    int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);

/// Returns the scale of the given uniform quantized type.
MLIR_CAPI_EXPORTED double mlirUniformQuantizedTypeGetScale(MlirType type);

/// Returns the zero point of the given uniform quantized type.
MLIR_CAPI_EXPORTED int64_t mlirUniformQuantizedTypeGetZeroPoint(MlirType type);

/// Returns `true` if the given uniform quantized type is fixed-point.
MLIR_CAPI_EXPORTED bool mlirUniformQuantizedTypeIsFixedPoint(MlirType type);

//===---------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a UniformQuantizedPerAxisType.
MLIR_CAPI_EXPORTED bool mlirTypeIsAUniformQuantizedPerAxisType(MlirType type);

/// Creates an instance of UniformQuantizedPerAxisType with the given parameters
/// in the same context as `storageType` and returns it. `scales` and
/// `zeroPoints` point to `nDims` number of elements. The instance is owned
/// by the context.
MLIR_CAPI_EXPORTED MlirType mlirUniformQuantizedPerAxisTypeGet(
    unsigned flags, MlirType storageType, MlirType expressedType,
    intptr_t nDims, double *scales, int64_t *zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin, int64_t storageTypeMax);

/// Returns the number of axes in the given quantized per-axis type.
MLIR_CAPI_EXPORTED intptr_t
mlirUniformQuantizedPerAxisTypeGetNumDims(MlirType type);

/// Returns `pos`-th scale of the given quantized per-axis type.
MLIR_CAPI_EXPORTED double mlirUniformQuantizedPerAxisTypeGetScale(MlirType type,
                                                                  intptr_t pos);

/// Returns `pos`-th zero point of the given quantized per-axis type.
MLIR_CAPI_EXPORTED int64_t
mlirUniformQuantizedPerAxisTypeGetZeroPoint(MlirType type, intptr_t pos);

/// Returns the index of the quantized dimension in the given quantized per-axis
/// type.
MLIR_CAPI_EXPORTED int32_t
mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(MlirType type);

/// Returns `true` if the given uniform quantized per-axis type is fixed-point.
MLIR_CAPI_EXPORTED bool
mlirUniformQuantizedPerAxisTypeIsFixedPoint(MlirType type);

//===---------------------------------------------------------------------===//
// CalibratedQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a CalibratedQuantizedType.
MLIR_CAPI_EXPORTED bool mlirTypeIsACalibratedQuantizedType(MlirType type);

/// Creates an instance of CalibratedQuantizedType with the given parameters
/// in the same context as `expressedType` and returns it. The instance is owned
/// by the context.
MLIR_CAPI_EXPORTED MlirType
mlirCalibratedQuantizedTypeGet(MlirType expressedType, double min, double max);

/// Returns the min value of the given calibrated quantized type.
MLIR_CAPI_EXPORTED double mlirCalibratedQuantizedTypeGetMin(MlirType type);

/// Returns the max value of the given calibrated quantized type.
MLIR_CAPI_EXPORTED double mlirCalibratedQuantizedTypeGetMax(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_QUANT_H
