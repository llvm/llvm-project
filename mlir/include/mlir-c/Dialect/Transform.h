//===-- mlir-c/Dialect/Transform.h - C API for Transform Dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_TRANSFORM_H
#define MLIR_C_DIALECT_TRANSFORM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Transform, transform);

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyOpType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyOpTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyParamType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyParamTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyValueType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyValueTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformOperationType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformOperationTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType
mlirTransformOperationTypeGet(MlirContext ctx, MlirStringRef operationName);

MLIR_CAPI_EXPORTED MlirStringRef
mlirTransformOperationTypeGetOperationName(MlirType type);

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformParamType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTransformParamTypeGet(MlirContext ctx,
                                                      MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTransformParamTypeGetType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_TRANSFORM_H
