//===-- mlir-c/Dialect/PDL.h - C API for PDL Dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_PDL_H
#define MLIR_C_DIALECT_PDL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PDL, pdl);

//===---------------------------------------------------------------------===//
// PDLType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLType(MlirType type);

//===---------------------------------------------------------------------===//
// AttributeType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLAttributeType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPDLAttributeTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLOperationType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPDLOperationTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// RangeType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLRangeType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPDLRangeTypeGet(MlirType elementType);

MLIR_CAPI_EXPORTED MlirType mlirPDLRangeTypeGetElementType(MlirType type);

//===---------------------------------------------------------------------===//
// TypeType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLTypeType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPDLTypeTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// ValueType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAPDLValueType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPDLValueTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_PDL_H
