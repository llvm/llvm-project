//===-- mlir-c/Dialect/GPU.h - C API for GPU dialect -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_GPU_H
#define MLIR_C_DIALECT_GPU_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

//===-------------------------------------------------------------------===//
// AsyncTokenType
//===-------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAGPUAsyncTokenType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirGPUAsyncTokenTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// ObjectAttr
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirAttributeIsAGPUObjectAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirGPUObjectAttrGet(MlirContext mlirCtx, MlirAttribute target, uint32_t format,
                     MlirStringRef objectStrRef, MlirAttribute mlirObjectProps);

MLIR_CAPI_EXPORTED MlirAttribute
mlirGPUObjectAttrGetTarget(MlirAttribute mlirObjectAttr);

MLIR_CAPI_EXPORTED uint32_t
mlirGPUObjectAttrGetFormat(MlirAttribute mlirObjectAttr);

MLIR_CAPI_EXPORTED MlirStringRef
mlirGPUObjectAttrGetObject(MlirAttribute mlirObjectAttr);

MLIR_CAPI_EXPORTED bool
mlirGPUObjectAttrHasProperties(MlirAttribute mlirObjectAttr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirGPUObjectAttrGetProperties(MlirAttribute mlirObjectAttr);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/GPU/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_GPU_H
