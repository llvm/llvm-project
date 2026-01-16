//===-- mlir-c/Dialect/AMDGPU.h - C API for AMDGPU dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_AMDGPU_H
#define MLIR_C_DIALECT_AMDGPU_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AMDGPU, amdgpu);

//===---------------------------------------------------------------------===//
// TDMBaseType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAAMDGPUTDMBaseType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirAMDGPUTDMBaseTypeGetTypeID();

MLIR_CAPI_EXPORTED MlirType mlirAMDGPUTDMBaseTypeGet(MlirContext ctx,
                                                     MlirType elementType);

//===---------------------------------------------------------------------===//
// TDMDescriptorType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAAMDGPUTDMDescriptorType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirAMDGPUTDMDescriptorTypeGetTypeID();

MLIR_CAPI_EXPORTED MlirType mlirAMDGPUTDMDescriptorTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// TDMGatherBaseType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAAMDGPUTDMGatherBaseType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirAMDGPUTDMGatherBaseTypeGetTypeID();

MLIR_CAPI_EXPORTED MlirType mlirAMDGPUTDMGatherBaseTypeGet(MlirContext ctx,
                                                           MlirType elementType,
                                                           MlirType indexType);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_AMDGPU_H
