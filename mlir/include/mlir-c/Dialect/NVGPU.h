//===-- mlir-c/Dialect/NVGPU.h - C API for NVGPU dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_NVGPU_H
#define MLIR_C_DIALECT_NVGPU_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(NVGPU, nvgpu);

//===---------------------------------------------------------------------===//
// TensorMapDescriptorType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsANVGPUTensorMapDescriptorType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirNVGPUTensorMapDescriptorTypeGet(
    MlirContext ctx, MlirType tensorMemrefType, int swizzle, int l2promo,
    int oobFill, int interleave);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_NVGPU_H
