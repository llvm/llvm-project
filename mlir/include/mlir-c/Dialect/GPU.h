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

#include "mlir-c/Registration.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/GPU/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_GPU_H
