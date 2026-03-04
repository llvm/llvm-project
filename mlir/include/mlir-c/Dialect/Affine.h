//===-- mlir-c/Dialect/Affine.h - C API for Affine Dialect --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_AFFINE_H
#define MLIR_C_DIALECT_AFFINE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/Affine/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_AFFINE_H
