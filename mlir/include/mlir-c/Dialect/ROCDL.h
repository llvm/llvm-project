//===-- mlir-c/Dialect/ROCDL.h - C API for ROCDL dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_ROCDL_H
#define MLIR_C_DIALECT_ROCDL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ROCDL, rocdl);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCDL_H
