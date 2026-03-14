//===-- mlir-c/Dialect/Async.h - C API for Async dialect ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_ASYNC_H
#define MLIR_C_DIALECT_ASYNC_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Async, async);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/Async/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_ASYNC_H
