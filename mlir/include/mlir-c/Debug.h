//===-- mlir-c/Debug.h - C API for MLIR/LLVM debugging functions --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Support.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Sets the global debugging flag.
MLIR_CAPI_EXPORTED void mlirEnableGlobalDebug(bool enable);

/// Retuns `true` if the global debugging flag is set, false otherwise.
MLIR_CAPI_EXPORTED bool mlirIsGlobalDebugEnabled();

/// Sets the current debug type, similarly to `-debug-only=type` in the
/// command-line tools. Note that global debug should be enabled for any output
/// to be produced.
MLIR_CAPI_EXPORTED void mlirSetGlobalDebugType(const char *type);

/// Sets multiple current debug types, similarly to `-debug-only=type1,type2" in
/// the command-line tools. Note that global debug should be enabled for any
/// output to be produced.
MLIR_CAPI_EXPORTED void mlirSetGlobalDebugTypes(const char **types, intptr_t n);

/// Checks if `type` is set as the current debug type.
MLIR_CAPI_EXPORTED bool mlirIsCurrentDebugType(const char *type);

#ifdef __cplusplus
}
#endif

#ifndef MLIR_C_DEBUG_H
#define MLIR_C_DEBUG_H
#endif // MLIR_C_DEBUG_H
