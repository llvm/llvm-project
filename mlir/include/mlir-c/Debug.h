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

/// Adds to the current debug type state, similarly to
/// `-debug-only=prev_type,new_type` in the command-line tools. Note that global
/// debug should be enabled for any output to be produced. A single append call
/// can be reverted with mlirPopAppendedGlobalDebugTypes.
MLIR_CAPI_EXPORTED void mlirAppendGlobalDebugType(const char *type);

/// Adds to the current debug type state, similarly to
/// `-debug-only=prev_type,new_type1,new_type2` in the command-line tools. Note
/// that global debug should be enabled for any output to be produced. A single
/// append call can be reverted with mlirPopAppendedGlobalDebugTypes.
MLIR_CAPI_EXPORTED void mlirAppendGlobalDebugTypes(const char **types,
                                                   intptr_t n);

/// Restores the current debug type state to its state before the last append
/// call. An appended state of `-debug-only=prev_type,new_type1,new_type2` would
/// be `-debug-only=prev_type` after this call.
MLIR_CAPI_EXPORTED void mlirPopAppendedGlobalDebugTypes();

/// Checks if `type` is set as the current debug type.
MLIR_CAPI_EXPORTED bool mlirIsCurrentDebugType(const char *type);

#ifdef __cplusplus
}
#endif

#ifndef MLIR_C_DEBUG_H
#define MLIR_C_DEBUG_H
#endif // MLIR_C_DEBUG_H
