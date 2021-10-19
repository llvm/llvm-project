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

#ifdef __cplusplus
}
#endif

#ifndef MLIR_C_DEBUG_H
#define MLIR_C_DEBUG_H
#endif // MLIR_C_DEBUG_H
