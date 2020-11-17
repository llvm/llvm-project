//===-- mlir-c/Registration.h - Registration functions for MLIR ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTRATION_H
#define MLIR_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects known to core MLIR with the provided Context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTRATION_H
