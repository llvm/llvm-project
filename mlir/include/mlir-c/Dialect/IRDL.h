//===-- mlir-c/Dialect/IRDL.h - C API for IRDL --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_IRDL_H
#define MLIR_C_DIALECT_IRDL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(IRDL, irdl);

/// Loads all IRDL dialects in the provided module, registering the dialects in
/// the module's associated context.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirLoadIRDLDialects(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_IRDL_H
