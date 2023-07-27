//===-- mlir-c/Dialect/Func.h - C API for Func dialect ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Func dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_FUNC_H
#define MLIR_C_DIALECT_FUNC_H

#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Func, func);

/// Sets the argument attribute 'name' of an argument at index 'pos'.
/// Asserts that the operation is a FuncOp.
MLIR_CAPI_EXPORTED void mlirFuncSetArgAttr(MlirOperation op, intptr_t pos,
                                           MlirStringRef name,
                                           MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_FUNC_H
