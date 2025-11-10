//===- PtrDialect.h - C interface for the Ptr dialect -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_PTR_H
#define MLIR_C_DIALECT_PTR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ptr, ptr);

//===----------------------------------------------------------------------===//
// MemorySpaceAttrInterface API.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Checks if the given type is a Ptr type.
MLIR_CAPI_EXPORTED bool mlirPtrTypeIsAPtrType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPtrGetPtrType(MlirAttribute memorySpace);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_PTR_H
