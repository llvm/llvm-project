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

//===----------------------------------------------------------------------===//
// Dialect registration declarations.
// Registration entry-points for each dialect are declared using the common
// MLIR_DECLARE_DIALECT_REGISTRATION_CAPI macro, which takes the dialect
// API name (i.e. "Func", "Tensor", "Linalg") and namespace (i.e. "func",
// "tensor", "linalg"). The following declarations are produced:
//
//   /// Gets the above hook methods in struct form for a dialect by namespace.
//   /// This is intended to facilitate dynamic lookup and registration of
//   /// dialects via a plugin facility based on shared library symbol lookup.
//   const MlirDialectHandle *mlirGetDialectHandle__{NAMESPACE}__();
//
// This is done via a common macro to facilitate future expansion to
// registration schemes.
//===----------------------------------------------------------------------===//

struct MlirDialectHandle {
  const void *ptr;
};
typedef struct MlirDialectHandle MlirDialectHandle;

#define MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Name, Namespace)                \
  MLIR_CAPI_EXPORTED MlirDialectHandle mlirGetDialectHandle__##Namespace##__()

/// Returns the namespace associated with the provided dialect handle.
MLIR_CAPI_EXPORTED
MlirStringRef mlirDialectHandleGetNamespace(MlirDialectHandle);

/// Inserts the dialect associated with the provided dialect handle into the
/// provided dialect registry
MLIR_CAPI_EXPORTED void mlirDialectHandleInsertDialect(MlirDialectHandle,
                                                       MlirDialectRegistry);

/// Registers the dialect associated with the provided dialect handle.
MLIR_CAPI_EXPORTED void mlirDialectHandleRegisterDialect(MlirDialectHandle,
                                                         MlirContext);

/// Loads the dialect associated with the provided dialect handle.
MLIR_CAPI_EXPORTED MlirDialect mlirDialectHandleLoadDialect(MlirDialectHandle,
                                                            MlirContext);

/// Registers all dialects known to core MLIR with the provided Context.
/// This is needed before creating IR for these Dialects.
/// TODO: Remove this function once the real registration API is finished.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirContext context);

/// Register all translations to LLVM IR for dialects that can support it.
MLIR_CAPI_EXPORTED void mlirRegisterAllLLVMTranslations(MlirContext context);

/// Register all compiler passes of MLIR.
MLIR_CAPI_EXPORTED void mlirRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTRATION_H
