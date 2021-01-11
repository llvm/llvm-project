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
// API name (i.e. "Standard", "Tensor", "Linalg") and namespace (i.e. "std",
// "tensor", "linalg"). The following declarations are produced:
//
//   /// Registers the dialect with the given context. This allows the
//   /// dialect to be loaded dynamically if needed when parsing. */
//   void mlirContextRegister{NAME}Dialect(MlirContext);
//
//   /// Loads the dialect into the given context. The dialect does _not_
//   /// have to be registered in advance.
//   MlirDialect mlirContextLoad{NAME}Dialect(MlirContext context);
//
//   /// Returns the namespace of the Standard dialect, suitable for loading it.
//   MlirStringRef mlir{NAME}DialectGetNamespace();
//
//   /// Gets the above hook methods in struct form for a dialect by namespace.
//   /// This is intended to facilitate dynamic lookup and registration of
//   /// dialects via a plugin facility based on shared library symbol lookup.
//   const MlirDialectRegistrationHooks *mlirGetDialectHooks__{NAMESPACE}__();
//
// This is done via a common macro to facilitate future expansion to
// registration schemes.
//===----------------------------------------------------------------------===//

#define MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Name, Namespace)                \
  MLIR_CAPI_EXPORTED void mlirContextRegister##Name##Dialect(                  \
      MlirContext context);                                                    \
  MLIR_CAPI_EXPORTED MlirDialect mlirContextLoad##Name##Dialect(               \
      MlirContext context);                                                    \
  MLIR_CAPI_EXPORTED MlirStringRef mlir##Name##DialectGetNamespace();          \
  MLIR_CAPI_EXPORTED const MlirDialectRegistrationHooks                        \
      *mlirGetDialectHooks__##Namespace##__()

/// Hooks for dynamic discovery of dialects.
typedef void (*MlirContextRegisterDialectHook)(MlirContext context);
typedef MlirDialect (*MlirContextLoadDialectHook)(MlirContext context);
typedef MlirStringRef (*MlirDialectGetNamespaceHook)();

/// Structure of dialect registration hooks.
struct MlirDialectRegistrationHooks {
  MlirContextRegisterDialectHook registerHook;
  MlirContextLoadDialectHook loadHook;
  MlirDialectGetNamespaceHook getNamespaceHook;
};
typedef struct MlirDialectRegistrationHooks MlirDialectRegistrationHooks;

/// Registers all dialects known to core MLIR with the provided Context.
/// This is needed before creating IR for these Dialects.
/// TODO: Remove this function once the real registration API is finished.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REGISTRATION_H
