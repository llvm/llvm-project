//===-- mlir-c/Pass.h - C API to Pass Management ------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_PASS_H
#define MLIR_C_PASS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirPass, void);
DEFINE_C_API_STRUCT(MlirPassManager, void);
DEFINE_C_API_STRUCT(MlirOpPassManager, void);

#undef DEFINE_C_API_STRUCT

/// Create a new top-level PassManager.
MLIR_CAPI_EXPORTED MlirPassManager mlirPassManagerCreate(MlirContext ctx);

/// Destroy the provided PassManager.
MLIR_CAPI_EXPORTED void mlirPassManagerDestroy(MlirPassManager passManager);

/// Checks if a PassManager is null.
static inline bool mlirPassManagerIsNull(MlirPassManager passManager) {
  return !passManager.ptr;
}

/// Cast a top-level PassManager to a generic OpPassManager.
MLIR_CAPI_EXPORTED MlirOpPassManager
mlirPassManagerGetAsOpPassManager(MlirPassManager passManager);

/// Run the provided `passManager` on the given `module`.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPassManagerRun(MlirPassManager passManager, MlirModule module);

/// Nest an OpPassManager under the top-level PassManager, the nested
/// passmanager will only run on operations matching the provided name.
/// The returned OpPassManager will be destroyed when the parent is destroyed.
/// To further nest more OpPassManager under the newly returned one, see
/// `mlirOpPassManagerNest` below.
MLIR_CAPI_EXPORTED MlirOpPassManager mlirPassManagerGetNestedUnder(
    MlirPassManager passManager, MlirStringRef operationName);

/// Nest an OpPassManager under the provided OpPassManager, the nested
/// passmanager will only run on operations matching the provided name.
/// The returned OpPassManager will be destroyed when the parent is destroyed.
MLIR_CAPI_EXPORTED MlirOpPassManager mlirOpPassManagerGetNestedUnder(
    MlirOpPassManager passManager, MlirStringRef operationName);

/// Add a pass and transfer ownership to the provided top-level mlirPassManager.
/// If the pass is not a generic operation pass or a ModulePass, a new
/// OpPassManager is implicitly nested under the provided PassManager.
MLIR_CAPI_EXPORTED void mlirPassManagerAddOwnedPass(MlirPassManager passManager,
                                                    MlirPass pass);

/// Add a pass and transfer ownership to the provided mlirOpPassManager. If the
/// pass is not a generic operation pass or matching the type of the provided
/// PassManager, a new OpPassManager is implicitly nested under the provided
/// PassManager.
MLIR_CAPI_EXPORTED void
mlirOpPassManagerAddOwnedPass(MlirOpPassManager passManager, MlirPass pass);

/// Print a textual MLIR pass pipeline by sending chunks of the string
/// representation and forwarding `userData to `callback`. Note that the
/// callback may be called several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirPrintPassPipeline(MlirOpPassManager passManager,
                                              MlirStringCallback callback,
                                              void *userData);

/// Parse a textual MLIR pass pipeline and add it to the provided OpPassManager.

MLIR_CAPI_EXPORTED MlirLogicalResult
mlirParsePassPipeline(MlirOpPassManager passManager, MlirStringRef pipeline);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_PASS_H
