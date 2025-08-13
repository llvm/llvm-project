//===-- mlir-c/ExecutionEngine.h - Execution engine management ---*- C -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides basic access to the MLIR JIT. This is minimalist and
// experimental at the moment.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_EXECUTIONENGINE_H
#define MLIR_C_EXECUTIONENGINE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirExecutionEngine, void);

#undef DEFINE_C_API_STRUCT

/// Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is
/// expected to be "translatable" to LLVM IR (only contains operations in
/// dialects that implement the `LLVMTranslationDialectInterface`). The module
/// ownership stays with the client and can be destroyed as soon as the call
/// returns. `optLevel` is the optimization level to be used for transformation
/// and code generation. LLVM passes at `optLevel` are run before code
/// generation. The number and array of paths corresponding to shared libraries
/// that will be loaded are specified via `numPaths` and `sharedLibPaths`
/// respectively.
/// TODO: figure out other options.
MLIR_CAPI_EXPORTED MlirExecutionEngine mlirExecutionEngineCreate(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump);

/// Destroy an ExecutionEngine instance.
MLIR_CAPI_EXPORTED void mlirExecutionEngineDestroy(MlirExecutionEngine jit);

/// Checks whether an execution engine is null.
static inline bool mlirExecutionEngineIsNull(MlirExecutionEngine jit) {
  return !jit.ptr;
}

/// Invoke a native function in the execution engine by name with the arguments
/// and result of the invoked function passed as an array of pointers. The
/// function must have been tagged with the `llvm.emit_c_interface` attribute.
/// Returns a failure if the execution fails for any reason (the function name
/// can't be resolved for instance).
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExecutionEngineInvokePacked(
    MlirExecutionEngine jit, MlirStringRef name, void **arguments);

/// Lookup the wrapper of the native function in the execution engine with the
/// given name, returns nullptr if the function can't be looked-up.
MLIR_CAPI_EXPORTED void *
mlirExecutionEngineLookupPacked(MlirExecutionEngine jit, MlirStringRef name);

/// Lookup a native function in the execution engine by name, returns nullptr
/// if the name can't be looked-up.
MLIR_CAPI_EXPORTED void *mlirExecutionEngineLookup(MlirExecutionEngine jit,
                                                   MlirStringRef name);

/// Register a symbol with the jit: this symbol will be accessible to the jitted
/// code.
MLIR_CAPI_EXPORTED void
mlirExecutionEngineRegisterSymbol(MlirExecutionEngine jit, MlirStringRef name,
                                  void *sym);

/// Dump as an object in `fileName`.
MLIR_CAPI_EXPORTED void
mlirExecutionEngineDumpToObjectFile(MlirExecutionEngine jit,
                                    MlirStringRef fileName);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_EXECUTIONENGINE_H
