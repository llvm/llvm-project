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
DEFINE_C_API_STRUCT(MlirExternalPass, void);
DEFINE_C_API_STRUCT(MlirPassManager, void);
DEFINE_C_API_STRUCT(MlirOpPassManager, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// PassManager/OpPassManager APIs.
//===----------------------------------------------------------------------===//

/// Create a new top-level PassManager with the default anchor.
MLIR_CAPI_EXPORTED MlirPassManager mlirPassManagerCreate(MlirContext ctx);

/// Create a new top-level PassManager anchored on `anchorOp`.
MLIR_CAPI_EXPORTED MlirPassManager
mlirPassManagerCreateOnOperation(MlirContext ctx, MlirStringRef anchorOp);

/// Destroy the provided PassManager.
MLIR_CAPI_EXPORTED void mlirPassManagerDestroy(MlirPassManager passManager);

/// Checks if a PassManager is null.
static inline bool mlirPassManagerIsNull(MlirPassManager passManager) {
  return !passManager.ptr;
}

/// Cast a top-level PassManager to a generic OpPassManager.
MLIR_CAPI_EXPORTED MlirOpPassManager
mlirPassManagerGetAsOpPassManager(MlirPassManager passManager);

/// Run the provided `passManager` on the given `op`.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPassManagerRunOnOp(MlirPassManager passManager, MlirOperation op);

/// Enable IR printing.
/// The treePrintingPath argument is an optional path to a directory
/// where the dumps will be produced. If it isn't provided then dumps
/// are produced to stderr.
MLIR_CAPI_EXPORTED void mlirPassManagerEnableIRPrinting(
    MlirPassManager passManager, bool printBeforeAll, bool printAfterAll,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, MlirOpPrintingFlags flags,
    MlirStringRef treePrintingPath);

/// Enable / disable verify-each.
MLIR_CAPI_EXPORTED void
mlirPassManagerEnableVerifier(MlirPassManager passManager, bool enable);

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

/// Parse a sequence of textual MLIR pass pipeline elements and add them to the
/// provided OpPassManager. If parsing fails an error message is reported using
/// the provided callback.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirOpPassManagerAddPipeline(
    MlirOpPassManager passManager, MlirStringRef pipelineElements,
    MlirStringCallback callback, void *userData);

/// Print a textual MLIR pass pipeline by sending chunks of the string
/// representation and forwarding `userData to `callback`. Note that the
/// callback may be called several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirPrintPassPipeline(MlirOpPassManager passManager,
                                              MlirStringCallback callback,
                                              void *userData);

/// Parse a textual MLIR pass pipeline and assign it to the provided
/// OpPassManager. If parsing fails an error message is reported using the
/// provided callback.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirParsePassPipeline(MlirOpPassManager passManager, MlirStringRef pipeline,
                      MlirStringCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// External Pass API.
//
// This API allows to define passes outside of MLIR, not necessarily in
// C++, and register them with the MLIR pass management infrastructure.
//
//===----------------------------------------------------------------------===//

/// Structure of external `MlirPass` callbacks.
/// All callbacks are required to be set unless otherwise specified.
struct MlirExternalPassCallbacks {
  /// This callback is called from the pass is created.
  /// This is analogous to a C++ pass constructor.
  void (*construct)(void *userData);

  /// This callback is called when the pass is destroyed
  /// This is analogous to a C++ pass destructor.
  void (*destruct)(void *userData);

  /// This callback is optional.
  /// The callback is called before the pass is run, allowing a chance to
  /// initialize any complex state necessary for running the pass.
  /// See Pass::initialize(MLIRContext *).
  MlirLogicalResult (*initialize)(MlirContext ctx, void *userData);

  /// This callback is called when the pass is cloned.
  /// See Pass::clonePass().
  void *(*clone)(void *userData);

  /// This callback is called when the pass is run.
  /// See Pass::runOnOperation().
  void (*run)(MlirOperation op, MlirExternalPass pass, void *userData);
};
typedef struct MlirExternalPassCallbacks MlirExternalPassCallbacks;

/// Creates an external `MlirPass` that calls the supplied `callbacks` using the
/// supplied `userData`. If `opName` is empty, the pass is a generic operation
/// pass. Otherwise it is an operation pass specific to the specified pass name.
MLIR_CAPI_EXPORTED MlirPass mlirCreateExternalPass(
    MlirTypeID passID, MlirStringRef name, MlirStringRef argument,
    MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    MlirExternalPassCallbacks callbacks, void *userData);

/// This signals that the pass has failed. This is only valid to call during
/// the `run` callback of `MlirExternalPassCallbacks`.
/// See Pass::signalPassFailure().
MLIR_CAPI_EXPORTED void mlirExternalPassSignalFailure(MlirExternalPass pass);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_PASS_H
