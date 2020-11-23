//===-- mlir-c/Diagnostics.h - MLIR Diagnostic subsystem C API ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C APIs accessing MLIR Diagnostics subsystem.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIAGNOSTICS_H
#define MLIR_C_DIAGNOSTICS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** An opaque reference to a diagnostic, always owned by the diagnostics engine
 * (context). Must not be stored outside of the diagnostic handler. */
struct MlirDiagnostic {
  void *ptr;
};
typedef struct MlirDiagnostic MlirDiagnostic;

/// Severity of a diagnostic.
enum MlirDiagnosticSeverity {
  MlirDiagnosticError,
  MlirDiagnosticWarning,
  MlirDiagnosticNote,
  MlirDiagnosticRemark
};
typedef enum MlirDiagnosticSeverity MlirDiagnosticSeverity;

/// Opaque identifier of a diagnostic handler, useful to detach a handler.
typedef uint64_t MlirDiagnosticHandlerID;

/** Diagnostic handler type. Accepts a reference to a diagnostic, which is only
 * guaranteed to be live during the call. The handler is passed the `userData`
 * that was provided when the handler was attached to a context. If the handler
 * processed the diagnostic completely, it is expected to return success.
 * Otherwise, it is expected to return failure to indicate that other handlers
 * should attempt to process the diagnostic. */
typedef MlirLogicalResult (*MlirDiagnosticHandler)(MlirDiagnostic,
                                                   void *userData);

/// Prints a diagnostic using the provided callback.
MLIR_CAPI_EXPORTED void mlirDiagnosticPrint(MlirDiagnostic diagnostic,
                                            MlirStringCallback callback,
                                            void *userData);

/// Returns the location at which the diagnostic is reported.
MLIR_CAPI_EXPORTED MlirLocation
mlirDiagnosticGetLocation(MlirDiagnostic diagnostic);

/// Returns the severity of the diagnostic.
MLIR_CAPI_EXPORTED MlirDiagnosticSeverity
mlirDiagnosticGetSeverity(MlirDiagnostic diagnostic);

/// Returns the number of notes attached to the diagnostic.
MLIR_CAPI_EXPORTED intptr_t
mlirDiagnosticGetNumNotes(MlirDiagnostic diagnostic);

/** Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a
 * valid zero-based index into the list of notes. */
MLIR_CAPI_EXPORTED MlirDiagnostic
mlirDiagnosticGetNote(MlirDiagnostic diagnostic, intptr_t pos);

/** Attaches the diagnostic handler to the context. Handlers are invoked in the
 * reverse order of attachment until one of them processes the diagnostic
 * completely. When a handler is invoked it is passed the `userData` that was
 * provided when it was attached. If non-NULL, `deleteUserData` is called once
 * the system no longer needs to call the handler (for instance after the
 * handler is detached or the context is destroyed). Returns an identifier that
 * can be used to detach the handler.
 */
MLIR_CAPI_EXPORTED MlirDiagnosticHandlerID mlirContextAttachDiagnosticHandler(
    MlirContext context, MlirDiagnosticHandler handler, void *userData,
    void (*deleteUserData)(void *));

/** Detaches an attached diagnostic handler from the context given its
 * identifier. */
MLIR_CAPI_EXPORTED void
mlirContextDetachDiagnosticHandler(MlirContext context,
                                   MlirDiagnosticHandlerID id);

/** Emits an error at the given location through the diagnostics engine. Used
 * for testing purposes. */
MLIR_CAPI_EXPORTED void mlirEmitError(MlirLocation location,
                                      const char *message);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIAGNOSTICS_H
