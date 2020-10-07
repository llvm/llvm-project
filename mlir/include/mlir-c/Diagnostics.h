/*===-- mlir-c/Diagnostics.h - MLIR Diagnostic subsystem C API ----*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C APIs accessing MLIR Diagnostics subsystem.      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_DIAGNOSTICS_H
#define MLIR_C_DIAGNOSTICS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** An opaque reference to a dignostic, always owned by the diagnostics engine
 * (context). Must not be stored outside of the diagnostic handler. */
struct MlirDiagnostic {
  void *ptr;
};
typedef struct MlirDiagnostic MlirDiagnostic;

/** Severity of a diagnostic. */
enum MlirDiagnosticSeverity {
  MlirDiagnosticError,
  MlirDiagnosticWarning,
  MlirDiagnosticNote,
  MlirDiagnosticRemark
};
typedef enum MlirDiagnosticSeverity MlirDiagnosticSeverity;

/** Opaque identifier of a diagnostic handler, useful to detach a handler. */
typedef uint64_t MlirDiagnosticHandlerID;

/** Diagnostic handler type. Acceps a reference to a diagnostic, which is only
 * guaranteed to be live during the call. If the handler processed the
 * diagnostic completely, it is expected to return success. Otherwise, it is
 * expected to return failure to indicate that other handlers should attempt to
 * process the diagnostic. */
typedef MlirLogicalResult (*MlirDiagnosticHandler)(MlirDiagnostic);

/** Prints a diagnostic using the provided callback. */
void mlirDiagnosticPrint(MlirDiagnostic diagnostic, MlirStringCallback callback,
                         void *userData);

/** Returns the location at which the diagnostic is reported. */
MlirLocation mlirDiagnosticGetLocation(MlirDiagnostic diagnostic);

/** Returns the severity of the diagnostic. */
MlirDiagnosticSeverity mlirDiagnosticGetSeverity(MlirDiagnostic diagnostic);

/** Returns the number of notes attached to the diagnostic. */
intptr_t mlirDiagnosticGetNumNotes(MlirDiagnostic diagnostic);

/** Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a
 * valid zero-based index into the list of notes. */
MlirDiagnostic mlirDiagnosticGetNote(MlirDiagnostic diagnostic, intptr_t pos);

/** Attaches the diagnostic handler to the context. Handlers are invoked in the
 * reverse order of attachment until one of them processes the diagnostic
 * completely. Returns an identifier that can be used to detach the handler. */
MlirDiagnosticHandlerID
mlirContextAttachDiagnosticHandler(MlirContext context,
                                   MlirDiagnosticHandler handler);

/** Detaches an attached diagnostic handler from the context given its
 * identifier. */
void mlirContextDetachDiagnosticHandler(MlirContext context,
                                        MlirDiagnosticHandlerID id);

/** Emits an error at the given location through the diagnostics engine. Used
 * for testing purposes. */
void mlirEmitError(MlirLocation location, const char *message);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIAGNOSTICS_H
