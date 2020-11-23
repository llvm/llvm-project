//===- Diagnostics.cpp - C Interface for MLIR Diagnostics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Diagnostics.h"
#include "mlir/CAPI/Diagnostics.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

void mlirDiagnosticPrint(MlirDiagnostic diagnostic, MlirStringCallback callback,
                         void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(diagnostic).print(stream);
}

MlirLocation mlirDiagnosticGetLocation(MlirDiagnostic diagnostic) {
  return wrap(unwrap(diagnostic).getLocation());
}

MlirDiagnosticSeverity mlirDiagnosticGetSeverity(MlirDiagnostic diagnostic) {
  switch (unwrap(diagnostic).getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    return MlirDiagnosticError;
  case mlir::DiagnosticSeverity::Warning:
    return MlirDiagnosticWarning;
  case mlir::DiagnosticSeverity::Note:
    return MlirDiagnosticNote;
  case mlir::DiagnosticSeverity::Remark:
    return MlirDiagnosticRemark;
  }
  llvm_unreachable("unhandled diagnostic severity");
}

// Notes are stored in a vector, so note iterator range is a pair of
// random access iterators, for which it is cheap to compute the size.
intptr_t mlirDiagnosticGetNumNotes(MlirDiagnostic diagnostic) {
  return static_cast<intptr_t>(llvm::size(unwrap(diagnostic).getNotes()));
}

// Notes are stored in a vector, so the iterator is a random access iterator,
// cheap to advance multiple steps at a time.
MlirDiagnostic mlirDiagnosticGetNote(MlirDiagnostic diagnostic, intptr_t pos) {
  return wrap(*std::next(unwrap(diagnostic).getNotes().begin(), pos));
}

static void deleteUserDataNoop(void *userData) {}

MlirDiagnosticHandlerID mlirContextAttachDiagnosticHandler(
    MlirContext context, MlirDiagnosticHandler handler, void *userData,
    void (*deleteUserData)(void *)) {
  assert(handler && "unexpected null diagnostic handler");
  if (deleteUserData == NULL)
    deleteUserData = deleteUserDataNoop;
  std::shared_ptr<void> sharedUserData(userData, deleteUserData);
  DiagnosticEngine::HandlerID id =
      unwrap(context)->getDiagEngine().registerHandler(
          [handler, sharedUserData](Diagnostic &diagnostic) {
            return unwrap(handler(wrap(diagnostic), sharedUserData.get()));
          });
  return static_cast<MlirDiagnosticHandlerID>(id);
}

void mlirContextDetachDiagnosticHandler(MlirContext context,
                                        MlirDiagnosticHandlerID id) {
  unwrap(context)->getDiagEngine().eraseHandler(
      static_cast<DiagnosticEngine::HandlerID>(id));
}

void mlirEmitError(MlirLocation location, const char *message) {
  emitError(unwrap(location)) << message;
}
