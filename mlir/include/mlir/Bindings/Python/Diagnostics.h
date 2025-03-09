//===- Diagnostics.h - Helpers for diagnostics in Python bindings ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_DIAGNOSTICS_H
#define MLIR_BINDINGS_PYTHON_DIAGNOSTICS_H

#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>

namespace mlir {
namespace python {

/// RAII scope intercepting all diagnostics into a string. The message must be
/// checked before this goes out of scope.
class CollectDiagnosticsToStringScope {
public:
  explicit CollectDiagnosticsToStringScope(MlirContext ctx) : context(ctx) {
    handlerID = mlirContextAttachDiagnosticHandler(ctx, &handler, &errorMessage,
                                                   /*deleteUserData=*/nullptr);
  }
  ~CollectDiagnosticsToStringScope() {
    mlirContextDetachDiagnosticHandler(context, handlerID);
  }

  [[nodiscard]] std::string takeMessage() {
    std::ostringstream stream;
    std::swap(stream, errorMessage);
    return stream.str();
  }

private:
  static MlirLogicalResult handler(MlirDiagnostic diag, void *data) {
    auto printer = +[](MlirStringRef message, void *data) {
      *static_cast<std::ostringstream *>(data)
          << std::string_view(message.data, message.length);
    };
    MlirLocation loc = mlirDiagnosticGetLocation(diag);
    *static_cast<std::ostringstream *>(data) << "at ";
    mlirLocationPrint(loc, printer, data);
    *static_cast<std::ostringstream *>(data) << ": ";
    mlirDiagnosticPrint(diag, printer, data);
    for (intptr_t i = 0; i < mlirDiagnosticGetNumNotes(diag); i++) {
      *static_cast<std::ostringstream *>(data) << "\n";
      MlirDiagnostic note = mlirDiagnosticGetNote(diag, i);
      handler(note, data);
    }
    return mlirLogicalResultSuccess();
  }

  MlirContext context;
  MlirDiagnosticHandlerID handlerID;
  std::ostringstream errorMessage;
};

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_DIAGNOSTICS_H
