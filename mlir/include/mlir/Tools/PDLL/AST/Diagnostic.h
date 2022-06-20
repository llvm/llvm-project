//===- Diagnostic.h - PDLL AST Diagnostics ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_AST_DIAGNOSTIC_H
#define MLIR_TOOLS_PDLL_AST_DIAGNOSTIC_H

#include <string>

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace pdll {
namespace ast {
class DiagnosticEngine;

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

/// This class provides a simple implementation of a PDLL diagnostic.
class Diagnostic {
public:
  using Severity = llvm::SourceMgr::DiagKind;

  /// Return the severity of this diagnostic.
  Severity getSeverity() const { return severity; }

  /// Return the message of this diagnostic.
  StringRef getMessage() const { return message; }

  /// Return the location of this diagnostic.
  SMRange getLocation() const { return location; }

  /// Return the notes of this diagnostic.
  auto getNotes() const { return llvm::make_pointee_range(notes); }

  /// Attach a note to this diagnostic.
  Diagnostic &attachNote(const Twine &msg,
                         Optional<SMRange> noteLoc = llvm::None) {
    assert(getSeverity() != Severity::DK_Note &&
           "cannot attach a Note to a Note");
    notes.emplace_back(
        new Diagnostic(Severity::DK_Note, noteLoc.value_or(location), msg));
    return *notes.back();
  }

  /// Allow an inflight diagnostic to be converted to 'failure', otherwise
  /// 'success' if this is an empty diagnostic.
  operator LogicalResult() const { return failure(); }

private:
  Diagnostic(Severity severity, SMRange loc, const Twine &msg)
      : severity(severity), message(msg.str()), location(loc) {}

  // Allow access to the constructor.
  friend DiagnosticEngine;

  /// The severity of this diagnostic.
  Severity severity;
  /// The message held by this diagnostic.
  std::string message;
  /// The raw location of this diagnostic.
  SMRange location;
  /// Any additional note diagnostics attached to this diagnostic.
  std::vector<std::unique_ptr<Diagnostic>> notes;
};

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

/// This class represents a diagnostic that is inflight and set to be reported.
/// This allows for last minute modifications of the diagnostic before it is
/// emitted by a DiagnosticEngine.
class InFlightDiagnostic {
public:
  InFlightDiagnostic() = default;
  InFlightDiagnostic(InFlightDiagnostic &&rhs)
      : owner(rhs.owner), impl(std::move(rhs.impl)) {
    // Reset the rhs diagnostic.
    rhs.impl.reset();
    rhs.abandon();
  }
  ~InFlightDiagnostic() {
    if (isInFlight())
      report();
  }

  /// Access the internal diagnostic.
  Diagnostic &operator*() { return *impl; }
  Diagnostic *operator->() { return &*impl; }

  /// Reports the diagnostic to the engine.
  void report();

  /// Abandons this diagnostic so that it will no longer be reported.
  void abandon() { owner = nullptr; }

  /// Allow an inflight diagnostic to be converted to 'failure', otherwise
  /// 'success' if this is an empty diagnostic.
  operator LogicalResult() const { return failure(isActive()); }

private:
  InFlightDiagnostic &operator=(const InFlightDiagnostic &) = delete;
  InFlightDiagnostic &operator=(InFlightDiagnostic &&) = delete;
  InFlightDiagnostic(DiagnosticEngine *owner, Diagnostic &&rhs)
      : owner(owner), impl(std::move(rhs)) {}

  /// Returns true if the diagnostic is still active, i.e. it has a live
  /// diagnostic.
  bool isActive() const { return impl.hasValue(); }

  /// Returns true if the diagnostic is still in flight to be reported.
  bool isInFlight() const { return owner; }

  // Allow access to the constructor.
  friend DiagnosticEngine;

  /// The engine that this diagnostic is to report to.
  DiagnosticEngine *owner = nullptr;

  /// The raw diagnostic that is inflight to be reported.
  Optional<Diagnostic> impl;
};

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

/// This class manages the construction and emission of PDLL diagnostics.
class DiagnosticEngine {
public:
  /// A function used to handle diagnostics emitted by the engine.
  using HandlerFn = llvm::unique_function<void(Diagnostic &)>;

  /// Emit an error to the diagnostic engine.
  InFlightDiagnostic emitError(SMRange loc, const Twine &msg) {
    return InFlightDiagnostic(
        this, Diagnostic(Diagnostic::Severity::DK_Error, loc, msg));
  }
  InFlightDiagnostic emitWarning(SMRange loc, const Twine &msg) {
    return InFlightDiagnostic(
        this, Diagnostic(Diagnostic::Severity::DK_Warning, loc, msg));
  }

  /// Report the given diagnostic.
  void report(Diagnostic &&diagnostic) {
    if (handler)
      handler(diagnostic);
  }

  /// Get the current handler function of this diagnostic engine.
  const HandlerFn &getHandlerFn() const { return handler; }

  /// Take the current handler function, resetting the current handler to null.
  HandlerFn takeHandlerFn() {
    HandlerFn oldHandler = std::move(handler);
    handler = {};
    return oldHandler;
  }

  /// Set the handler function for this diagnostic engine.
  void setHandlerFn(HandlerFn &&newHandler) { handler = std::move(newHandler); }

private:
  /// The registered diagnostic handler function.
  HandlerFn handler;
};

} // namespace ast
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_AST_DIAGNOSTIC_H
