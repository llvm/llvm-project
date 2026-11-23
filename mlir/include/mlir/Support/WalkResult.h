//===- WalkResult.h - Status of completed walk ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Result kind for completed walk.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_WALKRESULT_H
#define MLIR_SUPPORT_WALKRESULT_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Diagnostic;
class InFlightDiagnostic;

/// A utility result that is used to signal how to proceed with an ongoing walk:
///   * Interrupt: the walk will be interrupted and no more operations, regions
///   or blocks will be visited.
///   * Advance: the walk will continue.
///   * Skip: the walk of the current operation, region or block and their
///   nested elements that haven't been visited already will be skipped and will
///   continue with the next operation, region or block.
class WalkResult {
  enum ResultEnum { Interrupt, Advance, Skip } result;

public:
  WalkResult(ResultEnum result = Advance) : result(result) {}

  /// Allow LogicalResult to interrupt the walk on failure.
  WalkResult(LogicalResult result)
      : result(failed(result) ? Interrupt : Advance) {}

  /// Allow diagnostics to interrupt the walk.
  WalkResult(Diagnostic &&) : result(Interrupt) {}
  WalkResult(InFlightDiagnostic &&) : result(Interrupt) {}

  bool operator==(const WalkResult &rhs) const { return result == rhs.result; }
  bool operator!=(const WalkResult &rhs) const { return result != rhs.result; }

  static WalkResult interrupt() { return {Interrupt}; }
  static WalkResult advance() { return {Advance}; }
  static WalkResult skip() { return {Skip}; }

  /// Returns true if the walk was interrupted.
  bool wasInterrupted() const { return result == Interrupt; }

  /// Returns true if the walk was skipped.
  bool wasSkipped() const { return result == Skip; }
};

} // namespace mlir

#endif
