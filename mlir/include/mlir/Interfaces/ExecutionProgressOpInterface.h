//===- ExecutionProgressOpInterface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_EXECUTIONPROGRESSOPINTERFACE_H_
#define MLIR_INTERFACES_EXECUTIONPROGRESSOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

#include "mlir/Interfaces/ExecutionProgressOpInterface.h.inc"

namespace mlir {
/// Return "true" if the operation must progress.
///
/// Unregistered operations are treated conservatively: they may not
/// necessarily progress (i.e., return "false"). Registered operations are
/// assumed to progress by default. This can be overridden by the
/// ExecutionProgressOpInterface.
bool mustProgress(Operation *op);

/// Return "true" if the operation might not progress.
inline bool mightNotProgress(Operation *op) { return !mustProgress(op); }
} // namespace mlir

#endif // MLIR_INTERFACES_EXECUTIONPROGRESSOPINTERFACE_H_
