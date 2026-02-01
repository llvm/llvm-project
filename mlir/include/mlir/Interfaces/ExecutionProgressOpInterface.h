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
/// Return "true" if the operation must progress. Operations that do not
/// implement the ExecutionProgressOpInterface must progress by default. For
/// all other ops, the "must progress" property is queried from the interface.
bool mustProgress(Operation *op);
} // namespace mlir

#endif // MLIR_INTERFACES_EXECUTIONPROGRESSOPINTERFACE_H_
