//===- ExecutionProgressOpInterface.cpp -- Execution Progress Interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ExecutionProgressOpInterface.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/ExecutionProgressOpInterface.cpp.inc"
} // namespace mlir

bool mlir::mustProgress(Operation *op) {
  // Unregistered operations have unknown semantics, so we conservatively
  // assume that they do not necessarily progress.
  if (!op->getName().isRegistered())
    return false;
  // Registered operations are assumed to progress by default. This can be
  // overridden by the ExecutionProgressOpInterface.
  auto executionProgressOpInterface =
      dyn_cast<ExecutionProgressOpInterface>(op);
  if (!executionProgressOpInterface)
    return true;
  return executionProgressOpInterface.mustProgress();
}
