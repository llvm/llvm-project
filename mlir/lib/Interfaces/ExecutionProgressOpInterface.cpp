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
  auto executionProgressOpInterface =
      dyn_cast<ExecutionProgressOpInterface>(op);
  if (!executionProgressOpInterface)
    return false;
  return executionProgressOpInterface.mustProgress();
}
