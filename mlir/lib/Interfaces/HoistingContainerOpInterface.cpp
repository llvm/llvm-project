//===- HoistingContainerOpInterface.cpp -- Hoisting Container Op Interface -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/HoistingContainerOpInterface.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/HoistingContainerOpInterface.cpp.inc"
} // namespace mlir

bool mlir::canContainHoistedOps(Operation *op) {
  if (auto containerOp = dyn_cast<HoistingContainerOpInterface>(op))
    return containerOp.canContainHoistedOps();
  return false;
}
