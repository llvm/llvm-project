//===- HoistingContainerOpInterface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE
#define MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/HoistingContainerOpInterface.h.inc"

namespace mlir {
bool canContainHoistedOps(Operation *op);
} // namespace mlir

#endif // MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE
