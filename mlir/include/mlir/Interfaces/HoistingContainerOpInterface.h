//===- HoistingContainerOpInterface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE_H_
#define MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/HoistingContainerOpInterface.h.inc"

namespace mlir {
/// Returns true if the given operation implements HoistingContainerOpInterface
/// and its implementation allows hosting hoisted operations. Returns false
/// if the operation does not implement the interface, or if the operation
/// explicitly disallows hoisting.
bool canContainHoistedOps(Operation *op);
} // namespace mlir

#endif // MLIR_INTERFACES_HOISTING_CONTAINER_OP_INTERFACE_H_
