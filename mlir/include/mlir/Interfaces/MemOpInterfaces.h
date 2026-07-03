//===- MemOpInterfaces.h - Memory operation interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of interfaces for operations that interact
// with memory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MEMOPINTERFACES_H
#define MLIR_INTERFACES_MEMOPINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace detail {
/// Attempt to verify the given memory space cast operation.
LogicalResult verifyMemorySpaceCastOpInterface(Operation *op);

/// Tries to bubble-down inplace a `MemorySpaceCastOpInterface` operation
/// referenced by `operand`. On success, it returns `std::nullopt`. It
/// returns failure if `operand` doesn't reference a
/// `MemorySpaceCastOpInterface` op.
FailureOr<std::optional<SmallVector<Value>>>
bubbleDownInPlaceMemorySpaceCastImpl(OpOperand &operand, ValueRange results);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/MemOpInterfaces.h.inc"

#endif // MLIR_INTERFACES_MEMOPINTERFACES_H
