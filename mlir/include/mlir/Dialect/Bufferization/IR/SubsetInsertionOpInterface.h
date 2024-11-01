//===- SubsetInsertionOpInterface.h - Tensor Subsets ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_SUBSETINSERTIONOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_SUBSETINSERTIONOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace bufferization {
namespace detail {

/// Return the destination/"init" operand of the op if it implements the
/// `DestinationStyleOpInterface` and has exactly one "init" operand. Asserts
/// otherwise.
OpOperand &defaultGetDestinationOperand(Operation *op);

} // namespace detail
} // namespace bufferization
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/SubsetInsertionOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_SUBSETINSERTIONOPINTERFACE_H_
