//===- ControlFlowInterfaces.h - ControlFlow Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the branch interfaces defined in
// `ControlFlowInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_CONTROLFLOWINTERFACES_H
#define MLIR_ANALYSIS_CONTROLFLOWINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class BranchOpInterface;

namespace detail {
/// Erase an operand from a branch operation that is used as a successor
/// operand. `operandIndex` is the operand within `operands` to be erased.
void eraseBranchSuccessorOperand(OperandRange operands, unsigned operandIndex,
                                 Operation *op);

/// Return the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if `operandIndex` is within the range of `operands`, or None if
/// `operandIndex` isn't a successor operand index.
Optional<BlockArgument>
getBranchSuccessorArgument(Optional<OperandRange> operands,
                           unsigned operandIndex, Block *successor);

/// Verify that the given operands match those of the given successor block.
LogicalResult verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                            Optional<OperandRange> operands);
} // namespace detail

#include "mlir/Analysis/ControlFlowInterfaces.h.inc"
} // end namespace mlir

#endif // MLIR_ANALYSIS_CONTROLFLOWINTERFACES_H
