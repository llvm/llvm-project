//===- ControlFlowInterfaces.cpp - ControlFlow Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ControlFlowInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ControlFlowInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

/// Erase an operand from a branch operation that is used as a successor
/// operand. 'operandIndex' is the operand within 'operands' to be erased.
void mlir::detail::eraseBranchSuccessorOperand(OperandRange operands,
                                               unsigned operandIndex,
                                               Operation *op) {
  assert(operandIndex < operands.size() &&
         "invalid index for successor operands");

  // Erase the operand from the operation.
  size_t fullOperandIndex = operands.getBeginOperandIndex() + operandIndex;
  op->eraseOperand(fullOperandIndex);

  // If this operation has an OperandSegmentSizeAttr, keep it up to date.
  auto operandSegmentAttr =
      op->getAttrOfType<DenseElementsAttr>("operand_segment_sizes");
  if (!operandSegmentAttr)
    return;

  // Find the segment containing the full operand index and decrement it.
  // TODO: This seems like a general utility that could be added somewhere.
  SmallVector<int32_t, 4> values(operandSegmentAttr.getValues<int32_t>());
  unsigned currentSize = 0;
  for (unsigned i = 0, e = values.size(); i != e; ++i) {
    currentSize += values[i];
    if (fullOperandIndex < currentSize) {
      --values[i];
      break;
    }
  }
  op->setAttr("operand_segment_sizes",
              DenseIntElementsAttr::get(operandSegmentAttr.getType(), values));
}

/// Returns the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if 'operandIndex' is within the range of 'operands', or None if
/// `operandIndex` isn't a successor operand index.
Optional<BlockArgument> mlir::detail::getBranchSuccessorArgument(
    Optional<OperandRange> operands, unsigned operandIndex, Block *successor) {
  // Check that the operands are valid.
  if (!operands || operands->empty())
    return llvm::None;

  // Check to ensure that this operand is within the range.
  unsigned operandsStart = operands->getBeginOperandIndex();
  if (operandIndex < operandsStart ||
      operandIndex >= (operandsStart + operands->size()))
    return llvm::None;

  // Index the successor.
  unsigned argIndex = operandIndex - operandsStart;
  return successor->getArgument(argIndex);
}

/// Verify that the given operands match those of the given successor block.
LogicalResult
mlir::detail::verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                            Optional<OperandRange> operands) {
  if (!operands)
    return success();

  // Check the count.
  unsigned operandCount = operands->size();
  Block *destBB = op->getSuccessor(succNo);
  if (operandCount != destBB->getNumArguments())
    return op->emitError() << "branch has " << operandCount
                           << " operands for successor #" << succNo
                           << ", but target block has "
                           << destBB->getNumArguments();

  // Check the types.
  auto operandIt = operands->begin();
  for (unsigned i = 0; i != operandCount; ++i, ++operandIt) {
    if ((*operandIt).getType() != destBB->getArgument(i).getType())
      return op->emitError() << "type mismatch for bb argument #" << i
                             << " of successor #" << succNo;
  }
  return success();
}
