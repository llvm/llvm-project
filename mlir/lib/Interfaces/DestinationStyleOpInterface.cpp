//===- DestinationStyleOpInterface.cpp -- Destination style ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/DestinationStyleOpInterface.cpp.inc"
} // namespace mlir

namespace {
size_t getNumTensorResults(Operation *op) {
  size_t numTensorResults = 0;
  for (auto t : op->getResultTypes()) {
    if (isa<TensorType>(t)) {
      ++numTensorResults;
    }
  }
  return numTensorResults;
}
} // namespace

LogicalResult detail::verifyDestinationStyleOpInterface(Operation *op) {
  DestinationStyleOpInterface dstStyleOp =
      cast<DestinationStyleOpInterface>(op);

  SmallVector<OpOperand *> outputTensorOperands;
#ifndef NDEBUG
  int64_t lastOperandIdx;
  if (!dstStyleOp.getDpsInitsMutable().empty())
    lastOperandIdx =
        static_cast<int64_t>(
            dstStyleOp.getDpsInitsMutable().begin()->getOperandNumber()) -
        1;
#endif // NDEBUG
  for (OpOperand &operand : dstStyleOp.getDpsInitsMutable()) {
#ifndef NDEBUG
    // DPS inits must be consecutive operands. Since `getDpsInitsMutable`
    // returns a MutableArrayRef (that does not own the underlying data), it is
    // currently not possible to return non-consecutive operands and this check
    // just guards against future changes of this interface.
    assert(lastOperandIdx + 1 == operand.getOperandNumber() &&
           "DPS inits must be consecutive operands");
    ++lastOperandIdx;
#endif // NDEBUG
    Type type = operand.get().getType();
    if (isa<RankedTensorType>(type)) {
      outputTensorOperands.push_back(&operand);
    } else if (!isa<MemRefType>(type)) {
      return op->emitOpError("expected that operand #")
             << operand.getOperandNumber()
             << " is a ranked tensor or a ranked memref";
    }
  }

  // Verify the number of tensor results matches the number of output tensors.
  if (getNumTensorResults(op) != outputTensorOperands.size())
    return op->emitOpError("expected the number of tensor results (")
           << getNumTensorResults(op)
           << ") to be equal to the number of output tensors ("
           << outputTensorOperands.size() << ")";

  for (OpOperand *opOperand : outputTensorOperands) {
    OpResult result = dstStyleOp.getTiedOpResult(opOperand);
    if (result.getType() != opOperand->get().getType())
      return op->emitOpError("expected type of operand #")
             << opOperand->getOperandNumber() << " ("
             << opOperand->get().getType() << ")"
             << " to match type of corresponding result (" << result.getType()
             << ")";
  }
  return success();
}
