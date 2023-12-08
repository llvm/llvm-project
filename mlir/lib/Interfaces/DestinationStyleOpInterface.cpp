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
  for (OpOperand &operand : dstStyleOp.getDpsInitsMutable()) {
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
