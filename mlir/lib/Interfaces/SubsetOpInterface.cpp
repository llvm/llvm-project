//===- SubsetOpInterface.cpp - Tensor Subsets -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "mlir/Interfaces/SubsetOpInterface.cpp.inc"

using namespace mlir;

OpOperand &detail::defaultGetDestinationOperand(Operation *op) {
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dstOp && "getDestination must be implemented for non-DPS ops");
  assert(
      dstOp.getNumDpsInits() == 1 &&
      "getDestination must be implemented for ops with 0 or more than 1 init");
  return *dstOp.getDpsInitOperand(0);
}

OpResult detail::defaultGetUpdatedDestination(Operation *op) {
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dstOp && "getUpdatedDestination must be implemented for non-DPS ops");
  auto insertionOp = cast<SubsetInsertionOpInterface>(op);
  return dstOp.getTiedOpResult(&insertionOp.getDestinationOperand());
}

bool detail::defaultIsEquivalentSubset(
    Operation *op, Value candidate,
    function_ref<bool(Value, Value)> equivalenceFn) {
  assert(isa<SubsetInsertionOpInterface>(op) &&
         "expected SubsetInsertionOpInterface");
  if (!candidate.getDefiningOp<SubsetExtractionOpInterface>())
    return false;
  return cast<SubsetOpInterface>(op).operatesOnEquivalentSubset(
      candidate.getDefiningOp<SubsetOpInterface>(), equivalenceFn);
}

LogicalResult detail::verifySubsetOpInterface(SubsetOpInterface op) {
  if (!(isa<SubsetExtractionOpInterface>(op.getOperation()) ^
        isa<SubsetInsertionOpInterface>(op.getOperation())))
    return op->emitOpError(
        "SubsetOpInterface ops must implement either "
        "SubsetExtractionOpInterface or SubsetInsertionOpInterface");
  return success();
}

LogicalResult
detail::verifySubsetExtractionOpInterface(SubsetExtractionOpInterface op) {
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "SubsetExtractionOpInterface ops must have one result");
  return success();
}
