//===- SubsetInsertionOpInterface.cpp - Tensor Subsets --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/SubsetInsertionOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "mlir/Dialect/Bufferization/IR/SubsetInsertionOpInterface.cpp.inc"

using namespace mlir;

OpOperand &bufferization::detail::defaultGetDestinationOperand(Operation *op) {
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(op);
  assert(dstOp && "getDestination must be implemented for non-DPS ops");
  assert(
      dstOp.getNumDpsInits() == 1 &&
      "getDestination must be implemented for ops with 0 or more than 1 init");
  return *dstOp.getDpsInitOperand(0);
}
