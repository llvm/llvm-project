//===- UnstructuredControlFlow.cpp - Op Interface Helpers  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"

using namespace mlir;

SmallVector<OpOperand *>
mlir::bufferization::detail::getCallerOpOperands(BlockArgument bbArg) {
  SmallVector<OpOperand *> result;
  Block *block = bbArg.getOwner();
  for (Operation *caller : block->getUsers()) {
    auto branchOp = dyn_cast<BranchOpInterface>(caller);
    assert(branchOp && "expected that all callers implement BranchOpInterface");
    auto it = llvm::find(caller->getSuccessors(), block);
    assert(it != caller->getSuccessors().end() && "could not find successor");
    int64_t successorIdx = std::distance(caller->getSuccessors().begin(), it);
    SuccessorOperands operands = branchOp.getSuccessorOperands(successorIdx);
    assert(operands.getProducedOperandCount() == 0 &&
           "produced operands not supported");
    int64_t operandIndex =
        operands.getForwardedOperands().getBeginOperandIndex() +
        bbArg.getArgNumber();
    result.push_back(&caller->getOpOperand(operandIndex));
  }
  return result;
}
