//===- WasmSSAInterfaces.cpp - WasmSSA Interfaces -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the WasmSSA dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.cpp.inc"

namespace detail {
LogicalResult verifyLabelBranchingOpInterface(Operation *op) {
  auto branchInterface = dyn_cast<LabelBranchingOpInterface>(op);
  llvm::FailureOr<LabelLevelOpInterface> res =
      LabelBranchingOpInterface::getTargetOpFromBlock(
          op->getBlock(), branchInterface.getExitLevel());
  return res;
}

LogicalResult verifyConstantExpressionInterface(Operation *op) {
  Region &initializerRegion = op->getRegion(0);
  WalkResult resultState =
      initializerRegion.walk([&](Operation *currentOp) -> WalkResult {
        if (isa<ReturnOp>(currentOp) ||
            currentOp->hasTrait<ConstantExprOpTrait>())
          return WalkResult::advance();
        op->emitError("expected a constant initializer for this operator, got ")
            << currentOp;
        return WalkResult::interrupt();
      });
  return success(!resultState.wasInterrupted());
}

LogicalResult verifyLabelLevelInterface(Operation *op) {
  Block *target = cast<LabelLevelOpInterface>(op).getLabelTarget();
  Region *targetRegion = target->getParent();
  if (targetRegion != op->getParentRegion() &&
      targetRegion->getParentOp() != op)
    return op->emitError("target should be a block defined in same level than "
                         "operation or in its region.");
  return success();
}
} // namespace detail

llvm::FailureOr<LabelLevelOpInterface>
LabelBranchingOpInterface::getTargetOpFromBlock(::mlir::Block *block,
                                                uint32_t breakLevel) {
  LabelLevelOpInterface res{};
  for (size_t curLevel{0}; curLevel <= breakLevel; curLevel++) {
    res = dyn_cast_or_null<LabelLevelOpInterface>(block->getParentOp());
    if (!res)
      return failure();
    block = res->getBlock();
  }
  return res;
}
} // namespace mlir::wasmssa
