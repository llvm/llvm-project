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

namespace mlir::wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.cpp.inc"

namespace detail {
LogicalResult verifyWasmSSALabelBranchingOpInterface(Operation *op) {
  auto branchInterface = dyn_cast<WasmSSALabelBranchingOpInterface>(op);
  llvm::FailureOr<WasmSSALabelLevelOpInterface> res =
      WasmSSALabelBranchingOpInterface::getTargetOpFromBlock(
          op->getBlock(), branchInterface.getExitLevel());
  return success(succeeded(res));
}

LogicalResult verifyConstantExpressionInterface(Operation *op) {
  Region &initializerRegion = op->getRegion(0);
  WalkResult resultState =
      initializerRegion.walk([&](Operation *currentOp) -> WalkResult {
        if (isa<ReturnOp>(currentOp))
          return WalkResult::advance();
        if (auto interfaceOp =
                dyn_cast<ConstantExprCheckOpInterface>(currentOp)) {
          if (interfaceOp.isValidInConstantExpr().succeeded())
            return WalkResult::advance();
        }
        op->emitError("expected a constant initializer for this operator, got ")
            << currentOp;
        return WalkResult::interrupt();
      });
  return success(!resultState.wasInterrupted());
}
} // namespace detail

llvm::FailureOr<WasmSSALabelLevelOpInterface>
WasmSSALabelBranchingOpInterface::getTargetOpFromBlock(::mlir::Block *block,
                                                     uint32_t breakLevel) {
  WasmSSALabelLevelOpInterface res{};
  for (size_t curLevel{0}; curLevel <= breakLevel; curLevel++) {
    res = dyn_cast_or_null<WasmSSALabelLevelOpInterface>(block->getParentOp());
    if (!res)
      return failure();
    block = res->getBlock();
  }
  return res;
}
} // namespace mlir::wasmssa
