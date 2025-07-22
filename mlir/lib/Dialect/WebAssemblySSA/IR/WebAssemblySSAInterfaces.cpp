//===- WebAssemblySSAInterfaces.cpp - WebAssemblySSA Interfaces -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the WebAssemblySSA dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAInterfaces.h"
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSA.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

namespace mlir::wasmssa {
#include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAInterfaces.cpp.inc"

namespace detail {
LogicalResult verifyWasmSSALabelBranchingInterface(Operation *op) {
  auto branchInterface = dyn_cast<WasmSSALabelBranchingInterface>(op);
  llvm::FailureOr<WasmSSALabelLevelInterface> res =
      WasmSSALabelBranchingInterface::getTargetOpFromBlock(
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
                dyn_cast<WasmSSAConstantExprCheckInterface>(currentOp)) {
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

llvm::FailureOr<WasmSSALabelLevelInterface>
WasmSSALabelBranchingInterface::getTargetOpFromBlock(::mlir::Block *block,
                                                     uint32_t breakLevel) {
  WasmSSALabelLevelInterface res{};
  for (size_t curLevel{0}; curLevel <= breakLevel; curLevel++) {
    res = dyn_cast_or_null<WasmSSALabelLevelInterface>(block->getParentOp());
    if (!res)
      return failure();
    block = res->getBlock();
  }
  return res;
}
} // namespace mlir::wasmssa
