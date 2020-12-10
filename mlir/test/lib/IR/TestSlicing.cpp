//===- TestSlicing.cpp - Testing slice functionality ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple testing pass for slicing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

/// Create a function with the same signature as the parent function of `op`
/// with name being the function name and a `suffix`.
static LogicalResult createBackwardSliceFunction(Operation *op,
                                                 StringRef suffix) {
  FuncOp parentFuncOp = op->getParentOfType<FuncOp>();
  OpBuilder builder(parentFuncOp);
  Location loc = op->getLoc();
  std::string clonedFuncOpName = parentFuncOp.getName().str() + suffix.str();
  FuncOp clonedFuncOp =
      builder.create<FuncOp>(loc, clonedFuncOpName, parentFuncOp.getType());
  BlockAndValueMapping mapper;
  builder.setInsertionPointToEnd(clonedFuncOp.addEntryBlock());
  for (auto arg : enumerate(parentFuncOp.getArguments()))
    mapper.map(arg.value(), clonedFuncOp.getArgument(arg.index()));
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(op, &slice);
  for (Operation *slicedOp : slice)
    builder.clone(*slicedOp, mapper);
  builder.create<ReturnOp>(loc);
  return success();
}

namespace {
/// Pass to test slice generated from slice analysis.
struct SliceAnalysisTestPass
    : public PassWrapper<SliceAnalysisTestPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  SliceAnalysisTestPass() = default;
  SliceAnalysisTestPass(const SliceAnalysisTestPass &) {}
};
} // namespace

void SliceAnalysisTestPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto funcOps = module.getOps<FuncOp>();
  unsigned opNum = 0;
  for (auto funcOp : funcOps) {
    // TODO: For now this is just looking for Linalg ops. It can be generalized
    // to look for other ops using flags.
    funcOp.walk([&](Operation *op) {
      if (!isa<linalg::LinalgOp>(op))
        return WalkResult::advance();
      std::string append =
          std::string("__backward_slice__") + std::to_string(opNum);
      createBackwardSliceFunction(op, append);
      opNum++;
      return WalkResult::advance();
    });
  }
}

namespace mlir {
void registerSliceAnalysisTestPass() {
  PassRegistration<SliceAnalysisTestPass> pass(
      "slice-analysis-test", "Test Slice analysis functionality.");
}
} // namespace mlir
