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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

/// Create a function with the same signature as the parent function of `op`
/// with name being the function name and a `suffix`.
static LogicalResult createBackwardSliceFunction(Operation *op,
                                                 StringRef suffix,
                                                 bool omitBlockArguments) {
  func::FuncOp parentFuncOp = op->getParentOfType<func::FuncOp>();
  OpBuilder builder(parentFuncOp);
  Location loc = op->getLoc();
  std::string clonedFuncOpName = parentFuncOp.getName().str() + suffix.str();
  func::FuncOp clonedFuncOp = builder.create<func::FuncOp>(
      loc, clonedFuncOpName, parentFuncOp.getFunctionType());
  IRMapping mapper;
  builder.setInsertionPointToEnd(clonedFuncOp.addEntryBlock());
  for (const auto &arg : enumerate(parentFuncOp.getArguments()))
    mapper.map(arg.value(), clonedFuncOp.getArgument(arg.index()));
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = omitBlockArguments;
  // TODO: Make this default.
  options.omitUsesFromAbove = false;
  LogicalResult result = getBackwardSlice(op, &slice, options);
  assert(result.succeeded() && "expected a backward slice");
  (void)result;
  for (Operation *slicedOp : slice)
    builder.clone(*slicedOp, mapper);
  builder.create<func::ReturnOp>(loc);
  return success();
}

namespace {
/// Pass to test slice generated from slice analysis.
struct SliceAnalysisTestPass
    : public PassWrapper<SliceAnalysisTestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SliceAnalysisTestPass)

  StringRef getArgument() const final { return "slice-analysis-test"; }
  StringRef getDescription() const final {
    return "Test Slice analysis functionality.";
  }

  Option<bool> omitBlockArguments{
      *this, "omit-block-arguments",
      llvm::cl::desc("Test Slice analysis with multiple blocks but slice "
                     "omiting block arguments"),
      llvm::cl::init(true)};

  void runOnOperation() override;
  SliceAnalysisTestPass() = default;
  SliceAnalysisTestPass(const SliceAnalysisTestPass &) {}
};
} // namespace

void SliceAnalysisTestPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto funcOps = module.getOps<func::FuncOp>();
  unsigned opNum = 0;
  for (auto funcOp : funcOps) {
    // TODO: For now this is just looking for Linalg ops. It can be generalized
    // to look for other ops using flags.
    funcOp.walk([&](Operation *op) {
      if (!isa<linalg::LinalgOp>(op))
        return WalkResult::advance();
      std::string append =
          std::string("__backward_slice__") + std::to_string(opNum);
      (void)createBackwardSliceFunction(op, append, omitBlockArguments);
      opNum++;
      return WalkResult::advance();
    });
  }
}

namespace mlir {
void registerSliceAnalysisTestPass() {
  PassRegistration<SliceAnalysisTestPass>();
}
} // namespace mlir
