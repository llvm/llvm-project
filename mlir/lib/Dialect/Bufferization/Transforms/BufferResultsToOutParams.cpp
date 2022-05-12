//===- BufferResultsToOutParams.cpp - Calling convention conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// Updates the func op and entry block.
//
// Any args appended to the entry block are added to `appendedEntryArgs`.
static void updateFuncOp(func::FuncOp func,
                         SmallVectorImpl<BlockArgument> &appendedEntryArgs) {
  auto functionType = func.getFunctionType();

  // Collect information about the results will become appended arguments.
  SmallVector<Type, 6> erasedResultTypes;
  BitVector erasedResultIndices(functionType.getNumResults());
  for (const auto &resultType : llvm::enumerate(functionType.getResults())) {
    if (resultType.value().isa<BaseMemRefType>()) {
      erasedResultIndices.set(resultType.index());
      erasedResultTypes.push_back(resultType.value());
    }
  }

  // Add the new arguments to the function type.
  auto newArgTypes = llvm::to_vector<6>(
      llvm::concat<const Type>(functionType.getInputs(), erasedResultTypes));
  auto newFunctionType = FunctionType::get(func.getContext(), newArgTypes,
                                           functionType.getResults());
  func.setType(newFunctionType);

  // Transfer the result attributes to arg attributes.
  auto erasedIndicesIt = erasedResultIndices.set_bits_begin();
  for (int i = 0, e = erasedResultTypes.size(); i < e; ++i, ++erasedIndicesIt) {
    func.setArgAttrs(functionType.getNumInputs() + i,
                     func.getResultAttrs(*erasedIndicesIt));
  }

  // Erase the results.
  func.eraseResults(erasedResultIndices);

  // Add the new arguments to the entry block if the function is not external.
  if (func.isExternal())
    return;
  Location loc = func.getLoc();
  for (Type type : erasedResultTypes)
    appendedEntryArgs.push_back(func.front().addArgument(type, loc));
}

// Updates all ReturnOps in the scope of the given func::FuncOp by either
// keeping them as return values or copying the associated buffer contents into
// the given out-params.
static void updateReturnOps(func::FuncOp func,
                            ArrayRef<BlockArgument> appendedEntryArgs) {
  func.walk([&](func::ReturnOp op) {
    SmallVector<Value, 6> copyIntoOutParams;
    SmallVector<Value, 6> keepAsReturnOperands;
    for (Value operand : op.getOperands()) {
      if (operand.getType().isa<BaseMemRefType>())
        copyIntoOutParams.push_back(operand);
      else
        keepAsReturnOperands.push_back(operand);
    }
    OpBuilder builder(op);
    for (auto t : llvm::zip(copyIntoOutParams, appendedEntryArgs))
      builder.create<memref::CopyOp>(op.getLoc(), std::get<0>(t),
                                     std::get<1>(t));
    builder.create<func::ReturnOp>(op.getLoc(), keepAsReturnOperands);
    op.erase();
  });
}

// Updates all CallOps in the scope of the given ModuleOp by allocating
// temporary buffers for newly introduced out params.
static LogicalResult updateCalls(ModuleOp module) {
  bool didFail = false;
  module.walk([&](func::CallOp op) {
    SmallVector<Value, 6> replaceWithNewCallResults;
    SmallVector<Value, 6> replaceWithOutParams;
    for (OpResult result : op.getResults()) {
      if (result.getType().isa<BaseMemRefType>())
        replaceWithOutParams.push_back(result);
      else
        replaceWithNewCallResults.push_back(result);
    }
    SmallVector<Value, 6> outParams;
    OpBuilder builder(op);
    for (Value memref : replaceWithOutParams) {
      if (!memref.getType().cast<BaseMemRefType>().hasStaticShape()) {
        op.emitError()
            << "cannot create out param for dynamically shaped result";
        didFail = true;
        return;
      }
      Value outParam = builder.create<memref::AllocOp>(
          op.getLoc(), memref.getType().cast<MemRefType>());
      memref.replaceAllUsesWith(outParam);
      outParams.push_back(outParam);
    }

    auto newOperands = llvm::to_vector<6>(op.getOperands());
    newOperands.append(outParams.begin(), outParams.end());
    auto newResultTypes = llvm::to_vector<6>(llvm::map_range(
        replaceWithNewCallResults, [](Value v) { return v.getType(); }));
    auto newCall = builder.create<func::CallOp>(op.getLoc(), op.getCalleeAttr(),
                                                newResultTypes, newOperands);
    for (auto t : llvm::zip(replaceWithNewCallResults, newCall.getResults()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));
    op.erase();
  });

  return failure(didFail);
}

LogicalResult
mlir::bufferization::promoteBufferResultsToOutParams(ModuleOp module) {
  for (auto func : module.getOps<func::FuncOp>()) {
    SmallVector<BlockArgument, 6> appendedEntryArgs;
    updateFuncOp(func, appendedEntryArgs);
    if (func.isExternal())
      continue;
    updateReturnOps(func, appendedEntryArgs);
  }
  if (failed(updateCalls(module)))
    return failure();
  return success();
}

namespace {
struct BufferResultsToOutParamsPass
    : BufferResultsToOutParamsBase<BufferResultsToOutParamsPass> {
  void runOnOperation() override {
    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation())))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::bufferization::createBufferResultsToOutParamsPass() {
  return std::make_unique<BufferResultsToOutParamsPass>();
}
