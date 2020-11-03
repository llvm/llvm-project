//===- AsyncRegionRewriter.cpp - Implementation of GPU async rewriters ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU dialect pattern rewriters that make GPU op
// within a region execute asynchronously.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/GPU/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
namespace {
class GpuAsyncRegionPass : public GpuAsyncRegionPassBase<GpuAsyncRegionPass> {
  struct Callback;
  void runOnFunction() override;
};
} // namespace

// Region walk callback which makes GPU ops implementing the AsyncOpInterface
// execute asynchronously.
struct GpuAsyncRegionPass::Callback {
  // If `op` implements the AsyncOpInterface, insert a `gpu.wait async` to
  // create a current token (unless it already exists), and 'thread' that token
  // through the `op` so that it executes asynchronously.
  //
  // If `op` is a terminator or an op with side-effects, insert a `gpu.wait` to
  // host-synchronize execution.
  WalkResult operator()(Operation *op) {
    if (isa<gpu::LaunchOp>(op))
      return op->emitOpError("replace with gpu.launch_func first");
    if (isa<gpu::WaitOp>(op))
      return op->emitOpError("unexpected pre-existing gpu.wait");
    builder.setInsertionPoint(op);
    if (auto asyncOp = dyn_cast<gpu::AsyncOpInterface>(op))
      return rewriteAsyncOp(asyncOp); // Replace GPU op with async version.
    if (!currentToken)
      return success();
    if (!op->hasTrait<OpTrait::IsTerminator>() &&
        MemoryEffectOpInterface::hasNoEffect(op))
      return success();
    // Insert host synchronization before terminator or op with side effects.
    currentToken = createWaitOp(op->getLoc(), Type(), {currentToken});
    return success();
  }

  // Replaces asyncOp with a clone that returns a token.
  LogicalResult rewriteAsyncOp(gpu::AsyncOpInterface asyncOp) {
    auto *op = asyncOp.getOperation();
    if (asyncOp.getAsyncToken())
      // TODO: Support ops that are already async.
      return op->emitOpError("is already async");
    if (op->getNumRegions() > 0)
      return op->emitOpError("regions are not supported");

    // If there is no current token, insert a `gpu.wait async` without
    // dependencies to create one.
    if (!currentToken)
      currentToken = createWaitOp(op->getLoc(), tokenType, {});
    asyncOp.addAsyncDependency(currentToken);

    // Clone the op to return a token in addition to the other results.
    SmallVector<Type, 1> resultTypes = {tokenType};
    resultTypes.reserve(1 + op->getNumResults());
    copy(op->getResultTypes(), std::back_inserter(resultTypes));
    auto *newOp = Operation::create(op->getLoc(), op->getName(), resultTypes,
                                    op->getOperands(), op->getMutableAttrDict(),
                                    op->getSuccessors());

    // Replace the op with the async clone.
    auto results = newOp->getResults();
    currentToken = results.front();
    builder.insert(newOp);
    op->replaceAllUsesWith(results.drop_front());
    op->erase();

    return success();
  }

  Value createWaitOp(Location loc, Type resultType, ValueRange operands) {
    return builder.create<gpu::WaitOp>(loc, resultType, operands).asyncToken();
  }

  OpBuilder builder;
  const Type tokenType = builder.getType<gpu::AsyncTokenType>();
  // The token that represents the current asynchronous dependency. It's valid
  // range starts with a `gpu.wait async` op, and ends with a `gpu.wait` op.
  // In between, each gpu::AsyncOpInterface depends on the current token and
  // produces the new one.
  Value currentToken = {};
};

// Replaces synchronous GPU ops in the op's region with asynchronous ones and
// inserts the necessary synchronization (as gpu.wait ops). Assumes sequential
// execution semantics and that no GPU ops are asynchronous yet.
void GpuAsyncRegionPass::runOnFunction() {
  Callback callback{OpBuilder(&getContext())};
  if (getFunction().getRegion().walk(callback).wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createGpuAsyncRegionPass() {
  return std::make_unique<GpuAsyncRegionPass>();
}
