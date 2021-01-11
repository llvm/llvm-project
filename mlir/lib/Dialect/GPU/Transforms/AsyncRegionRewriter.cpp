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
#include "mlir/Dialect/Async/IR/Async.h"
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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace {
class GpuAsyncRegionPass : public GpuAsyncRegionPassBase<GpuAsyncRegionPass> {
  struct ThreadTokenCallback;
  struct DeferWaitCallback;
  void runOnFunction() override;
};
} // namespace

static bool isTerminator(Operation *op) { return !op->isKnownNonTerminator(); }
static bool hasSideEffects(Operation *op) {
  return !MemoryEffectOpInterface::hasNoEffect(op);
}

// Region walk callback which makes GPU ops implementing the AsyncOpInterface
// execute asynchronously.
struct GpuAsyncRegionPass::ThreadTokenCallback {
  ThreadTokenCallback(MLIRContext &context) : builder(&context) {}

  // If `op` implements the AsyncOpInterface, insert a `gpu.wait async` to
  // create a current token (unless it already exists), and 'thread' that token
  // through the `op` so that it executes asynchronously.
  //
  // If `op` is a terminator or an op with side-effects, insert a `gpu.wait` to
  // host-synchronize execution. A `!gpu.async.token` will therefore only be
  // used inside of its block and GPU execution will always synchronize with
  // the host at block boundaries.
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
    // Insert host synchronization before terminator or op with side effects.
    if (isTerminator(op) || hasSideEffects(op))
      currentToken = createWaitOp(op->getLoc(), Type(), {currentToken});
    return success();
  }

private:
  // Replaces asyncOp with a clone that returns a token.
  LogicalResult rewriteAsyncOp(gpu::AsyncOpInterface asyncOp) {
    auto *op = asyncOp.getOperation();
    if (asyncOp.getAsyncToken())
      // TODO: Support ops that are already async.
      return op->emitOpError("is already async");
    if (op->getNumRegions() > 0)
      return op->emitOpError("regions are not supported");

    auto tokenType = builder.getType<gpu::AsyncTokenType>();

    // If there is no current token, insert a `gpu.wait async` without
    // dependencies to create one.
    if (!currentToken)
      currentToken = createWaitOp(op->getLoc(), tokenType, {});
    asyncOp.addAsyncDependency(currentToken);

    // Clone the op to return a token in addition to the other results.
    SmallVector<Type, 1> resultTypes;
    resultTypes.reserve(1 + op->getNumResults());
    copy(op->getResultTypes(), std::back_inserter(resultTypes));
    resultTypes.push_back(tokenType);
    auto *newOp = Operation::create(op->getLoc(), op->getName(), resultTypes,
                                    op->getOperands(), op->getAttrDictionary(),
                                    op->getSuccessors());

    // Replace the op with the async clone.
    auto results = newOp->getResults();
    currentToken = results.back();
    builder.insert(newOp);
    op->replaceAllUsesWith(results.drop_back());
    op->erase();

    return success();
  }

  Value createWaitOp(Location loc, Type resultType, ValueRange operands) {
    return builder.create<gpu::WaitOp>(loc, resultType, operands).asyncToken();
  }

  OpBuilder builder;

  // The token that represents the current asynchronous dependency. It's valid
  // range starts with a `gpu.wait async` op, and ends with a `gpu.wait` op.
  // In between, each gpu::AsyncOpInterface depends on the current token and
  // produces the new one.
  Value currentToken = {};
};

// Callback for `async.execute` ops which tries to push the contained
// synchronous `gpu.wait` op to the dependencies of the `async.execute`.
struct GpuAsyncRegionPass::DeferWaitCallback {
  // If the `executeOp`s token is used only in `async.execute` or `async.await`
  // ops, add the region's last `gpu.wait` op to the worklist if it is
  // synchronous and is the last op with side effects.
  void operator()(async::ExecuteOp executeOp) {
    if (!areAllUsersExecuteOrAwait(executeOp.token()))
      return;
    // async.execute's region is currently restricted to one block.
    for (auto &op : llvm::reverse(executeOp.getBody()->without_terminator())) {
      if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
        if (!waitOp.asyncToken())
          worklist.push_back(waitOp);
        return;
      }
      if (hasSideEffects(&op))
        return;
    }
  }

  // The destructor performs the actual rewrite work.
  ~DeferWaitCallback() {
    for (size_t i = 0; i < worklist.size(); ++i) {
      auto waitOp = worklist[i];
      auto executeOp = waitOp->getParentOfType<async::ExecuteOp>();
      auto numDependencies = waitOp.asyncDependencies().size();

      // Erase `gpu.wait` and return async dependencies from region instead.
      auto &yieldOp = executeOp.getBody()->getOperations().back();
      yieldOp.insertOperands(yieldOp.getNumOperands(),
                             waitOp.asyncDependencies());
      waitOp.erase();
      auto asyncTokens = addAsyncTokenResults(executeOp, numDependencies);

      // Add the async dependency to each user of the `async.execute` token.
      for (Operation *user : executeOp.token().getUsers())
        addAsyncDependencyAfter(asyncTokens, user);
    }
  }

private:
  // Append `count` `!async.value<!gpu.async.token>` results to `executeOp`.
  static ValueRange addAsyncTokenResults(async::ExecuteOp &executeOp,
                                         unsigned count) {
    auto numResults = executeOp.getNumResults() + count;

    // Construct new result type list with `count` additional types.
    SmallVector<Type, 2> resultTypes;
    resultTypes.reserve(numResults);
    transform(executeOp.getResultTypes(), std::back_inserter(resultTypes),
              [](Type type) {
                // Extract value type from !async.value.
                if (auto valueType = type.dyn_cast<async::ValueType>())
                  return valueType.getValueType();
                assert(type.isa<async::TokenType>() && "expected token type");
                return type;
              });
    OpBuilder builder(executeOp);
    auto tokenType = builder.getType<gpu::AsyncTokenType>();
    resultTypes.resize(numResults, tokenType);

    // Clone executeOp with the extra `!gpu.async.token` results.
    auto newOp = builder.create<async::ExecuteOp>(
        executeOp.getLoc(), TypeRange{resultTypes}.drop_front() /*drop token*/,
        executeOp.dependencies(), executeOp.operands());
    BlockAndValueMapping mapper;
    newOp.getRegion().getBlocks().clear();
    executeOp.getRegion().cloneInto(&newOp.getRegion(), mapper);

    // Replace executeOp with cloned one.
    executeOp.getOperation()->replaceAllUsesWith(
        newOp.getResults().drop_back(count));
    executeOp.erase();
    executeOp = newOp;

    // Return the new result values.
    return executeOp.getResults().take_back(count);
  }

  // Returns whether all token users are either 'async.execute' or 'async.await'
  // ops. This is used as a requirement for pushing 'gpu.wait' ops from a
  // 'async.execute' body to it's users. Specifically, we do not allow
  // terminator users, because it could mean that the `async.execute` is inside
  // control flow code.
  static bool areAllUsersExecuteOrAwait(Value token) {
    return llvm::all_of(token.getUsers(), [](Operation *user) {
      return isa<async::ExecuteOp, async::AwaitOp>(user);
    });
  }

  // Add the `asyncToken` as dependency as needed after `op`.
  void addAsyncDependencyAfter(ValueRange asyncTokens, Operation *op) {
    OpBuilder builder(op->getContext());
    auto loc = op->getLoc();

    Block::iterator it;
    SmallVector<Value, 1> tokens;
    tokens.reserve(asyncTokens.size());
    TypeSwitch<Operation *>(op)
        .Case<async::AwaitOp>([&](auto awaitOp) {
          // Add async.await ops to wait for the !gpu.async.tokens.
          builder.setInsertionPointAfter(op);
          for (auto asyncToken : asyncTokens)
            tokens.push_back(
                builder.create<async::AwaitOp>(loc, asyncToken).result());
          // Set `it` after the inserted async.await ops.
          it = builder.getInsertionPoint();
        })
        .Case<async::ExecuteOp>([&](auto executeOp) {
          // Set `it` to the beginning of the region and add asyncTokens to the
          // async.execute operands.
          it = executeOp.getBody()->begin();
          executeOp.operandsMutable().append(asyncTokens);
          SmallVector<Type, 1> tokenTypes(
              asyncTokens.size(), builder.getType<gpu::AsyncTokenType>());
          copy(executeOp.getBody()->addArguments(tokenTypes),
               std::back_inserter(tokens));
        });

    // Advance `it` to terminator or op with side-effects.
    it = std::find_if(it, Block::iterator(), [](Operation &op) {
      return isTerminator(&op) || hasSideEffects(&op);
    });

    // If `op` implements the AsyncOpInterface, add `token` to the list of async
    // dependencies.
    if (auto asyncOp = dyn_cast<gpu::AsyncOpInterface>(*it)) {
      for (auto token : tokens)
        asyncOp.addAsyncDependency(token);
      return;
    }

    // Otherwise, insert a gpu.wait before 'it'.
    builder.setInsertionPoint(it->getBlock(), it);
    auto waitOp = builder.create<gpu::WaitOp>(loc, Type{}, tokens);

    // If the new waitOp is at the end of an async.execute region, add it to the
    // worklist. 'operator()(executeOp)' would do the same, but this is faster.
    auto executeOp = dyn_cast<async::ExecuteOp>(it->getParentOp());
    if (executeOp && areAllUsersExecuteOrAwait(executeOp.token()) &&
        !it->getNextNode())
      worklist.push_back(waitOp);
  }

  SmallVector<gpu::WaitOp, 8> worklist;
};

// Replaces synchronous GPU ops in the op's region with asynchronous ones and
// inserts the necessary synchronization (as gpu.wait ops). Assumes sequential
// execution semantics and that no GPU ops are asynchronous yet.
void GpuAsyncRegionPass::runOnFunction() {
  if (getFunction()
          .getRegion()
          .walk(ThreadTokenCallback(getContext()))
          .wasInterrupted())
    return signalPassFailure();

  // Collect gpu.wait ops that we can move out of async.execute regions.
  getFunction().getRegion().walk(DeferWaitCallback());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createGpuAsyncRegionPass() {
  return std::make_unique<GpuAsyncRegionPass>();
}
