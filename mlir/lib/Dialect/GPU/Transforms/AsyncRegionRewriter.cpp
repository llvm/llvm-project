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

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_GPUASYNCREGIONPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class GpuAsyncRegionPass
    : public impl::GpuAsyncRegionPassBase<GpuAsyncRegionPass> {
  struct ThreadTokenCallback;
  struct DeferWaitCallback;
  struct SingleTokenUseCallback;
  void runOnOperation() override;
};
} // namespace

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}
static bool hasSideEffects(Operation *op) { return !isMemoryEffectFree(op); }

// Region walk callback which makes GPU ops implementing the AsyncOpInterface
// execute asynchronously.
struct GpuAsyncRegionPass::ThreadTokenCallback {
  ThreadTokenCallback(MLIRContext &context) : builder(&context) {}

  WalkResult operator()(Block *block) {
    for (Operation &op : make_early_inc_range(*block)) {
      if (failed(visit(&op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }

private:
  // If `op` implements the AsyncOpInterface, insert a `gpu.wait async` to
  // create a current token (unless it already exists), and 'thread' that token
  // through the `op` so that it executes asynchronously.
  //
  // If `op` is a terminator or an op with side-effects, insert a `gpu.wait` to
  // host-synchronize execution. A `!gpu.async.token` will therefore only be
  // used inside of its block and GPU execution will always synchronize with
  // the host at block boundaries.
  LogicalResult visit(Operation *op) {
    if (isa<gpu::LaunchOp>(op))
      return op->emitOpError("replace with gpu.launch_func first");
    if (auto waitOp = llvm::dyn_cast<gpu::WaitOp>(op)) {
      if (currentToken)
        waitOp.addAsyncDependency(currentToken);
      currentToken = waitOp.getAsyncToken();
      return success();
    }
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

  // Replaces asyncOp with a clone that returns a token.
  LogicalResult rewriteAsyncOp(gpu::AsyncOpInterface asyncOp) {
    auto *op = asyncOp.getOperation();
    auto tokenType = builder.getType<gpu::AsyncTokenType>();

    // If there is no current token, insert a `gpu.wait async` without
    // dependencies to create one.
    if (!currentToken)
      currentToken = createWaitOp(op->getLoc(), tokenType, {});
    asyncOp.addAsyncDependency(currentToken);

    // Return early if op returns a token already.
    currentToken = asyncOp.getAsyncToken();
    if (currentToken)
      return success();

    // Clone the op to return a token in addition to the other results.
    SmallVector<Type, 1> resultTypes;
    resultTypes.reserve(1 + op->getNumResults());
    copy(op->getResultTypes(), std::back_inserter(resultTypes));
    resultTypes.push_back(tokenType);
    auto *newOp = Operation::create(op->getLoc(), op->getName(), resultTypes,
                                    op->getOperands(), op->getAttrDictionary(),
                                    op->getPropertiesStorage(),
                                    op->getSuccessors(), op->getNumRegions());

    // Clone regions into new op.
    IRMapping mapping;
    for (auto pair : llvm::zip_first(op->getRegions(), newOp->getRegions()))
      std::get<0>(pair).cloneInto(&std::get<1>(pair), mapping);

    // Replace the op with the async clone.
    auto results = newOp->getResults();
    currentToken = results.back();
    builder.insert(newOp);
    op->replaceAllUsesWith(results.drop_back());
    op->erase();

    return success();
  }

  Value createWaitOp(Location loc, Type resultType, ValueRange operands) {
    return builder.create<gpu::WaitOp>(loc, resultType, operands)
        .getAsyncToken();
  }

  OpBuilder builder;

  // The token that represents the current asynchronous dependency. It's valid
  // range starts with a `gpu.wait async` op, and ends with a `gpu.wait` op.
  // In between, each gpu::AsyncOpInterface depends on the current token and
  // produces the new one.
  Value currentToken = {};
};

/// Erases `executeOp` and returns a clone with additional `results`.
async::ExecuteOp addExecuteResults(async::ExecuteOp executeOp,
                                   ValueRange results) {
  // Add values to async.yield op.
  Operation *yieldOp = executeOp.getBody()->getTerminator();
  yieldOp->insertOperands(yieldOp->getNumOperands(), results);

  // Construct new result type list with additional types.
  SmallVector<Type, 2> resultTypes;
  resultTypes.reserve(executeOp.getNumResults() + results.size());
  transform(executeOp.getResultTypes(), std::back_inserter(resultTypes),
            [](Type type) {
              // Extract value type from !async.value.
              if (auto valueType = dyn_cast<async::ValueType>(type))
                return valueType.getValueType();
              assert(isa<async::TokenType>(type) && "expected token type");
              return type;
            });
  transform(results, std::back_inserter(resultTypes),
            [](Value value) { return value.getType(); });

  // Clone executeOp with the extra results.
  OpBuilder builder(executeOp);
  auto newOp = builder.create<async::ExecuteOp>(
      executeOp.getLoc(), TypeRange{resultTypes}.drop_front() /*drop token*/,
      executeOp.getDependencies(), executeOp.getBodyOperands());
  IRMapping mapper;
  newOp.getRegion().getBlocks().clear();
  executeOp.getRegion().cloneInto(&newOp.getRegion(), mapper);

  // Replace executeOp with cloned one.
  executeOp.getOperation()->replaceAllUsesWith(
      newOp.getResults().drop_back(results.size()));
  executeOp.erase();

  return newOp;
}

// Callback for `async.execute` ops which tries to push the contained
// synchronous `gpu.wait` op to the dependencies of the `async.execute`.
struct GpuAsyncRegionPass::DeferWaitCallback {
  // If the `executeOp`s token is used only in `async.execute` or `async.await`
  // ops, add the region's last `gpu.wait` op to the worklist if it is
  // synchronous and is the last op with side effects.
  void operator()(async::ExecuteOp executeOp) {
    if (!areAllUsersExecuteOrAwait(executeOp.getToken()))
      return;
    // async.execute's region is currently restricted to one block.
    for (auto &op : llvm::reverse(executeOp.getBody()->without_terminator())) {
      if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
        if (!waitOp.getAsyncToken())
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

      // Erase `gpu.wait` and return async dependencies from execute op instead.
      SmallVector<Value, 4> dependencies = waitOp.getAsyncDependencies();
      waitOp.erase();
      executeOp = addExecuteResults(executeOp, dependencies);

      // Add the async dependency to each user of the `async.execute` token.
      auto asyncTokens = executeOp.getResults().take_back(dependencies.size());
      SmallVector<Operation *, 4> users(executeOp.getToken().user_begin(),
                                        executeOp.getToken().user_end());
      for (Operation *user : users)
        addAsyncDependencyAfter(asyncTokens, user);
    }
  }

private:
  // Returns whether all token users are either 'async.execute' or 'async.await'
  // ops. This is used as a requirement for pushing 'gpu.wait' ops from a
  // 'async.execute' body to it's users. Specifically, we do not allow
  // terminator users, because it could mean that the `async.execute` is inside
  // control flow code.
  static bool areAllUsersExecuteOrAwait(Value token) {
    return !token.use_empty() &&
           llvm::all_of(token.getUsers(), [](Operation *user) {
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
                builder.create<async::AwaitOp>(loc, asyncToken).getResult());
          // Set `it` after the inserted async.await ops.
          it = builder.getInsertionPoint();
        })
        .Case<async::ExecuteOp>([&](auto executeOp) {
          // Set `it` to the beginning of the region and add asyncTokens to the
          // async.execute operands.
          it = executeOp.getBody()->begin();
          executeOp.getBodyOperandsMutable().append(asyncTokens);
          SmallVector<Type, 1> tokenTypes(
              asyncTokens.size(), builder.getType<gpu::AsyncTokenType>());
          SmallVector<Location, 1> tokenLocs(asyncTokens.size(),
                                             executeOp.getLoc());
          copy(executeOp.getBody()->addArguments(tokenTypes, tokenLocs),
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
    if (executeOp && areAllUsersExecuteOrAwait(executeOp.getToken()) &&
        !it->getNextNode())
      worklist.push_back(waitOp);
  }

  SmallVector<gpu::WaitOp, 8> worklist;
};

// Callback for `async.execute` ops which repeats !gpu.async.token results
// so that each of them is only used once.
struct GpuAsyncRegionPass::SingleTokenUseCallback {
  void operator()(async::ExecuteOp executeOp) {
    // Extract !gpu.async.token results which have multiple uses.
    auto multiUseResults = llvm::make_filter_range(
        executeOp.getBodyResults(), [](OpResult result) {
          if (result.use_empty() || result.hasOneUse())
            return false;
          auto valueType = dyn_cast<async::ValueType>(result.getType());
          return valueType &&
                 isa<gpu::AsyncTokenType>(valueType.getValueType());
        });
    if (multiUseResults.empty())
      return;

    // Indices within !async.execute results (i.e. without the async.token).
    SmallVector<int, 4> indices;
    transform(multiUseResults, std::back_inserter(indices),
              [](OpResult result) {
                return result.getResultNumber() - 1; // Index without token.
              });

    for (auto index : indices) {
      assert(!executeOp.getBodyResults()[index].getUses().empty());
      // Repeat async.yield token result, one for each use after the first one.
      auto uses = llvm::drop_begin(executeOp.getBodyResults()[index].getUses());
      auto count = std::distance(uses.begin(), uses.end());
      auto yieldOp = cast<async::YieldOp>(executeOp.getBody()->getTerminator());
      SmallVector<Value, 4> operands(count, yieldOp.getOperand(index));
      executeOp = addExecuteResults(executeOp, operands);
      // Update 'uses' to refer to the new executeOp.
      uses = llvm::drop_begin(executeOp.getBodyResults()[index].getUses());
      auto results = executeOp.getBodyResults().take_back(count);
      for (auto pair : llvm::zip(uses, results))
        std::get<0>(pair).set(std::get<1>(pair));
    }
  }
};

// Replaces synchronous GPU ops in the op's region with asynchronous ones and
// inserts the necessary synchronization (as gpu.wait ops). Assumes sequential
// execution semantics and that no GPU ops are asynchronous yet.
void GpuAsyncRegionPass::runOnOperation() {
  if (getOperation()->walk(ThreadTokenCallback(getContext())).wasInterrupted())
    return signalPassFailure();

  // Collect gpu.wait ops that we can move out of async.execute regions.
  getOperation().getRegion().walk(DeferWaitCallback());
  // Makes each !gpu.async.token returned from async.execute op have single use.
  getOperation().getRegion().walk(SingleTokenUseCallback());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createGpuAsyncRegionPass() {
  return std::make_unique<GpuAsyncRegionPass>();
}
