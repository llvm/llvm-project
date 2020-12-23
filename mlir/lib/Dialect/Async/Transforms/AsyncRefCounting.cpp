//===- AsyncRefCounting.cpp - Implementation of Async Ref Counting --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements automatic reference counting for Async dialect data
// types.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-ref-counting"

namespace {

class AsyncRefCountingPass : public AsyncRefCountingBase<AsyncRefCountingPass> {
public:
  AsyncRefCountingPass() = default;
  void runOnFunction() override;

private:
  /// Adds an automatic reference counting to the `value`.
  ///
  /// All values are semantically created with a reference count of +1 and it is
  /// the responsibility of the last async value user to drop reference count.
  ///
  /// Async values created when:
  ///   1. Operation returns async result (e.g. the result of an
  ///      `async.execute`).
  ///   2. Async value passed in as a block argument.
  ///
  /// To implement automatic reference counting, we must insert a +1 reference
  /// before each `async.execute` operation using the value, and drop it after
  /// the last use inside the async body region (we currently drop the reference
  /// before the `async.yield` terminator).
  ///
  /// Automatic reference counting algorithm outline:
  ///
  /// 1. `ReturnLike` operations forward the reference counted values without
  ///     modifying the reference count.
  ///
  /// 2. Use liveness analysis to find blocks in the CFG where the lifetime of
  ///    reference counted values ends, and insert `drop_ref` operations after
  ///    the last use of the value.
  ///
  /// 3. Insert `add_ref` before the `async.execute` operation capturing the
  ///    value, and pairing `drop_ref` before the async body region terminator,
  ///    to release the captured reference counted value when execution
  ///    completes.
  ///
  /// 4. If the reference counted value is passed only to some of the block
  ///    successors, insert `drop_ref` operations in the beginning of the blocks
  ///    that do not have reference counted value uses.
  ///
  ///
  /// Example:
  ///
  ///   %token = ...
  ///   async.execute {
  ///     async.await %token : !async.token   // await #1
  ///     async.yield
  ///   }
  ///   async.await %token : !async.token     // await #2
  ///
  /// Based on the liveness analysis await #2 is the last use of the %token,
  /// however the execution of the async region can be delayed, and to guarantee
  /// that the %token is still alive when await #1 executes we need to
  /// explicitly extend its lifetime using `add_ref` operation.
  ///
  /// After automatic reference counting:
  ///
  ///   %token = ...
  ///
  ///   // Make sure that %token is alive inside async.execute.
  ///   async.add_ref %token {count = 1 : i32} : !async.token
  ///
  ///   async.execute {
  ///     async.await %token : !async.token   // await #1
  ///
  ///     // Drop the extra reference added to keep %token alive.
  ///     async.drop_ref %token {count = 1 : i32} : !async.token
  ///
  ///     async.yied
  ///   }
  ///   async.await %token : !async.token     // await #2
  ///
  ///   // Drop the reference after the last use of %token.
  ///   async.drop_ref %token {count = 1 : i32} : !async.token
  ///
  LogicalResult addAutomaticRefCounting(Value value);
};

} // namespace

LogicalResult AsyncRefCountingPass::addAutomaticRefCounting(Value value) {
  MLIRContext *ctx = value.getContext();
  OpBuilder builder(ctx);

  // Set inserton point after the operation producing a value, or at the
  // beginning of the block if the value defined by the block argument.
  if (Operation *op = value.getDefiningOp())
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPointToStart(value.getParentBlock());

  Location loc = value.getLoc();
  auto i32 = IntegerType::get(ctx, 32);

  // Drop the reference count immediately if the value has no uses.
  if (value.getUses().empty()) {
    builder.create<DropRefOp>(loc, value, IntegerAttr::get(i32, 1));
    return success();
  }

  // Use liveness analysis to find the placement of `drop_ref`operation.
  auto liveness = getAnalysis<Liveness>();

  // We analyse only the blocks of the region that defines the `value`, and do
  // not check nested blocks attached to operations.
  //
  // By analyzing only the `definingRegion` CFG we potentially loose an
  // opportunity to drop the reference count earlier and can extend the lifetime
  // of reference counted value longer then it is really required.
  //
  // We also assume that all nested regions finish their execution before the
  // completion of the owner operation. The only exception to this rule is
  // `async.execute` operation, which is handled explicitly below.
  Region *definingRegion = value.getParentRegion();

  // ------------------------------------------------------------------------ //
  // Find blocks where the `value` dies: the value is in `liveIn` set and not
  // in the `liveOut` set. We place `drop_ref` immediately after the last use
  // of the `value` in such regions.
  // ------------------------------------------------------------------------ //

  // Last users of the `value` inside all blocks where the value dies.
  llvm::SmallSet<Operation *, 4> lastUsers;

  for (Block &block : definingRegion->getBlocks()) {
    const LivenessBlockInfo *blockLiveness = liveness.getLiveness(&block);

    // Value in live input set or was defined in the block.
    bool liveIn = blockLiveness->isLiveIn(value) ||
                  blockLiveness->getBlock() == value.getParentBlock();
    if (!liveIn)
      continue;

    // Value is in the live out set.
    bool liveOut = blockLiveness->isLiveOut(value);
    if (liveOut)
      continue;

    // We proved that `value` dies in the `block`. Now find the last use of the
    // `value` inside the `block`.

    // Find any user of the `value` inside the block (including uses in nested
    // regions attached to the operations in the block).
    Operation *userInTheBlock = nullptr;
    for (Operation *user : value.getUsers()) {
      userInTheBlock = block.findAncestorOpInBlock(*user);
      if (userInTheBlock)
        break;
    }

    // Values with zero users handled explicitly in the beginning, if the value
    // is in live out set it must have at least one use in the block.
    assert(userInTheBlock && "value must have a user in the block");

    // Find the last user of the `value` in the block;
    Operation *lastUser = blockLiveness->getEndOperation(value, userInTheBlock);
    assert(lastUsers.count(lastUser) == 0 && "last users must be unique");
    lastUsers.insert(lastUser);
  }

  // Process all the last users of the `value` inside each block where the value
  // dies.
  for (Operation *lastUser : lastUsers) {
    // Return like operations forward reference count.
    if (lastUser->hasTrait<OpTrait::ReturnLike>())
      continue;

    // We can't currently handle other types of terminators.
    if (lastUser->hasTrait<OpTrait::IsTerminator>())
      return lastUser->emitError() << "async reference counting can't handle "
                                      "terminators that are not ReturnLike";

    // Add a drop_ref immediately after the last user.
    builder.setInsertionPointAfter(lastUser);
    builder.create<DropRefOp>(loc, value, IntegerAttr::get(i32, 1));
  }

  // ------------------------------------------------------------------------ //
  // Find blocks where the `value` is in `liveOut` set, however it is not in
  // the `liveIn` set of all successors. If the `value` is not in the successor
  // `liveIn` set, we add a `drop_ref` to the beginning of it.
  // ------------------------------------------------------------------------ //

  // Successors that we'll need a `drop_ref` for the `value`.
  llvm::SmallSet<Block *, 4> dropRefSuccessors;

  for (Block &block : definingRegion->getBlocks()) {
    const LivenessBlockInfo *blockLiveness = liveness.getLiveness(&block);

    // Skip the block if value is not in the `liveOut` set.
    if (!blockLiveness->isLiveOut(value))
      continue;

    // Find successors that do not have `value` in the `liveIn` set.
    for (Block *successor : block.getSuccessors()) {
      const LivenessBlockInfo *succLiveness = liveness.getLiveness(successor);

      if (!succLiveness->isLiveIn(value))
        dropRefSuccessors.insert(successor);
    }
  }

  // Drop reference in all successor blocks that do not have the `value` in
  // their `liveIn` set.
  for (Block *dropRefSuccessor : dropRefSuccessors) {
    builder.setInsertionPointToStart(dropRefSuccessor);
    builder.create<DropRefOp>(loc, value, IntegerAttr::get(i32, 1));
  }

  // ------------------------------------------------------------------------ //
  // Find all `async.execute` operation that take `value` as an operand
  // (dependency token or async value), or capture implicitly by the nested
  // region. Each `async.execute` operation will require `add_ref` operation
  // to keep all captured values alive until it will finish its execution.
  // ------------------------------------------------------------------------ //

  llvm::SmallSet<ExecuteOp, 4> executeOperations;

  auto trackAsyncExecute = [&](Operation *op) {
    if (auto execute = dyn_cast<ExecuteOp>(op))
      executeOperations.insert(execute);
  };

  for (Operation *user : value.getUsers()) {
    // Follow parent operations up until the operation in the `definingRegion`.
    while (user->getParentRegion() != definingRegion) {
      trackAsyncExecute(user);
      user = user->getParentOp();
      assert(user != nullptr && "value user lies outside of the value region");
    }

    // Don't forget to process the parent in the `definingRegion` (can be the
    // original user operation itself).
    trackAsyncExecute(user);
  }

  // Process all `async.execute` operations capturing `value`.
  for (ExecuteOp execute : executeOperations) {
    // Add a reference before the execute operation to keep the reference
    // counted alive before the async region completes execution.
    builder.setInsertionPoint(execute.getOperation());
    builder.create<AddRefOp>(loc, value, IntegerAttr::get(i32, 1));

    // Drop the reference inside the async region before completion.
    OpBuilder executeBuilder = OpBuilder::atBlockTerminator(execute.getBody());
    executeBuilder.create<DropRefOp>(loc, value, IntegerAttr::get(i32, 1));
  }

  return success();
}

void AsyncRefCountingPass::runOnFunction() {
  FuncOp func = getFunction();

  // Check that we do not have explicit `add_ref` or `drop_ref` in the IR
  // because otherwise automatic reference counting will produce incorrect
  // results.
  WalkResult refCountingWalk = func.walk([&](Operation *op) -> WalkResult {
    if (isa<AddRefOp, DropRefOp>(op))
      return op->emitError() << "explicit reference counting is not supported";
    return WalkResult::advance();
  });

  if (refCountingWalk.wasInterrupted())
    signalPassFailure();

  // Add reference counting to block arguments.
  WalkResult blockWalk = func.walk([&](Block *block) -> WalkResult {
    for (BlockArgument arg : block->getArguments())
      if (isRefCounted(arg.getType()))
        if (failed(addAutomaticRefCounting(arg)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (blockWalk.wasInterrupted())
    signalPassFailure();

  // Add reference counting to operation results.
  WalkResult opWalk = func.walk([&](Operation *op) -> WalkResult {
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      if (isRefCounted(op->getResultTypes()[i]))
        if (failed(addAutomaticRefCounting(op->getResult(i))))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (opWalk.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAsyncRefCountingPass() {
  return std::make_unique<AsyncRefCountingPass>();
}
