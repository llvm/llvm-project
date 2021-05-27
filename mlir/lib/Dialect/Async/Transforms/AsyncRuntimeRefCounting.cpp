//===- AsyncRuntimeRefCounting.cpp - Async Runtime Ref Counting -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements automatic reference counting for Async runtime
// operations and types.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-runtime-ref-counting"

namespace {

class AsyncRuntimeRefCountingPass
    : public AsyncRuntimeRefCountingBase<AsyncRuntimeRefCountingPass> {
public:
  AsyncRuntimeRefCountingPass() = default;
  void runOnOperation() override;

private:
  /// Adds an automatic reference counting to the `value`.
  ///
  /// All values (token, group or value) are semantically created with a
  /// reference count of +1 and it is the responsibility of the async value user
  /// to place the `add_ref` and `drop_ref` operations to ensure that the value
  /// is destroyed after the last use.
  ///
  /// The function returns failure if it can't deduce the locations where
  /// to place the reference counting operations.
  ///
  /// Async values "semantically created" when:
  ///   1. Operation returns async result (e.g. `async.runtime.create`)
  ///   2. Async value passed in as a block argument (or function argument,
  ///      because function arguments are just entry block arguments)
  ///
  /// Passing async value as a function argument (or block argument) does not
  /// really mean that a new async value is created, it only means that the
  /// caller of a function transfered ownership of `+1` reference to the callee.
  /// It is convenient to think that from the callee perspective async value was
  /// "created" with `+1` reference by the block argument.
  ///
  /// Automatic reference counting algorithm outline:
  ///
  /// #1 Insert `drop_ref` operations after last use of the `value`.
  /// #2 Insert `add_ref` operations before functions calls with reference
  ///    counted `value` operand (newly created `+1` reference will be
  ///    transferred to the callee).
  /// #3 Verify that divergent control flow does not lead to leaked reference
  ///    counted objects.
  ///
  /// Async runtime reference counting optimization pass will optimize away
  /// some of the redundant `add_ref` and `drop_ref` operations inserted by this
  /// strategy (see `async-runtime-ref-counting-opt`).
  LogicalResult addAutomaticRefCounting(Value value);

  /// (#1) Adds the `drop_ref` operation after the last use of the `value`
  /// relying on the liveness analysis.
  ///
  /// If the `value` is in the block `liveIn` set and it is not in the block
  /// `liveOut` set, it means that it "dies" in the block. We find the last
  /// use of the value in such block and:
  ///
  ///   1. If the last user is a `ReturnLike` operation we do nothing, because
  ///      it forwards the ownership to the caller.
  ///   2. Otherwise we add a `drop_ref` operation immediately after the last
  ///      use.
  LogicalResult addDropRefAfterLastUse(Value value);

  /// (#2) Adds the `add_ref` operation before the function call taking `value`
  /// operand to ensure that the value passed to the function entry block
  /// has a `+1` reference count.
  LogicalResult addAddRefBeforeFunctionCall(Value value);

  /// (#3) Adds the `drop_ref` operation to account for successor blocks with
  /// divergent `liveIn` property: `value` is not in the `liveIn` set of all
  /// successor blocks.
  ///
  /// Example:
  ///
  ///   ^entry:
  ///     %token = async.runtime.create : !async.token
  ///     cond_br %cond, ^bb1, ^bb2
  ///   ^bb1:
  ///     async.runtime.await %token
  ///     async.runtime.drop_ref %token
  ///     br ^bb2
  ///   ^bb2:
  ///     return
  ///
  /// In this example ^bb2 does not have `value` in the `liveIn` set, so we have
  /// to branch into a special "reference counting block" from the ^entry that
  /// will have a `drop_ref` operation, and then branch into the ^bb2.
  ///
  /// After transformation:
  ///
  ///   ^entry:
  ///     %token = async.runtime.create : !async.token
  ///     cond_br %cond, ^bb1, ^reference_counting
  ///   ^bb1:
  ///     async.runtime.await %token
  ///     async.runtime.drop_ref %token
  ///     br ^bb2
  ///   ^reference_counting:
  ///     async.runtime.drop_ref %token
  ///     br ^bb2
  ///   ^bb2:
  ///     return
  ///
  /// An exception to this rule are blocks with `async.coro.suspend` terminator,
  /// because in Async to LLVM lowering it is guaranteed that the control flow
  /// will jump into the resume block, and then follow into the cleanup and
  /// suspend blocks.
  ///
  /// Example:
  ///
  ///  ^entry(%value: !async.value<f32>):
  ///     async.runtime.await_and_resume %value, %hdl : !async.value<f32>
  ///     async.coro.suspend %ret, ^suspend, ^resume, ^cleanup
  ///   ^resume:
  ///     %0 = async.runtime.load %value
  ///     br ^cleanup
  ///   ^cleanup:
  ///     ...
  ///   ^suspend:
  ///     ...
  ///
  /// Although cleanup and suspend blocks do not have the `value` in the
  /// `liveIn` set, it is guaranteed that execution will eventually continue in
  /// the resume block (we never explicitly destroy coroutines).
  LogicalResult addDropRefInDivergentLivenessSuccessor(Value value);
};

} // namespace

LogicalResult AsyncRuntimeRefCountingPass::addDropRefAfterLastUse(Value value) {
  OpBuilder builder(value.getContext());
  Location loc = value.getLoc();

  // Use liveness analysis to find the placement of `drop_ref`operation.
  auto &liveness = getAnalysis<Liveness>();

  // We analyse only the blocks of the region that defines the `value`, and do
  // not check nested blocks attached to operations.
  //
  // By analyzing only the `definingRegion` CFG we potentially loose an
  // opportunity to drop the reference count earlier and can extend the lifetime
  // of reference counted value longer then it is really required.
  //
  // We also assume that all nested regions finish their execution before the
  // completion of the owner operation. The only exception to this rule is
  // `async.execute` operation, and we verify that they are lowered to the
  // `async.runtime` operations before adding automatic reference counting.
  Region *definingRegion = value.getParentRegion();

  // Last users of the `value` inside all blocks where the value dies.
  llvm::SmallSet<Operation *, 4> lastUsers;

  // Find blocks in the `definingRegion` that have users of the `value` (if
  // there are multiple users in the block, which one will be selected is
  // undefined). User operation might be not the actual user of the value, but
  // the operation in the block that has a "real user" in one of the attached
  // regions.
  llvm::DenseMap<Block *, Operation *> usersInTheBlocks;

  for (Operation *user : value.getUsers()) {
    Block *userBlock = user->getBlock();
    Block *ancestor = definingRegion->findAncestorBlockInRegion(*userBlock);
    usersInTheBlocks[ancestor] = ancestor->findAncestorOpInBlock(*user);
    assert(ancestor && "ancestor block must be not null");
    assert(usersInTheBlocks[ancestor] && "ancestor op must be not null");
  }

  // Find blocks where the `value` dies: the value is in `liveIn` set and not
  // in the `liveOut` set. We place `drop_ref` immediately after the last use
  // of the `value` in such regions (after handling few special cases).
  //
  // We do not traverse all the blocks in the `definingRegion`, because the
  // `value` can be in the live in set only if it has users in the block, or it
  // is defined in the block.
  //
  // Values with zero users (only definition) handled explicitly above.
  for (auto &blockAndUser : usersInTheBlocks) {
    Block *block = blockAndUser.getFirst();
    Operation *userInTheBlock = blockAndUser.getSecond();

    const LivenessBlockInfo *blockLiveness = liveness.getLiveness(block);

    // Value must be in the live input set or defined in the block.
    assert(blockLiveness->isLiveIn(value) ||
           blockLiveness->getBlock() == value.getParentBlock());

    // If value is in the live out set, it means it doesn't "die" in the block.
    if (blockLiveness->isLiveOut(value))
      continue;

    // At this point we proved that `value` dies in the `block`. Find the last
    // use of the `value` inside the `block`, this is where it "dies".
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
    builder.create<RuntimeDropRefOp>(loc, value, builder.getI32IntegerAttr(1));
  }

  return success();
}

LogicalResult
AsyncRuntimeRefCountingPass::addAddRefBeforeFunctionCall(Value value) {
  OpBuilder builder(value.getContext());
  Location loc = value.getLoc();

  for (Operation *user : value.getUsers()) {
    if (!isa<CallOp>(user))
      continue;

    // Add a reference before the function call to pass the value at `+1`
    // reference to the function entry block.
    builder.setInsertionPoint(user);
    builder.create<RuntimeAddRefOp>(loc, value, builder.getI32IntegerAttr(1));
  }

  return success();
}

LogicalResult
AsyncRuntimeRefCountingPass::addDropRefInDivergentLivenessSuccessor(
    Value value) {
  using BlockSet = llvm::SmallPtrSet<Block *, 4>;

  OpBuilder builder(value.getContext());

  // If a block has successors with different `liveIn` property of the `value`,
  // record block successors that do not thave the `value` in the `liveIn` set.
  llvm::SmallDenseMap<Block *, BlockSet> divergentLivenessBlocks;

  // Use liveness analysis to find the placement of `drop_ref`operation.
  auto &liveness = getAnalysis<Liveness>();

  // Because we only add `drop_ref` operations to the region that defines the
  // `value` we can only process CFG for the same region.
  Region *definingRegion = value.getParentRegion();

  // Collect blocks with successors with mismatching `liveIn` sets.
  for (Block &block : definingRegion->getBlocks()) {
    const LivenessBlockInfo *blockLiveness = liveness.getLiveness(&block);

    // Skip the block if value is not in the `liveOut` set.
    if (!blockLiveness || !blockLiveness->isLiveOut(value))
      continue;

    BlockSet liveInSuccessors;   // `value` is in `liveIn` set
    BlockSet noLiveInSuccessors; // `value` is not in the `liveIn` set

    // Collect successors that do not have `value` in the `liveIn` set.
    for (Block *successor : block.getSuccessors()) {
      const LivenessBlockInfo *succLiveness = liveness.getLiveness(successor);
      if (succLiveness && succLiveness->isLiveIn(value))
        liveInSuccessors.insert(successor);
      else
        noLiveInSuccessors.insert(successor);
    }

    // Block has successors with different `liveIn` property of the `value`.
    if (!liveInSuccessors.empty() && !noLiveInSuccessors.empty())
      divergentLivenessBlocks.try_emplace(&block, noLiveInSuccessors);
  }

  // Try to insert `dropRef` operations to handle blocks with divergent liveness
  // in successors blocks.
  for (auto kv : divergentLivenessBlocks) {
    Block *block = kv.getFirst();
    BlockSet &successors = kv.getSecond();

    // Coroutine suspension is a special case terminator for wich we do not
    // need to create additional reference counting (see details above).
    Operation *terminator = block->getTerminator();
    if (isa<CoroSuspendOp>(terminator))
      continue;

    // We only support successor blocks with empty block argument list.
    auto hasArgs = [](Block *block) { return !block->getArguments().empty(); };
    if (llvm::any_of(successors, hasArgs))
      return terminator->emitOpError()
             << "successor have different `liveIn` property of the reference "
                "counted value";

    // Make sure that `dropRef` operation is called when branched into the
    // successor block without `value` in the `liveIn` set.
    for (Block *successor : successors) {
      // If successor has a unique predecessor, it is safe to create `dropRef`
      // operations directly in the successor block.
      //
      // Otherwise we need to create a special block for reference counting
      // operations, and branch from it to the original successor block.
      Block *refCountingBlock = nullptr;

      if (successor->getUniquePredecessor() == block) {
        refCountingBlock = successor;
      } else {
        refCountingBlock = &successor->getParent()->emplaceBlock();
        refCountingBlock->moveBefore(successor);
        OpBuilder builder = OpBuilder::atBlockEnd(refCountingBlock);
        builder.create<BranchOp>(value.getLoc(), successor);
      }

      OpBuilder builder = OpBuilder::atBlockBegin(refCountingBlock);
      builder.create<RuntimeDropRefOp>(value.getLoc(), value,
                                       builder.getI32IntegerAttr(1));

      // No need to update the terminator operation.
      if (successor == refCountingBlock)
        continue;

      // Update terminator `successor` block to `refCountingBlock`.
      for (auto pair : llvm::enumerate(terminator->getSuccessors()))
        if (pair.value() == successor)
          terminator->setSuccessor(refCountingBlock, pair.index());
    }
  }

  return success();
}

LogicalResult
AsyncRuntimeRefCountingPass::addAutomaticRefCounting(Value value) {
  OpBuilder builder(value.getContext());
  Location loc = value.getLoc();

  // Set inserton point after the operation producing a value, or at the
  // beginning of the block if the value defined by the block argument.
  if (Operation *op = value.getDefiningOp())
    builder.setInsertionPointAfter(op);
  else
    builder.setInsertionPointToStart(value.getParentBlock());

  // Drop the reference count immediately if the value has no uses.
  if (value.getUses().empty()) {
    builder.create<RuntimeDropRefOp>(loc, value, builder.getI32IntegerAttr(1));
    return success();
  }

  // Add `drop_ref` operations based on the liveness analysis.
  if (failed(addDropRefAfterLastUse(value)))
    return failure();

  // Add `add_ref` operations before function calls.
  if (failed(addAddRefBeforeFunctionCall(value)))
    return failure();

  // Add `drop_ref` operations to successors with divergent `value` liveness.
  if (failed(addDropRefInDivergentLivenessSuccessor(value)))
    return failure();

  return success();
}

void AsyncRuntimeRefCountingPass::runOnOperation() {
  Operation *op = getOperation();

  // Check that we do not have high level async operations in the IR because
  // otherwise automatic reference counting will produce incorrect results after
  // execute operations will be lowered to `async.runtime`
  WalkResult executeOpWalk = op->walk([&](Operation *op) -> WalkResult {
    if (!isa<ExecuteOp, AwaitOp, AwaitAllOp, YieldOp>(op))
      return WalkResult::advance();

    return op->emitError()
           << "async operations must be lowered to async runtime operations";
  });

  if (executeOpWalk.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Add reference counting to block arguments.
  WalkResult blockWalk = op->walk([&](Block *block) -> WalkResult {
    for (BlockArgument arg : block->getArguments())
      if (isRefCounted(arg.getType()))
        if (failed(addAutomaticRefCounting(arg)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (blockWalk.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Add reference counting to operation results.
  WalkResult opWalk = op->walk([&](Operation *op) -> WalkResult {
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      if (isRefCounted(op->getResultTypes()[i]))
        if (failed(addAutomaticRefCounting(op->getResult(i))))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (opWalk.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createAsyncRuntimeRefCountingPass() {
  return std::make_unique<AsyncRuntimeRefCountingPass>();
}
