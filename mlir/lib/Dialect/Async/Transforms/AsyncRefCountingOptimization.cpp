//===- AsyncRefCountingOptimization.cpp - Async Ref Counting --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimize Async dialect reference counting operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-ref-counting"

namespace {

class AsyncRefCountingOptimizationPass
    : public AsyncRefCountingOptimizationBase<
          AsyncRefCountingOptimizationPass> {
public:
  AsyncRefCountingOptimizationPass() = default;
  void runOnFunction() override;

private:
  LogicalResult optimizeReferenceCounting(Value value);
};

} // namespace

LogicalResult
AsyncRefCountingOptimizationPass::optimizeReferenceCounting(Value value) {
  Region *definingRegion = value.getParentRegion();

  // Find all users of the `value` inside each block, including operations that
  // do not use `value` directly, but have a direct use inside nested region(s).
  //
  // Example:
  //
  //  ^bb1:
  //    %token = ...
  //    scf.if %cond {
  //      ^bb2:
  //      async.await %token : !async.token
  //    }
  //
  // %token has a use inside ^bb2 (`async.await`) and inside ^bb1 (`scf.if`).
  //
  // In addition to the operation that uses the `value` we also keep track if
  // this user is an `async.execute` operation itself, or has `async.execute`
  // operations in the nested regions that do use the `value`.

  struct UserInfo {
    Operation *operation;
    bool hasExecuteUser;
  };

  struct BlockUsersInfo {
    llvm::SmallVector<AddRefOp, 4> addRefs;
    llvm::SmallVector<DropRefOp, 4> dropRefs;
    llvm::SmallVector<UserInfo, 4> users;
  };

  llvm::DenseMap<Block *, BlockUsersInfo> blockUsers;

  auto updateBlockUsersInfo = [&](UserInfo user) {
    BlockUsersInfo &info = blockUsers[user.operation->getBlock()];
    info.users.push_back(user);

    if (auto addRef = dyn_cast<AddRefOp>(user.operation))
      info.addRefs.push_back(addRef);
    if (auto dropRef = dyn_cast<DropRefOp>(user.operation))
      info.dropRefs.push_back(dropRef);
  };

  for (Operation *user : value.getUsers()) {
    bool isAsyncUser = isa<ExecuteOp>(user);

    while (user->getParentRegion() != definingRegion) {
      updateBlockUsersInfo({user, isAsyncUser});
      user = user->getParentOp();
      isAsyncUser |= isa<ExecuteOp>(user);
      assert(user != nullptr && "value user lies outside of the value region");
    }

    updateBlockUsersInfo({user, isAsyncUser});
  }

  // Sort all operations found in the block.
  auto preprocessBlockUsersInfo = [](BlockUsersInfo &info) -> BlockUsersInfo & {
    auto isBeforeInBlock = [](Operation *a, Operation *b) -> bool {
      return a->isBeforeInBlock(b);
    };
    llvm::sort(info.addRefs, isBeforeInBlock);
    llvm::sort(info.dropRefs, isBeforeInBlock);
    llvm::sort(info.users, [&](UserInfo a, UserInfo b) -> bool {
      return isBeforeInBlock(a.operation, b.operation);
    });

    return info;
  };

  // Find and erase matching pairs of `add_ref` / `drop_ref` operations in the
  // blocks that modify the reference count of the `value`.
  for (auto &kv : blockUsers) {
    BlockUsersInfo &info = preprocessBlockUsersInfo(kv.second);

    // Find all cancellable pairs first and erase them later to keep all
    // pointers in the `info` valid until the end.
    //
    // Mapping from `dropRef.getOperation()` to `addRef.getOperation()`.
    llvm::SmallDenseMap<Operation *, Operation *> cancellable;

    for (AddRefOp addRef : info.addRefs) {
      for (DropRefOp dropRef : info.dropRefs) {
        // `drop_ref` operation after the `add_ref` with matching count.
        if (dropRef.count() != addRef.count() ||
            dropRef->isBeforeInBlock(addRef.getOperation()))
          continue;

        // `drop_ref` was already marked for removal.
        if (cancellable.find(dropRef.getOperation()) != cancellable.end())
          continue;

        // Check `value` users between `addRef` and `dropRef` in the `block`.
        Operation *addRefOp = addRef.getOperation();
        Operation *dropRefOp = dropRef.getOperation();

        // If there is a "regular" user after the `async.execute` user it is
        // unsafe to erase cancellable reference counting operations pair,
        // because async region can complete before the "regular" user and
        // destroy the reference counted value.
        bool hasExecuteUser = false;
        bool unsafeToCancel = false;

        for (UserInfo &user : info.users) {
          Operation *op = user.operation;

          // `user` operation lies after `addRef` ...
          if (op == addRefOp || op->isBeforeInBlock(addRefOp))
            continue;
          // ... and before `dropRef`.
          if (op == dropRefOp || dropRefOp->isBeforeInBlock(op))
            break;

          bool isRegularUser = !user.hasExecuteUser;
          bool isExecuteUser = user.hasExecuteUser;

          // It is unsafe to cancel `addRef` / `dropRef` pair.
          if (isRegularUser && hasExecuteUser) {
            unsafeToCancel = true;
            break;
          }

          hasExecuteUser |= isExecuteUser;
        }

        // Mark the pair of reference counting operations for removal.
        if (!unsafeToCancel)
          cancellable[dropRef.getOperation()] = addRef.getOperation();

        // If it us unsafe to cancel `addRef <-> dropRef` pair at this point,
        // all the following pairs will be also unsafe.
        break;
      }
    }

    // Erase all cancellable `addRef <-> dropRef` operation pairs.
    for (auto &kv : cancellable) {
      kv.first->erase();
      kv.second->erase();
    }
  }

  return success();
}

void AsyncRefCountingOptimizationPass::runOnFunction() {
  FuncOp func = getFunction();

  // Optimize reference counting for values defined by block arguments.
  WalkResult blockWalk = func.walk([&](Block *block) -> WalkResult {
    for (BlockArgument arg : block->getArguments())
      if (isRefCounted(arg.getType()))
        if (failed(optimizeReferenceCounting(arg)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (blockWalk.wasInterrupted())
    signalPassFailure();

  // Optimize reference counting for values defined by operation results.
  WalkResult opWalk = func.walk([&](Operation *op) -> WalkResult {
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      if (isRefCounted(op->getResultTypes()[i]))
        if (failed(optimizeReferenceCounting(op->getResult(i))))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (opWalk.wasInterrupted())
    signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAsyncRefCountingOptimizationPass() {
  return std::make_unique<AsyncRefCountingOptimizationPass>();
}
