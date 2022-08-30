//===- AsyncRuntimeRefCountingOpt.cpp - Async Ref Counting --------------===//
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

#include "mlir/Dialect/Async/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ASYNCRUNTIMEREFCOUNTINGOPTPASS
#include "mlir/Dialect/Async/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "async-ref-counting"

using namespace mlir;
using namespace mlir::async;

namespace {

class AsyncRuntimeRefCountingOptPass
    : public impl::AsyncRuntimeRefCountingOptPassBase<
          AsyncRuntimeRefCountingOptPass> {
public:
  AsyncRuntimeRefCountingOptPass() = default;
  void runOnOperation() override;

private:
  LogicalResult optimizeReferenceCounting(
      Value value, llvm::SmallDenseMap<Operation *, Operation *> &cancellable);
};

} // namespace

LogicalResult AsyncRuntimeRefCountingOptPass::optimizeReferenceCounting(
    Value value, llvm::SmallDenseMap<Operation *, Operation *> &cancellable) {
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
  //      async.runtime.await %token : !async.token
  //    }
  //
  // %token has a use inside ^bb2 (`async.runtime.await`) and inside ^bb1
  // (`scf.if`).

  struct BlockUsersInfo {
    llvm::SmallVector<RuntimeAddRefOp, 4> addRefs;
    llvm::SmallVector<RuntimeDropRefOp, 4> dropRefs;
    llvm::SmallVector<Operation *, 4> users;
  };

  llvm::DenseMap<Block *, BlockUsersInfo> blockUsers;

  auto updateBlockUsersInfo = [&](Operation *user) {
    BlockUsersInfo &info = blockUsers[user->getBlock()];
    info.users.push_back(user);

    if (auto addRef = dyn_cast<RuntimeAddRefOp>(user))
      info.addRefs.push_back(addRef);
    if (auto dropRef = dyn_cast<RuntimeDropRefOp>(user))
      info.dropRefs.push_back(dropRef);
  };

  for (Operation *user : value.getUsers()) {
    while (user->getParentRegion() != definingRegion) {
      updateBlockUsersInfo(user);
      user = user->getParentOp();
      assert(user != nullptr && "value user lies outside of the value region");
    }

    updateBlockUsersInfo(user);
  }

  // Sort all operations found in the block.
  auto preprocessBlockUsersInfo = [](BlockUsersInfo &info) -> BlockUsersInfo & {
    auto isBeforeInBlock = [](Operation *a, Operation *b) -> bool {
      return a->isBeforeInBlock(b);
    };
    llvm::sort(info.addRefs, isBeforeInBlock);
    llvm::sort(info.dropRefs, isBeforeInBlock);
    llvm::sort(info.users, [&](Operation *a, Operation *b) -> bool {
      return isBeforeInBlock(a, b);
    });

    return info;
  };

  // Find and erase matching pairs of `add_ref` / `drop_ref` operations in the
  // blocks that modify the reference count of the `value`.
  for (auto &kv : blockUsers) {
    BlockUsersInfo &info = preprocessBlockUsersInfo(kv.second);

    for (RuntimeAddRefOp addRef : info.addRefs) {
      for (RuntimeDropRefOp dropRef : info.dropRefs) {
        // `drop_ref` operation after the `add_ref` with matching count.
        if (dropRef.count() != addRef.count() ||
            dropRef->isBeforeInBlock(addRef.getOperation()))
          continue;

        // When reference counted value passed to a function as an argument,
        // function takes ownership of +1 reference and it will drop it before
        // returning.
        //
        // Example:
        //
        //   %token = ... : !async.token
        //
        //   async.runtime.add_ref %token {count = 1 : i64} : !async.token
        //   call @pass_token(%token: !async.token, ...)
        //
        //   async.await %token : !async.token
        //   async.runtime.drop_ref %token {count = 1 : i64} : !async.token
        //
        // In this example if we'll cancel a pair of reference counting
        // operations we might end up with a deallocated token when we'll
        // reach `async.await` operation.
        Operation *firstFunctionCallUser = nullptr;
        Operation *lastNonFunctionCallUser = nullptr;

        for (Operation *user : info.users) {
          // `user` operation lies after `addRef` ...
          if (user == addRef || user->isBeforeInBlock(addRef))
            continue;
          // ... and before `dropRef`.
          if (user == dropRef || dropRef->isBeforeInBlock(user))
            break;

          // Find the first function call user of the reference counted value.
          Operation *functionCall = dyn_cast<func::CallOp>(user);
          if (functionCall &&
              (!firstFunctionCallUser ||
               functionCall->isBeforeInBlock(firstFunctionCallUser))) {
            firstFunctionCallUser = functionCall;
            continue;
          }

          // Find the last regular user of the reference counted value.
          if (!functionCall &&
              (!lastNonFunctionCallUser ||
               lastNonFunctionCallUser->isBeforeInBlock(user))) {
            lastNonFunctionCallUser = user;
            continue;
          }
        }

        // Non function call user after the function call user of the reference
        // counted value.
        if (firstFunctionCallUser && lastNonFunctionCallUser &&
            firstFunctionCallUser->isBeforeInBlock(lastNonFunctionCallUser))
          continue;

        // Try to cancel the pair of `add_ref` and `drop_ref` operations.
        auto emplaced = cancellable.try_emplace(dropRef.getOperation(),
                                                addRef.getOperation());

        if (!emplaced.second) // `drop_ref` was already marked for removal
          continue;           // go to the next `drop_ref`

        if (emplaced.second) // successfully cancelled `add_ref` <-> `drop_ref`
          break;             // go to the next `add_ref`
      }
    }
  }

  return success();
}

void AsyncRuntimeRefCountingOptPass::runOnOperation() {
  Operation *op = getOperation();

  // Mapping from `dropRef.getOperation()` to `addRef.getOperation()`.
  //
  // Find all cancellable pairs of operation and erase them in the end to keep
  // all iterators valid while we are walking the function operations.
  llvm::SmallDenseMap<Operation *, Operation *> cancellable;

  // Optimize reference counting for values defined by block arguments.
  WalkResult blockWalk = op->walk([&](Block *block) -> WalkResult {
    for (BlockArgument arg : block->getArguments())
      if (isRefCounted(arg.getType()))
        if (failed(optimizeReferenceCounting(arg, cancellable)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (blockWalk.wasInterrupted())
    signalPassFailure();

  // Optimize reference counting for values defined by operation results.
  WalkResult opWalk = op->walk([&](Operation *op) -> WalkResult {
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      if (isRefCounted(op->getResultTypes()[i]))
        if (failed(optimizeReferenceCounting(op->getResult(i), cancellable)))
          return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (opWalk.wasInterrupted())
    signalPassFailure();

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << cancellable.size()
                 << " cancellable reference counting operations\n";
  });

  // Erase all cancellable `add_ref <-> drop_ref` operation pairs.
  for (auto &kv : cancellable) {
    kv.first->erase();
    kv.second->erase();
  }
}

std::unique_ptr<Pass> mlir::createAsyncRuntimeRefCountingOptPass() {
  return std::make_unique<AsyncRuntimeRefCountingOptPass>();
}
