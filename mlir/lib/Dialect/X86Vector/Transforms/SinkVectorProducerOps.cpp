//===- SinkVectorProducerOps.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

static FailureOr<llvm::SmallVector<Operation *>>
getSameBlockUsers(Operation *op) {
  llvm::SmallVector<Operation *> opUsers;
  for (OpResult result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      // Check prod and users belongs to same block.
      if (op->getBlock() != user->getBlock())
        return failure();
      opUsers.push_back(user);
    }
  }

  return opUsers;
}

// Prevent pathological looping:
// If two/three producers are used by same consumer, will end in looping of
// moving the producers.
// For example:
// %1 = prod1
// %2 = prod2
// %3 = prod3
// %4 = op %1, %2, %3
static bool checkLooping(Operation *op) {
  llvm::SmallVector<Operation *> operations;
  operations.push_back(op);

  // Retrive the next immediate operation until it is a vector.load or
  // a vector.transfer_read
  Operation *nextOp = op->getNextNode();
  while (nextOp) {
    if (isa<vector::LoadOp>(nextOp) || isa<vector::TransferReadOp>(nextOp)) {
      operations.push_back(op);
    } else {
      break;
    }
    nextOp = nextOp->getNextNode();
  }

  // If all the loads or transfer_reads have same immediate nextOp as its
  // user, then it loops.
  for (Operation *op : operations) {
    FailureOr<llvm::SmallVector<Operation *>> users = getSameBlockUsers(op);
    if (failed(users))
      return false;

    if (!llvm::is_contained(*users, nextOp))
      return false;
  }

  return true;
}

/// Sink vector producers forward to reduce live ranges.
/// This pattern applies to ops such as vector.load and vector.transfer_read.
template <typename producerOp>
struct SinkVectorProducerOps final : public OpRewritePattern<producerOp> {
  using OpRewritePattern<producerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(producerOp op,
                                PatternRewriter &rewriter) const override {

    auto users = getSameBlockUsers(op);
    if (failed(users))
      return failure();

    if (checkLooping(op))
      return failure();

    llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> prodsAllUsers;
    llvm::DenseMap<Operation *, Operation *> prodsFirstUser;

    llvm::SmallVector<Operation *> opUsers = *users;
    prodsAllUsers.try_emplace(op, opUsers);

    // Iterate until the last instruction to find the first users of all
    // producers within the block.
    Operation *nextOp = op;

    while ((nextOp = nextOp->getNextNode())) {

      if (isa<vector::LoadOp>(nextOp) || isa<vector::TransferReadOp>(nextOp)) {
        auto nextUsers = getSameBlockUsers(nextOp);

        if (failed(nextUsers))
          continue;
        llvm::SmallVector<Operation *> nextOpUsers = *nextUsers;
        prodsAllUsers.try_emplace(nextOp, nextOpUsers);
      } else {
        llvm::SmallVector<Operation *> operations;

        for (auto &entry : prodsAllUsers) {
          llvm::SmallVector<Operation *> &users = entry.second;

          if (llvm::is_contained(users, nextOp)) {
            Operation *operation = entry.first;
            operations.push_back(operation);
            prodsFirstUser.try_emplace(operation, nextOp);
          }
        }

        for (Operation *op : operations) {
          prodsAllUsers.erase(op);
        }
      }
    }

    // Move all the loads or transfer_reads before its first use.
    for (auto &entry : prodsFirstUser) {
      Operation *prod = entry.first;
      Operation *consumer = entry.second;

      prod->moveBefore(consumer);
    }

    return success();
  }
};

void x86vector::populateSinkVectorProducerOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SinkVectorProducerOps<vector::TransferReadOp>,
               SinkVectorProducerOps<vector::LoadOp>>(patterns.getContext());
}
