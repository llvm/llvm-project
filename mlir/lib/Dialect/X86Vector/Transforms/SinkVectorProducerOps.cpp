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

/// Sink vector producers forward to reduce live ranges.
/// This pattern applies to ops such as vector.load and vector.transfer_read.
template <typename producerOp>
struct SinkVectorProducerOps final : public OpRewritePattern<producerOp> {
  using OpRewritePattern<producerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(producerOp op,
                                PatternRewriter &rewriter) const override {

    // Collect all users of the producer op.
    llvm::SmallVector<Operation *> opUsers;
    for (OpResult result : op->getResults())
      for (Operation *user : result.getUsers())
        opUsers.push_back(user);

    // If there are no users, nothing to sink.
    if (opUsers.empty())
      return failure();

    // If the next op is already a user, do not move.
    Operation *nextOp = op->getNextNode();
    if (llvm::is_contained(opUsers, nextOp))
      return failure();

    // Prevent pathological looping:
    // If two producers are used by same consumer, will end in looping of
    // moving the producers.
    // For example:
    // %1 = prod1
    // %2 = prod2
    // %3 = op %1, %2
    llvm::SmallVector<Operation *> nextOpUsers;
    for (OpResult result : nextOp->getResults())
      for (Operation *user : result.getUsers())
        nextOpUsers.push_back(user);

    // Both producers have one same users.
    if (opUsers.size() == 1 && nextOpUsers.size() != 1 &&
        llvm::is_contained(opUsers, nextOpUsers.front()))
      return failure();

    // Get the first user of both the current and next operation.
    Operation *opFirstUser = op->getNextNode();
    Operation *nextOpFirstUser = op->getNextNode();

    while (opFirstUser) {
      if (llvm::is_contained(opUsers, opFirstUser))
        break;

      opFirstUser = opFirstUser->getNextNode();
    }

    while (nextOpFirstUser) {
      if (llvm::is_contained(nextOpUsers, nextOpFirstUser))
        break;

      nextOpFirstUser = nextOpFirstUser->getNextNode();
    }

    if (!opFirstUser)
      return failure();

    // The Op first user and next Op first user are same. Break here to
    // to avoid the shift cycle looping.
    if (opFirstUser == nextOpFirstUser)
      return failure();

    // Both ops must be in the same block to safely move.
    if (op->getBlock() != opFirstUser->getBlock())
      return failure();

    // Move producer immediately before its first user.
    op->moveBefore(opFirstUser);

    // Move the nextOp to its first user
    if (nextOpFirstUser && (nextOpFirstUser->getBlock() == nextOp->getBlock()))
      nextOp->moveBefore(nextOpFirstUser);

    return success();
  }
};

void x86vector::populateSinkVectorProducerOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SinkVectorProducerOps<vector::TransferReadOp>,
               SinkVectorProducerOps<vector::LoadOp>>(patterns.getContext());
}
