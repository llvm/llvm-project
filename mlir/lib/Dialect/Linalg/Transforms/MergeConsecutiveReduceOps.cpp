//===- MergeConsecutiveReduceOps.cpp - Merge linalg.reduce ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGMERGECONSECUTIVEREDUCEOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-merge-consecutive-reduce-ops"

namespace {
struct MergeConsecutiveReduceOp : OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp consumer,
                                PatternRewriter &rewriter) const override {
    if (consumer.getNumDpsInputs() != 1) {
      return rewriter.notifyMatchFailure(
          consumer, "only supports second reduce op with one input");
    }
    Value input = consumer.getDpsInputs().front();
    if (!input.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          consumer, "does not support producer result with multiple users");
    }
    auto producer = input.getDefiningOp<ReduceOp>();
    if (!producer) {
      return rewriter.notifyMatchFailure(consumer,
                                         "does not find consecutive reduces");
    }
    if (consumer->getBlock() != producer->getBlock()) {
      return rewriter.notifyMatchFailure(
          consumer, "does not support reduce in different blocks");
    }
    if (!OperationEquivalence::isRegionEquivalentTo(
            &consumer.getRegion(), &producer.getRegion(),
            OperationEquivalence::Flags::IgnoreLocations)) {
      return rewriter.notifyMatchFailure(
          consumer, "reduce operation regions are not equal");
    }
    SmallVector<unsigned> prodDims, consDims;
    producer.getReductionDims(prodDims);
    consumer.getReductionDims(consDims);
    auto maxRank =
        cast<ShapedType>(producer.getDpsInputs()[0].getType()).getRank();

    auto dims = mergeConsecutiveReduceDims(prodDims, consDims, maxRank);
    rewriter.setInsertionPointAfter(consumer);
    auto newReduce = ReduceOp::create(
        rewriter, consumer->getLoc(), TypeRange(consumer->getResults()),
        producer.getInputs(), consumer.getInits(), dims);
    Region &newRegion = newReduce.getRegion();
    IRMapping mapping;
    consumer.getRegion().cloneInto(&newRegion, newRegion.begin(), mapping);

    rewriter.replaceOp(consumer, newReduce);
    rewriter.eraseOp(producer);
    return success();
  }

  /// Merge two reduce dims of consecutive reduce ops, returning the merged dims
  /// that apply to the original reduce input.
  SmallVector<int64_t> mergeConsecutiveReduceDims(ArrayRef<unsigned> prodDims,
                                                  ArrayRef<unsigned> consDims,
                                                  unsigned maxRank) const {
    BitVector availableMask(maxRank, true);
    for (unsigned dim : prodDims)
      availableMask[dim] = false;
    SmallVector<unsigned> remainingDimIndex;
    for (unsigned i = 0; i < maxRank; i++)
      if (availableMask[i])
        remainingDimIndex.push_back(i);
    SmallVector<int64_t> newDims(prodDims);
    for (unsigned dim : consDims)
      newDims.push_back(remainingDimIndex[dim]);
    llvm::sort(newDims.begin(), newDims.end());
    return newDims;
  }
};

struct LinalgMergeConsecutiveReduceOpsPass
    : public impl::LinalgMergeConsecutiveReduceOpsPassBase<
          LinalgMergeConsecutiveReduceOpsPass> {
  using impl::LinalgMergeConsecutiveReduceOpsPassBase<
      LinalgMergeConsecutiveReduceOpsPass>::
      LinalgMergeConsecutiveReduceOpsPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<MergeConsecutiveReduceOp>(op->getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
