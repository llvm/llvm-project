//===- StageSparseOperations.cpp - stage sparse ops rewriting rules -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

struct GuardSparseAlloc
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern<bufferization::AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp op,
                                PatternRewriter &rewriter) const override {
    // Only rewrite sparse allocations.
    if (!getSparseTensorEncoding(op.getResult().getType()))
      return failure();

    // Only rewrite sparse allocations that escape the method
    // without any chance of a finalizing operation in between.
    // Here we assume that sparse tensor setup never crosses
    // method boundaries. The current rewriting only repairs
    // the most obvious allocate-call/return cases.
    if (!llvm::all_of(op->getUses(), [](OpOperand &use) {
          return isa<func::ReturnOp, func::CallOp, func::CallIndirectOp>(
              use.getOwner());
        }))
      return failure();

    // Guard escaping empty sparse tensor allocations with a finalizing
    // operation that leaves the underlying storage in a proper state
    // before the tensor escapes across the method boundary.
    rewriter.setInsertionPointAfter(op);
    auto load = rewriter.create<LoadOp>(op.getLoc(), op.getResult(), true);
    rewriter.replaceAllUsesExcept(op, load, load);
    return success();
  }
};

template <typename StageWithSortOp>
struct StageUnorderedSparseOps : public OpRewritePattern<StageWithSortOp> {
  using OpRewritePattern<StageWithSortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StageWithSortOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value tmpBuf = nullptr;
    auto itOp = llvm::cast<StageWithSortSparseOp>(op.getOperation());
    LogicalResult stageResult = itOp.stageWithSort(rewriter, tmpBuf);
    // Deallocate tmpBuf.
    // TODO: Delegate to buffer deallocation pass in the future.
    if (succeeded(stageResult) && tmpBuf)
      rewriter.create<bufferization::DeallocTensorOp>(loc, tmpBuf);

    return stageResult;
  }
};
} // namespace

void mlir::populateStageSparseOperationsPatterns(RewritePatternSet &patterns) {
  patterns.add<GuardSparseAlloc, StageUnorderedSparseOps<ConvertOp>,
               StageUnorderedSparseOps<ConcatenateOp>>(patterns.getContext());
}
