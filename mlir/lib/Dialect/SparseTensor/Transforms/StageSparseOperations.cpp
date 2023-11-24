//===- StageSparseOperations.cpp - stage sparse ops rewriting rules -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

template <typename StageWithSortOp>
struct StageUnorderedSparseOps : public OpRewritePattern<StageWithSortOp> {
  using OpRewritePattern<StageWithSortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StageWithSortOp op,
                                PatternRewriter &rewriter) const override {
    return llvm::cast<StageWithSortSparseOp>(op.getOperation())
        .stageWithSort(rewriter);
  }
};
} // namespace

void mlir::populateStageSparseOperationsPatterns(RewritePatternSet &patterns) {
  patterns.add<StageUnorderedSparseOps<ConvertOp>,
               StageUnorderedSparseOps<ConcatenateOp>>(patterns.getContext());
}
