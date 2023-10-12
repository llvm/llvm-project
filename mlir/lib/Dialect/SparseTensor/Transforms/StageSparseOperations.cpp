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

struct StageUnorderedConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement it as an Interface, this can be reused from other
    // operations too (e.g., concatenate, reshape, etc).

    if (op.directConvertable() || op.isSortCOOConvert())
      return failure();

    Location loc = op.getLoc();
    SparseTensorType srcStt = getSparseTensorType(op.getSource());
    SparseTensorType dstStt = getSparseTensorType(op.getDest());

    // Just to make sure that convert to dense tensor is always direct.
    assert(!dstStt.isAllDense());

    // source -> coo
    // The tmp COO must be unordered, otherwise it is a direct conversion.
    assert(!(srcStt.hasSameDimToLvl(dstStt) && srcStt.isAllOrdered()));
    (void)srcStt; // to silence warning when assertion is disabled

    Type srcCOOTp = getCOOFromTypeWithOrdering(
        dstStt.getRankedTensorType(), dstStt.getDimToLvl(), /*ordered=*/false);
    Value srcCOO = rewriter.create<ConvertOp>(loc, srcCOOTp, op.getSource());

    // -> sort
    Type dstCOOTp = getCOOFromTypeWithOrdering(
        dstStt.getRankedTensorType(), dstStt.getDimToLvl(), /*ordered=*/true);
    // TODO: this should be a sort_coo operation.
    Value dstCOO = rewriter.create<ConvertOp>(loc, dstCOOTp, srcCOO);

    // -> dest.
    if (dstCOO.getType() == op.getType()) {
      rewriter.replaceOp(op, dstCOO);
    } else {
      // Need an extra conversion if the target type is not COO.
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getDest().getType(),
                                             dstCOO);
    }
    // TODO: deallocate extra COOs, we should probably delegate it to buffer
    // deallocation pass.

    return success();
  }
};
} // namespace

void mlir::populateStageSparseOperationsPatterns(RewritePatternSet &patterns) {
  patterns.add<StageUnorderedConvert>(patterns.getContext());
}
