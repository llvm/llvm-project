//===- ConcatOpPatterns.cpp - Patterns related to tensor.concat lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Decompose `tensor.concat` into `tensor.empty` and a chain of slice inserts.
///
/// %concat = tensor.concat dim(1) %0, %1 :
///         (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
///
/// Becomes
///
/// %empty = tensor.empty() : tensor<2x7xf32>
/// %insert0 = tensor.insert_slice %0 into %empty[0, 0][2, 3][1, 1]
/// %concat = tensor.insert_slice %1 into %insert0[0, 3][2, 4][1, 1]
struct DecomposeTensorConcatOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {

    Location loc = concatOp.getLoc();
    FailureOr<Value> dest =
        tensor::getOrCreateDestination(rewriter, loc, concatOp->getResult(0));
    if (failed(dest))
      return failure();

    auto empty = dest->getDefiningOp<tensor::EmptyOp>();
    if (!empty)
      return failure();

    int64_t dim = concatOp.getDim();
    Value dimValue = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dim));

    int64_t rank = concatOp.getResultType().getRank();
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> dimOffsets;
    dimOffsets.push_back(rewriter.getIndexAttr(0));

    for (auto input : concatOp.getInputs().drop_front()) {
      Value size = rewriter.createOrFold<tensor::DimOp>(loc, input, dimValue);
      Value currentOffset =
          getValueOrCreateConstantIndexOp(rewriter, loc, dimOffsets.back());
      Value total =
          rewriter.createOrFold<arith::AddIOp>(loc, currentOffset, size);
      dimOffsets.push_back(getAsOpFoldResult(total));
    }

    Value result = *dest;

    for (auto [input, offset] : llvm::zip(concatOp.getInputs(), dimOffsets)) {
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, input);
      offsets[dim] = offset;
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, input, result, offsets, sizes, strides);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        concatOp, concatOp.getResultType(), result);
    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorConcatPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorConcatOp>(patterns.getContext());
}
