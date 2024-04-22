//===- TransposeMatmul.cpp - Convert Linalg matmul to transposed variants -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is intended to be a simple high-level (target-agnostic) matmul
// transposition transformation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "linalg-transpose-matmul"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Pattern to replace
///
///   linalg.matmul(a, b)
///
/// with
///
///   linalg.matmul_transpose_a(linalg.transpose(a), b)
///
/// By default A is transposed. If `transposeA` is set to false then B is
/// transposed.
struct TransposeMatmul final : public OpRewritePattern<linalg::MatmulOp> {
  TransposeMatmul(MLIRContext *ctx, bool transposeA)
      : OpRewritePattern(ctx), transposeA(transposeA) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!bufferization::hasTensorSemantics(matmulOp))
      return rewriter.notifyMatchFailure(
          matmulOp, "only matmul ops with tensors are supported");

    Location loc = matmulOp.getLoc();
    Value input = matmulOp.getInputs()[transposeA ? 0 : 1];
    auto type = cast<ShapedType>(input.getType());

    SmallVector<Value> dynamicDims;
    if (type.isDynamicDim(1))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, 1));
    if (type.isDynamicDim(0))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));

    auto shape = type.getShape();
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{shape[1], shape[0]}, type.getElementType(),
        dynamicDims);
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, input, empty, ArrayRef<int64_t>{1, 0});
    if (transposeA)
      rewriter.replaceOpWithNewOp<linalg::MatmulTransposeAOp>(
          matmulOp, matmulOp.getResultTypes(),
          ValueRange{transposeOp->getResult(0), matmulOp.getInputs()[1]},
          matmulOp.getOutputs());
    else
      rewriter.replaceOpWithNewOp<linalg::MatmulTransposeBOp>(
          matmulOp, matmulOp.getResultTypes(),
          ValueRange{matmulOp.getInputs()[0], transposeOp->getResult(0)},
          matmulOp.getOutputs());

    return success();
  }

private:
  bool transposeA;
};

/// Pattern to replace
///
///   linalg.batch_matmul(a, b)
///
/// with
///
///   linalg.batch_matmul_transpose_a(linalg.transpose(a), b)
///
/// Only the non-batch dimensions are transposed. By default A is transposed. If
/// `transposeA` is set to false then B is transposed.
struct TransposeBatchMatmul final
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  TransposeBatchMatmul(MLIRContext *ctx, bool transposeA)
      : OpRewritePattern(ctx), transposeA(transposeA) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (!bufferization::hasTensorSemantics(batchMatmulOp))
      return rewriter.notifyMatchFailure(
          batchMatmulOp, "only matmul ops with tensors are supported");

    Location loc = batchMatmulOp.getLoc();
    Value input = batchMatmulOp.getInputs()[transposeA ? 0 : 1];
    auto type = cast<ShapedType>(input.getType());

    SmallVector<Value> dynamicDims;
    if (type.isDynamicDim(0))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
    if (type.isDynamicDim(2))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, 2));
    if (type.isDynamicDim(1))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, input, 1));

    auto shape = type.getShape();
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{shape[2], shape[1], shape[0]},
        type.getElementType(), dynamicDims);
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, input, empty, ArrayRef<int64_t>{2, 1, 0});
    if (transposeA)
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulTransposeAOp>(
          batchMatmulOp, batchMatmulOp.getResultTypes(),
          ValueRange{transposeOp->getResult(0), batchMatmulOp.getInputs()[1]},
          batchMatmulOp.getOutputs());
    else
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulTransposeBOp>(
          batchMatmulOp, batchMatmulOp.getResultTypes(),
          ValueRange{batchMatmulOp.getInputs()[0], transposeOp->getResult(0)},
          batchMatmulOp.getOutputs());

    return success();
  }

private:
  bool transposeA;
};
} // namespace

void mlir::linalg::populateTransposeMatmulPatterns(RewritePatternSet &patterns,
                                                   bool transposeA) {
  patterns.add<TransposeMatmul, TransposeBatchMatmul>(patterns.getContext(),
                                                      transposeA);
}
