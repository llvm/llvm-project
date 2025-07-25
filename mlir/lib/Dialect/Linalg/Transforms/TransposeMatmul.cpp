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

#define DEBUG_TYPE "linalg-transpose-matmul"

using namespace mlir;
using namespace mlir::linalg;

/// Pattern to replace
///
///   linalg.matmul(a, b)
///
/// with
///
///   linalg.matmul_transpose_a(linalg.transpose(a), b)
///
/// By default the LHS is transposed. Set `transposeLHS=false` to
/// transpose RHS instead.
FailureOr<Operation *> mlir::linalg::transposeMatmul(RewriterBase &rewriter,
                                                     linalg::MatmulOp matmulOp,
                                                     bool transposeLHS) {
  // Check to not let go the matmul with extended semantic, through this
  // transform.
  if (matmulOp.hasUserDefinedMaps()) {
    return rewriter.notifyMatchFailure(
        matmulOp, "only matmul ops with non-extended semantics are supported");
  }

  if (!matmulOp.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(
        matmulOp, "only matmul ops with tensors are supported");

  Location loc = matmulOp.getLoc();
  Value input = matmulOp.getInputs()[transposeLHS ? 0 : 1];
  auto type = cast<ShapedType>(input.getType());

  SmallVector<Value> dynamicDims;
  if (type.isDynamicDim(1))
    dynamicDims.push_back(tensor::DimOp::create(rewriter, loc, input, 1));
  if (type.isDynamicDim(0))
    dynamicDims.push_back(tensor::DimOp::create(rewriter, loc, input, 0));

  ArrayRef<int64_t> shape = type.getShape();
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{shape[1], shape[0]}, type.getElementType(),
      dynamicDims);
  auto transposeOp = rewriter.create<linalg::TransposeOp>(
      loc, input, empty, ArrayRef<int64_t>{1, 0});
  Operation *newMatmulOp;
  if (transposeLHS) {
    newMatmulOp = MatmulTransposeAOp::create(
        rewriter, loc, matmulOp.getResultTypes(),
        ValueRange{transposeOp->getResult(0), matmulOp.getInputs()[1]},
        matmulOp.getOutputs());
  } else {
    newMatmulOp = MatmulTransposeBOp::create(
        rewriter, loc, matmulOp.getResultTypes(),
        ValueRange{matmulOp.getInputs()[0], transposeOp->getResult(0)},
        matmulOp.getOutputs());
  }
  rewriter.replaceOp(matmulOp, newMatmulOp);
  return newMatmulOp;
}

/// Pattern to replace
///
///   linalg.batch_matmul(a, b)
///
/// with
///
///   linalg.batch_matmul_transpose_a(linalg.transpose(a), b)
///
/// Only the non-batch dimensions are transposed. By default the LHS is
/// transposed. Set `transposeLHS=false` to transpose RHS instead.
FailureOr<Operation *>
mlir::linalg::transposeBatchMatmul(RewriterBase &rewriter,
                                   linalg::BatchMatmulOp batchMatmulOp,
                                   bool transposeLHS) {
  if (batchMatmulOp.hasUserDefinedMaps()) {
    return rewriter.notifyMatchFailure(
        batchMatmulOp, "ops with user-defined maps are not supported");
  }

  if (!batchMatmulOp.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(
        batchMatmulOp, "only matmul ops with tensors are supported");

  Location loc = batchMatmulOp.getLoc();
  Value input = batchMatmulOp.getInputs()[transposeLHS ? 0 : 1];
  auto type = cast<ShapedType>(input.getType());

  SmallVector<Value> dynamicDims;
  if (type.isDynamicDim(0))
    dynamicDims.push_back(tensor::DimOp::create(rewriter, loc, input, 0));
  if (type.isDynamicDim(2))
    dynamicDims.push_back(tensor::DimOp::create(rewriter, loc, input, 2));
  if (type.isDynamicDim(1))
    dynamicDims.push_back(tensor::DimOp::create(rewriter, loc, input, 1));

  ArrayRef<int64_t> shape = type.getShape();
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, ArrayRef<int64_t>{shape[0], shape[2], shape[1]},
      type.getElementType(), dynamicDims);
  auto transposeOp = rewriter.create<linalg::TransposeOp>(
      loc, input, empty, ArrayRef<int64_t>{0, 2, 1});
  Operation *newMatmulOp;
  if (transposeLHS) {
    newMatmulOp = BatchMatmulTransposeAOp::create(
        rewriter, loc, batchMatmulOp.getResultTypes(),
        ValueRange{transposeOp->getResult(0), batchMatmulOp.getInputs()[1]},
        batchMatmulOp.getOutputs());
  } else {
    newMatmulOp = BatchMatmulTransposeBOp::create(
        rewriter, loc, batchMatmulOp.getResultTypes(),
        ValueRange{batchMatmulOp.getInputs()[0], transposeOp->getResult(0)},
        batchMatmulOp.getOutputs());
  }
  rewriter.replaceOp(batchMatmulOp, newMatmulOp);
  return newMatmulOp;
}

namespace {
struct TransposeMatmul final : public OpRewritePattern<linalg::MatmulOp> {
  TransposeMatmul(MLIRContext *ctx, bool transposeLHS)
      : OpRewritePattern(ctx), transposeLHS(transposeLHS) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(transposeMatmul(rewriter, op, transposeLHS))) {
      return failure();
    }
    return success();
  }

private:
  bool transposeLHS;
};

struct TransposeBatchMatmul final
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  TransposeBatchMatmul(MLIRContext *ctx, bool transposeLHS)
      : OpRewritePattern(ctx), transposeLHS(transposeLHS) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(transposeBatchMatmul(rewriter, op, transposeLHS))) {
      return failure();
    }
    return success();
  }

private:
  bool transposeLHS;
};
} // namespace

void mlir::linalg::populateTransposeMatmulPatterns(RewritePatternSet &patterns,
                                                   bool transposeLHS) {
  patterns.add<TransposeMatmul, TransposeBatchMatmul>(patterns.getContext(),
                                                      transposeLHS);
}
