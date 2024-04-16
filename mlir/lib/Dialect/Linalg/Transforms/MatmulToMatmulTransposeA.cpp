//===- MatmulToMatmulTransposeA.cpp - Linalg matmul to matmul_transpose_a -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This rewrite and pass transposes the A matrix of a `linalg.matmul` operation
// with the aim of the memory accesses becoming contiguous.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGMATMULTOMATMULTRANSPOSEAPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-matmul-to-matmul-transpose-a"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// Pattern to replace `linalg.matmul(a, b)` with
/// `linalg.matmul_transpose_a(linalg.transpose(a), b)`.
struct MatmulToMatmulTransposeA final
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!bufferization::hasTensorSemantics(matmulOp))
      return rewriter.notifyMatchFailure(
          matmulOp, "only matmul ops with tensors are supported");

    Value a = matmulOp.getInputs()[0];
    auto aType = cast<ShapedType>(a.getType());
    if (aType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          matmulOp, "only 2-D matmul ops are supported");

    Location loc = matmulOp.getLoc();

    SmallVector<Value> dynamicDims;
    if (aType.isDynamicDim(1))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, a, 1));
    if (aType.isDynamicDim(0))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, a, 0));

    auto aShape = aType.getShape();
    SmallVector<int64_t> transposedShape{aShape[1], aShape[0]};
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, transposedShape, aType.getElementType(), dynamicDims);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto transposeAOp =
        rewriter.create<linalg::TransposeOp>(loc, a, empty, perm);
    rewriter.replaceOpWithNewOp<linalg::MatmulTransposeAOp>(
        matmulOp, matmulOp.getResultTypes(),
        ValueRange{transposeAOp->getResult(0), matmulOp.getInputs()[1]},
        matmulOp.getOutputs());

    return success();
  }
};
} // namespace

void mlir::linalg::populateMatmulToMatmulTransposeAPattern(
    RewritePatternSet &patterns) {
  patterns.add<MatmulToMatmulTransposeA>(patterns.getContext());
}

namespace {
struct LinalgMatmulToMatmulTransposeAPass
    : public impl::LinalgMatmulToMatmulTransposeAPassBase<
          LinalgMatmulToMatmulTransposeAPass> {
  using impl::LinalgMatmulToMatmulTransposeAPassBase<
      LinalgMatmulToMatmulTransposeAPass>::
      LinalgMatmulToMatmulTransposeAPassBase;
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateMatmulToMatmulTransposeAPattern(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
} // namespace
