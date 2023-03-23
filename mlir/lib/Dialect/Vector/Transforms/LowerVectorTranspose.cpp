//===- LowerVectorTranspose.cpp - Lower 'vector.transpose' operation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.transpose' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "vector-shape-cast-lowering"

using namespace mlir;
using namespace mlir::vector;

/// Given a 'transpose' pattern, prune the rightmost dimensions that are not
/// transposed.
static void pruneNonTransposedDims(ArrayRef<int64_t> transpose,
                                   SmallVectorImpl<int64_t> &result) {
  size_t numTransposedDims = transpose.size();
  for (size_t transpDim : llvm::reverse(transpose)) {
    if (transpDim != numTransposedDims - 1)
      break;
    numTransposedDims--;
  }

  result.append(transpose.begin(), transpose.begin() + numTransposedDims);
}

namespace {
/// Progressive lowering of TransposeOp.
/// One:
///   %x = vector.transpose %y, [1, 0]
/// is replaced by:
///   %z = arith.constant dense<0.000000e+00>
///   %0 = vector.extract %y[0, 0]
///   %1 = vector.insert %0, %z [0, 0]
///   ..
///   %x = vector.insert .., .. [.., ..]
class TransposeOpLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOpLowering(vector::VectorTransformsOptions vectorTransformOptions,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value input = op.getVector();
    VectorType inputType = op.getSourceVectorType();
    VectorType resType = op.getResultVectorType();

    // Set up convenience transposition table.
    SmallVector<int64_t> transp;
    for (auto attr : op.getTransp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());

    if (vectorTransformOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Shuffle &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0)
      return rewriter.notifyMatchFailure(
          op, "Options specifies lowering to shuffle");

    // Handle a true 2-D matrix transpose differently when requested.
    if (vectorTransformOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Flat &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0) {
      Type flattenedType =
          VectorType::get(resType.getNumElements(), resType.getElementType());
      auto matrix =
          rewriter.create<vector::ShapeCastOp>(loc, flattenedType, input);
      auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
      auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
      Value trans = rewriter.create<vector::FlatTransposeOp>(
          loc, flattenedType, matrix, rows, columns);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
      return success();
    }

    // Generate unrolled extract/insert ops. We do not unroll the rightmost
    // (i.e., highest-order) dimensions that are not transposed and leave them
    // in vector form to improve performance. Therefore, we prune those
    // dimensions from the shape/transpose data structures used to generate the
    // extract/insert ops.
    SmallVector<int64_t> prunedTransp;
    pruneNonTransposedDims(transp, prunedTransp);
    size_t numPrunedDims = transp.size() - prunedTransp.size();
    auto prunedInShape = inputType.getShape().drop_back(numPrunedDims);
    auto prunedInStrides = computeStrides(prunedInShape);

    // Generates the extract/insert operations for every scalar/vector element
    // of the leftmost transposed dimensions. We traverse every transpose
    // element using a linearized index that we delinearize to generate the
    // appropriate indices for the extract/insert operations.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resType, rewriter.getZeroAttr(resType));
    int64_t numTransposedElements = ShapedType::getNumElements(prunedInShape);

    for (int64_t linearIdx = 0; linearIdx < numTransposedElements;
         ++linearIdx) {
      auto extractIdxs = delinearize(linearIdx, prunedInStrides);
      SmallVector<int64_t> insertIdxs(extractIdxs);
      applyPermutationToVector(insertIdxs, prunedTransp);
      Value extractOp =
          rewriter.create<vector::ExtractOp>(loc, input, extractIdxs);
      result =
          rewriter.create<vector::InsertOp>(loc, extractOp, result, insertIdxs);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};

/// Rewrite a 2-D vector.transpose as a sequence of:
///   vector.shape_cast 2D -> 1D
///   vector.shuffle
///   vector.shape_cast 1D -> 2D
class TransposeOp2DToShuffleLowering
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  TransposeOp2DToShuffleLowering(
      vector::VectorTransformsOptions vectorTransformOptions,
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        vectorTransformOptions(vectorTransformOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose");

    SmallVector<int64_t> transp;
    for (auto attr : op.getTransp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());
    if (transp[0] != 1 && transp[1] != 0)
      return rewriter.notifyMatchFailure(op, "Not a 2D transpose permutation");

    if (vectorTransformOptions.vectorTransposeLowering !=
        VectorTransposeLowering::Shuffle)
      return rewriter.notifyMatchFailure(op, "Options do not ask for Shuffle");

    int64_t m = srcType.getShape().front(), n = srcType.getShape().back();
    Value casted = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get({m * n}, srcType.getElementType()),
        op.getVector());
    SmallVector<int64_t> mask;
    mask.reserve(m * n);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i)
        mask.push_back(i * n + j);

    Value shuffled =
        rewriter.create<vector::ShuffleOp>(loc, casted, casted, mask);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), shuffled);

    return success();
  }

private:
  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformOptions;
};
} // namespace

void mlir::vector::populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit) {
  patterns.add<TransposeOpLowering, TransposeOp2DToShuffleLowering>(
      options, patterns.getContext(), benefit);
}
