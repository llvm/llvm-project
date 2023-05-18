//===- LowerVectorShapeCast.cpp - Lower 'vector.shape_cast' operation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.shape_cast' operation.
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
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
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

namespace {
/// ShapeOp 2D -> 1D downcast serves the purpose of flattening 2-D to 1-D
/// vectors progressively on the way to target llvm.matrix intrinsics.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract from 2-D + vector.insert_strided_slice offset into 1-D
class ShapeCastOp2DDownCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    if (sourceVectorType.getRank() != 2 || resultVectorType.getRank() != 1)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = sourceVectorType.getShape()[1];
    for (int64_t i = 0, e = sourceVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.getSource(), i);
      desc = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vec, desc,
          /*offsets=*/i * mostMinorVectorSize, /*strides=*/1);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// ShapeOp 1D -> 2D upcast serves the purpose of unflattening 2-D from 1-D
/// vectors progressively.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract_strided_slice from 1-D + vector.insert into 2-D
/// Note that 1-D extract_strided_slice are lowered to efficient vector.shuffle.
class ShapeCastOp2DUpCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    if (sourceVectorType.getRank() != 1 || resultVectorType.getRank() != 2)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = resultVectorType.getShape()[1];
    for (int64_t i = 0, e = resultVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getSource(), /*offsets=*/i * mostMinorVectorSize,
          /*sizes=*/mostMinorVectorSize,
          /*strides=*/1);
      desc = rewriter.create<vector::InsertOp>(loc, vec, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

static void incIdx(llvm::MutableArrayRef<int64_t> idx, VectorType tp,
                   int dimIdx, int initialStep = 1) {
  int step = initialStep;
  for (int d = dimIdx; d >= 0; d--) {
    idx[d] += step;
    if (idx[d] >= tp.getDimSize(d)) {
      idx[d] = 0;
      step = 1;
    } else {
      break;
    }
  }
}

// We typically should not lower general shape cast operations into data
// movement instructions, since the assumption is that these casts are
// optimized away during progressive lowering. For completeness, however,
// we fall back to a reference implementation that moves all elements
// into the right place if we get here.
class ShapeCastOpRewritePattern : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    // Special case 2D / 1D lowerings with better implementations.
    // TODO: make is ND / 1D to allow generic ND -> 1D -> MD.
    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if ((srcRank == 2 && resRank == 1) || (srcRank == 1 && resRank == 2))
      return failure();

    // Generic ShapeCast lowering path goes all the way down to unrolled scalar
    // extract/insert chains.
    // TODO: consider evolving the semantics to only allow 1D source or dest and
    // drop this potentially very expensive lowering.
    // Compute number of elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t r = 0; r < srcRank; r++)
      numElts *= sourceVectorType.getDimSize(r);
    // Replace with data movement operations:
    //    x[0,0,0] = y[0,0]
    //    x[0,0,1] = y[0,1]
    //    x[0,1,0] = y[0,2]
    // etc., incrementing the two index vectors "row-major"
    // within the source and result shape.
    SmallVector<int64_t> srcIdx(srcRank);
    SmallVector<int64_t> resIdx(resRank);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, srcRank - 1);
        incIdx(resIdx, resultVectorType, resRank - 1);
      }

      Value extract;
      if (srcRank == 0) {
        // 0-D vector special case
        assert(srcIdx.empty() && "Unexpected indices for 0-D vector");
        extract = rewriter.create<vector::ExtractElementOp>(
            loc, op.getSourceVectorType().getElementType(), op.getSource());
      } else {
        extract =
            rewriter.create<vector::ExtractOp>(loc, op.getSource(), srcIdx);
      }

      if (resRank == 0) {
        // 0-D vector special case
        assert(resIdx.empty() && "Unexpected indices for 0-D vector");
        result = rewriter.create<vector::InsertElementOp>(loc, extract, result);
      } else {
        result =
            rewriter.create<vector::InsertOp>(loc, extract, result, resIdx);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// A shape_cast lowering for scalable vectors with a single trailing scalable
/// dimension. This is similar to the general shape_cast lowering but makes use
/// of vector.scalable.insert and vector.scalable.extract to move elements a
/// subvector at a time.
///
/// E.g.:
/// ```
/// // Flatten scalable vector
/// %0 = vector.shape_cast %arg0 : vector<2x1x[4]xi32> to vector<[8]xi32>
/// ```
/// is rewritten to:
/// ```
/// // Flatten scalable vector
/// %c = arith.constant dense<0> : vector<[8]xi32>
/// %0 = vector.extract %arg0[0, 0] : vector<[4]xi32> from vector<2x1x[4]xi32>
/// %1 = vector.scalable.insert %0, %c[0] : vector<[4]xi32> into vector<[8]xi32>
/// %2 = vector.extract %arg0[1, 0] : vector<[4]xi32> from vector<2x1x[4]xi32>
/// %3 = vector.scalable.insert %2, %1[4] : vector<[4]xi32> into vector<[8]xi32>
/// ```
/// or:
/// ```
/// // Un-flatten scalable vector
/// %0 = vector.shape_cast %arg0 : vector<[8]xi32> to vector<2x1x[4]xi32>
/// ```
/// is rewritten to:
/// ```
/// // Un-flatten scalable vector
/// %c = arith.constant dense<0> : vector<2x1x[4]xi32>
/// %0 = vector.scalable.extract %arg0[0] : vector<[4]xi32> from vector<[8]xi32>
/// %1 = vector.insert %0, %c [0, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
/// %2 = vector.scalable.extract %arg0[4] : vector<[4]xi32> from vector<[8]xi32>
/// %3 = vector.insert %2, %1 [1, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
/// ```
class ScalableShapeCastOpRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    auto srcRank = sourceVectorType.getRank();
    auto resRank = resultVectorType.getRank();

    // This can only lower shape_casts where both the source and result types
    // have a single trailing scalable dimension. This is because there are no
    // legal representation of other scalable types in LLVM (and likely won't be
    // soon). There are also (currently) no operations that can index or extract
    // from >= 2D scalable vectors or scalable vectors of fixed vectors.
    if (!isTrailingDimScalable(sourceVectorType) ||
        !isTrailingDimScalable(resultVectorType)) {
      return failure();
    }

    // The sizes of the trailing dimension of the source and result vectors, the
    // size of subvector to move, and the number of elements in the vectors.
    // These are "min" sizes as they are the size when vscale == 1.
    auto minSourceTrailingSize = sourceVectorType.getShape().back();
    auto minResultTrailingSize = resultVectorType.getShape().back();
    auto minExtractionSize =
        std::min(minSourceTrailingSize, minResultTrailingSize);
    int64_t minNumElts = 1;
    for (auto size : sourceVectorType.getShape())
      minNumElts *= size;

    // The subvector type to move from the source to the result. Note that this
    // is a scalable vector. This rewrite will generate code in terms of the
    // "min" size (vscale == 1 case), that scales to any vscale.
    auto extractionVectorType = VectorType::get(
        {minExtractionSize}, sourceVectorType.getElementType(), {true});

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));

    SmallVector<int64_t> srcIdx(srcRank);
    SmallVector<int64_t> resIdx(resRank);

    // TODO: Try rewriting this with StaticTileOffsetRange (from IndexingUtils)
    // once D150000 lands.
    Value currentResultScalableVector;
    Value currentSourceScalableVector;
    for (int64_t i = 0; i < minNumElts; i += minExtractionSize) {
      // 1. Extract a scalable subvector from the source vector.
      if (!currentSourceScalableVector) {
        if (srcRank != 1) {
          currentSourceScalableVector = rewriter.create<vector::ExtractOp>(
              loc, op.getSource(), llvm::ArrayRef(srcIdx).drop_back());
        } else {
          currentSourceScalableVector = op.getSource();
        }
      }
      Value sourceSubVector = currentSourceScalableVector;
      if (minExtractionSize < minSourceTrailingSize) {
        sourceSubVector = rewriter.create<vector::ScalableExtractOp>(
            loc, extractionVectorType, sourceSubVector, srcIdx.back());
      }

      // 2. Insert the scalable subvector into the result vector.
      if (!currentResultScalableVector) {
        if (minExtractionSize == minResultTrailingSize) {
          currentResultScalableVector = sourceSubVector;
        } else if (resRank != 1) {
          currentResultScalableVector = rewriter.create<vector::ExtractOp>(
              loc, result, llvm::ArrayRef(resIdx).drop_back());
        } else {
          currentResultScalableVector = result;
        }
      }
      if (minExtractionSize < minResultTrailingSize) {
        currentResultScalableVector = rewriter.create<vector::ScalableInsertOp>(
            loc, sourceSubVector, currentResultScalableVector, resIdx.back());
      }

      // 3. Update the source and result scalable vectors if needed.
      if (resIdx.back() + minExtractionSize >= minResultTrailingSize &&
          currentResultScalableVector != result) {
        // Finished row of result. Insert complete scalable vector into result
        // (n-D) vector.
        result = rewriter.create<vector::InsertOp>(
            loc, currentResultScalableVector, result,
            llvm::ArrayRef(resIdx).drop_back());
        currentResultScalableVector = {};
      }
      if (srcIdx.back() + minExtractionSize >= minSourceTrailingSize) {
        // Finished row of source.
        currentSourceScalableVector = {};
      }

      // 4. Increment the insert/extract indices, stepping by minExtractionSize
      // for the trailing dimensions.
      incIdx(srcIdx, sourceVectorType, srcRank - 1, minExtractionSize);
      incIdx(resIdx, resultVectorType, resRank - 1, minExtractionSize);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  static bool isTrailingDimScalable(VectorType type) {
    return type.getRank() >= 1 && type.getScalableDims().back() &&
           !llvm::is_contained(type.getScalableDims().drop_back(), true);
  }
};

} // namespace

void mlir::vector::populateVectorShapeCastLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ShapeCastOp2DDownCastRewritePattern,
               ShapeCastOp2DUpCastRewritePattern, ShapeCastOpRewritePattern,
               ScalableShapeCastOpRewritePattern>(patterns.getContext(),
                                                  benefit);
}
