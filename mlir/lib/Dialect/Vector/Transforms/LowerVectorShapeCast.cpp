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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB//IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-shape-cast-lowering"

using namespace mlir;
using namespace mlir::vector;

/// Increments n-D `indices` by `step` starting from the innermost dimension.
static void incIdx(SmallVectorImpl<int64_t> &indices, VectorType vecType,
                   int step = 1) {
  for (int dim : llvm::reverse(llvm::seq<int>(0, indices.size()))) {
    assert(indices[dim] < vecType.getDimSize(dim) &&
           "Indices are out of bound");
    indices[dim] += step;
    if (indices[dim] < vecType.getDimSize(dim))
      break;

    indices[dim] = 0;
    step = 1;
  }
}

namespace {
/// ShapeOp n-D -> 1-D downcast serves the purpose of flattening N-D to 1-D
/// vectors progressively. This iterates over the n-1 major dimensions of the
/// n-D vector and performs rewrites into:
///   vector.extract from n-D + vector.insert_strided_slice offset into 1-D
class ShapeCastOpNDDownCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if (srcRank < 2 || resRank != 1)
      return failure();

    // Compute the number of 1-D vector elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t dim = 0; dim < srcRank - 1; ++dim)
      numElts *= sourceVectorType.getDimSize(dim);

    auto loc = op.getLoc();
    SmallVector<int64_t> srcIdx(srcRank - 1, 0);
    SmallVector<int64_t> resIdx(resRank, 0);
    int64_t extractSize = sourceVectorType.getShape().back();
    Value result = rewriter.create<ub::PoisonOp>(loc, resultVectorType);

    // Compute the indices of each 1-D vector element of the source extraction
    // and destination slice insertion and generate such instructions.
    for (int64_t i = 0; i < numElts; ++i) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, /*step=*/1);
        incIdx(resIdx, resultVectorType, /*step=*/extractSize);
      }

      Value extract =
          rewriter.create<vector::ExtractOp>(loc, op.getSource(), srcIdx);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, extract, result,
          /*offsets=*/resIdx, /*strides=*/1);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// ShapeOp 1-D -> n-D upcast serves the purpose of unflattening n-D from 1-D
/// vectors progressively. This iterates over the n-1 major dimension of the n-D
/// vector and performs rewrites into:
///   vector.extract_strided_slice from 1-D + vector.insert into n-D
/// Note that 1-D extract_strided_slice are lowered to efficient vector.shuffle.
class ShapeCastOpNDUpCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if (srcRank != 1 || resRank < 2)
      return failure();

    // Compute the number of 1-D vector elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t dim = 0; dim < resRank - 1; ++dim)
      numElts *= resultVectorType.getDimSize(dim);

    // Compute the indices of each 1-D vector element of the source slice
    // extraction and destination insertion and generate such instructions.
    auto loc = op.getLoc();
    SmallVector<int64_t> srcIdx(srcRank, 0);
    SmallVector<int64_t> resIdx(resRank - 1, 0);
    int64_t extractSize = resultVectorType.getShape().back();
    Value result = rewriter.create<ub::PoisonOp>(loc, resultVectorType);
    for (int64_t i = 0; i < numElts; ++i) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, /*step=*/extractSize);
        incIdx(resIdx, resultVectorType, /*step=*/1);
      }

      Value extract = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getSource(), /*offsets=*/srcIdx, /*sizes=*/extractSize,
          /*strides=*/1);
      result = rewriter.create<vector::InsertOp>(loc, extract, result, resIdx);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

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

    // Special case for n-D / 1-D lowerings with better implementations.
    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if ((srcRank > 1 && resRank == 1) || (srcRank == 1 && resRank > 1))
      return failure();

    // Generic ShapeCast lowering path goes all the way down to unrolled scalar
    // extract/insert chains.
    int64_t numElts = 1;
    for (int64_t r = 0; r < srcRank; r++)
      numElts *= sourceVectorType.getDimSize(r);
    // Replace with data movement operations:
    //    x[0,0,0] = y[0,0]
    //    x[0,0,1] = y[0,1]
    //    x[0,1,0] = y[0,2]
    // etc., incrementing the two index vectors "row-major"
    // within the source and result shape.
    SmallVector<int64_t> srcIdx(srcRank, 0);
    SmallVector<int64_t> resIdx(resRank, 0);
    Value result = rewriter.create<ub::PoisonOp>(loc, resultVectorType);
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType);
        incIdx(resIdx, resultVectorType);
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
    // from >= 2-D scalable vectors or scalable vectors of fixed vectors.
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

    Value result = rewriter.create<ub::PoisonOp>(loc, resultVectorType);
    SmallVector<int64_t> srcIdx(srcRank, 0);
    SmallVector<int64_t> resIdx(resRank, 0);

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
      incIdx(srcIdx, sourceVectorType, /*step=*/minExtractionSize);
      incIdx(resIdx, resultVectorType, /*step=*/minExtractionSize);
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
  patterns.add<ShapeCastOpNDDownCastRewritePattern,
               ShapeCastOpNDUpCastRewritePattern, ShapeCastOpRewritePattern,
               ScalableShapeCastOpRewritePattern>(patterns.getContext(),
                                                  benefit);
}
