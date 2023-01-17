//===- VectorMultiDimReductionTransforms.cpp - Multi-Reduction Transforms -===//
//
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements target-independent rewrites of MultiDimReductionOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-multi-reduction"

using namespace mlir;

/// This file implements the following transformations as composable atomic
/// patterns.

/// Converts vector.multi_reduction into inner-most/outer-most reduction form
/// by using vector.transpose
class InnerOuterDimReductionConversion
    : public OpRewritePattern<vector::MultiDimReductionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit InnerOuterDimReductionConversion(
      MLIRContext *context, vector::VectorMultiReductionLowering options,
      PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<vector::MultiDimReductionOp>(context, benefit),
        useInnerDimsForReduction(
            options == vector::VectorMultiReductionLowering::InnerReduction) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
    } else {
      rootOp = multiReductionOp;
    }

    auto src = multiReductionOp.getSource();
    auto loc = multiReductionOp.getLoc();
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();

    // Separate reduction and parallel dims
    auto reductionDimsRange =
        multiReductionOp.getReductionDims().getAsValueRange<IntegerAttr>();
    auto reductionDims = llvm::to_vector<4>(llvm::map_range(
        reductionDimsRange, [](const APInt &a) { return a.getZExtValue(); }));
    llvm::SmallDenseSet<int64_t> reductionDimsSet(reductionDims.begin(),
                                                  reductionDims.end());
    int64_t reductionSize = reductionDims.size();
    SmallVector<int64_t, 4> parallelDims;
    for (int64_t i = 0; i < srcRank; ++i)
      if (!reductionDimsSet.contains(i))
        parallelDims.push_back(i);

    // Add transpose only if inner-most/outer-most dimensions are not parallel
    // and there are parallel dims.
    if (parallelDims.empty())
      return failure();
    if (useInnerDimsForReduction &&
        (parallelDims ==
         llvm::to_vector<4>(llvm::seq<int64_t>(0, parallelDims.size()))))
      return failure();

    if (!useInnerDimsForReduction &&
        (parallelDims !=
         llvm::to_vector<4>(llvm::seq<int64_t>(0, parallelDims.size()))))
      return failure();

    SmallVector<int64_t, 4> indices;
    if (useInnerDimsForReduction) {
      indices.append(parallelDims.begin(), parallelDims.end());
      indices.append(reductionDims.begin(), reductionDims.end());
    } else {
      indices.append(reductionDims.begin(), reductionDims.end());
      indices.append(parallelDims.begin(), parallelDims.end());
    }

    // If masked, transpose the original mask.
    Value transposedMask;
    if (maskableOp.isMasked()) {
      transposedMask = rewriter.create<vector::TransposeOp>(
          loc, maskableOp.getMaskingOp().getMask(), indices);
    }

    // Transpose reduction source.
    auto transposeOp = rewriter.create<vector::TransposeOp>(loc, src, indices);
    SmallVector<bool> reductionMask(srcRank, false);
    for (int i = 0; i < reductionSize; ++i) {
      if (useInnerDimsForReduction)
        reductionMask[srcRank - i - 1] = true;
      else
        reductionMask[i] = true;
    }

    Operation *newMultiRedOp = rewriter.create<vector::MultiDimReductionOp>(
        multiReductionOp.getLoc(), transposeOp.getResult(),
        multiReductionOp.getAcc(), reductionMask, multiReductionOp.getKind());
    newMultiRedOp =
        mlir::vector::maskOperation(rewriter, newMultiRedOp, transposedMask);

    rewriter.replaceOp(rootOp, newMultiRedOp->getResult(0));
    return success();
  }

private:
  const bool useInnerDimsForReduction;
};

/// Reduces the rank of vector.multi_reduction nd -> 2d given all reduction
/// dimensions are either inner most or outer most.
class ReduceMultiDimReductionRank
    : public OpRewritePattern<vector::MultiDimReductionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit ReduceMultiDimReductionRank(
      MLIRContext *context, vector::VectorMultiReductionLowering options,
      PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<vector::MultiDimReductionOp>(context, benefit),
        useInnerDimsForReduction(
            options == vector::VectorMultiReductionLowering::InnerReduction) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
    } else {
      rootOp = multiReductionOp;
    }

    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    auto srcShape = multiReductionOp.getSourceVectorType().getShape();
    auto loc = multiReductionOp.getLoc();

    // If rank less than 2, nothing to do.
    if (srcRank < 2)
      return failure();

    // If already rank-2 ["parallel", "reduce"] or ["reduce", "parallel"] bail.
    SmallVector<bool> reductionMask = multiReductionOp.getReductionMask();
    if (srcRank == 2 && reductionMask.front() != reductionMask.back())
      return failure();

    // 1. Separate reduction and parallel dims.
    SmallVector<int64_t, 4> parallelDims, parallelShapes;
    SmallVector<int64_t, 4> reductionDims, reductionShapes;
    for (const auto &it : llvm::enumerate(reductionMask)) {
      int64_t i = it.index();
      bool isReduction = it.value();
      if (isReduction) {
        reductionDims.push_back(i);
        reductionShapes.push_back(srcShape[i]);
      } else {
        parallelDims.push_back(i);
        parallelShapes.push_back(srcShape[i]);
      }
    }

    // 2. Compute flattened parallel and reduction sizes.
    int flattenedParallelDim = 0;
    int flattenedReductionDim = 0;
    if (!parallelShapes.empty()) {
      flattenedParallelDim = 1;
      for (auto d : parallelShapes)
        flattenedParallelDim *= d;
    }
    if (!reductionShapes.empty()) {
      flattenedReductionDim = 1;
      for (auto d : reductionShapes)
        flattenedReductionDim *= d;
    }
    // We must at least have some parallel or some reduction.
    assert((flattenedParallelDim || flattenedReductionDim) &&
           "expected at least one parallel or reduction dim");

    // 3. Fail if reduction/parallel dims are not contiguous.
    // Check parallelDims are exactly [0 .. size).
    int64_t counter = 0;
    if (useInnerDimsForReduction &&
        llvm::any_of(parallelDims, [&](int64_t i) { return i != counter++; }))
      return failure();
    // Check parallelDims are exactly {reductionDims.size()} + [0 .. size).
    counter = reductionDims.size();
    if (!useInnerDimsForReduction &&
        llvm::any_of(parallelDims, [&](int64_t i) { return i != counter++; }))
      return failure();

    // 4. Shape cast to collapse consecutive parallel (resp. reduction dim) into
    // a single parallel (resp. reduction) dim.
    SmallVector<bool, 2> mask;
    SmallVector<int64_t, 2> vectorShape;
    if (flattenedParallelDim) {
      mask.push_back(false);
      vectorShape.push_back(flattenedParallelDim);
    }
    if (flattenedReductionDim) {
      mask.push_back(true);
      vectorShape.push_back(flattenedReductionDim);
    }
    if (!useInnerDimsForReduction && vectorShape.size() == 2) {
      std::swap(mask.front(), mask.back());
      std::swap(vectorShape.front(), vectorShape.back());
    }

    Value newVectorMask;
    if (maskableOp.isMasked()) {
      Value vectorMask = maskableOp.getMaskingOp().getMask();
      auto maskCastedType = VectorType::get(
          vectorShape,
          vectorMask.getType().cast<VectorType>().getElementType());
      newVectorMask =
          rewriter.create<vector::ShapeCastOp>(loc, maskCastedType, vectorMask);
    }

    auto castedType = VectorType::get(
        vectorShape, multiReductionOp.getSourceVectorType().getElementType());
    Value cast = rewriter.create<vector::ShapeCastOp>(
        loc, castedType, multiReductionOp.getSource());

    Value acc = multiReductionOp.getAcc();
    if (flattenedParallelDim) {
      auto accType = VectorType::get(
          {flattenedParallelDim},
          multiReductionOp.getSourceVectorType().getElementType());
      acc = rewriter.create<vector::ShapeCastOp>(loc, accType, acc);
    }
    // 6. Creates the flattened form of vector.multi_reduction with inner/outer
    // most dim as reduction.
    Operation *newMultiDimRedOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, cast, acc, mask, multiReductionOp.getKind());
    newMultiDimRedOp =
        mlir::vector::maskOperation(rewriter, newMultiDimRedOp, newVectorMask);

    // 7. If there are no parallel shapes, the result is a scalar.
    // TODO: support 0-d vectors when available.
    if (parallelShapes.empty()) {
      rewriter.replaceOp(rootOp, newMultiDimRedOp->getResult(0));
      return success();
    }

    // 8. Creates shape cast for the output n-D -> 2-D.
    VectorType outputCastedType = VectorType::get(
        parallelShapes,
        multiReductionOp.getSourceVectorType().getElementType());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        rootOp, outputCastedType, newMultiDimRedOp->getResult(0));
    return success();
  }

private:
  const bool useInnerDimsForReduction;
};

/// Unrolls vector.multi_reduction with outermost reductions
/// and combines results
struct TwoDimMultiReductionToElementWise
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    if (maskableOp.isMasked())
      // TODO: Support masking.
      return failure();

    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-2 ["parallel", "reduce"] or bail.
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.isReducedDim(1) || !multiReductionOp.isReducedDim(0))
      return failure();

    auto loc = multiReductionOp.getLoc();
    ArrayRef<int64_t> srcShape =
        multiReductionOp.getSourceVectorType().getShape();

    Type elementType = getElementTypeOrSelf(multiReductionOp.getDestType());
    if (!elementType.isIntOrIndexOrFloat())
      return failure();

    Value result = multiReductionOp.getAcc();
    for (int64_t i = 0; i < srcShape[0]; i++) {
      auto operand = rewriter.create<vector::ExtractOp>(
          loc, multiReductionOp.getSource(), i);
      result = makeArithReduction(rewriter, loc, multiReductionOp.getKind(),
                                  operand, result);
    }

    rewriter.replaceOp(multiReductionOp, result);
    return success();
  }
};

/// Converts 2d vector.multi_reduction with inner most reduction dimension into
/// a sequence of vector.reduction ops.
struct TwoDimMultiReductionToReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.isReducedDim(0) || !multiReductionOp.isReducedDim(1))
      return failure();

    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
    } else {
      rootOp = multiReductionOp;
    }

    auto loc = multiReductionOp.getLoc();
    Value result = rewriter.create<arith::ConstantOp>(
        loc, multiReductionOp.getDestType(),
        rewriter.getZeroAttr(multiReductionOp.getDestType()));
    int outerDim = multiReductionOp.getSourceVectorType().getShape()[0];

    for (int i = 0; i < outerDim; ++i) {
      auto v = rewriter.create<vector::ExtractOp>(
          loc, multiReductionOp.getSource(), ArrayRef<int64_t>{i});
      auto acc = rewriter.create<vector::ExtractOp>(
          loc, multiReductionOp.getAcc(), ArrayRef<int64_t>{i});
      Operation *reductionOp = rewriter.create<vector::ReductionOp>(
          loc, multiReductionOp.getKind(), v, acc);

      // If masked, slice the mask and mask the new reduction operation.
      if (maskableOp.isMasked()) {
        Value mask = rewriter.create<vector::ExtractOp>(
            loc, maskableOp.getMaskingOp().getMask(), ArrayRef<int64_t>{i});
        reductionOp = mlir::vector::maskOperation(rewriter, reductionOp, mask);
      }

      result = rewriter.create<vector::InsertElementOp>(
          loc, reductionOp->getResult(0), result,
          rewriter.create<arith::ConstantIndexOp>(loc, i));
    }

    rewriter.replaceOp(rootOp, result);
    return success();
  }
};

/// Converts 1d vector.multi_reduction with a single reduction dimension to a 2d
/// form with both a single parallel and reduction dimension.
/// This is achieved with a simple vector.shape_cast that inserts a leading 1.
/// The case with a single parallel dimension is a noop and folds away
/// separately.
struct OneDimMultiReductionToTwoDim
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    if (maskableOp.isMasked())
      // TODO: Support masking.
      return failure();

    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-1 or bail.
    if (srcRank != 1)
      return failure();

    auto loc = multiReductionOp.getLoc();
    auto srcVectorType = multiReductionOp.getSourceVectorType();
    auto srcShape = srcVectorType.getShape();
    auto castedType = VectorType::get(ArrayRef<int64_t>{1, srcShape.back()},
                                      srcVectorType.getElementType());
    auto accType =
        VectorType::get(ArrayRef<int64_t>{1}, srcVectorType.getElementType());
    assert(!multiReductionOp.getDestType().isa<VectorType>() &&
           "multi_reduction with a single dimension expects a scalar result");

    // If the unique dim is reduced and we insert a parallel in front, we need a
    // {false, true} mask.
    SmallVector<bool, 2> mask{false, true};

    /// vector.extract(vector.multi_reduce(vector.shape_cast(v, 1xk)), 0)
    Value cast = rewriter.create<vector::ShapeCastOp>(
        loc, castedType, multiReductionOp.getSource());
    Value castAcc = rewriter.create<vector::BroadcastOp>(
        loc, accType, multiReductionOp.getAcc());
    Value reduced = rewriter.create<vector::MultiDimReductionOp>(
        loc, cast, castAcc, mask, multiReductionOp.getKind());
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(multiReductionOp, reduced,
                                                   ArrayRef<int64_t>{0});
    return success();
  }
};

void mlir::vector::populateVectorMultiReductionLoweringPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options,
    PatternBenefit benefit) {
  patterns.add<InnerOuterDimReductionConversion, ReduceMultiDimReductionRank>(
      patterns.getContext(), options, benefit);
  patterns.add<OneDimMultiReductionToTwoDim>(patterns.getContext(), benefit);
  if (options == VectorMultiReductionLowering ::InnerReduction)
    patterns.add<TwoDimMultiReductionToReduction>(patterns.getContext(),
                                                  benefit);
  else
    patterns.add<TwoDimMultiReductionToElementWise>(patterns.getContext(),
                                                    benefit);
}
