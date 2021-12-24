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

#include "mlir/Dialect/Vector/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  explicit InnerOuterDimReductionConversion(
      MLIRContext *context, vector::VectorMultiReductionLowering options)
      : mlir::OpRewritePattern<vector::MultiDimReductionOp>(context),
        useInnerDimsForReduction(
            options == vector::VectorMultiReductionLowering::InnerReduction) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto src = multiReductionOp.source();
    auto loc = multiReductionOp.getLoc();
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();

    // Separate reduction and parallel dims
    auto reductionDimsRange =
        multiReductionOp.reduction_dims().getAsValueRange<IntegerAttr>();
    auto reductionDims = llvm::to_vector<4>(llvm::map_range(
        reductionDimsRange, [](APInt a) { return a.getZExtValue(); }));
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
    auto transposeOp = rewriter.create<vector::TransposeOp>(loc, src, indices);
    SmallVector<bool> reductionMask(srcRank, false);
    for (int i = 0; i < reductionSize; ++i) {
      if (useInnerDimsForReduction)
        reductionMask[srcRank - i - 1] = true;
      else
        reductionMask[i] = true;
    }
    rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
        multiReductionOp, transposeOp.result(), reductionMask,
        multiReductionOp.kind());
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
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  explicit ReduceMultiDimReductionRank(
      MLIRContext *context, vector::VectorMultiReductionLowering options)
      : mlir::OpRewritePattern<vector::MultiDimReductionOp>(context),
        useInnerDimsForReduction(
            options == vector::VectorMultiReductionLowering::InnerReduction) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
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
    for (auto it : llvm::enumerate(reductionMask)) {
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
    if (parallelShapes.size() > 0) {
      flattenedParallelDim = 1;
      for (auto d : parallelShapes)
        flattenedParallelDim *= d;
    }
    if (reductionShapes.size() > 0) {
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
    auto castedType = VectorType::get(
        vectorShape, multiReductionOp.getSourceVectorType().getElementType());
    Value cast = rewriter.create<vector::ShapeCastOp>(
        loc, castedType, multiReductionOp.source());

    // 5. Creates the flattened form of vector.multi_reduction with inner/outer
    // most dim as reduction.
    auto newOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, cast, mask, multiReductionOp.kind());

    // 6. If there are no parallel shapes, the result is a scalar.
    // TODO: support 0-d vectors when available.
    if (parallelShapes.empty()) {
      rewriter.replaceOp(multiReductionOp, newOp.dest());
      return success();
    }

    // 7. Creates shape cast for the output n-D -> 2-D
    VectorType outputCastedType = VectorType::get(
        parallelShapes,
        multiReductionOp.getSourceVectorType().getElementType());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        multiReductionOp, outputCastedType, newOp.dest());
    return success();
  }

private:
  const bool useInnerDimsForReduction;
};

/// Unrolls vector.multi_reduction with outermost reductions
/// and combines results
struct TwoDimMultiReductionToElementWise
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
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

    Value condition;
    Value result =
        rewriter.create<vector::ExtractOp>(loc, multiReductionOp.source(), 0)
            .getResult();
    for (int64_t i = 1; i < srcShape[0]; i++) {
      auto operand =
          rewriter.create<vector::ExtractOp>(loc, multiReductionOp.source(), i);
      switch (multiReductionOp.kind()) {
      case vector::CombiningKind::ADD:
        if (elementType.isIntOrIndex())
          result = rewriter.create<arith::AddIOp>(loc, operand, result);
        else
          result = rewriter.create<arith::AddFOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MUL:
        if (elementType.isIntOrIndex())
          result = rewriter.create<arith::MulIOp>(loc, operand, result);
        else
          result = rewriter.create<arith::MulFOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MINUI:
        result = rewriter.create<arith::MinUIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MINSI:
        result = rewriter.create<arith::MinSIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MINF:
        result = rewriter.create<arith::MinFOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MAXUI:
        result = rewriter.create<arith::MaxUIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MAXSI:
        result = rewriter.create<arith::MaxSIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::MAXF:
        result = rewriter.create<arith::MaxFOp>(loc, operand, result);
        break;
      case vector::CombiningKind::AND:
        result = rewriter.create<arith::AndIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::OR:
        result = rewriter.create<arith::OrIOp>(loc, operand, result);
        break;
      case vector::CombiningKind::XOR:
        result = rewriter.create<arith::XOrIOp>(loc, operand, result);
        break;
      }
    }

    rewriter.replaceOp(multiReductionOp, result);
    return success();
  }
};

/// Converts 2d vector.multi_reduction with inner most reduction dimension into
/// a sequence of vector.reduction ops.
struct TwoDimMultiReductionToReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.isReducedDim(0) || !multiReductionOp.isReducedDim(1))
      return failure();

    auto loc = multiReductionOp.getLoc();
    Value result = rewriter.create<ConstantOp>(
        loc, multiReductionOp.getDestType(),
        rewriter.getZeroAttr(multiReductionOp.getDestType()));
    int outerDim = multiReductionOp.getSourceVectorType().getShape()[0];

    // TODO: Add vector::CombiningKind attribute instead of string to
    // vector.reduction.
    auto getKindStr = [](vector::CombiningKind kind) {
      switch (kind) {
      case vector::CombiningKind::ADD:
        return "add";
      case vector::CombiningKind::MUL:
        return "mul";
      case vector::CombiningKind::MINUI:
        return "minui";
      case vector::CombiningKind::MINSI:
        return "minsi";
      case vector::CombiningKind::MINF:
        return "minf";
      case vector::CombiningKind::MAXUI:
        return "maxui";
      case vector::CombiningKind::MAXSI:
        return "maxsi";
      case vector::CombiningKind::MAXF:
        return "maxf";
      case vector::CombiningKind::AND:
        return "and";
      case vector::CombiningKind::OR:
        return "or";
      case vector::CombiningKind::XOR:
        return "xor";
      }
      llvm_unreachable("unknown combining kind");
    };

    for (int i = 0; i < outerDim; ++i) {
      auto v = rewriter.create<vector::ExtractOp>(
          loc, multiReductionOp.source(), ArrayRef<int64_t>{i});
      auto reducedValue = rewriter.create<vector::ReductionOp>(
          loc, getElementTypeOrSelf(multiReductionOp.getDestType()),
          rewriter.getStringAttr(getKindStr(multiReductionOp.kind())), v,
          ValueRange{});
      result = rewriter.create<vector::InsertElementOp>(
          loc, reducedValue, result,
          rewriter.create<arith::ConstantIndexOp>(loc, i));
    }
    rewriter.replaceOp(multiReductionOp, result);
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
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-1 or bail.
    if (srcRank != 1)
      return failure();

    auto loc = multiReductionOp.getLoc();
    auto srcVectorType = multiReductionOp.getSourceVectorType();
    auto srcShape = srcVectorType.getShape();
    auto castedType = VectorType::get(ArrayRef<int64_t>{1, srcShape.back()},
                                      srcVectorType.getElementType());
    assert(!multiReductionOp.getDestType().isa<VectorType>() &&
           "multi_reduction with a single dimension expects a scalar result");

    // If the unique dim is reduced and we insert a parallel in front, we need a
    // {false, true} mask.
    SmallVector<bool, 2> mask{false, true};

    /// vector.extract(vector.multi_reduce(vector.shape_cast(v, 1xk)), 0)
    Value cast = rewriter.create<vector::ShapeCastOp>(
        loc, castedType, multiReductionOp.source());
    Value reduced = rewriter.create<vector::MultiDimReductionOp>(
        loc, cast, mask, multiReductionOp.kind());
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(multiReductionOp, reduced,
                                                   ArrayRef<int64_t>{0});
    return success();
  }
};

void mlir::vector::populateVectorMultiReductionLoweringPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options) {
  patterns.add<InnerOuterDimReductionConversion, ReduceMultiDimReductionRank>(
      patterns.getContext(), options);
  patterns.add<OneDimMultiReductionToTwoDim>(patterns.getContext());
  if (options == VectorMultiReductionLowering ::InnerReduction)
    patterns.add<TwoDimMultiReductionToReduction>(patterns.getContext());
  else
    patterns.add<TwoDimMultiReductionToElementWise>(patterns.getContext());
}
