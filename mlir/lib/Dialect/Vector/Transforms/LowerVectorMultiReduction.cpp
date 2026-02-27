//===- LowerVectorMultiReduction.cpp - Lower `vector.multi_reduction` op --===//
//
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.multi_reduction' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_LOWERVECTORMULTIREDUCTION
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

#define DEBUG_TYPE "vector-multi-reduction"

using namespace mlir;

namespace {
/// This file implements the following transformations as composable atomic
/// patterns.

/// Converts vector.multi_reduction into inner-most/outer-most reduction form
/// by using vector.transpose
class InnerOuterDimReductionConversion
    : public OpRewritePattern<vector::MultiDimReductionOp> {
public:
  using Base::Base;

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
    ArrayRef<int64_t> reductionDims = multiReductionOp.getReductionDims();
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
        (parallelDims == llvm::to_vector<4>(llvm::seq<int64_t>(
                             reductionDims.size(),
                             parallelDims.size() + reductionDims.size()))))
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
      transposedMask = vector::TransposeOp::create(
          rewriter, loc, maskableOp.getMaskingOp().getMask(), indices);
    }

    // Transpose reduction source.
    auto transposeOp = vector::TransposeOp::create(rewriter, loc, src, indices);
    SmallVector<bool> reductionMask(srcRank, false);
    for (int i = 0; i < reductionSize; ++i) {
      if (useInnerDimsForReduction)
        reductionMask[srcRank - i - 1] = true;
      else
        reductionMask[i] = true;
    }

    Operation *newMultiRedOp = vector::MultiDimReductionOp::create(
        rewriter, multiReductionOp.getLoc(), transposeOp.getResult(),
        multiReductionOp.getAcc(), reductionMask, multiReductionOp.getKind());
    newMultiRedOp =
        mlir::vector::maskOperation(rewriter, newMultiRedOp, transposedMask);

    rewriter.replaceOp(rootOp, newMultiRedOp->getResult(0));
    return success();
  }

private:
  const bool useInnerDimsForReduction;
};

/// Flattens vector.multi_reduction to 2D
///
/// Given all reduction dimensions are either inner most or outer most,
/// flattens all reduction and parallel dimensions so that there are only 2Ds.
///
/// BEFORE
///     vector.multi_reduction <add>, %vec, %acc [2, 3] : vector<2x3x4x5xi32> to
///     vector<2x3xi32>
/// AFTER
///     %vec_sc = vector.shape_cast %vec
///     %acc_sc = vector.shape_cast %acc
///     %res = vector.multi_reduction <add>, %vec_sc, %acc_cs [1] :
///     vector<6x20xi32> to vector<6xi32> %res_sc = vector.shape_cast %res
class FlattenMultiReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
public:
  using Base::Base;

  explicit FlattenMultiReduction(MLIRContext *context,
                                 vector::VectorMultiReductionLowering options,
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
    auto srcScalableDims =
        multiReductionOp.getSourceVectorType().getScalableDims();
    auto loc = multiReductionOp.getLoc();

    // If rank less than 2, nothing to do.
    if (srcRank < 2)
      return failure();

    // Allow only 1 scalable dimensions. Otherwise we could end-up with e.g.
    // `vscale * vscale` that's currently not modelled.
    if (llvm::count(srcScalableDims, true) > 1)
      return failure();

    // If already rank-2 ["parallel", "reduce"] or ["reduce", "parallel"] bail.
    SmallVector<bool> reductionMask = multiReductionOp.getReductionMask();
    if (srcRank == 2 && reductionMask.front() != reductionMask.back())
      return failure();

    // 1. Separate reduction and parallel dims.
    SmallVector<int64_t, 4> parallelDims, parallelShapes;
    SmallVector<bool, 4> parallelScalableDims;
    SmallVector<int64_t, 4> reductionDims, reductionShapes;
    bool isReductionDimScalable = false;
    for (const auto &it : llvm::enumerate(reductionMask)) {
      int64_t i = it.index();
      bool isReduction = it.value();
      if (isReduction) {
        reductionDims.push_back(i);
        reductionShapes.push_back(srcShape[i]);
        isReductionDimScalable |= srcScalableDims[i];
      } else {
        parallelDims.push_back(i);
        parallelShapes.push_back(srcShape[i]);
        parallelScalableDims.push_back(srcScalableDims[i]);
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
    SmallVector<bool, 2> scalableDims;
    SmallVector<int64_t, 2> vectorShape;
    bool isParallelDimScalable = llvm::is_contained(parallelScalableDims, true);
    if (flattenedParallelDim) {
      mask.push_back(false);
      vectorShape.push_back(flattenedParallelDim);
      scalableDims.push_back(isParallelDimScalable);
    }
    if (flattenedReductionDim) {
      mask.push_back(true);
      vectorShape.push_back(flattenedReductionDim);
      scalableDims.push_back(isReductionDimScalable);
    }
    if (!useInnerDimsForReduction && vectorShape.size() == 2) {
      std::swap(mask.front(), mask.back());
      std::swap(vectorShape.front(), vectorShape.back());
      std::swap(scalableDims.front(), scalableDims.back());
    }

    Value newVectorMask;
    if (maskableOp.isMasked()) {
      Value vectorMask = maskableOp.getMaskingOp().getMask();
      auto maskCastedType = VectorType::get(
          vectorShape,
          llvm::cast<VectorType>(vectorMask.getType()).getElementType());
      newVectorMask = vector::ShapeCastOp::create(rewriter, loc, maskCastedType,
                                                  vectorMask);
    }

    auto castedType = VectorType::get(
        vectorShape, multiReductionOp.getSourceVectorType().getElementType(),
        scalableDims);
    Value cast = vector::ShapeCastOp::create(rewriter, loc, castedType,
                                             multiReductionOp.getSource());

    Value acc = multiReductionOp.getAcc();
    if (flattenedParallelDim) {
      auto accType = VectorType::get(
          {flattenedParallelDim},
          multiReductionOp.getSourceVectorType().getElementType(),
          /*scalableDims=*/{isParallelDimScalable});
      acc = vector::ShapeCastOp::create(rewriter, loc, accType, acc);
    }
    // 6. Creates the flattened form of vector.multi_reduction with inner/outer
    // most dim as reduction.
    Operation *newMultiDimRedOp = vector::MultiDimReductionOp::create(
        rewriter, loc, cast, acc, mask, multiReductionOp.getKind());
    newMultiDimRedOp =
        mlir::vector::maskOperation(rewriter, newMultiDimRedOp, newVectorMask);

    // 7. If there are no parallel shapes, the result is a scalar.
    // TODO: support 0-d vectors when available.
    if (parallelShapes.empty()) {
      rewriter.replaceOp(rootOp, newMultiDimRedOp->getResult(0));
      return success();
    }

    // 8. Shape cast the flattened result back to the original n-D parallel
    // shape.
    VectorType outputCastedType = VectorType::get(
        parallelShapes, multiReductionOp.getSourceVectorType().getElementType(),
        parallelScalableDims);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        rootOp, outputCastedType, newMultiDimRedOp->getResult(0));
    return success();
  }

private:
  const bool useInnerDimsForReduction;
};

/// Lowers 2D vector.multi_reduction to a squence of Arith Ops
///
/// The reduction dimension must be the outer-most dimension.
///
/// BEFORE:
///
///  %1 = vector.multi_reduction <mul>, %src, %acc [0] : vector<4x2xf32> to
///  vector<2xf32>
///
/// AFTER:
///
///   // Prod 1.
///   %vec_0 = vector.extract %src[0] : vector<2xf32> from vector<4x2xf32>
///   %mul_0 = arith.mulf %vec_0, %acc : vector<2xf32>
///
///   // Prod 2.
///   %vec_1 = vector.extract %src[1] : vector<2xf32> from vector<4x2xf32>
///   %mul_2 = arith.mulf %vec_1, %mul_0 : vector<2xf32>
///
///   // Prod 3.
///   %vec_3 = vector.extract %src[2] : vector<2xf32> from vector<4x2xf32>
///   %mul_3 = arith.mulf %vec_3, %mul_2 : vector<2xf32>
///
///   // Prod 4.
///   %vec_4 = vector.extract %src[3] : vector<2xf32> from vector<4x2xf32>
///   %res = arith.mulf %vec_4, %mul_3 : vector<2xf32>
struct TwoDimMultiReductionToElementWise
    : public vector::MaskableOpRewritePattern<vector::MultiDimReductionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::MultiDimReductionOp multiReductionOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-2 ["parallel", "reduce"] or bail.
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.isReducedDim(1) || !multiReductionOp.isReducedDim(0))
      return failure();

    Value mask = maskingOp ? maskingOp.getMask() : Value();

    auto loc = multiReductionOp.getLoc();
    Value source = multiReductionOp.getSource();
    ArrayRef<int64_t> srcShape =
        multiReductionOp.getSourceVectorType().getShape();
    int outerDim = srcShape[0];

    Value result = multiReductionOp.getAcc();
    for (int64_t i = 0; i < outerDim; i++) {
      auto v = vector::ExtractOp::create(rewriter, loc, source, i);
      Value m = mask ? Value(vector::ExtractOp::create(rewriter, loc, mask, i))
                     : nullptr;
      result = makeArithReduction(rewriter, loc, multiReductionOp.getKind(), v,
                                  result, /*fastmath=*/nullptr, m);
    }

    return result;
  }
};

/// Lowers 2D vector.multi_reduction to a squence of vector.reduction Ops
///
/// The reduction dimension must be the inner-most dimension.
///
/// BEFORE:
///  vector.multi_reduction <mul>, %src, %acc [1] : vector<2x4xf32> to
///  vector<2xf32>
///
/// AFTER:
///   // 1st reduction
///   %v_0 = vector.extract %src[0] : vector<4xf32> from vector<2x4xf32>
///   %a_0 = vector.extract %acc[0] : f32 from vector<2xf32>
///   %red_1 = vector.reduction <mul>, %v_0, %a_1 : vector<4xf32> into f32
///   %res_tmp = vector.insert %red_1, %res [0] : f32 into vector<2xf32>
///
///   // 2nd reduction
///   %v_1 = vector.extract %src[1] : vector<4xf32> from vector<2x4xf32>
///   %a_1 = vector.extract %acc[1] : f32 from vector<2xf32>
///   %red_2 = vector.reduction <mul>, %v_1, %a_1 : vector<4xf32> into f32
///   %res_final = vector.insert %red_2, %res_tmp [1] : f32 into vector<2xf32>
struct TwoDimMultiReductionToReduction
    : public vector::MaskableOpRewritePattern<vector::MultiDimReductionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::MultiDimReductionOp multiReductionOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-2 ["reduce", "parallel"] or bail.
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.isReducedDim(0) || !multiReductionOp.isReducedDim(1))
      return failure();

    Value mask = maskingOp ? maskingOp.getMask() : nullptr;

    auto loc = multiReductionOp.getLoc();
    Value source = multiReductionOp.getSource();
    Value acc = multiReductionOp.getAcc();
    int outerDim = multiReductionOp.getSourceVectorType().getShape()[0];

    Value result = arith::ConstantOp::create(
        rewriter, loc, multiReductionOp.getDestType(),
        rewriter.getZeroAttr(multiReductionOp.getDestType()));

    SmallVector<Value> vectors(outerDim);
    for (int64_t i = 0; i < outerDim; ++i) {
      Value v = vector::ExtractOp::create(rewriter, loc, source, i);
      Value a = vector::ExtractOp::create(rewriter, loc, acc, i);

      Operation *reductionOp = vector::ReductionOp::create(
          rewriter, loc, multiReductionOp.getKind(), v, a);

      if (mask) {
        Value m = vector::ExtractOp::create(rewriter, loc, mask, i);
        reductionOp = mlir::vector::maskOperation(rewriter, reductionOp, m);
      }

      result = vector::InsertOp::create(rewriter, loc,
                                        reductionOp->getResult(0), result, i);
    }

    return result;
  }
};

/// Converts 1d vector.multi_reduction with a single reduction dimension to a 2d
/// form with both a single parallel and reduction dimension.
/// This is achieved with a simple vector.shape_cast that inserts a leading 1.
/// The case with a single parallel dimension is a noop and folds away
/// separately.
struct OneDimMultiReductionToTwoDim
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    // Rank-1 or bail.
    if (srcRank != 1)
      return failure();

    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(multiReductionOp.getOperation());
    Operation *rootOp;
    Value mask;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
      mask = maskableOp.getMaskingOp().getMask();
    } else {
      rootOp = multiReductionOp;
    }

    auto loc = multiReductionOp.getLoc();
    auto srcVectorType = multiReductionOp.getSourceVectorType();
    auto srcShape = srcVectorType.getShape();
    auto castedType = VectorType::get(
        ArrayRef<int64_t>{1, srcShape.back()}, srcVectorType.getElementType(),
        ArrayRef<bool>{false, srcVectorType.getScalableDims().back()});

    auto accType =
        VectorType::get(ArrayRef<int64_t>{1}, srcVectorType.getElementType());
    assert(!llvm::isa<VectorType>(multiReductionOp.getDestType()) &&
           "multi_reduction with a single dimension expects a scalar result");

    // If the unique dim is reduced and we insert a parallel in front, we need a
    // {false, true} mask.
    SmallVector<bool, 2> reductionMask{false, true};

    /// vector.extract(vector.multi_reduce(vector.shape_cast(v, 1xk)), 0)
    Value cast = vector::ShapeCastOp::create(rewriter, loc, castedType,
                                             multiReductionOp.getSource());
    Value castAcc = vector::BroadcastOp::create(rewriter, loc, accType,
                                                multiReductionOp.getAcc());
    Value castMask;
    if (maskableOp.isMasked()) {
      auto maskType = llvm::cast<VectorType>(mask.getType());
      auto castMaskType = VectorType::get(
          ArrayRef<int64_t>{1, maskType.getShape().back()},
          maskType.getElementType(),
          ArrayRef<bool>{false, maskType.getScalableDims().back()});
      castMask = vector::BroadcastOp::create(rewriter, loc, castMaskType, mask);
    }

    Operation *newOp = vector::MultiDimReductionOp::create(
        rewriter, loc, cast, castAcc, reductionMask,
        multiReductionOp.getKind());
    newOp = vector::maskOperation(rewriter, newOp, castMask);

    rewriter.replaceOpWithNewOp<vector::ExtractOp>(rootOp, newOp->getResult(0),
                                                   ArrayRef<int64_t>{0});
    return success();
  }
};

struct LowerVectorMultiReductionPass
    : public vector::impl::LowerVectorMultiReductionBase<
          LowerVectorMultiReductionPass> {
  LowerVectorMultiReductionPass(vector::VectorMultiReductionLowering option) {
    this->loweringStrategy = option;
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);
    mlir::vector::populateVectorMultiReductionReorderAndExpandPatterns(
        patterns, this->loweringStrategy);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet flatteningPatterns(context);
    mlir::vector::populateVectorMultiReductionFlatteningPatterns(
        flatteningPatterns, this->loweringStrategy);
    if (failed(applyPatternsGreedily(op, std::move(flatteningPatterns))))
      signalPassFailure();

    RewritePatternSet unrollingPatterns(context);
    mlir::vector::populateVectorMultiReductionUnrollingPatterns(
        unrollingPatterns, this->loweringStrategy);
    if (failed(applyPatternsGreedily(op, std::move(unrollingPatterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
};

} // namespace

void mlir::vector::populateVectorMultiReductionReorderAndExpandPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options,
    PatternBenefit benefit) {
  patterns.add<OneDimMultiReductionToTwoDim>(patterns.getContext(), benefit);
  patterns.add<InnerOuterDimReductionConversion>(patterns.getContext(), options,
                                                 benefit);
}

void mlir::vector::populateVectorMultiReductionFlatteningPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options,
    PatternBenefit benefit) {
  patterns.add<FlattenMultiReduction>(patterns.getContext(), options, benefit);
}

void mlir::vector::populateVectorMultiReductionUnrollingPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options,
    PatternBenefit benefit) {
  if (options == VectorMultiReductionLowering ::InnerReduction)
    patterns.add<TwoDimMultiReductionToReduction>(patterns.getContext(),
                                                  benefit);
  else
    patterns.add<TwoDimMultiReductionToElementWise>(patterns.getContext(),
                                                    benefit);
}

std::unique_ptr<Pass> vector::createLowerVectorMultiReductionPass(
    vector::VectorMultiReductionLowering option) {
  return std::make_unique<LowerVectorMultiReductionPass>(option);
}
