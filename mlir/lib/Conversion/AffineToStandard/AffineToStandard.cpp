//===- AffineToStandard.cpp - Lower affine constructs to primitives -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers affine constructs (If and For statements, AffineApply
// operations) within a function into their standard If and For equivalent ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vector;

/// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate.  The predicate defines which
/// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
/// `cmpi` operation followed by the `select` operation:
///
///   %cond   = arith.cmpi "predicate" %v0, %v1
///   %result = select %cond, %v0, %v1
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
static Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<arith::CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<arith::SelectOp>(loc, cmpOp.getResult(), value,
                                            *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to the affine map in the upper bound
/// applied to the respective operands, and compute the minimum value across
/// the results.
Value mlir::lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMin(builder, op.getLoc(), op.getUpperBoundMap(),
                           op.getUpperBoundOperands());
}

/// Emit instructions that correspond to the affine map in the lower bound
/// applied to the respective operands, and compute the maximum value across
/// the results.
Value mlir::lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  return lowerAffineMapMax(builder, op.getLoc(), op.getLowerBoundMap(),
                           op.getLowerBoundOperands());
}

namespace {
class AffineMinLowering : public OpRewritePattern<AffineMinOp> {
public:
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp op,
                                PatternRewriter &rewriter) const override {
    Value reduced =
        lowerAffineMapMin(rewriter, op.getLoc(), op.map(), op.operands());
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

class AffineMaxLowering : public OpRewritePattern<AffineMaxOp> {
public:
  using OpRewritePattern<AffineMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMaxOp op,
                                PatternRewriter &rewriter) const override {
    Value reduced =
        lowerAffineMapMax(rewriter, op.getLoc(), op.map(), op.operands());
    if (!reduced)
      return failure();

    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Affine yields ops are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<scf::ParallelOp>(op->getParentOp())) {
      // scf.parallel does not yield any values via its terminator scf.yield but
      // models reductions differently using additional ops in its region.
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.operands());
    return success();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.region(), scfForOp.getRegion(),
                                scfForOp.getRegion().end());
    rewriter.replaceOp(op, scfForOp.getResults());
    return success();
  }
};

/// Convert an `affine.parallel` (loop nest) operation into a `scf.parallel`
/// operation.
class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 8> steps;
    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    SmallVector<Value, 8> identityVals;
    // Emit IR computing the lower and upper bound by expanding the map
    // expression.
    lowerBoundTuple.reserve(op.getNumDims());
    upperBoundTuple.reserve(op.getNumDims());
    for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
      Value lower = lowerAffineMapMax(rewriter, loc, op.getLowerBoundMap(i),
                                      op.getLowerBoundsOperands());
      if (!lower)
        return rewriter.notifyMatchFailure(op, "couldn't convert lower bounds");
      lowerBoundTuple.push_back(lower);

      Value upper = lowerAffineMapMin(rewriter, loc, op.getUpperBoundMap(i),
                                      op.getUpperBoundsOperands());
      if (!upper)
        return rewriter.notifyMatchFailure(op, "couldn't convert upper bounds");
      upperBoundTuple.push_back(upper);
    }
    steps.reserve(op.steps().size());
    for (int64_t step : op.steps())
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, step));

    // Get the terminator op.
    Operation *affineParOpTerminator = op.getBody()->getTerminator();
    scf::ParallelOp parOp;
    if (op.results().empty()) {
      // Case with no reduction operations/return values.
      parOp = rewriter.create<scf::ParallelOp>(loc, lowerBoundTuple,
                                               upperBoundTuple, steps,
                                               /*bodyBuilderFn=*/nullptr);
      rewriter.eraseBlock(parOp.getBody());
      rewriter.inlineRegionBefore(op.region(), parOp.getRegion(),
                                  parOp.getRegion().end());
      rewriter.replaceOp(op, parOp.getResults());
      return success();
    }
    // Case with affine.parallel with reduction operations/return values.
    // scf.parallel handles the reduction operation differently unlike
    // affine.parallel.
    ArrayRef<Attribute> reductions = op.reductions().getValue();
    for (auto pair : llvm::zip(reductions, op.getResultTypes())) {
      // For each of the reduction operations get the identity values for
      // initialization of the result values.
      Attribute reduction = std::get<0>(pair);
      Type resultType = std::get<1>(pair);
      Optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              static_cast<uint64_t>(reduction.cast<IntegerAttr>().getInt()));
      assert(reductionOp && "Reduction operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = reductionOp.getValue();
      identityVals.push_back(
          arith::getIdentityValue(reductionOpValue, resultType, rewriter, loc));
    }
    parOp = rewriter.create<scf::ParallelOp>(
        loc, lowerBoundTuple, upperBoundTuple, steps, identityVals,
        /*bodyBuilderFn=*/nullptr);

    //  Copy the body of the affine.parallel op.
    rewriter.eraseBlock(parOp.getBody());
    rewriter.inlineRegionBefore(op.region(), parOp.getRegion(),
                                parOp.getRegion().end());
    assert(reductions.size() == affineParOpTerminator->getNumOperands() &&
           "Unequal number of reductions and operands.");
    for (unsigned i = 0, end = reductions.size(); i < end; i++) {
      // For each of the reduction operations get the respective mlir::Value.
      Optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              reductions[i].cast<IntegerAttr>().getInt());
      assert(reductionOp && "Reduction Operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = reductionOp.getValue();
      rewriter.setInsertionPoint(&parOp.getBody()->back());
      auto reduceOp = rewriter.create<scf::ReduceOp>(
          loc, affineParOpTerminator->getOperand(i));
      rewriter.setInsertionPointToEnd(&reduceOp.getReductionOperator().front());
      Value reductionResult = arith::getReductionOp(
          reductionOpValue, rewriter, loc,
          reduceOp.getReductionOperator().front().getArgument(0),
          reduceOp.getReductionOperator().front().getArgument(1));
      rewriter.create<scf::ReduceReturnOp>(loc, reductionResult);
    }
    rewriter.replaceOp(op, parOp.getResults());
    return success();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value zeroConstant = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                         operandsRef.take_front(numDims),
                                         operandsRef.drop_front(numDims));
      if (!affResult)
        return failure();
      auto pred =
          isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value cmpVal =
          rewriter.create<arith::CmpIOp>(loc, pred, affResult, zeroConstant);
      cond = cond
                 ? rewriter.create<arith::AndIOp>(loc, cond, cmpVal).getResult()
                 : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                        /*width=*/1);

    bool hasElseRegion = !op.elseRegion().empty();
    auto ifOp = rewriter.create<scf::IfOp>(loc, op.getResultTypes(), cond,
                                           hasElseRegion);
    rewriter.inlineRegionBefore(op.thenRegion(), &ifOp.getThenRegion().back());
    rewriter.eraseBlock(&ifOp.getThenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.elseRegion(),
                                  &ifOp.getElseRegion().back());
      rewriter.eraseBlock(&ifOp.getElseRegion().back());
    }

    // Replace the Affine IfOp finally.
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.prefetch' operation to its operands,
/// and feed the results to a newly created 'memref.prefetch' operation (which
/// replaces the original 'affine.prefetch').
class AffinePrefetchLowering : public OpRewritePattern<AffinePrefetchOp> {
public:
  using OpRewritePattern<AffinePrefetchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffinePrefetchOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affinePrefetchOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build memref.prefetch memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::PrefetchOp>(
        op, op.memref(), *resultOperands, op.isWrite(), op.localityHint(),
        op.isDataCache());
    return success();
  }
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine maps from an 'affine.dma_start' operation to each of their
/// respective map operands, and feed the results to a newly created
/// 'memref.dma_start' operation (which replaces the original
/// 'affine.dma_start').
class AffineDmaStartLowering : public OpRewritePattern<AffineDmaStartOp> {
public:
  using OpRewritePattern<AffineDmaStartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaStartOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Expand affine map for DMA source memref.
    auto maybeExpandedSrcMap = expandAffineMap(
        rewriter, op.getLoc(), op.getSrcMap(),
        operandsRef.drop_front(op.getSrcMemRefOperandIndex() + 1));
    if (!maybeExpandedSrcMap)
      return failure();
    // Expand affine map for DMA destination memref.
    auto maybeExpandedDstMap = expandAffineMap(
        rewriter, op.getLoc(), op.getDstMap(),
        operandsRef.drop_front(op.getDstMemRefOperandIndex() + 1));
    if (!maybeExpandedDstMap)
      return failure();
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap = expandAffineMap(
        rewriter, op.getLoc(), op.getTagMap(),
        operandsRef.drop_front(op.getTagMemRefOperandIndex() + 1));
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_start operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaStartOp>(
        op, op.getSrcMemRef(), *maybeExpandedSrcMap, op.getDstMemRef(),
        *maybeExpandedDstMap, op.getNumElements(), op.getTagMemRef(),
        *maybeExpandedTagMap, op.getStride(), op.getNumElementsPerStride());
    return success();
  }
};

/// Apply the affine map from an 'affine.dma_wait' operation tag memref,
/// and feed the results to a newly created 'memref.dma_wait' operation (which
/// replaces the original 'affine.dma_wait').
class AffineDmaWaitLowering : public OpRewritePattern<AffineDmaWaitOp> {
public:
  using OpRewritePattern<AffineDmaWaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineDmaWaitOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map for DMA tag memref.
    SmallVector<Value, 8> indices(op.getTagIndices());
    auto maybeExpandedTagMap =
        expandAffineMap(rewriter, op.getLoc(), op.getTagMap(), indices);
    if (!maybeExpandedTagMap)
      return failure();

    // Build memref.dma_wait operation with affine map results.
    rewriter.replaceOpWithNewOp<memref::DmaWaitOp>(
        op, op.getTagMemRef(), *maybeExpandedTagMap, op.getNumElements());
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_load' operation to its operands,
/// and feed the results to a newly created 'vector.load' operation (which
/// replaces the original 'affine.vector_load').
class AffineVectorLoadLowering : public OpRewritePattern<AffineVectorLoadOp> {
public:
  using OpRewritePattern<AffineVectorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<vector::LoadOp>(
        op, op.getVectorType(), op.getMemRef(), *resultOperands);
    return success();
  }
};

/// Apply the affine map from an 'affine.vector_store' operation to its
/// operands, and feed the results to a newly created 'vector.store' operation
/// (which replaces the original 'affine.vector_store').
class AffineVectorStoreLowering : public OpRewritePattern<AffineVectorStoreOp> {
public:
  using OpRewritePattern<AffineVectorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineVectorStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineVectorStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

} // namespace

void mlir::populateAffineToStdConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AffineApplyLowering,
      AffineDmaStartLowering,
      AffineDmaWaitLowering,
      AffineLoadLowering,
      AffineMinLowering,
      AffineMaxLowering,
      AffineParallelLowering,
      AffinePrefetchLowering,
      AffineStoreLowering,
      AffineForLowering,
      AffineIfLowering,
      AffineYieldOpLowering>(patterns.getContext());
  // clang-format on
}

void mlir::populateAffineToVectorConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AffineVectorLoadLowering,
      AffineVectorStoreLowering>(patterns.getContext());
  // clang-format on
}

namespace {
class LowerAffinePass : public ConvertAffineToStandardBase<LowerAffinePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateAffineToVectorConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,
                           scf::SCFDialect, VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
std::unique_ptr<Pass> mlir::createLowerAffinePass() {
  return std::make_unique<LowerAffinePass>();
}
