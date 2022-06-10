//===- VectorDistribute.cpp - patterns to do vector distribution ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Transforms/SideEffectUtils.h"

using namespace mlir;
using namespace mlir::vector;

static LogicalResult
rewriteWarpOpToScfFor(RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
                      const WarpExecuteOnLane0LoweringOptions &options) {
  assert(warpOp.getBodyRegion().hasOneBlock() &&
         "expected WarpOp with single block");
  Block *warpOpBody = &warpOp.getBodyRegion().front();
  Location loc = warpOp.getLoc();

  // Passed all checks. Start rewriting.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);

  // Create scf.if op.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value isLane0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpOp.getLaneid(), c0);
  auto ifOp = rewriter.create<scf::IfOp>(loc, isLane0,
                                         /*withElseRegion=*/false);
  rewriter.eraseOp(ifOp.thenBlock()->getTerminator());

  // Store vectors that are defined outside of warpOp into the scratch pad
  // buffer.
  SmallVector<Value> bbArgReplacements;
  for (const auto &it : llvm::enumerate(warpOp.getArgs())) {
    Value val = it.value();
    Value bbArg = warpOpBody->getArgument(it.index());

    rewriter.setInsertionPoint(ifOp);
    Value buffer = options.warpAllocationFn(warpOp->getLoc(), rewriter, warpOp,
                                            bbArg.getType());

    // Store arg vector into buffer.
    rewriter.setInsertionPoint(ifOp);
    auto vectorType = val.getType().cast<VectorType>();
    int64_t storeSize = vectorType.getShape()[0];
    Value storeOffset = rewriter.create<arith::MulIOp>(
        loc, warpOp.getLaneid(),
        rewriter.create<arith::ConstantIndexOp>(loc, storeSize));
    rewriter.create<vector::StoreOp>(loc, val, buffer, storeOffset);

    // Load bbArg vector from buffer.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    auto bbArgType = bbArg.getType().cast<VectorType>();
    Value loadOp = rewriter.create<vector::LoadOp>(loc, bbArgType, buffer, c0);
    bbArgReplacements.push_back(loadOp);
  }

  // Insert sync after all the stores and before all the loads.
  if (!warpOp.getArgs().empty()) {
    rewriter.setInsertionPoint(ifOp);
    options.warpSyncronizationFn(warpOp->getLoc(), rewriter, warpOp);
  }

  // Move body of warpOp to ifOp.
  rewriter.mergeBlocks(warpOpBody, ifOp.thenBlock(), bbArgReplacements);

  // Rewrite terminator and compute replacements of WarpOp results.
  SmallVector<Value> replacements;
  auto yieldOp = cast<vector::YieldOp>(ifOp.thenBlock()->getTerminator());
  Location yieldLoc = yieldOp.getLoc();
  for (const auto &it : llvm::enumerate(yieldOp.operands())) {
    Value val = it.value();
    Type resultType = warpOp->getResultTypes()[it.index()];
    rewriter.setInsertionPoint(ifOp);
    Value buffer = options.warpAllocationFn(warpOp->getLoc(), rewriter, warpOp,
                                            val.getType());

    // Store yielded value into buffer.
    rewriter.setInsertionPoint(yieldOp);
    if (val.getType().isa<VectorType>())
      rewriter.create<vector::StoreOp>(yieldLoc, val, buffer, c0);
    else
      rewriter.create<memref::StoreOp>(yieldLoc, val, buffer, c0);

    // Load value from buffer (after warpOp).
    rewriter.setInsertionPointAfter(ifOp);
    if (resultType == val.getType()) {
      // Result type and yielded value type are the same. This is a broadcast.
      // E.g.:
      // %r = vector.warp_execute_on_lane_0(...) -> (f32) {
      //   vector.yield %cst : f32
      // }
      // Both types are f32. The constant %cst is broadcasted to all lanes.
      // This is described in more detail in the documentation of the op.
      Value loadOp = rewriter.create<memref::LoadOp>(loc, buffer, c0);
      replacements.push_back(loadOp);
    } else {
      auto loadedVectorType = resultType.cast<VectorType>();
      int64_t loadSize = loadedVectorType.getShape()[0];

      // loadOffset = laneid * loadSize
      Value loadOffset = rewriter.create<arith::MulIOp>(
          loc, warpOp.getLaneid(),
          rewriter.create<arith::ConstantIndexOp>(loc, loadSize));
      Value loadOp = rewriter.create<vector::LoadOp>(loc, loadedVectorType,
                                                     buffer, loadOffset);
      replacements.push_back(loadOp);
    }
  }

  // Insert sync after all the stores and before all the loads.
  if (!yieldOp.operands().empty()) {
    rewriter.setInsertionPointAfter(ifOp);
    options.warpSyncronizationFn(warpOp->getLoc(), rewriter, warpOp);
  }

  // Delete terminator and add empty scf.yield.
  rewriter.eraseOp(yieldOp);
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  rewriter.create<scf::YieldOp>(yieldLoc);

  // Compute replacements for WarpOp results.
  rewriter.replaceOp(warpOp, replacements);

  return success();
}

/// Helper to create a new WarpExecuteOnLane0Op with different signature.
static WarpExecuteOnLane0Op moveRegionToNewWarpOpAndReplaceReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);
  auto newWarpOp = rewriter.create<WarpExecuteOnLane0Op>(
      warpOp.getLoc(), newReturnTypes, warpOp.getLaneid(), warpOp.getWarpSize(),
      warpOp.getArgs(), warpOp.getBody()->getArgumentTypes());

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  rewriter.inlineRegionBefore(opBody, newOpBody, newOpBody.begin());
  auto yield =
      cast<vector::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());

  rewriter.updateRootInPlace(
      yield, [&]() { yield.operandsMutable().assign(newYieldedValues); });
  return newWarpOp;
}

/// Helper to create a new WarpExecuteOnLane0Op region with extra outputs.
static WarpExecuteOnLane0Op moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes) {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  types.append(newReturnTypes.begin(), newReturnTypes.end());
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  SmallVector<Value> yieldValues(yield.getOperands().begin(),
                                 yield.getOperands().end());
  yieldValues.append(newYieldedValues.begin(), newYieldedValues.end());
  WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
      rewriter, warpOp, yieldValues, types);
  rewriter.replaceOp(warpOp,
                     newWarpOp.getResults().take_front(warpOp.getNumResults()));
  return newWarpOp;
}

/// Helper to know if an op can be hoisted out of the region.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return llvm::all_of(op->getOperands(), definedOutside) &&
         isSideEffectFree(op) && op->getNumRegions() == 0;
}

namespace {

struct WarpOpToScfForPattern : public OpRewritePattern<WarpExecuteOnLane0Op> {
  WarpOpToScfForPattern(MLIRContext *context,
                        const WarpExecuteOnLane0LoweringOptions &options,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<WarpExecuteOnLane0Op>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    return rewriteWarpOpToScfFor(rewriter, warpOp, options);
  }

private:
  const WarpExecuteOnLane0LoweringOptions &options;
};

/// Distribute transfer_write ops based on the affine map returned by
/// `distributionMapFn`.
/// Example:
/// ```
/// %0 = vector.warp_execute_on_lane_0(%id){
///   ...
///   vector.transfer_write %v, %A[%c0] : vector<32xf32>, memref<128xf32>
///   vector.yield
/// }
/// ```
/// To
/// ```
/// %r:3 = vector.warp_execute_on_lane_0(%id) -> (vector<1xf32>) {
///   ...
///   vector.yield %v : vector<32xf32>
/// }
/// vector.transfer_write %v, %A[%id] : vector<1xf32>, memref<128xf32>
struct WarpOpTransferWrite : public OpRewritePattern<vector::TransferWriteOp> {
  WarpOpTransferWrite(MLIRContext *ctx, DistributionMapFn fn,
                      PatternBenefit b = 1)
      : OpRewritePattern<vector::TransferWriteOp>(ctx, b),
        distributionMapFn(fn) {}

  /// Distribute the TransferWriteOp. Only 1D distributions and vector dims that
  /// are multiples of the distribution ratio are supported at the moment.
  LogicalResult tryDistributeOp(RewriterBase &rewriter,
                                vector::TransferWriteOp writeOp,
                                WarpExecuteOnLane0Op warpOp) const {
    AffineMap map = distributionMapFn(writeOp);
    SmallVector<int64_t> targetShape(writeOp.getVectorType().getShape().begin(),
                                     writeOp.getVectorType().getShape().end());
    assert(map.getNumResults() == 1 &&
           "multi-dim distribution not implemented yet");
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      unsigned position = map.getDimPosition(i);
      if (targetShape[position] % warpOp.getWarpSize() != 0)
        return failure();
      targetShape[position] = targetShape[position] / warpOp.getWarpSize();
    }
    VectorType targetType =
        VectorType::get(targetShape, writeOp.getVectorType().getElementType());

    SmallVector<Value> yieldValues = {writeOp.getVector()};
    SmallVector<Type> retTypes = {targetType};
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Move op outside of region: Insert clone at the insertion point and delete
    // the old op.
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    rewriter.eraseOp(writeOp);

    rewriter.setInsertionPoint(newWriteOp);
    AffineMap indexMap = map.compose(newWriteOp.getPermutationMap());
    Location loc = newWriteOp.getLoc();
    SmallVector<Value> indices(newWriteOp.getIndices().begin(),
                               newWriteOp.getIndices().end());
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(newWarpOp.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale =
          getAffineConstantExpr(targetShape[vectorPos], newWarpOp.getContext());
      indices[indexPos] =
          makeComposedAffineApply(rewriter, loc, d0 + scale * d1,
                                  {indices[indexPos], newWarpOp.getLaneid()});
    }
    newWriteOp.getVectorMutable().assign(newWarpOp.getResults().back());
    newWriteOp.getIndicesMutable().assign(indices);

    return success();
  }

  /// Extract TransferWriteOps of vector<1x> into a separate warp op.
  LogicalResult tryExtractOp(RewriterBase &rewriter,
                             vector::TransferWriteOp writeOp,
                             WarpExecuteOnLane0Op warpOp) const {
    Location loc = writeOp.getLoc();
    VectorType vecType = writeOp.getVectorType();

    // Only vector<1x> is supported at the moment.
    if (vecType.getShape().size() != 1 || vecType.getShape()[0] != 1)
      return failure();

    // Do not process warp ops that contain only TransferWriteOps.
    if (llvm::all_of(warpOp.getOps(), [](Operation &op) {
          return isa<vector::TransferWriteOp, vector::YieldOp>(&op);
        }))
      return failure();

    SmallVector<Value> yieldValues = {writeOp.getVector()};
    SmallVector<Type> retTypes = {vecType};
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Create a second warp op that contains only writeOp.
    auto secondWarpOp = rewriter.create<WarpExecuteOnLane0Op>(
        loc, TypeRange(), newWarpOp.getLaneid(), newWarpOp.getWarpSize());
    Block &body = secondWarpOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&body);
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    newWriteOp.getVectorMutable().assign(
        newWarpOp.getResult(newWarpOp.getNumResults() - 1));
    rewriter.eraseOp(writeOp);
    rewriter.create<vector::YieldOp>(newWarpOp.getLoc());
    return success();
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // Ops with mask not supported yet.
    if (writeOp.getMask())
      return failure();

    auto warpOp = dyn_cast<WarpExecuteOnLane0Op>(writeOp->getParentOp());
    if (!warpOp)
      return failure();

    // There must be no op with a side effect after writeOp.
    Operation *nextOp = writeOp.getOperation();
    while ((nextOp = nextOp->getNextNode()))
      if (!isSideEffectFree(nextOp))
        return failure();

    if (!llvm::all_of(writeOp->getOperands(), [&](Value value) {
          return writeOp.getVector() == value ||
                 warpOp.isDefinedOutsideOfRegion(value);
        }))
      return failure();

    if (succeeded(tryDistributeOp(rewriter, writeOp, warpOp)))
      return success();

    if (succeeded(tryExtractOp(rewriter, writeOp, warpOp)))
      return success();

    return failure();
  }

private:
  DistributionMapFn distributionMapFn;
};

} // namespace

void mlir::vector::populateWarpExecuteOnLane0OpToScfForPattern(
    RewritePatternSet &patterns,
    const WarpExecuteOnLane0LoweringOptions &options) {
  patterns.add<WarpOpToScfForPattern>(patterns.getContext(), options);
}

void mlir::vector::populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, DistributionMapFn distributionMapFn) {
  patterns.add<WarpOpTransferWrite>(patterns.getContext(), distributionMapFn);
}

void mlir::vector::moveScalarUniformCode(WarpExecuteOnLane0Op warpOp) {
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return result.getType().isa<VectorType>();
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}
