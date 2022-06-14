//===- VectorDistribute.cpp - patterns to do vector distribution ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(WarpExecuteOnLane0Op warpOp,
                                std::function<bool(Operation *)> fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

// Clones `op` into a new operation that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(RewriterBase &rewriter,
                                              Location loc, Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return rewriter.create(res);
}

/// Currently the distribution map is implicit based on the vector shape. In the
/// future it will be part of the op.
/// Example:
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (vector<1x16x2xf32>) {
///   ...
///   vector.yield %3 : vector<32x16x64xf32>
/// }
/// ```
/// Would have an implicit map of:
/// `(d0, d1, d2) -> (d0, d2)`
static AffineMap calculateImplicitMap(Value yield, Value ret) {
  auto srcType = yield.getType().cast<VectorType>();
  auto dstType = ret.getType().cast<VectorType>();
  SmallVector<AffineExpr> perm;
  // Check which dimensions of the yield value are different than the dimensions
  // of the result to know the distributed dimensions. Then associate each
  // distributed dimension to an ID in order.
  for (unsigned i = 0, e = srcType.getRank(); i < e; i++) {
    if (srcType.getDimSize(i) != dstType.getDimSize(i))
      perm.push_back(getAffineDimExpr(i, yield.getContext()));
  }
  auto map = AffineMap::get(srcType.getRank(), 0, perm, yield.getContext());
  return map;
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

/// Sink out elementwise op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %3 = arith.addf %1, %2 : vector<32xf32>
///   vector.yield %3 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %r:3 = vector.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %4 = arith.addf %2, %3 : vector<32xf32>
///   vector.yield %4, %2, %3 : vector<32xf32>, vector<32xf32>,
///   vector<32xf32>
/// }
/// %0 = arith.addf %r#1, %r#2 : vector<1xf32>
struct WarpOpElementwise : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(warpOp, [](Operation *op) {
      return OpTrait::hasElementwiseMappableTraits(op);
    });
    if (!yieldOperand)
      return failure();
    Operation *elementWise = yieldOperand->get().getDefiningOp();
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    Location loc = warpOp.getLoc();
    for (OpOperand &operand : elementWise->getOpOperands()) {
      Type targetType;
      if (auto vecType = distributedVal.getType().dyn_cast<VectorType>()) {
        // If the result type is a vector, the operands must also be vectors.
        auto operandType = operand.get().getType().cast<VectorType>();
        targetType =
            VectorType::get(vecType.getShape(), operandType.getElementType());
      } else {
        auto operandType = operand.get().getType();
        assert(!operandType.isa<VectorType>() &&
               "unexpected yield of vector from op with scalar result type");
        targetType = operandType;
      }
      retTypes.push_back(targetType);
      yieldValues.push_back(operand.get());
    }
    unsigned numResults = warpOp.getNumResults();
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : llvm::seq(unsigned(0), elementWise->getNumOperands())) {
      newOperands[i] = newWarpOp.getResult(i + numResults);
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, loc, elementWise, newOperands,
        {newWarpOp.getResult(operandIndex).getType()});
    newWarpOp.getResult(operandIndex).replaceAllUsesWith(newOp->getResult(0));
    return success();
  }
};

/// Sink out transfer_read op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
//    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
//    vector<32xf32>
///   vector.yield %2 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %dead = vector.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
///   vector<32xf32> vector.yield %2 : vector<32xf32>
/// }
/// %0 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<1xf32>
struct WarpOpTransferRead : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::TransferReadOp>(op); });
    if (!operand)
      return failure();
    auto read = operand->get().getDefiningOp<vector::TransferReadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(read.getIndices().begin(),
                                  read.getIndices().end());
    AffineMap map = calculateImplicitMap(read.getResult(), distributedVal);
    AffineMap indexMap = map.compose(read.getPermutationMap());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      int64_t scale =
          distributedVal.getType().cast<VectorType>().getDimSize(vectorPos);
      indices[indexPos] =
          makeComposedAffineApply(rewriter, read.getLoc(), d0 + scale * d1,
                                  {indices[indexPos], warpOp.getLaneid()});
    }
    Value newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), distributedVal.getType(), read.getSource(), indices,
        read.getPermutationMapAttr(), read.getPadding(), read.getMask(),
        read.getInBoundsAttr());
    distributedVal.replaceAllUsesWith(newRead);
    return success();
  }
};

/// Remove any result that has no use along with the matching yieldOp operand.
// TODO: Move this in WarpExecuteOnLane0Op canonicalization.
struct WarpOpDeadResult : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValues;
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        continue;
      resultTypes.push_back(result.getType());
      yieldValues.push_back(yield.getOperand(result.getResultNumber()));
    }
    if (yield.getNumOperands() == yieldValues.size())
      return failure();
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
        rewriter, warpOp, yieldValues, resultTypes);
    unsigned resultIndex = 0;
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        continue;
      result.replaceAllUsesWith(newWarpOp.getResult(resultIndex++));
    }
    rewriter.eraseOp(warpOp);
    return success();
  }
};

// If an operand is directly yielded out of the region we can forward it
// directly and it doesn't need to go through the region.
struct WarpOpForwardOperand : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValues;
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Value valForwarded;
    unsigned resultIndex;
    for (OpOperand &operand : yield->getOpOperands()) {
      Value result = warpOp.getResult(operand.getOperandNumber());
      if (result.use_empty())
        continue;

      // Assume all the values coming from above are uniform.
      if (!warpOp.getBodyRegion().isAncestor(operand.get().getParentRegion())) {
        if (result.getType() != operand.get().getType())
          continue;
        valForwarded = operand.get();
        resultIndex = operand.getOperandNumber();
        break;
      }
      auto arg = operand.get().dyn_cast<BlockArgument>();
      if (!arg || arg.getOwner()->getParentOp() != warpOp.getOperation())
        continue;
      Value warpOperand = warpOp.getArgs()[arg.getArgNumber()];
      if (result.getType() != warpOperand.getType())
        continue;
      valForwarded = warpOperand;
      resultIndex = operand.getOperandNumber();
      break;
    }
    if (!valForwarded)
      return failure();
    warpOp.getResult(resultIndex).replaceAllUsesWith(valForwarded);
    return success();
  }
};

struct WarpOpBroadcast : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::BroadcastOp>(op); });
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto broadcastOp = operand->get().getDefiningOp<vector::BroadcastOp>();
    Location loc = broadcastOp.getLoc();
    auto destVecType =
        warpOp->getResultTypes()[operandNumber].cast<VectorType>();
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {broadcastOp.getSource()},
        {broadcastOp.getSource().getType()});
    rewriter.setInsertionPointAfter(newWarpOp);
    Value broadcasted = rewriter.create<vector::BroadcastOp>(
        loc, destVecType, newWarpOp->getResults().back());
    newWarpOp->getResult(operandNumber).replaceAllUsesWith(broadcasted);

    return success();
  }
};

/// Sink scf.for region out of WarpExecuteOnLane0Op. This can be done only if
/// the scf.ForOp is the last operation in the region so that it doesn't change
/// the order of execution. This creates a new scf.for region after the
/// WarpExecuteOnLane0Op. The new scf.for region will contain a new
/// WarpExecuteOnLane0Op region. Example:
/// ```
/// %w = vector.warp_execute_on_lane_0(%laneid) -> (vector<4xf32>) {
///   ...
///   %v1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %v)
///   -> (vector<128xf32>) {
///     ...
///     scf.yield %r : vector<128xf32>
///   }
///   vector.yield %v1 : vector<128xf32>
/// }
/// ```
/// To:
/// %w0 = vector.warp_execute_on_lane_0(%arg0) -> (vector<4xf32>) {
///   ...
///   vector.yield %v : vector<128xf32>
/// }
/// %w = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%varg = %q0)
///   -> (vector<4xf32>) {
///     %iw = vector.warp_execute_on_lane_0(%laneid)
///     args(%varg : vector<4xf32>) -> (vector<4xf32>) {
///     ^bb0(%arg: vector<128xf32>):
///       ...
///       vector.yield %ir : vector<128xf32>
///     }
///     scf.yield %iw : vector<4xf32>
///  }
/// ```
struct WarpOpScfForOp : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    // Only pick up forOp if it is the last op in the region.
    Operation *lastNode = yield->getPrevNode();
    auto forOp = dyn_cast_or_null<scf::ForOp>(lastNode);
    if (!forOp)
      return failure();
    SmallVector<Value> newOperands;
    SmallVector<unsigned> resultIdx;
    // Collect all the outputs coming from the forOp.
    for (OpOperand &yieldOperand : yield->getOpOperands()) {
      if (yieldOperand.get().getDefiningOp() != forOp.getOperation())
        continue;
      auto forResult = yieldOperand.get().cast<OpResult>();
      newOperands.push_back(warpOp.getResult(yieldOperand.getOperandNumber()));
      yieldOperand.set(forOp.getIterOperands()[forResult.getResultNumber()]);
      resultIdx.push_back(yieldOperand.getOperandNumber());
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // Create a new for op outside the region with a WarpExecuteOnLane0Op region
    // inside.
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newOperands);
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    auto innerWarp = rewriter.create<WarpExecuteOnLane0Op>(
        warpOp.getLoc(), newForOp.getResultTypes(), warpOp.getLaneid(),
        warpOp.getWarpSize(), newForOp.getRegionIterArgs(),
        forOp.getResultTypes());

    SmallVector<Value> argMapping;
    argMapping.push_back(newForOp.getInductionVar());
    for (Value args : innerWarp.getBody()->getArguments()) {
      argMapping.push_back(args);
    }
    SmallVector<Value> yieldOperands;
    for (Value operand : forOp.getBody()->getTerminator()->getOperands())
      yieldOperands.push_back(operand);
    rewriter.eraseOp(forOp.getBody()->getTerminator());
    rewriter.mergeBlocks(forOp.getBody(), innerWarp.getBody(), argMapping);
    rewriter.setInsertionPoint(innerWarp.getBody(), innerWarp.getBody()->end());
    rewriter.create<vector::YieldOp>(innerWarp.getLoc(), yieldOperands);
    rewriter.setInsertionPointAfter(innerWarp);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), innerWarp.getResults());
    rewriter.eraseOp(forOp);
    // Replace the warpOp result coming from the original ForOp.
    for (const auto &res : llvm::enumerate(resultIdx)) {
      warpOp.getResult(res.value())
          .replaceAllUsesWith(newForOp.getResult(res.index()));
      newForOp->setOperand(res.index() + 3, warpOp.getResult(res.value()));
    }
    return success();
  }
};

/// A pattern that extracts vector.reduction ops from a WarpExecuteOnLane0Op.
/// The vector is reduced in parallel. Currently limited to vector<32x...>
/// values. Every lane reduces two scalars, 5 times in a row.
/// E.g.:
/// ```
/// %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
///   %0 = "some_def"() : () -> (vector<32xf32>)
///   %1 = vector.reduction "add", %0 : vector<32xf32> into f32
///   vector_ext.yield %1 : f32
/// }
/// ```
/// is lowered to:
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<1xf32>) {
///   %1 = "some_def"() : () -> (vector<32xf32>)
///   vector_ext.yield %1 : vector<32xf32>
/// }
/// %a = vector.extract %0[0] : vector<1xf32>
/// %r0, %s0 = gpu.shuffle xor %e, %c1, %c32 : f32
/// %a0 = arith.addf %a, %r0 : f32
/// %r1, %s1 = gpu.shuffle xor %a0, %c2, %c32 : f32
/// %a1 = arith.addf %a0, %r1 : f32
/// ...
/// %r4, %s4 = gpu.shuffle xor %a3, %c16, %c32 : f32
/// %r = arith.addf %a3, %r4 : f32
/// ```
struct ReductionToGPUWarpShuffle
    : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::ReductionOp>(op); });
    if (!yieldOperand)
      return failure();

    auto reductionOp =
        cast<vector::ReductionOp>(yieldOperand->get().getDefiningOp());
    auto vectorType = reductionOp.getVector().getType().cast<VectorType>();
    // Only rank 1 vectors supported.
    if (vectorType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          warpOp, "Only rank 1 reductions can be distributed.");
    // Only warp_size-sized vectors supported.
    if (static_cast<uint64_t>(vectorType.getShape()[0]) != warpOp.getWarpSize())
      return rewriter.notifyMatchFailure(
          warpOp, "Reduction vector dimension must match was size.");
    // Only f32 and i32 element types are supported.
    if (!reductionOp.getType().isF32() &&
        !reductionOp.getType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          warpOp,
          "Reduction distribution currently only supports 32bits types.");

    Location yieldLoc = yieldOperand->getOwner()->getLoc();

    // Return vector that will be reduced from the WarpExecuteOnLane0Op.
    unsigned operandIndex = yieldOperand->getOperandNumber();
    SmallVector<Value> yieldValues = {reductionOp.getVector()};
    SmallVector<Type> retTypes = {VectorType::get({1}, reductionOp.getType())};
    unsigned numResults = warpOp.getNumResults();
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Every lane has one scalar value. These should be reduced.
    Value laneValVec = newWarpOp.getResult(numResults);
    Value laneVal = rewriter.create<vector::ExtractOp>(yieldLoc, laneValVec, 0);

    // Parallel reduction using butterfly shuffles.
    for (uint64_t i = 1; i < newWarpOp.getWarpSize(); i <<= 1) {
      Value shuffled =
          rewriter
              .create<gpu::ShuffleOp>(reductionOp.getLoc(), laneVal, i,
                                      /*width=*/newWarpOp.getWarpSize(),
                                      /*mode=*/gpu::ShuffleMode::XOR)
              .result();
      laneVal = makeArithReduction(rewriter, reductionOp.getLoc(),
                                   reductionOp.getKind(), laneVal, shuffled);
    }

    newWarpOp.getResult(operandIndex).replaceAllUsesWith(laneVal);
    return success();
  }
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

void mlir::vector::populatePropagateWarpVectorDistributionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<WarpOpElementwise, WarpOpTransferRead, WarpOpDeadResult,
               WarpOpBroadcast, WarpOpForwardOperand, WarpOpScfForOp>(
      patterns.getContext());
}

void mlir::vector::populateReductionToGPUWarpShufflePatterns(
    RewritePatternSet &patterns) {
  patterns.add<ReductionToGPUWarpShuffle>(patterns.getContext());
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
