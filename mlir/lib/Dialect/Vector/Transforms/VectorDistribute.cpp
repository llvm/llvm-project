//===- VectorDistribute.cpp - patterns to do vector distribution ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/SideEffectUtils.h"
#include "llvm/ADT/SetVector.h"
#include <utility>

using namespace mlir;
using namespace mlir::vector;

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
static AffineMap calculateImplicitMap(VectorType sequentialType,
                                      VectorType distributedType) {
  SmallVector<AffineExpr> perm;
  perm.reserve(1);
  // Check which dimensions of the sequential type are different than the
  // dimensions of the distributed type to know the distributed dimensions. Then
  // associate each distributed dimension to an ID in order.
  for (unsigned i = 0, e = sequentialType.getRank(); i < e; i++) {
    if (sequentialType.getDimSize(i) != distributedType.getDimSize(i))
      perm.push_back(getAffineDimExpr(i, distributedType.getContext()));
  }
  auto map = AffineMap::get(sequentialType.getRank(), 0, perm,
                            distributedType.getContext());
  assert(map.getNumResults() <= 1 &&
         "only support distribution along one dimension for now.");
  return map;
}

namespace {

/// Helper struct to create the load / store operations that permit transit
/// through the parallel / sequential and the sequential / parallel boundaries
/// when performing `rewriteWarpOpToScfFor`.
///
/// The vector distribution dimension is inferred from the vector types.
struct DistributedLoadStoreHelper {
  DistributedLoadStoreHelper(Value sequentialVal, Value distributedVal,
                             Value laneId, Value zero)
      : sequentialVal(sequentialVal), distributedVal(distributedVal),
        laneId(laneId), zero(zero) {
    sequentialVectorType = sequentialVal.getType().dyn_cast<VectorType>();
    distributedVectorType = distributedVal.getType().dyn_cast<VectorType>();
    if (sequentialVectorType && distributedVectorType)
      distributionMap =
          calculateImplicitMap(sequentialVectorType, distributedVectorType);
  }

  Value buildDistributedOffset(RewriterBase &b, Location loc, int64_t index) {
    int64_t distributedSize = distributedVectorType.getDimSize(index);
    AffineExpr tid = getAffineSymbolExpr(0, b.getContext());
    return b.createOrFold<AffineApplyOp>(loc, tid * distributedSize,
                                         ArrayRef<Value>{laneId});
  }

  /// Create a store during the process of distributing the
  /// `vector.warp_execute_on_thread_0` op.
  /// Vector distribution assumes the following convention regarding the
  /// temporary buffers that are created to transition values. This **must**
  /// be properly specified in the `options.warpAllocationFn`:
  ///   1. scalars of type T transit through a memref<1xT>.
  ///   2. vectors of type V<shapexT> transit through a memref<shapexT>
  Operation *buildStore(RewriterBase &b, Location loc, Value val,
                        Value buffer) {
    assert((val == distributedVal || val == sequentialVal) &&
           "Must store either the preregistered distributed or the "
           "preregistered sequential value.");
    // Scalar case can directly use memref.store.
    if (!val.getType().isa<VectorType>())
      return b.create<memref::StoreOp>(loc, val, buffer, zero);

    // Vector case must use vector::TransferWriteOp which will later lower to
    //   vector.store of memref.store depending on further lowerings.
    int64_t rank = sequentialVectorType.getRank();
    SmallVector<Value> indices(rank, zero);
    if (val == distributedVal) {
      for (auto dimExpr : distributionMap.getResults()) {
        int64_t index = dimExpr.cast<AffineDimExpr>().getPosition();
        indices[index] = buildDistributedOffset(b, loc, index);
      }
    }
    SmallVector<bool> inBounds(indices.size(), true);
    return b.create<vector::TransferWriteOp>(
        loc, val, buffer, indices,
        ArrayRef<bool>(inBounds.begin(), inBounds.end()));
  }

  /// Create a load during the process of distributing the
  /// `vector.warp_execute_on_thread_0` op.
  /// Vector distribution assumes the following convention regarding the
  /// temporary buffers that are created to transition values. This **must**
  /// be properly specified in the `options.warpAllocationFn`:
  ///   1. scalars of type T transit through a memref<1xT>.
  ///   2. vectors of type V<shapexT> transit through a memref<shapexT>
  ///
  /// When broadcastMode is true, the load is not distributed to account for
  /// the broadcast semantics of the `vector.warp_execute_on_lane_0` op.
  ///
  /// Example:
  ///
  /// ```
  ///   %r = vector.warp_execute_on_lane_0(...) -> (f32) {
  ///     vector.yield %cst : f32
  ///   }
  ///   // Both types are f32. The constant %cst is broadcasted to all lanes.
  /// ```
  /// This behavior described in more detail in the documentation of the op.
  Value buildLoad(RewriterBase &b, Location loc, Type type, Value buffer) {

    // Scalar case can directly use memref.store.
    if (!type.isa<VectorType>())
      return b.create<memref::LoadOp>(loc, buffer, zero);

    // Other cases must be vector atm.
    // Vector case must use vector::TransferReadOp which will later lower to
    //   vector.read of memref.read depending on further lowerings.
    assert((type == distributedVectorType || type == sequentialVectorType) &&
           "Must store either the preregistered distributed or the "
           "preregistered sequential type.");
    SmallVector<Value> indices(sequentialVectorType.getRank(), zero);
    if (type == distributedVectorType) {
      for (auto dimExpr : distributionMap.getResults()) {
        int64_t index = dimExpr.cast<AffineDimExpr>().getPosition();
        indices[index] = buildDistributedOffset(b, loc, index);
      }
    }
    SmallVector<bool> inBounds(indices.size(), true);
    return b.create<vector::TransferReadOp>(
        loc, type.cast<VectorType>(), buffer, indices,
        ArrayRef<bool>(inBounds.begin(), inBounds.end()));
  }

  Value sequentialVal, distributedVal, laneId, zero;
  VectorType sequentialVectorType, distributedVectorType;
  AffineMap distributionMap;
};

} // namespace

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
  Block &newOpFirstBlock = newOpBody.front();
  rewriter.inlineRegionBefore(opBody, newOpBody, newOpBody.begin());
  rewriter.eraseBlock(&newOpFirstBlock);
  assert(newWarpOp.getWarpRegion().hasOneBlock() &&
         "expected WarpOp with single block");

  auto yield =
      cast<vector::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());

  rewriter.updateRootInPlace(
      yield, [&]() { yield.operandsMutable().assign(newYieldedValues); });
  return newWarpOp;
}

/// Helper to create a new WarpExecuteOnLane0Op region with extra outputs.
/// `indices` return the index of each new output.
static WarpExecuteOnLane0Op moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    llvm::SmallVector<size_t> &indices) {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  llvm::SmallSetVector<Value, 32> yieldValues(yield.getOperands().begin(),
                                              yield.getOperands().end());
  for (auto newRet : llvm::zip(newYieldedValues, newReturnTypes)) {
    if (yieldValues.insert(std::get<0>(newRet))) {
      types.push_back(std::get<1>(newRet));
      indices.push_back(yieldValues.size() - 1);
    } else {
      // If the value already exit the region don't create a new output.
      for (auto &yieldOperand : llvm::enumerate(yieldValues.getArrayRef())) {
        if (yieldOperand.value() == std::get<0>(newRet)) {
          indices.push_back(yieldOperand.index());
          break;
        }
      }
    }
  }
  yieldValues.insert(newYieldedValues.begin(), newYieldedValues.end());
  WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
      rewriter, warpOp, yieldValues.getArrayRef(), types);
  rewriter.replaceOp(warpOp,
                     newWarpOp.getResults().take_front(warpOp.getNumResults()));
  return newWarpOp;
}

/// Helper to know if an op can be hoisted out of the region.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return llvm::all_of(op->getOperands(), definedOutside) &&
         isMemoryEffectFree(op) && op->getNumRegions() == 0;
}

/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(WarpExecuteOnLane0Op warpOp,
                                const std::function<bool(Operation *)> &fn) {
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

namespace {

/// Rewrite a WarpExecuteOnLane0Op into a predicated scf.if op where the single
/// thread `laneId` executes the entirety of the computation.
///
/// After the transformation:
///   - the IR within the scf.if op can be thought of as executing sequentially
///     (from the point of view of threads along `laneId`).
///   - the IR outside of the scf.if op can be thought of as executing in
///     parallel (from the point of view of threads along `laneId`).
///
/// Values that need to transit through the parallel / sequential and the
/// sequential / parallel boundaries do so via reads and writes to a temporary
/// memory location.
///
/// The transformation proceeds in multiple steps:
///   1. Create the scf.if op.
///   2. Insert appropriate (alloc, write)-pairs before the scf.if and reads
///      within the scf.if to transit the values captured from above.
///   3. Synchronize before the scf.if to ensure all writes inserted in 2. are
///      consistent within the scf.if.
///   4. Move the body of the WarpExecuteOnLane0Op inside the scf.if.
///   5. Insert appropriate writes within scf.if and reads after the scf.if to
///      transit the values returned by the op.
///   6. Synchronize after the scf.if to ensure all writes inserted in 5. are
///      consistent after the scf.if.
///   7. Perform late cleanups.
///
/// All this assumes the vector distribution occurs along the most minor
/// distributed vector dimension.
struct WarpOpToScfIfPattern : public OpRewritePattern<WarpExecuteOnLane0Op> {
  WarpOpToScfIfPattern(MLIRContext *context,
                       const WarpExecuteOnLane0LoweringOptions &options,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<WarpExecuteOnLane0Op>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    assert(warpOp.getBodyRegion().hasOneBlock() &&
           "expected WarpOp with single block");
    Block *warpOpBody = &warpOp.getBodyRegion().front();
    Location loc = warpOp.getLoc();

    // Passed all checks. Start rewriting.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(warpOp);

    // Step 1: Create scf.if op.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isLane0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, warpOp.getLaneid(), c0);
    auto ifOp = rewriter.create<scf::IfOp>(loc, isLane0,
                                           /*withElseRegion=*/false);
    rewriter.eraseOp(ifOp.thenBlock()->getTerminator());

    // Step 2: insert appropriate (alloc, write)-pairs before the scf.if and
    // reads within the scf.if to transit the values captured from above.
    SmallVector<Value> bbArgReplacements;
    for (const auto &it : llvm::enumerate(warpOp.getArgs())) {
      Value sequentialVal = warpOpBody->getArgument(it.index());
      Value distributedVal = it.value();
      DistributedLoadStoreHelper helper(sequentialVal, distributedVal,
                                        warpOp.getLaneid(), c0);

      // Create buffer before the ifOp.
      rewriter.setInsertionPoint(ifOp);
      Value buffer = options.warpAllocationFn(loc, rewriter, warpOp,
                                              sequentialVal.getType());
      // Store distributed vector into buffer, before the ifOp.
      helper.buildStore(rewriter, loc, distributedVal, buffer);
      // Load sequential vector from buffer, inside the ifOp.
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      bbArgReplacements.push_back(
          helper.buildLoad(rewriter, loc, sequentialVal.getType(), buffer));
    }

    // Step 3. Insert sync after all the stores and before all the loads.
    if (!warpOp.getArgs().empty()) {
      rewriter.setInsertionPoint(ifOp);
      options.warpSyncronizationFn(loc, rewriter, warpOp);
    }

    // Step 4. Move body of warpOp to ifOp.
    rewriter.mergeBlocks(warpOpBody, ifOp.thenBlock(), bbArgReplacements);

    // Step 5. Insert appropriate writes within scf.if and reads after the
    // scf.if to transit the values returned by the op.
    // TODO: at this point, we can reuse the shared memory from previous
    // buffers.
    SmallVector<Value> replacements;
    auto yieldOp = cast<vector::YieldOp>(ifOp.thenBlock()->getTerminator());
    Location yieldLoc = yieldOp.getLoc();
    for (const auto &it : llvm::enumerate(yieldOp.operands())) {
      Value sequentialVal = it.value();
      Value distributedVal = warpOp->getResult(it.index());
      DistributedLoadStoreHelper helper(sequentialVal, distributedVal,
                                        warpOp.getLaneid(), c0);

      // Create buffer before the ifOp.
      rewriter.setInsertionPoint(ifOp);
      Value buffer = options.warpAllocationFn(loc, rewriter, warpOp,
                                              sequentialVal.getType());

      // Store yielded value into buffer, inside the ifOp, before the
      // terminator.
      rewriter.setInsertionPoint(yieldOp);
      helper.buildStore(rewriter, loc, sequentialVal, buffer);

      // Load distributed value from buffer, after  the warpOp.
      rewriter.setInsertionPointAfter(ifOp);
      // Result type and yielded value type are the same. This is a broadcast.
      // E.g.:
      // %r = vector.warp_execute_on_lane_0(...) -> (f32) {
      //   vector.yield %cst : f32
      // }
      // Both types are f32. The constant %cst is broadcasted to all lanes.
      // This is described in more detail in the documentation of the op.
      replacements.push_back(
          helper.buildLoad(rewriter, loc, distributedVal.getType(), buffer));
    }

    // Step 6. Insert sync after all the stores and before all the loads.
    if (!yieldOp.operands().empty()) {
      rewriter.setInsertionPointAfter(ifOp);
      options.warpSyncronizationFn(loc, rewriter, warpOp);
    }

    // Step 7. Delete terminator and add empty scf.yield.
    rewriter.eraseOp(yieldOp);
    rewriter.setInsertionPointToEnd(ifOp.thenBlock());
    rewriter.create<scf::YieldOp>(yieldLoc);

    // Compute replacements for WarpOp results.
    rewriter.replaceOp(warpOp, replacements);

    return success();
  }

private:
  const WarpExecuteOnLane0LoweringOptions &options;
};

/// Clone `writeOp` assumed to be nested under `warpOp` into a new warp execute
/// op with the proper return type.
/// The new write op is updated to write the result of the new warp execute op.
/// The old `writeOp` is deleted.
static vector::TransferWriteOp cloneWriteOp(RewriterBase &rewriter,
                                            WarpExecuteOnLane0Op warpOp,
                                            vector::TransferWriteOp writeOp,
                                            VectorType targetType) {
  assert(writeOp->getParentOp() == warpOp &&
         "write must be nested immediately under warp");
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<size_t> newRetIndices;
  WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
      rewriter, warpOp, ValueRange{{writeOp.getVector()}},
      TypeRange{targetType}, newRetIndices);
  rewriter.setInsertionPointAfter(newWarpOp);
  auto newWriteOp =
      cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
  rewriter.eraseOp(writeOp);
  newWriteOp.getVectorMutable().assign(newWarpOp.getResult(newRetIndices[0]));
  return newWriteOp;
}

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
        distributionMapFn(std::move(fn)) {}

  /// Distribute the TransferWriteOp. Only 1D distributions and vector dims that
  /// are multiples of the distribution ratio are supported at the moment.
  LogicalResult tryDistributeOp(RewriterBase &rewriter,
                                vector::TransferWriteOp writeOp,
                                WarpExecuteOnLane0Op warpOp) const {
    VectorType writtenVectorType = writeOp.getVectorType();

    // 1. If the write is 0-D, we just clone it into a new WarpExecuteOnLane0Op
    // to separate it from the rest.
    if (writtenVectorType.getRank() == 0)
      return failure();

    // 2. Compute the distribution map.
    AffineMap map = distributionMapFn(writeOp);
    if (map.getNumResults() != 1)
      return writeOp->emitError("multi-dim distribution not implemented yet");

    // 3. Compute the targetType using the distribution map.
    SmallVector<int64_t> targetShape(writtenVectorType.getShape().begin(),
                                     writtenVectorType.getShape().end());
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      unsigned position = map.getDimPosition(i);
      if (targetShape[position] % warpOp.getWarpSize() != 0)
        return failure();
      targetShape[position] = targetShape[position] / warpOp.getWarpSize();
    }
    VectorType targetType =
        VectorType::get(targetShape, writtenVectorType.getElementType());

    // 4. clone the write into a new WarpExecuteOnLane0Op to separate it from
    // the rest.
    vector::TransferWriteOp newWriteOp =
        cloneWriteOp(rewriter, warpOp, writeOp, targetType);

    // 5. Reindex the write using the distribution map.
    auto newWarpOp =
        newWriteOp.getVector().getDefiningOp<WarpExecuteOnLane0Op>();
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
      auto scale = rewriter.getAffineConstantExpr(targetShape[vectorPos]);
      indices[indexPos] =
          makeComposedAffineApply(rewriter, loc, d0 + scale * d1,
                                  {indices[indexPos], newWarpOp.getLaneid()});
    }
    newWriteOp.getIndicesMutable().assign(indices);

    return success();
  }

  /// Extract TransferWriteOps of vector<1x> into a separate warp op.
  LogicalResult tryExtractOp(RewriterBase &rewriter,
                             vector::TransferWriteOp writeOp,
                             WarpExecuteOnLane0Op warpOp) const {
    Location loc = writeOp.getLoc();
    VectorType vecType = writeOp.getVectorType();

    // Only sink out vector of 1 element for now to not serialize large vector
    // store. This can later be controlled by user.
    if (vecType.getNumElements() != 1)
      return failure();

    // Do not process warp ops that contain only TransferWriteOps.
    if (llvm::all_of(warpOp.getOps(), [](Operation &op) {
          return isa<vector::TransferWriteOp, vector::YieldOp>(&op);
        }))
      return failure();

    SmallVector<Value> yieldValues = {writeOp.getVector()};
    SmallVector<Type> retTypes = {vecType};
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Create a second warp op that contains only writeOp.
    auto secondWarpOp = rewriter.create<WarpExecuteOnLane0Op>(
        loc, TypeRange(), newWarpOp.getLaneid(), newWarpOp.getWarpSize());
    Block &body = secondWarpOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&body);
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    newWriteOp.getVectorMutable().assign(newWarpOp.getResult(newRetIndices[0]));
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
      if (!isMemoryEffectFree(nextOp))
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
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : llvm::seq(unsigned(0), elementWise->getNumOperands())) {
      newOperands[i] = newWarpOp.getResult(newRetIndices[i]);
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

/// Sink out splat constant op feeding into a warp op yield.
/// ```
/// %0 = vector.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %cst = arith.constant dense<2.0> : vector<32xf32>
///   vector.yield %cst : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// vector.warp_execute_on_lane_0(%arg0 {
///   ...
/// }
/// %0 = arith.constant dense<2.0> : vector<1xf32>
struct WarpOpConstant : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(
        warpOp, [](Operation *op) { return isa<arith::ConstantOp>(op); });
    if (!yieldOperand)
      return failure();
    auto constantOp = yieldOperand->get().getDefiningOp<arith::ConstantOp>();
    auto dense = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (!dense)
      return failure();
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Attribute scalarAttr = dense.getSplatValue<Attribute>();
    Attribute newAttr = DenseElementsAttr::get(
        warpOp.getResult(operandIndex).getType(), scalarAttr);
    Location loc = warpOp.getLoc();
    rewriter.setInsertionPointAfter(warpOp);
    Value distConstant = rewriter.create<arith::ConstantOp>(loc, newAttr);
    warpOp.getResult(operandIndex).replaceAllUsesWith(distConstant);
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
    // Don't duplicate transfer_read ops when distributing.
    if (!read.getResult().hasOneUse())
      return failure();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(read.getIndices().begin(),
                                  read.getIndices().end());
    auto sequentialType = read.getResult().getType().cast<VectorType>();
    auto distributedType = distributedVal.getType().cast<VectorType>();
    AffineMap map = calculateImplicitMap(sequentialType, distributedType);
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
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(warpOp->getNumResults());
    SmallVector<Value> newYieldValues;
    newYieldValues.reserve(warpOp->getNumResults());
    DenseMap<Value, int64_t> dedupYieldOperandPositionMap;
    DenseMap<OpResult, int64_t> dedupResultPositionMap;
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());

    // Some values may be yielded multiple times and correspond to multiple
    // results. Deduplicating occurs by taking each result with its matching
    // yielded value, and:
    //   1. recording the unique first position at which the value is yielded.
    //   2. recording for the result, the first position at which the dedup'ed
    //      value is yielded.
    //   3. skipping from the new result types / new yielded values any result
    //      that has no use or whose yielded value has already been seen.
    for (OpResult result : warpOp.getResults()) {
      Value yieldOperand = yield.getOperand(result.getResultNumber());
      auto it = dedupYieldOperandPositionMap.insert(
          std::make_pair(yieldOperand, newResultTypes.size()));
      dedupResultPositionMap.insert(std::make_pair(result, it.first->second));
      if (result.use_empty() || !it.second)
        continue;
      newResultTypes.push_back(result.getType());
      newYieldValues.push_back(yieldOperand);
    }
    // No modification, exit early.
    if (yield.getNumOperands() == newYieldValues.size())
      return failure();
    // Move the body of the old warpOp to a new warpOp.
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
        rewriter, warpOp, newYieldValues, newResultTypes);
    // Replace results of the old warpOp by the new, deduplicated results.
    SmallVector<Value> newValues;
    newValues.reserve(warpOp->getNumResults());
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        newValues.push_back(Value());
      else
        newValues.push_back(
            newWarpOp.getResult(dedupResultPositionMap.lookup(result)));
    }
    rewriter.replaceOp(warpOp, newValues);
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
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {broadcastOp.getSource()},
        {broadcastOp.getSource().getType()}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value broadcasted = rewriter.create<vector::BroadcastOp>(
        loc, destVecType, newWarpOp->getResult(newRetIndices[0]));
    newWarpOp->getResult(operandNumber).replaceAllUsesWith(broadcasted);
    return success();
  }
};

/// Pattern to move out vector.extract of single element vector. Those don't
/// need to be distributed and can just be propagated outside of the region.
struct WarpOpExtract : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::ExtractOp>(op); });
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto extractOp = operand->get().getDefiningOp<vector::ExtractOp>();
    if (extractOp.getVectorType().getNumElements() != 1)
      return failure();
    Location loc = extractOp.getLoc();
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {extractOp.getVector()}, {extractOp.getVectorType()},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value newExtract = rewriter.create<vector::ExtractOp>(
        loc, newWarpOp->getResult(newRetIndices[0]), extractOp.getPosition());
    newWarpOp->getResult(operandNumber).replaceAllUsesWith(newExtract);
    return success();
  }
};

/// Pattern to move out vector.extractelement of 0-D tensors. Those don't
/// need to be distributed and can just be propagated outside of the region.
struct WarpOpExtractElement : public OpRewritePattern<WarpExecuteOnLane0Op> {
  using OpRewritePattern<WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, [](Operation *op) {
      return isa<vector::ExtractElementOp>(op);
    });
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto extractOp = operand->get().getDefiningOp<vector::ExtractElementOp>();
    if (extractOp.getVectorType().getRank() != 0)
      return failure();
    Location loc = extractOp.getLoc();
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {extractOp.getVector()}, {extractOp.getVectorType()},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value newExtract = rewriter.create<vector::ExtractElementOp>(
        loc, newWarpOp->getResult(newRetIndices[0]));
    newWarpOp->getResult(operandNumber).replaceAllUsesWith(newExtract);
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
    if (!innerWarp.getResults().empty())
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
/// The vector is reduced in parallel. Currently limited to vector size matching
/// the warpOp size. E.g.:
/// ```
/// %r = vector_ext.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
///   %0 = "some_def"() : () -> (vector<32xf32>)
///   %1 = vector.reduction "add", %0 : vector<32xf32> into f32
///   vector_ext.yield %1 : f32
/// }
/// ```
/// is lowered to:
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
///   %1 = "some_def"() : () -> (vector<32xf32>)
///   vector_ext.yield %1 : vector<32xf32>
/// }
/// %a = vector.extract %0[0] : vector<1xf32>
/// %r = ("warp.reduction %a")
/// ```
struct WarpOpReduction : public OpRewritePattern<WarpExecuteOnLane0Op> {
  WarpOpReduction(MLIRContext *context,
                  DistributedReductionFn distributedReductionFn,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<WarpExecuteOnLane0Op>(context, benefit),
        distributedReductionFn(std::move(distributedReductionFn)) {}

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
    if (vectorType.getShape()[0] % warpOp.getWarpSize() != 0)
      return rewriter.notifyMatchFailure(
          warpOp, "Reduction vector dimension must match was size.");
    // Only f32 and i32 element types are supported.
    if (!reductionOp.getType().isF32() &&
        !reductionOp.getType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          warpOp,
          "Reduction distribution currently only supports 32bits types.");

    int64_t numElements = vectorType.getShape()[0] / warpOp.getWarpSize();
    // Return vector that will be reduced from the WarpExecuteOnLane0Op.
    unsigned operandIndex = yieldOperand->getOperandNumber();
    SmallVector<Value> yieldValues = {reductionOp.getVector()};
    SmallVector<Type> retTypes = {
        VectorType::get({numElements}, reductionOp.getType())};
    if (reductionOp.getAcc()) {
      yieldValues.push_back(reductionOp.getAcc());
      retTypes.push_back(reductionOp.getAcc().getType());
    }
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);

    Value laneValVec = newWarpOp.getResult(newRetIndices[0]);
    // First reduce on a single thread.
    Value perLaneReduction = rewriter.create<vector::ReductionOp>(
        reductionOp.getLoc(), reductionOp.getKind(), laneValVec);
    // Then distribute across threads.
    Value fullReduce =
        distributedReductionFn(reductionOp.getLoc(), rewriter, perLaneReduction,
                               reductionOp.getKind(), newWarpOp.getWarpSize());
    if (reductionOp.getAcc()) {
      fullReduce = vector::makeArithReduction(
          rewriter, reductionOp.getLoc(), reductionOp.getKind(), fullReduce,
          newWarpOp.getResult(newRetIndices[1]));
    }
    newWarpOp.getResult(operandIndex).replaceAllUsesWith(fullReduce);
    return success();
  }

private:
  DistributedReductionFn distributedReductionFn;
};

} // namespace

void mlir::vector::populateWarpExecuteOnLane0OpToScfForPattern(
    RewritePatternSet &patterns,
    const WarpExecuteOnLane0LoweringOptions &options, PatternBenefit benefit) {
  patterns.add<WarpOpToScfIfPattern>(patterns.getContext(), options, benefit);
}

void mlir::vector::populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, const DistributionMapFn &distributionMapFn,
    PatternBenefit benefit) {
  patterns.add<WarpOpTransferWrite>(patterns.getContext(), distributionMapFn,
                                    benefit);
}

void mlir::vector::populatePropagateWarpVectorDistributionPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<WarpOpElementwise, WarpOpTransferRead, WarpOpDeadResult,
               WarpOpBroadcast, WarpOpExtract, WarpOpExtractElement,
               WarpOpForwardOperand, WarpOpScfForOp, WarpOpConstant>(
      patterns.getContext(), benefit);
}

void mlir::vector::populateDistributeReduction(
    RewritePatternSet &patterns,
    const DistributedReductionFn &distributedReductionFn,
    PatternBenefit benefit) {
  patterns.add<WarpOpReduction>(patterns.getContext(), distributedReductionFn,
                                benefit);
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
