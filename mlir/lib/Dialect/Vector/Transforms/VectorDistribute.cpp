//===- VectorDistribute.cpp - patterns to do vector distribution ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <utility>

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::gpu;

/// Currently the distribution map is implicit based on the vector shape. In the
/// future it will be part of the op.
/// Example:
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1x16x2xf32>) {
///   ...
///   gpu.yield %3 : vector<32x16x64xf32>
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
    sequentialVectorType = dyn_cast<VectorType>(sequentialVal.getType());
    distributedVectorType = dyn_cast<VectorType>(distributedVal.getType());
    if (sequentialVectorType && distributedVectorType)
      distributionMap =
          calculateImplicitMap(sequentialVectorType, distributedVectorType);
  }

  Value buildDistributedOffset(RewriterBase &b, Location loc, int64_t index) {
    int64_t distributedSize = distributedVectorType.getDimSize(index);
    AffineExpr tid = getAffineSymbolExpr(0, b.getContext());
    return b.createOrFold<affine::AffineApplyOp>(loc, tid * distributedSize,
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
    if (!isa<VectorType>(val.getType()))
      return b.create<memref::StoreOp>(loc, val, buffer, zero);

    // Vector case must use vector::TransferWriteOp which will later lower to
    //   vector.store of memref.store depending on further lowerings.
    int64_t rank = sequentialVectorType.getRank();
    SmallVector<Value> indices(rank, zero);
    if (val == distributedVal) {
      for (auto dimExpr : distributionMap.getResults()) {
        int64_t index = cast<AffineDimExpr>(dimExpr).getPosition();
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
  /// the broadcast semantics of the `gpu.warp_execute_on_lane_0` op.
  ///
  /// Example:
  ///
  /// ```
  ///   %r = gpu.warp_execute_on_lane_0(...) -> (f32) {
  ///     gpu.yield %cst : f32
  ///   }
  ///   // Both types are f32. The constant %cst is broadcasted to all lanes.
  /// ```
  /// This behavior described in more detail in the documentation of the op.
  Value buildLoad(RewriterBase &b, Location loc, Type type, Value buffer) {

    // Scalar case can directly use memref.store.
    if (!isa<VectorType>(type))
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
        int64_t index = cast<AffineDimExpr>(dimExpr).getPosition();
        indices[index] = buildDistributedOffset(b, loc, index);
      }
    }
    SmallVector<bool> inBounds(indices.size(), true);
    return b.create<vector::TransferReadOp>(
        loc, cast<VectorType>(type), buffer, indices,
        ArrayRef<bool>(inBounds.begin(), inBounds.end()));
  }

  Value sequentialVal, distributedVal, laneId, zero;
  VectorType sequentialVectorType, distributedVectorType;
  AffineMap distributionMap;
};

} // namespace

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
struct WarpOpToScfIfPattern : public WarpDistributionPattern {
  WarpOpToScfIfPattern(MLIRContext *context,
                       const WarpExecuteOnLane0LoweringOptions &options,
                       PatternBenefit benefit = 1)
      : WarpDistributionPattern(context, benefit), options(options) {}

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
    auto yieldOp = cast<gpu::YieldOp>(ifOp.thenBlock()->getTerminator());
    Location yieldLoc = yieldOp.getLoc();
    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
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
      // %r = gpu.warp_execute_on_lane_0(...) -> (f32) {
      //   gpu.yield %cst : f32
      // }
      // Both types are f32. The constant %cst is broadcasted to all lanes.
      // This is described in more detail in the documentation of the op.
      replacements.push_back(
          helper.buildLoad(rewriter, loc, distributedVal.getType(), buffer));
    }

    // Step 6. Insert sync after all the stores and before all the loads.
    if (!yieldOp.getOperands().empty()) {
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

/// Return the distributed vector type based on the original type and the
/// distribution map. The map is expected to have a dimension equal to the
/// original type rank and should be a projection where the results are the
/// distributed dimensions. The number of results should be equal to the number
/// of warp sizes which is currently limited to 1.
/// Example: For a vector<16x32x64> distributed with a map(d0, d1, d2) -> (d1)
/// and a warp size of 16 would distribute the second dimension (associated to
/// d1) and return vector<16x2x64>
static VectorType getDistributedType(VectorType originalType, AffineMap map,
                                     int64_t warpSize) {
  SmallVector<int64_t> targetShape(originalType.getShape());
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    unsigned position = map.getDimPosition(i);
    if (targetShape[position] % warpSize != 0) {
      if (warpSize % targetShape[position] != 0) {
        return VectorType();
      }
      warpSize /= targetShape[position];
      targetShape[position] = 1;
      continue;
    }
    targetShape[position] = targetShape[position] / warpSize;
    warpSize = 1;
    break;
  }
  if (warpSize != 1) {
    return VectorType();
  }
  VectorType targetType =
      VectorType::get(targetShape, originalType.getElementType());
  return targetType;
}

/// Distribute transfer_write ops based on the affine map returned by
/// `distributionMapFn`. Writes of size more than `maxNumElementToExtract`
/// will not be distributed (it should be less than the warp size).
///
/// Example:
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%id){
///   ...
///   vector.transfer_write %v, %A[%c0] : vector<32xf32>, memref<128xf32>
///   gpu.yield
/// }
/// ```
/// To
/// ```
/// %r:3 = gpu.warp_execute_on_lane_0(%id) -> (vector<1xf32>) {
///   ...
///   gpu.yield %v : vector<32xf32>
/// }
/// vector.transfer_write %v, %A[%id] : vector<1xf32>, memref<128xf32>
struct WarpOpTransferWrite : public WarpDistributionPattern {
  WarpOpTransferWrite(MLIRContext *ctx, DistributionMapFn fn,
                      unsigned maxNumElementsToExtract, PatternBenefit b = 1)
      : WarpDistributionPattern(ctx, b), distributionMapFn(std::move(fn)),
        maxNumElementsToExtract(maxNumElementsToExtract) {}

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

    // 2. Compute the distributed type.
    AffineMap map = distributionMapFn(writeOp.getVector());
    VectorType targetType =
        getDistributedType(writtenVectorType, map, warpOp.getWarpSize());
    if (!targetType)
      return failure();

    // 2.5 Compute the distributed type for the new mask;
    VectorType maskType;
    if (writeOp.getMask()) {
      // TODO: Distribution of masked writes with non-trivial permutation maps
      // requires the distribution of the mask to elementwise match the
      // distribution of the permuted written vector. Currently the details
      // of which lane is responsible for which element is captured strictly
      // by shape information on the warp op, and thus requires materializing
      // the permutation in IR.
      if (!writeOp.getPermutationMap().isMinorIdentity())
        return failure();
      maskType =
          getDistributedType(writeOp.getMaskType(), map, warpOp.getWarpSize());
    }

    // 3. clone the write into a new WarpExecuteOnLane0Op to separate it from
    // the rest.
    vector::TransferWriteOp newWriteOp =
        cloneWriteOp(rewriter, warpOp, writeOp, targetType, maskType);

    // 4. Reindex the write using the distribution map.
    auto newWarpOp =
        newWriteOp.getVector().getDefiningOp<WarpExecuteOnLane0Op>();

    // Delinearize the lane id based on the way threads are divided across the
    // vector. To get the number of threads per vector dimension, divide the
    // sequential size by the distributed size along each dim.
    rewriter.setInsertionPoint(newWriteOp);
    SmallVector<OpFoldResult> delinearizedIdSizes;
    for (auto [seqSize, distSize] :
         llvm::zip_equal(writtenVectorType.getShape(), targetType.getShape())) {
      assert(seqSize % distSize == 0 && "Invalid distributed vector shape");
      delinearizedIdSizes.push_back(rewriter.getIndexAttr(seqSize / distSize));
    }
    SmallVector<Value> delinearized;
    if (map.getNumResults() > 1) {
      delinearized = rewriter
                         .create<mlir::affine::AffineDelinearizeIndexOp>(
                             newWarpOp.getLoc(), newWarpOp.getLaneid(),
                             delinearizedIdSizes)
                         .getResults();
    } else {
      // If there is only one map result, we can elide the delinearization
      // op and use the lane id directly.
      delinearized.append(targetType.getRank(), newWarpOp.getLaneid());
    }

    AffineMap indexMap = map.compose(newWriteOp.getPermutationMap());
    Location loc = newWriteOp.getLoc();
    SmallVector<Value> indices(newWriteOp.getIndices().begin(),
                               newWriteOp.getIndices().end());
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(newWarpOp.getContext(), d0, d1);
      auto indexExpr = dyn_cast<AffineDimExpr>(std::get<0>(it));
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = cast<AffineDimExpr>(std::get<1>(it)).getPosition();
      Value laneId = delinearized[vectorPos];
      auto scale =
          rewriter.getAffineConstantExpr(targetType.getDimSize(vectorPos));
      indices[indexPos] = affine::makeComposedAffineApply(
          rewriter, loc, d0 + scale * d1, {indices[indexPos], laneId});
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

    if (vecType.getNumElements() > maxNumElementsToExtract) {
      return rewriter.notifyMatchFailure(
          warpOp,
          llvm::formatv(
              "writes more elements ({0}) than allowed to extract ({1})",
              vecType.getNumElements(), maxNumElementsToExtract));
    }

    // Do not process warp ops that contain only TransferWriteOps.
    if (llvm::all_of(warpOp.getOps(),
                     llvm::IsaPred<vector::TransferWriteOp, gpu::YieldOp>))
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
    newWriteOp.getValueToStoreMutable().assign(
        newWarpOp.getResult(newRetIndices[0]));
    rewriter.eraseOp(writeOp);
    rewriter.create<gpu::YieldOp>(newWarpOp.getLoc());
    return success();
  }

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<gpu::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Operation *lastNode = yield->getPrevNode();
    auto writeOp = dyn_cast_or_null<vector::TransferWriteOp>(lastNode);
    if (!writeOp)
      return failure();

    Value maybeMask = writeOp.getMask();
    if (!llvm::all_of(writeOp->getOperands(), [&](Value value) {
          return writeOp.getVector() == value ||
                 (maybeMask && maybeMask == value) ||
                 warpOp.isDefinedOutsideOfRegion(value);
        }))
      return failure();

    if (succeeded(tryDistributeOp(rewriter, writeOp, warpOp)))
      return success();

    // Masked writes not supported for extraction.
    if (writeOp.getMask())
      return failure();

    if (succeeded(tryExtractOp(rewriter, writeOp, warpOp)))
      return success();

    return failure();
  }

private:
  /// Clone `writeOp` assumed to be nested under `warpOp` into a new warp
  /// execute op with the proper return type. The new write op is updated to
  /// write the result of the new warp execute op. The old `writeOp` is deleted.
  vector::TransferWriteOp cloneWriteOp(RewriterBase &rewriter,
                                       WarpExecuteOnLane0Op warpOp,
                                       vector::TransferWriteOp writeOp,
                                       VectorType targetType,
                                       VectorType maybeMaskType) const {
    assert(writeOp->getParentOp() == warpOp &&
           "write must be nested immediately under warp");
    OpBuilder::InsertionGuard g(rewriter);
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp;
    if (maybeMaskType) {
      newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, ValueRange{writeOp.getVector(), writeOp.getMask()},
          TypeRange{targetType, maybeMaskType}, newRetIndices);
    } else {
      newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, ValueRange{{writeOp.getVector()}},
          TypeRange{targetType}, newRetIndices);
    }
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    rewriter.eraseOp(writeOp);
    newWriteOp.getValueToStoreMutable().assign(
        newWarpOp.getResult(newRetIndices[0]));
    if (maybeMaskType)
      newWriteOp.getMaskMutable().assign(newWarpOp.getResult(newRetIndices[1]));
    return newWriteOp;
  }

  DistributionMapFn distributionMapFn;
  unsigned maxNumElementsToExtract = 1;
};

/// Sink out elementwise op feeding into a warp op yield.
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %3 = arith.addf %1, %2 : vector<32xf32>
///   gpu.yield %3 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %r:3 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %4 = arith.addf %2, %3 : vector<32xf32>
///   gpu.yield %4, %2, %3 : vector<32xf32>, vector<32xf32>,
///   vector<32xf32>
/// }
/// %0 = arith.addf %r#1, %r#2 : vector<1xf32>
struct WarpOpElementwise : public WarpDistributionPattern {
  using Base::Base;
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
      if (auto vecType = dyn_cast<VectorType>(distributedVal.getType())) {
        // If the result type is a vector, the operands must also be vectors.
        auto operandType = cast<VectorType>(operand.get().getType());
        targetType =
            VectorType::get(vecType.getShape(), operandType.getElementType());
      } else {
        auto operandType = operand.get().getType();
        assert(!isa<VectorType>(operandType) &&
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
    rewriter.replaceAllUsesWith(newWarpOp.getResult(operandIndex),
                                newOp->getResult(0));
    return success();
  }
};

/// Sink out splat constant op feeding into a warp op yield.
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %cst = arith.constant dense<2.0> : vector<32xf32>
///   gpu.yield %cst : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// gpu.warp_execute_on_lane_0(%arg0 {
///   ...
/// }
/// %0 = arith.constant dense<2.0> : vector<1xf32>
struct WarpOpConstant : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<arith::ConstantOp>);
    if (!yieldOperand)
      return failure();
    auto constantOp = yieldOperand->get().getDefiningOp<arith::ConstantOp>();
    auto dense = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!dense)
      return failure();
    // Notify the rewriter that the warp op is changing (see the comment on
    // the WarpOpTransferRead pattern).
    rewriter.startOpModification(warpOp);
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Attribute scalarAttr = dense.getSplatValue<Attribute>();
    auto newAttr = DenseElementsAttr::get(
        cast<ShapedType>(warpOp.getResult(operandIndex).getType()), scalarAttr);
    Location loc = warpOp.getLoc();
    rewriter.setInsertionPointAfter(warpOp);
    Value distConstant = rewriter.create<arith::ConstantOp>(loc, newAttr);
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIndex), distConstant);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};

/// Sink out transfer_read op feeding into a warp op yield.
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
//    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
//    vector<32xf32>
///   gpu.yield %2 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %dead = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
///   vector<32xf32> gpu.yield %2 : vector<32xf32>
/// }
/// %0 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<1xf32>
struct WarpOpTransferRead : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    // Try to find a distributable yielded read. Note that this pattern can
    // still fail at the end after distribution, in which case this might have
    // missed another distributable read.
    OpOperand *operand = getWarpResult(warpOp, [](Operation *op) {
      // Don't duplicate transfer_read ops when distributing.
      return isa<vector::TransferReadOp>(op) && op->hasOneUse();
    });
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a vector.transfer_read op");
    auto read = operand->get().getDefiningOp<vector::TransferReadOp>();

    // Source must be defined outside of the region.
    if (!warpOp.isDefinedOutsideOfRegion(read.getBase()))
      return rewriter.notifyMatchFailure(
          read, "source must be defined outside of the region");

    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(read.getIndices().begin(),
                                  read.getIndices().end());
    auto sequentialType = cast<VectorType>(read.getResult().getType());
    auto distributedType = cast<VectorType>(distributedVal.getType());
    AffineMap map = calculateImplicitMap(sequentialType, distributedType);
    AffineMap indexMap = map.compose(read.getPermutationMap());

    // Try to delinearize the lane ID to match the rank expected for
    // distribution.
    SmallVector<Value> delinearizedIds;
    if (!delinearizeLaneId(rewriter, read.getLoc(), sequentialType.getShape(),
                           distributedType.getShape(), warpOp.getWarpSize(),
                           warpOp.getLaneid(), delinearizedIds)) {
      return rewriter.notifyMatchFailure(
          read, "cannot delinearize lane ID for distribution");
    }
    assert(!delinearizedIds.empty() || map.getNumResults() == 0);

    // Distribute indices and the mask (if present).
    OpBuilder::InsertionGuard g(rewriter);
    SmallVector<Value> additionalResults(indices.begin(), indices.end());
    SmallVector<Type> additionalResultTypes(indices.size(),
                                            rewriter.getIndexType());
    additionalResults.push_back(read.getPadding());
    additionalResultTypes.push_back(read.getPadding().getType());

    bool hasMask = false;
    if (read.getMask()) {
      hasMask = true;
      // TODO: Distribution of masked reads with non-trivial permutation maps
      // requires the distribution of the mask to elementwise match the
      // distribution of the permuted written vector. Currently the details
      // of which lane is responsible for which element is captured strictly
      // by shape information on the warp op, and thus requires materializing
      // the permutation in IR.
      if (!mlir::compressUnusedDims(read.getPermutationMap()).isIdentity())
        return rewriter.notifyMatchFailure(
            read, "non-trivial permutation maps not supported");
      VectorType maskType =
          getDistributedType(read.getMaskType(), map, warpOp.getWarpSize());
      additionalResults.push_back(read.getMask());
      additionalResultTypes.push_back(maskType);
    }

    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, additionalResults, additionalResultTypes,
        newRetIndices);
    distributedVal = newWarpOp.getResult(operandIndex);

    // Distributed indices were appended first.
    SmallVector<Value> newIndices;
    for (int64_t i = 0, e = indices.size(); i < e; ++i)
      newIndices.push_back(newWarpOp.getResult(newRetIndices[i]));

    rewriter.setInsertionPointAfter(newWarpOp);
    for (auto it : llvm::zip_equal(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = dyn_cast<AffineDimExpr>(std::get<0>(it));
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = cast<AffineDimExpr>(std::get<1>(it)).getPosition();
      int64_t scale = distributedType.getDimSize(vectorPos);
      newIndices[indexPos] = affine::makeComposedAffineApply(
          rewriter, read.getLoc(), d0 + scale * d1,
          {newIndices[indexPos], delinearizedIds[vectorPos]});
    }

    // Distributed padding value was appended right after the indices.
    Value newPadding = newWarpOp.getResult(newRetIndices[indices.size()]);
    // Distributed mask value was added at the end (if the op has a mask).
    Value newMask =
        hasMask ? newWarpOp.getResult(newRetIndices[newRetIndices.size() - 1])
                : Value();
    auto newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), distributedVal.getType(), read.getBase(), newIndices,
        read.getPermutationMapAttr(), newPadding, newMask,
        read.getInBoundsAttr());

    rewriter.replaceAllUsesWith(distributedVal, newRead);
    return success();
  }
};

/// Remove any result that has no use along with the matching yieldOp operand.
// TODO: Move this in WarpExecuteOnLane0Op canonicalization.
struct WarpOpDeadResult : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(warpOp->getNumResults());
    SmallVector<Value> newYieldValues;
    newYieldValues.reserve(warpOp->getNumResults());
    DenseMap<Value, int64_t> dedupYieldOperandPositionMap;
    DenseMap<OpResult, int64_t> dedupResultPositionMap;
    auto yield = cast<gpu::YieldOp>(
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

    // Simplify the new warp op after dropping dead results.
    newWarpOp.getBody()->walk([&](Operation *op) {
      if (isOpTriviallyDead(op))
        rewriter.eraseOp(op);
    });

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
struct WarpOpForwardOperand : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<gpu::YieldOp>(
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
      auto arg = dyn_cast<BlockArgument>(operand.get());
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
    // Notify the rewriter that the warp op is changing (see the comment on
    // the WarpOpTransferRead pattern).
    rewriter.startOpModification(warpOp);
    rewriter.replaceAllUsesWith(warpOp.getResult(resultIndex), valForwarded);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};

struct WarpOpBroadcast : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::BroadcastOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto broadcastOp = operand->get().getDefiningOp<vector::BroadcastOp>();
    Location loc = broadcastOp.getLoc();
    auto destVecType =
        cast<VectorType>(warpOp->getResultTypes()[operandNumber]);
    Value broadcastSrc = broadcastOp.getSource();
    Type broadcastSrcType = broadcastSrc.getType();

    // Check that the broadcast actually spans a set of values uniformly across
    // all threads. In other words, check that each thread can reconstruct
    // their own broadcast.
    // For that we simply check that the broadcast we want to build makes sense.
    if (vector::isBroadcastableTo(broadcastSrcType, destVecType) !=
        vector::BroadcastableToResult::Success)
      return failure();
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {broadcastSrc}, {broadcastSrcType}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value broadcasted = rewriter.create<vector::BroadcastOp>(
        loc, destVecType, newWarpOp->getResult(newRetIndices[0]));
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                broadcasted);
    return success();
  }
};

/// Pattern to move shape cast out of the warp op. shape cast is basically a
/// no-op for warp distribution; we need to handle the shape though.
struct WarpOpShapeCast : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ShapeCastOp>);
    if (!operand)
      return failure();

    auto oldCastOp = operand->get().getDefiningOp<vector::ShapeCastOp>();

    unsigned int operandNumber = operand->getOperandNumber();
    auto castDistributedType =
        cast<VectorType>(warpOp->getResultTypes()[operandNumber]);
    VectorType castOriginalType = oldCastOp.getSourceVectorType();
    VectorType castResultType = castDistributedType;

    // We expect the distributed type to have a smaller rank than the original
    // type. Prepend with size-one dimensions to make them the same.
    unsigned castDistributedRank = castDistributedType.getRank();
    unsigned castOriginalRank = castOriginalType.getRank();
    if (castDistributedRank < castOriginalRank) {
      SmallVector<int64_t> shape(castOriginalRank - castDistributedRank, 1);
      llvm::append_range(shape, castDistributedType.getShape());
      castDistributedType =
          VectorType::get(shape, castDistributedType.getElementType());
    }

    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {oldCastOp.getSource()}, {castDistributedType},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value newCast = rewriter.create<vector::ShapeCastOp>(
        oldCastOp.getLoc(), castResultType,
        newWarpOp->getResult(newRetIndices[0]));
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), newCast);
    return success();
  }
};

/// Sink out vector.create_mask op feeding into a warp op yield.
/// ```
/// %0 = ...
/// %1 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %mask = vector.create_mask %0 : vector<32xi1>
///   gpu.yield %mask : vector<32xi1>
/// }
/// ```
/// To
/// ```
/// %0 = ...
/// gpu.warp_execute_on_lane_0(%arg0) {
///   ...
/// }
/// %cmp = arith.cmpi ult, %laneid, %0
/// %ub = arith.select %cmp, %c0, %c1
/// %1 = vector.create_mask %ub : vector<1xi1>
struct WarpOpCreateMask : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<vector::CreateMaskOp>);
    if (!yieldOperand)
      return failure();

    auto mask = yieldOperand->get().getDefiningOp<vector::CreateMaskOp>();

    // Early exit if any values needed for calculating the new mask indices
    // are defined inside the warp op.
    if (!llvm::all_of(mask->getOperands(), [&](Value value) {
          return warpOp.isDefinedOutsideOfRegion(value);
        }))
      return failure();

    Location loc = mask.getLoc();
    unsigned operandIndex = yieldOperand->getOperandNumber();

    auto distType = cast<VectorType>(warpOp.getResult(operandIndex).getType());
    VectorType seqType = mask.getVectorType();
    ArrayRef<int64_t> seqShape = seqType.getShape();
    ArrayRef<int64_t> distShape = distType.getShape();

    rewriter.setInsertionPointAfter(warpOp);

    // Delinearize the lane ID for constructing the distributed mask sizes.
    SmallVector<Value> delinearizedIds;
    if (!delinearizeLaneId(rewriter, loc, seqShape, distShape,
                           warpOp.getWarpSize(), warpOp.getLaneid(),
                           delinearizedIds))
      return rewriter.notifyMatchFailure(
          mask, "cannot delinearize lane ID for distribution");
    assert(!delinearizedIds.empty());

    // Notify the rewriter that the warp op is changing (see the comment on
    // the WarpOpTransferRead pattern).
    rewriter.startOpModification(warpOp);

    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    SmallVector<Value> newOperands;
    for (int i = 0, e = distShape.size(); i < e; ++i) {
      // Get `mask_dim_range_upper_limit[i] - lane_id[i] * dist_sizes[i]` to
      // find the distance from the largest mask index owned by this lane to the
      // original mask size. `vector.create_mask` implicitly clamps mask
      // operands to the range [0, mask_vector_size[i]], or in other words, the
      // mask sizes are always in the range [0, mask_vector_size[i]).
      Value maskDimIdx = affine::makeComposedAffineApply(
          rewriter, loc, s1 - s0 * distShape[i],
          {delinearizedIds[i], mask.getOperand(i)});
      newOperands.push_back(maskDimIdx);
    }

    auto newMask =
        rewriter.create<vector::CreateMaskOp>(loc, distType, newOperands);
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIndex), newMask);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};

/// Pattern to move out vector.extract of single element vector. Those don't
/// need to be distributed and can just be propagated outside of the region.
struct WarpOpExtract : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ExtractOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto extractOp = operand->get().getDefiningOp<vector::ExtractOp>();
    VectorType extractSrcType = extractOp.getSourceVectorType();
    Location loc = extractOp.getLoc();

    // For 1-d or 0-d source cases, we rely on WarpOpExtractScalar pattern.
    if (extractSrcType.getRank() <= 1) {
      return failure();
    }

    // All following cases are 2d or higher dimensional source vectors.

    if (warpOp.getResult(operandNumber).getType() == operand->get().getType()) {
      // There is no distribution, this is a broadcast. Simply move the extract
      // out of the warp op.
      // TODO: This could be optimized. E.g., in case of a scalar result, let
      // one lane extract and shuffle the result to all other lanes (same as
      // the 1d case).
      SmallVector<size_t> newRetIndices;
      WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, {extractOp.getVector()},
          {extractOp.getSourceVectorType()}, newRetIndices);
      rewriter.setInsertionPointAfter(newWarpOp);
      Value distributedVec = newWarpOp->getResult(newRetIndices[0]);
      // Extract from distributed vector.
      Value newExtract = rewriter.create<vector::ExtractOp>(
          loc, distributedVec, extractOp.getMixedPosition());
      rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                  newExtract);
      return success();
    }

    // Find the distributed dimension. There should be exactly one.
    auto distributedType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    auto yieldedType = cast<VectorType>(operand->get().getType());
    int64_t distributedDim = -1;
    for (int64_t i = 0; i < yieldedType.getRank(); ++i) {
      if (distributedType.getDimSize(i) != yieldedType.getDimSize(i)) {
        // Keep this assert here in case WarpExecuteOnLane0Op gets extended to
        // support distributing multiple dimensions in the future.
        assert(distributedDim == -1 && "found multiple distributed dims");
        distributedDim = i;
      }
    }
    assert(distributedDim != -1 && "could not find distributed dimension");
    (void)distributedDim;

    // Yield source vector from warp op.
    SmallVector<int64_t> newDistributedShape(extractSrcType.getShape());
    for (int i = 0; i < distributedType.getRank(); ++i)
      newDistributedShape[i + extractOp.getNumIndices()] =
          distributedType.getDimSize(i);
    auto newDistributedType =
        VectorType::get(newDistributedShape, distributedType.getElementType());
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {extractOp.getVector()}, {newDistributedType},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value distributedVec = newWarpOp->getResult(newRetIndices[0]);
    // Extract from distributed vector.
    Value newExtract = rewriter.create<vector::ExtractOp>(
        loc, distributedVec, extractOp.getMixedPosition());
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                newExtract);
    return success();
  }
};

/// Pattern to move out vector.extract with a scalar result.
/// Only supports 1-D and 0-D sources for now.
struct WarpOpExtractScalar : public WarpDistributionPattern {
  WarpOpExtractScalar(MLIRContext *ctx, WarpShuffleFromIdxFn fn,
                      PatternBenefit b = 1)
      : WarpDistributionPattern(ctx, b), warpShuffleFromIdxFn(std::move(fn)) {}
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ExtractOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto extractOp = operand->get().getDefiningOp<vector::ExtractOp>();
    VectorType extractSrcType = extractOp.getSourceVectorType();
    // Only supports 1-D or 0-D sources for now.
    if (extractSrcType.getRank() > 1) {
      return rewriter.notifyMatchFailure(
          extractOp, "only 0-D or 1-D source supported for now");
    }
    // TODO: Supported shuffle types should be parameterizable, similar to
    // `WarpShuffleFromIdxFn`.
    if (!extractSrcType.getElementType().isF32() &&
        !extractSrcType.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          extractOp, "only f32/i32 element types are supported");
    bool is0dOrVec1Extract = extractSrcType.getNumElements() == 1;
    Type elType = extractSrcType.getElementType();
    VectorType distributedVecType;
    if (!is0dOrVec1Extract) {
      assert(extractSrcType.getRank() == 1 &&
             "expected that extract src rank is 0 or 1");
      if (extractSrcType.getShape()[0] % warpOp.getWarpSize() != 0)
        return failure();
      int64_t elementsPerLane =
          extractSrcType.getShape()[0] / warpOp.getWarpSize();
      distributedVecType = VectorType::get({elementsPerLane}, elType);
    } else {
      distributedVecType = extractSrcType;
    }
    // Yield source vector and position (if present) from warp op.
    SmallVector<Value> additionalResults{extractOp.getVector()};
    SmallVector<Type> additionalResultTypes{distributedVecType};
    additionalResults.append(
        SmallVector<Value>(extractOp.getDynamicPosition()));
    additionalResultTypes.append(
        SmallVector<Type>(extractOp.getDynamicPosition().getTypes()));

    Location loc = extractOp.getLoc();
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, additionalResults, additionalResultTypes,
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value distributedVec = newWarpOp->getResult(newRetIndices[0]);

    // 0d extract: The new warp op broadcasts the source vector to all lanes.
    // All lanes extract the scalar.
    if (is0dOrVec1Extract) {
      Value newExtract;
      SmallVector<int64_t> indices(extractSrcType.getRank(), 0);
      newExtract =
          rewriter.create<vector::ExtractOp>(loc, distributedVec, indices);
      rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                  newExtract);
      return success();
    }

    int64_t staticPos = extractOp.getStaticPosition()[0];
    OpFoldResult pos = ShapedType::isDynamic(staticPos)
                           ? (newWarpOp->getResult(newRetIndices[1]))
                           : OpFoldResult(rewriter.getIndexAttr(staticPos));
    // 1d extract: Distribute the source vector. One lane extracts and shuffles
    // the value to all other lanes.
    int64_t elementsPerLane = distributedVecType.getShape()[0];
    AffineExpr sym0 = getAffineSymbolExpr(0, rewriter.getContext());
    // tid of extracting thread: pos / elementsPerLane
    Value broadcastFromTid = affine::makeComposedAffineApply(
        rewriter, loc, sym0.ceilDiv(elementsPerLane), pos);
    // Extract at position: pos % elementsPerLane
    Value newPos =
        elementsPerLane == 1
            ? rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult()
            : affine::makeComposedAffineApply(rewriter, loc,
                                              sym0 % elementsPerLane, pos);
    Value extracted =
        rewriter.create<vector::ExtractOp>(loc, distributedVec, newPos);

    // Shuffle the extracted value to all lanes.
    Value shuffled = warpShuffleFromIdxFn(
        loc, rewriter, extracted, broadcastFromTid, newWarpOp.getWarpSize());
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), shuffled);
    return success();
  }

private:
  WarpShuffleFromIdxFn warpShuffleFromIdxFn;
};

/// Pattern to convert vector.extractelement to vector.extract.
struct WarpOpExtractElement : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ExtractElementOp>);
    if (!operand)
      return failure();
    auto extractOp = operand->get().getDefiningOp<vector::ExtractElementOp>();
    SmallVector<OpFoldResult> indices;
    if (auto pos = extractOp.getPosition()) {
      indices.push_back(pos);
    }
    rewriter.setInsertionPoint(extractOp);
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        extractOp, extractOp.getVector(), indices);
    return success();
  }
};

/// Pattern to move out vector.insert with a scalar input.
/// Only supports 1-D and 0-D destinations for now.
struct WarpOpInsertScalar : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, llvm::IsaPred<vector::InsertOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto insertOp = operand->get().getDefiningOp<vector::InsertOp>();
    VectorType vecType = insertOp.getDestVectorType();
    VectorType distrType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());

    // Only supports 1-D or 0-D destinations for now.
    if (vecType.getRank() > 1) {
      return rewriter.notifyMatchFailure(
          insertOp, "only 0-D or 1-D source supported for now");
    }

    // Yield destination vector, source scalar and position from warp op.
    SmallVector<Value> additionalResults{insertOp.getDest(),
                                         insertOp.getValueToStore()};
    SmallVector<Type> additionalResultTypes{
        distrType, insertOp.getValueToStore().getType()};
    additionalResults.append(SmallVector<Value>(insertOp.getDynamicPosition()));
    additionalResultTypes.append(
        SmallVector<Type>(insertOp.getDynamicPosition().getTypes()));

    Location loc = insertOp.getLoc();
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, additionalResults, additionalResultTypes,
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value distributedVec = newWarpOp->getResult(newRetIndices[0]);
    Value newSource = newWarpOp->getResult(newRetIndices[1]);
    rewriter.setInsertionPointAfter(newWarpOp);

    OpFoldResult pos;
    if (vecType.getRank() != 0) {
      int64_t staticPos = insertOp.getStaticPosition()[0];
      pos = ShapedType::isDynamic(staticPos)
                ? (newWarpOp->getResult(newRetIndices[2]))
                : OpFoldResult(rewriter.getIndexAttr(staticPos));
    }

    // This condition is always true for 0-d vectors.
    if (vecType == distrType) {
      Value newInsert;
      SmallVector<OpFoldResult> indices;
      if (pos) {
        indices.push_back(pos);
      }
      newInsert = rewriter.create<vector::InsertOp>(loc, newSource,
                                                    distributedVec, indices);
      // Broadcast: Simply move the vector.insert op out.
      rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                  newInsert);
      return success();
    }

    // This is a distribution. Only one lane should insert.
    int64_t elementsPerLane = distrType.getShape()[0];
    AffineExpr sym0 = getAffineSymbolExpr(0, rewriter.getContext());
    // tid of extracting thread: pos / elementsPerLane
    Value insertingLane = affine::makeComposedAffineApply(
        rewriter, loc, sym0.ceilDiv(elementsPerLane), pos);
    // Insert position: pos % elementsPerLane
    OpFoldResult newPos = affine::makeComposedFoldedAffineApply(
        rewriter, loc, sym0 % elementsPerLane, pos);
    Value isInsertingLane = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, newWarpOp.getLaneid(), insertingLane);
    Value newResult =
        rewriter
            .create<scf::IfOp>(
                loc, isInsertingLane,
                /*thenBuilder=*/
                [&](OpBuilder &builder, Location loc) {
                  Value newInsert = builder.create<vector::InsertOp>(
                      loc, newSource, distributedVec, newPos);
                  builder.create<scf::YieldOp>(loc, newInsert);
                },
                /*elseBuilder=*/
                [&](OpBuilder &builder, Location loc) {
                  builder.create<scf::YieldOp>(loc, distributedVec);
                })
            .getResult(0);
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), newResult);
    return success();
  }
};

struct WarpOpInsert : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, llvm::IsaPred<vector::InsertOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto insertOp = operand->get().getDefiningOp<vector::InsertOp>();
    Location loc = insertOp.getLoc();

    // For 1-d or 0-d destination cases, we rely on WarpOpInsertScalar pattern.
    if (insertOp.getDestVectorType().getRank() <= 1) {
      return failure();
    }

    // All following cases are 2d or higher dimensional source vectors.

    if (warpOp.getResult(operandNumber).getType() == operand->get().getType()) {
      // There is no distribution, this is a broadcast. Simply move the insert
      // out of the warp op.
      SmallVector<size_t> newRetIndices;
      WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, {insertOp.getValueToStore(), insertOp.getDest()},
          {insertOp.getValueToStoreType(), insertOp.getDestVectorType()},
          newRetIndices);
      rewriter.setInsertionPointAfter(newWarpOp);
      Value distributedSrc = newWarpOp->getResult(newRetIndices[0]);
      Value distributedDest = newWarpOp->getResult(newRetIndices[1]);
      Value newResult = rewriter.create<vector::InsertOp>(
          loc, distributedSrc, distributedDest, insertOp.getMixedPosition());
      rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                  newResult);
      return success();
    }

    // Find the distributed dimension. There should be exactly one.
    auto distrDestType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    auto yieldedType = cast<VectorType>(operand->get().getType());
    int64_t distrDestDim = -1;
    for (int64_t i = 0; i < yieldedType.getRank(); ++i) {
      if (distrDestType.getDimSize(i) != yieldedType.getDimSize(i)) {
        // Keep this assert here in case WarpExecuteOnLane0Op gets extended to
        // support distributing multiple dimensions in the future.
        assert(distrDestDim == -1 && "found multiple distributed dims");
        distrDestDim = i;
      }
    }
    assert(distrDestDim != -1 && "could not find distributed dimension");

    // Compute the distributed source vector type.
    VectorType srcVecType = cast<VectorType>(insertOp.getValueToStoreType());
    SmallVector<int64_t> distrSrcShape(srcVecType.getShape());
    // E.g.: vector.insert %s, %d [2] : vector<96xf32> into vector<128x96xf32>
    // Case 1: distrDestDim = 1 (dim of size 96). In that case, each lane will
    //         insert a smaller vector<3xf32>.
    // Case 2: distrDestDim = 0 (dim of size 128) => distrSrcDim = -1. In that
    //         case, one lane will insert the source vector<96xf32>. The other
    //         lanes will not do anything.
    int64_t distrSrcDim = distrDestDim - insertOp.getNumIndices();
    if (distrSrcDim >= 0)
      distrSrcShape[distrSrcDim] = distrDestType.getDimSize(distrDestDim);
    auto distrSrcType =
        VectorType::get(distrSrcShape, distrDestType.getElementType());

    // Yield source and dest vectors from warp op.
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {insertOp.getValueToStore(), insertOp.getDest()},
        {distrSrcType, distrDestType}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value distributedSrc = newWarpOp->getResult(newRetIndices[0]);
    Value distributedDest = newWarpOp->getResult(newRetIndices[1]);

    // Insert into the distributed vector.
    Value newResult;
    if (distrSrcDim >= 0) {
      // Every lane inserts a small piece.
      newResult = rewriter.create<vector::InsertOp>(
          loc, distributedSrc, distributedDest, insertOp.getMixedPosition());
    } else {
      // One lane inserts the entire source vector.
      int64_t elementsPerLane = distrDestType.getDimSize(distrDestDim);
      SmallVector<OpFoldResult> pos = insertOp.getMixedPosition();
      SmallVector<int64_t> newPos = getAsIntegers(pos);
      // tid of inserting lane: pos / elementsPerLane
      Value insertingLane = rewriter.create<arith::ConstantIndexOp>(
          loc, newPos[distrDestDim] / elementsPerLane);
      Value isInsertingLane = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, newWarpOp.getLaneid(), insertingLane);
      // Insert position: pos % elementsPerLane
      newPos[distrDestDim] %= elementsPerLane;
      auto insertingBuilder = [&](OpBuilder &builder, Location loc) {
        Value newInsert = builder.create<vector::InsertOp>(
            loc, distributedSrc, distributedDest, newPos);
        builder.create<scf::YieldOp>(loc, newInsert);
      };
      auto nonInsertingBuilder = [&](OpBuilder &builder, Location loc) {
        builder.create<scf::YieldOp>(loc, distributedDest);
      };
      newResult = rewriter
                      .create<scf::IfOp>(loc, isInsertingLane,
                                         /*thenBuilder=*/insertingBuilder,
                                         /*elseBuilder=*/nonInsertingBuilder)
                      .getResult(0);
    }

    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), newResult);
    return success();
  }
};

struct WarpOpInsertElement : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::InsertElementOp>);
    if (!operand)
      return failure();
    auto insertOp = operand->get().getDefiningOp<vector::InsertElementOp>();
    SmallVector<OpFoldResult> indices;
    if (auto pos = insertOp.getPosition()) {
      indices.push_back(pos);
    }
    rewriter.setInsertionPoint(insertOp);
    rewriter.replaceOpWithNewOp<vector::InsertOp>(
        insertOp, insertOp.getSource(), insertOp.getDest(), indices);
    return success();
  }
};

/// Sink scf.for region out of WarpExecuteOnLane0Op. This can be done only if
/// the scf.ForOp is the last operation in the region so that it doesn't
/// change the order of execution. This creates a new scf.for region after the
/// WarpExecuteOnLane0Op. The new scf.for region will contain a new
/// WarpExecuteOnLane0Op region. Example:
/// ```
/// %w = gpu.warp_execute_on_lane_0(%laneid) -> (vector<4xf32>) {
///   ...
///   %v1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %v)
///   -> (vector<128xf32>) {
///     ...
///     scf.yield %r : vector<128xf32>
///   }
///   gpu.yield %v1 : vector<128xf32>
/// }
/// ```
/// To:
/// %w0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<4xf32>) {
///   ...
///   gpu.yield %v : vector<128xf32>
/// }
/// %w = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%varg = %q0)
///   -> (vector<4xf32>) {
///     %iw = gpu.warp_execute_on_lane_0(%laneid)
///     args(%varg : vector<4xf32>) -> (vector<4xf32>) {
///     ^bb0(%arg: vector<128xf32>):
///       ...
///       gpu.yield %ir : vector<128xf32>
///     }
///     scf.yield %iw : vector<4xf32>
///  }
/// ```
struct WarpOpScfForOp : public WarpDistributionPattern {

  WarpOpScfForOp(MLIRContext *ctx, DistributionMapFn fn, PatternBenefit b = 1)
      : WarpDistributionPattern(ctx, b), distributionMapFn(std::move(fn)) {}
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<gpu::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    // Only pick up forOp if it is the last op in the region.
    Operation *lastNode = yield->getPrevNode();
    auto forOp = dyn_cast_or_null<scf::ForOp>(lastNode);
    if (!forOp)
      return failure();
    // Collect Values that come from the warp op but are outside the forOp.
    // Those Value needs to be returned by the original warpOp and passed to
    // the new op.
    llvm::SmallSetVector<Value, 32> escapingValues;
    SmallVector<Type> inputTypes;
    SmallVector<Type> distTypes;
    mlir::visitUsedValuesDefinedAbove(
        forOp.getBodyRegion(), [&](OpOperand *operand) {
          Operation *parent = operand->get().getParentRegion()->getParentOp();
          if (warpOp->isAncestor(parent)) {
            if (!escapingValues.insert(operand->get()))
              return;
            Type distType = operand->get().getType();
            if (auto vecType = dyn_cast<VectorType>(distType)) {
              AffineMap map = distributionMapFn(operand->get());
              distType = getDistributedType(vecType, map, warpOp.getWarpSize());
            }
            inputTypes.push_back(operand->get().getType());
            distTypes.push_back(distType);
          }
        });

    if (llvm::is_contained(distTypes, Type{}))
      return failure();

    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, escapingValues.getArrayRef(), distTypes,
        newRetIndices);
    yield = cast<gpu::YieldOp>(
        newWarpOp.getBodyRegion().getBlocks().begin()->getTerminator());

    SmallVector<Value> newOperands;
    SmallVector<unsigned> resultIdx;
    // Collect all the outputs coming from the forOp.
    for (OpOperand &yieldOperand : yield->getOpOperands()) {
      if (yieldOperand.get().getDefiningOp() != forOp.getOperation())
        continue;
      auto forResult = cast<OpResult>(yieldOperand.get());
      newOperands.push_back(
          newWarpOp.getResult(yieldOperand.getOperandNumber()));
      yieldOperand.set(forOp.getInitArgs()[forResult.getResultNumber()]);
      resultIdx.push_back(yieldOperand.getOperandNumber());
    }

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Create a new for op outside the region with a WarpExecuteOnLane0Op
    // region inside.
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newOperands);
    rewriter.setInsertionPointToStart(newForOp.getBody());

    SmallVector<Value> warpInput(newForOp.getRegionIterArgs().begin(),
                                 newForOp.getRegionIterArgs().end());
    SmallVector<Type> warpInputType(forOp.getResultTypes().begin(),
                                    forOp.getResultTypes().end());
    llvm::SmallDenseMap<Value, int64_t> argIndexMapping;
    for (auto [i, retIdx] : llvm::enumerate(newRetIndices)) {
      warpInput.push_back(newWarpOp.getResult(retIdx));
      argIndexMapping[escapingValues[i]] = warpInputType.size();
      warpInputType.push_back(inputTypes[i]);
    }
    auto innerWarp = rewriter.create<WarpExecuteOnLane0Op>(
        newWarpOp.getLoc(), newForOp.getResultTypes(), newWarpOp.getLaneid(),
        newWarpOp.getWarpSize(), warpInput, warpInputType);

    SmallVector<Value> argMapping;
    argMapping.push_back(newForOp.getInductionVar());
    for (Value args : innerWarp.getBody()->getArguments()) {
      argMapping.push_back(args);
    }
    argMapping.resize(forOp.getBody()->getNumArguments());
    SmallVector<Value> yieldOperands;
    for (Value operand : forOp.getBody()->getTerminator()->getOperands())
      yieldOperands.push_back(operand);
    rewriter.eraseOp(forOp.getBody()->getTerminator());
    rewriter.mergeBlocks(forOp.getBody(), innerWarp.getBody(), argMapping);
    rewriter.setInsertionPointToEnd(innerWarp.getBody());
    rewriter.create<gpu::YieldOp>(innerWarp.getLoc(), yieldOperands);
    rewriter.setInsertionPointAfter(innerWarp);
    if (!innerWarp.getResults().empty())
      rewriter.create<scf::YieldOp>(forOp.getLoc(), innerWarp.getResults());
    rewriter.eraseOp(forOp);
    // Replace the warpOp result coming from the original ForOp.
    for (const auto &res : llvm::enumerate(resultIdx)) {
      rewriter.replaceAllUsesWith(newWarpOp.getResult(res.value()),
                                  newForOp.getResult(res.index()));
      newForOp->setOperand(res.index() + 3, newWarpOp.getResult(res.value()));
    }
    newForOp.walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        auto it = argIndexMapping.find(operand.get());
        if (it == argIndexMapping.end())
          continue;
        operand.set(innerWarp.getBodyRegion().getArgument(it->second));
      }
    });

    // Finally, hoist out any now uniform code from the inner warp op.
    mlir::vector::moveScalarUniformCode(innerWarp);
    return success();
  }

private:
  DistributionMapFn distributionMapFn;
};

/// A pattern that extracts vector.reduction ops from a WarpExecuteOnLane0Op.
/// The vector is reduced in parallel. Currently limited to vector size
/// matching the warpOp size. E.g.:
/// ```
/// %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
///   %0 = "some_def"() : () -> (vector<32xf32>)
///   %1 = vector.reduction "add", %0 : vector<32xf32> into f32
///   gpu.yield %1 : f32
/// }
/// ```
/// is lowered to:
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
///   %1 = "some_def"() : () -> (vector<32xf32>)
///   gpu.yield %1 : vector<32xf32>
/// }
/// %a = vector.extract %0[0] : f32 from vector<1xf32>
/// %r = ("warp.reduction %a")
/// ```
struct WarpOpReduction : public WarpDistributionPattern {
  WarpOpReduction(MLIRContext *context,
                  DistributedReductionFn distributedReductionFn,
                  PatternBenefit benefit = 1)
      : WarpDistributionPattern(context, benefit),
        distributedReductionFn(std::move(distributedReductionFn)) {}

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ReductionOp>);
    if (!yieldOperand)
      return failure();

    auto reductionOp =
        cast<vector::ReductionOp>(yieldOperand->get().getDefiningOp());
    auto vectorType = cast<VectorType>(reductionOp.getVector().getType());
    // Only rank 1 vectors supported.
    if (vectorType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          warpOp, "Only rank 1 reductions can be distributed.");
    // Only warp_size-sized vectors supported.
    if (vectorType.getShape()[0] % warpOp.getWarpSize() != 0)
      return rewriter.notifyMatchFailure(
          warpOp, "Reduction vector dimension must match was size.");
    if (!reductionOp.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(
          warpOp, "Reduction distribution currently only supports floats and "
                  "integer types.");

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

    // Obtain data to reduce for a single lane.
    Value laneValVec = newWarpOp.getResult(newRetIndices[0]);
    // Distribute and reduce across threads.
    Value fullReduce =
        distributedReductionFn(reductionOp.getLoc(), rewriter, laneValVec,
                               reductionOp.getKind(), newWarpOp.getWarpSize());
    if (reductionOp.getAcc()) {
      fullReduce = vector::makeArithReduction(
          rewriter, reductionOp.getLoc(), reductionOp.getKind(), fullReduce,
          newWarpOp.getResult(newRetIndices[1]));
    }
    rewriter.replaceAllUsesWith(newWarpOp.getResult(operandIndex), fullReduce);
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
    unsigned maxNumElementsToExtract, PatternBenefit benefit) {
  patterns.add<WarpOpTransferWrite>(patterns.getContext(), distributionMapFn,
                                    maxNumElementsToExtract, benefit);
}

void mlir::vector::populatePropagateWarpVectorDistributionPatterns(
    RewritePatternSet &patterns, const DistributionMapFn &distributionMapFn,
    const WarpShuffleFromIdxFn &warpShuffleFromIdxFn, PatternBenefit benefit,
    PatternBenefit readBenefit) {
  patterns.add<WarpOpTransferRead>(patterns.getContext(), readBenefit);
  patterns.add<WarpOpElementwise, WarpOpDeadResult, WarpOpBroadcast,
               WarpOpShapeCast, WarpOpExtract, WarpOpForwardOperand,
               WarpOpConstant, WarpOpExtractElement, WarpOpInsertElement,
               WarpOpInsertScalar, WarpOpInsert, WarpOpCreateMask>(
      patterns.getContext(), benefit);
  patterns.add<WarpOpExtractScalar>(patterns.getContext(), warpShuffleFromIdxFn,
                                    benefit);
  patterns.add<WarpOpScfForOp>(patterns.getContext(), distributionMapFn,
                               benefit);
}

void mlir::vector::populateDistributeReduction(
    RewritePatternSet &patterns,
    const DistributedReductionFn &distributedReductionFn,
    PatternBenefit benefit) {
  patterns.add<WarpOpReduction>(patterns.getContext(), distributedReductionFn,
                                benefit);
}

/// Helper to know if an op can be hoisted out of the region.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return llvm::all_of(op->getOperands(), definedOutside) &&
         isMemoryEffectFree(op) && op->getNumRegions() == 0;
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
      return isa<VectorType>(result.getType());
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}
