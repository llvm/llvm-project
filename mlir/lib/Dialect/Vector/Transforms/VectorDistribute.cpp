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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
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

/// Given a sequential and distributed vector type, returns the distributed
/// dimension. This function expects that only a single dimension is
/// distributed.
static int getDistributedDim(VectorType sequentialType,
                             VectorType distributedType) {
  assert(sequentialType.getRank() == distributedType.getRank() &&
         "sequential and distributed vector types must have the same rank");
  int64_t distributedDim = -1;
  for (int64_t i = 0; i < sequentialType.getRank(); ++i) {
    if (distributedType.getDimSize(i) != sequentialType.getDimSize(i)) {
      // Keep this assert here in case WarpExecuteOnLane0Op gets extended to
      // support distributing multiple dimensions in the future.
      assert(distributedDim == -1 && "found multiple distributed dims");
      distributedDim = i;
    }
  }
  return distributedDim;
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
      return memref::StoreOp::create(b, loc, val, buffer, zero);

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
    return vector::TransferWriteOp::create(
        b, loc, val, buffer, indices,
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
      return memref::LoadOp::create(b, loc, buffer, zero);

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
    return vector::TransferReadOp::create(
        b, loc, cast<VectorType>(type), buffer, indices,
        /*padding=*/std::nullopt,
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
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value isLane0 = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, warpOp.getLaneid(), c0);
    auto ifOp = scf::IfOp::create(rewriter, loc, isLane0,
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
    scf::YieldOp::create(rewriter, yieldLoc);

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

/// Given a warpOp that contains ops with regions, the corresponding op's
/// "inner" region and the distributionMapFn, get all values used by the op's
/// region that are defined within the warpOp, but outside the inner region.
/// Return the set of values, their types and their distributed types.
std::tuple<llvm::SmallSetVector<Value, 32>, SmallVector<Type>,
           SmallVector<Type>>
getInnerRegionEscapingValues(WarpExecuteOnLane0Op warpOp, Region &innerRegion,
                             DistributionMapFn distributionMapFn) {
  llvm::SmallSetVector<Value, 32> escapingValues;
  SmallVector<Type> escapingValueTypes;
  SmallVector<Type> escapingValueDistTypes; // to yield from the new warpOp
  if (innerRegion.empty())
    return {std::move(escapingValues), std::move(escapingValueTypes),
            std::move(escapingValueDistTypes)};
  mlir::visitUsedValuesDefinedAbove(innerRegion, [&](OpOperand *operand) {
    Operation *parent = operand->get().getParentRegion()->getParentOp();
    if (warpOp->isAncestor(parent)) {
      if (!escapingValues.insert(operand->get()))
        return;
      Type distType = operand->get().getType();
      if (auto vecType = dyn_cast<VectorType>(distType)) {
        AffineMap map = distributionMapFn(operand->get());
        distType = getDistributedType(vecType, map, warpOp.getWarpSize());
      }
      escapingValueTypes.push_back(operand->get().getType());
      escapingValueDistTypes.push_back(distType);
    }
  });
  return {std::move(escapingValues), std::move(escapingValueTypes),
          std::move(escapingValueDistTypes)};
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
      delinearized = mlir::affine::AffineDelinearizeIndexOp::create(
                         rewriter, newWarpOp.getLoc(), newWarpOp.getLaneid(),
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
    auto secondWarpOp = WarpExecuteOnLane0Op::create(rewriter, loc, TypeRange(),
                                                     newWarpOp.getLaneid(),
                                                     newWarpOp.getWarpSize());
    Block &body = secondWarpOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&body);
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    newWriteOp.getValueToStoreMutable().assign(
        newWarpOp.getResult(newRetIndices[0]));
    rewriter.eraseOp(writeOp);
    gpu::YieldOp::create(rewriter, newWarpOp.getLoc());
    return success();
  }

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    gpu::YieldOp yield = warpOp.getTerminator();
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
    Value distConstant = arith::ConstantOp::create(rewriter, loc, newAttr);
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIndex), distConstant);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};

/// Sink out step op feeding into a warp op yield.
/// Vector step op is treated similar to arith.constant, apart from
/// the result that represents a sequence [0, vec_size).
/// Due to the to vec_size == warp_size limitation,
/// we can simply wrap the lane id into a vector (i.e., broadcast).
/// Supporting vec_size != warp_size may involve preserving the step
/// result and using additional arith ops (the exact details are TBD).
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<1xindex>) {
///   ...
///   %cst = vector.step : vector<32xindex>
///   gpu.yield %cst : vector<1xindex>
/// }
/// ```
/// To
/// ```
/// gpu.warp_execute_on_lane_0(%arg0) {
///   ...
/// }
/// %lane_id_vec = vector.broadcast %arg0 : index to vector<1xindex>
struct WarpOpStep final : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<vector::StepOp>);
    if (!yieldOperand)
      return failure();
    const unsigned operandIdx = yieldOperand->getOperandNumber();
    auto stepOp = yieldOperand->get().getDefiningOp<vector::StepOp>();
    VectorType resTy = stepOp.getResult().getType();
    if (resTy.getNumElements() != static_cast<int64_t>(warpOp.getWarpSize()))
      return rewriter.notifyMatchFailure(
          warpOp,
          llvm::formatv("Expected result size ({0}) to be of warp size ({1})",
                        resTy.getNumElements(), warpOp.getWarpSize()));
    VectorType newVecTy =
        cast<VectorType>(warpOp.getResult(operandIdx).getType());
    rewriter.setInsertionPointAfter(warpOp);
    Value laneIdVec = vector::BroadcastOp::create(rewriter, warpOp.getLoc(),
                                                  newVecTy, warpOp.getLaneid());
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIdx), laneIdVec);
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
    auto newRead = vector::TransferReadOp::create(
        rewriter, read.getLoc(), distributedVal.getType(), read.getBase(),
        newIndices, read.getPermutationMapAttr(), newPadding, newMask,
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
    gpu::YieldOp yield = warpOp.getTerminator();

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
    gpu::YieldOp yield = warpOp.getTerminator();
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
    Value broadcasted = vector::BroadcastOp::create(
        rewriter, loc, destVecType, newWarpOp->getResult(newRetIndices[0]));
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
    Value newCast = vector::ShapeCastOp::create(
        rewriter, oldCastOp.getLoc(), castResultType,
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
        vector::CreateMaskOp::create(rewriter, loc, distType, newOperands);
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIndex), newMask);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};

/// Sink out insert_strided_slice op feeding into a warp op yield.
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<8x1xf32>) {
///   ...
///   %src = ... : vector<4x32xf32>
///   %dest = ... : vector<8x32xf32>
///   %insert = vector.insert_strided_slice %src, %dest, offsets = [0, 0],
///     strides = [1, 1] : vector<4x32xf32> into vector<8x32xf32>
///   gpu.yield %insert : vector<8x32xf32>
/// }
/// ```
/// To
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<4x1xf32>,
/// vector<8x1xf32>) {
///   ...
///   %src = ... : vector<4x32xf32>
///   %dest = ... : vector<8x32xf32>
///   gpu.yield %src, %dest : vector<4x16xf32>, vector<8x16xf32>
/// }
/// %insert = vector.insert_strided_slice %0#0, %0#1,
///   offsets = [0, 0], strides = [1, 1] : vector<4x1xf32> into vector<8x1xf32>
/// ```
/// NOTE: Current support assumes that both src and dest vectors are distributed
/// to lanes and sinking the insert op does not require any cross lane
/// communication.
struct WarpOpInsertStridedSlice : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::InsertStridedSliceOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto insertOp =
        operand->get().getDefiningOp<vector::InsertStridedSliceOp>();
    auto distributedType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    // Distributed type must be 2D or higher.
    // TODO: Support 1D distributed types.
    if (distributedType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          insertOp, "result vector type must be 2D or higher");
    // Find the distributed dimension of the dest vector. There should be
    // exactly one.
    auto yieldedType = cast<VectorType>(operand->get().getType());
    int64_t destDistributedDim =
        getDistributedDim(yieldedType, distributedType);
    assert(destDistributedDim != -1 && "could not find distributed dimension");

    VectorType srcType = insertOp.getSourceVectorType();
    VectorType destType = insertOp.getDestVectorType();
    // Currently we require that both source (kD) and dest (nD) vectors are
    // distributed. This requires that distributedDim (d) is contained in the
    // last k dims of the dest vector (d >= n - k).
    // TODO: Add support for case where source vector is not distributed.
    int64_t sourceDistributedDim =
        destDistributedDim - (destType.getRank() - srcType.getRank());
    if (sourceDistributedDim < 0)
      return rewriter.notifyMatchFailure(
          insertOp,
          "distributed dimension must be in the last k dims of dest vector");
    // Distributed dimension must be fully inserted.
    if (srcType.getDimSize(sourceDistributedDim) !=
        destType.getDimSize(destDistributedDim))
      return rewriter.notifyMatchFailure(
          insertOp, "distributed dimension must be fully inserted");
    SmallVector<int64_t> newSourceDistShape(
        insertOp.getSourceVectorType().getShape());
    newSourceDistShape[sourceDistributedDim] =
        distributedType.getDimSize(destDistributedDim);
    auto newSourceTy =
        VectorType::get(newSourceDistShape, distributedType.getElementType());
    VectorType newDestTy = distributedType;
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {insertOp.getValueToStore(), insertOp.getDest()},
        {newSourceTy, newDestTy}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value distributedSource = newWarpOp->getResult(newRetIndices[0]);
    Value distributedDest = newWarpOp->getResult(newRetIndices[1]);
    // Create a new insert strided slice op that inserts distributed source into
    // distributed dest.
    Value newInsert = vector::InsertStridedSliceOp::create(
        rewriter, insertOp.getLoc(), distributedDest.getType(),
        distributedSource, distributedDest, insertOp.getOffsets(),
        insertOp.getStrides());
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), newInsert);
    return success();
  }
};

/// Sink out extract_strided_slice op feeding into a warp op yield.
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<16x1xf32>) {
///   ...
///   %src = ... : vector<64x32xf32>
///   %extract = vector.extract_strided_slice %src, offsets = [0], sizes = [16],
///     strides = [1] : vector<64x32xf32> to vector<16x32xf32>
///   gpu.yield %extract : vector<16x32xf32>
/// }
/// ```
/// To
/// ```
/// %0 = gpu.warp_execute_on_lane_0(%arg0) -> (vector<64x1xf32>) {
///   ...
///   %src = ... : vector<64x32xf32>
///   gpu.yield %src : vector<64x32xf32>
/// }
/// %extract = vector.extract_strided_slice %0, offsets = [0], sizes = [16],
///   strides = [1] : vector<64x1xf32> to vector<16x1xf32>
/// ```
/// NOTE: Current support assumes that the extraction happens only on non
/// distributed dimensions (does not require cross lane communication).
struct WarpOpExtractStridedSlice : public WarpDistributionPattern {
  using Base::Base;
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ExtractStridedSliceOp>);
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto extractOp =
        operand->get().getDefiningOp<vector::ExtractStridedSliceOp>();
    auto distributedType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    // Distributed type must be 2D or higher.
    // TODO: Support 1D distributed types.
    if (distributedType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          extractOp, "result vector type must be 2D or higher");

    // Find the distributed dimension. There should be exactly one.
    auto yieldedType = cast<VectorType>(operand->get().getType());
    int64_t distributedDim = getDistributedDim(yieldedType, distributedType);
    assert(distributedDim != -1 && "could not find distributed dimension");

    int64_t numOfExtractedDims =
        static_cast<int64_t>(extractOp.getSizes().size());
    // If the distributed dim is included in the extracted dims,  then we make
    // sure distributed dim is fully extracted. If distributed dim is not
    // included in extracted dims, it is guaranteed to be fully extracted (i.e.
    // distributed dim comes after all the extracted dims)
    // TODO: Partial extraction from distributed dimension require cross lane
    // communication.
    if (distributedDim < numOfExtractedDims) {
      int64_t distributedDimOffset =
          llvm::cast<IntegerAttr>(extractOp.getOffsets()[distributedDim])
              .getInt();
      int64_t distributedDimSize =
          llvm::cast<IntegerAttr>(extractOp.getSizes()[distributedDim])
              .getInt();
      if (distributedDimOffset != 0 ||
          distributedDimSize != yieldedType.getDimSize(distributedDim))
        return rewriter.notifyMatchFailure(
            extractOp, "distributed dimension must be fully extracted");
    }
    SmallVector<int64_t> newDistributedShape(
        extractOp.getSourceVectorType().getShape());
    newDistributedShape[distributedDim] =
        distributedType.getDimSize(distributedDim);
    auto newDistributedType =
        VectorType::get(newDistributedShape, distributedType.getElementType());
    SmallVector<size_t> newRetIndices;
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {extractOp.getVector()}, {newDistributedType},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Attribute> distributedSizes = llvm::map_to_vector(
        extractOp.getSizes(), [](Attribute attr) { return attr; });
    // Update the distributed sizes to match the distributed type.
    if (distributedDim < static_cast<int64_t>(distributedSizes.size()))
      distributedSizes[distributedDim] = rewriter.getI64IntegerAttr(
          distributedType.getDimSize(distributedDim));

    // Create a new extract strided slice op that extracts from the
    // distributed vector.
    Value distributedVec = newWarpOp->getResult(newRetIndices[0]);
    Value newExtract = vector::ExtractStridedSliceOp::create(
        rewriter, extractOp.getLoc(), distributedType, distributedVec,
        extractOp.getOffsets(),
        ArrayAttr::get(rewriter.getContext(), distributedSizes),
        extractOp.getStrides());
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                newExtract);
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
      Value newExtract = vector::ExtractOp::create(
          rewriter, loc, distributedVec, extractOp.getMixedPosition());
      rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber),
                                  newExtract);
      return success();
    }

    // Find the distributed dimension. There should be exactly one.
    auto distributedType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    auto yieldedType = cast<VectorType>(operand->get().getType());
    int64_t distributedDim = getDistributedDim(yieldedType, distributedType);
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
    Value newExtract = vector::ExtractOp::create(rewriter, loc, distributedVec,
                                                 extractOp.getMixedPosition());
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
          vector::ExtractOp::create(rewriter, loc, distributedVec, indices);
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
            ? arith::ConstantIndexOp::create(rewriter, loc, 0).getResult()
            : affine::makeComposedAffineApply(rewriter, loc,
                                              sym0 % elementsPerLane, pos);
    Value extracted =
        vector::ExtractOp::create(rewriter, loc, distributedVec, newPos);

    // Shuffle the extracted value to all lanes.
    Value shuffled = warpShuffleFromIdxFn(
        loc, rewriter, extracted, broadcastFromTid, newWarpOp.getWarpSize());
    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), shuffled);
    return success();
  }

private:
  WarpShuffleFromIdxFn warpShuffleFromIdxFn;
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
      newInsert = vector::InsertOp::create(rewriter, loc, newSource,
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
    Value isInsertingLane =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                              newWarpOp.getLaneid(), insertingLane);
    Value newResult =
        scf::IfOp::create(
            rewriter, loc, isInsertingLane,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              Value newInsert = vector::InsertOp::create(
                  builder, loc, newSource, distributedVec, newPos);
              scf::YieldOp::create(builder, loc, newInsert);
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              scf::YieldOp::create(builder, loc, distributedVec);
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
      Value newResult = vector::InsertOp::create(rewriter, loc, distributedSrc,
                                                 distributedDest,
                                                 insertOp.getMixedPosition());
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
      newResult = vector::InsertOp::create(rewriter, loc, distributedSrc,
                                           distributedDest,
                                           insertOp.getMixedPosition());
    } else {
      // One lane inserts the entire source vector.
      int64_t elementsPerLane = distrDestType.getDimSize(distrDestDim);
      SmallVector<OpFoldResult> pos = insertOp.getMixedPosition();
      SmallVector<int64_t> newPos = getAsIntegers(pos);
      // tid of inserting lane: pos / elementsPerLane
      Value insertingLane = arith::ConstantIndexOp::create(
          rewriter, loc, newPos[distrDestDim] / elementsPerLane);
      Value isInsertingLane =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                newWarpOp.getLaneid(), insertingLane);
      // Insert position: pos % elementsPerLane
      newPos[distrDestDim] %= elementsPerLane;
      auto insertingBuilder = [&](OpBuilder &builder, Location loc) {
        Value newInsert = vector::InsertOp::create(builder, loc, distributedSrc,
                                                   distributedDest, newPos);
        scf::YieldOp::create(builder, loc, newInsert);
      };
      auto nonInsertingBuilder = [&](OpBuilder &builder, Location loc) {
        scf::YieldOp::create(builder, loc, distributedDest);
      };
      newResult = scf::IfOp::create(rewriter, loc, isInsertingLane,
                                    /*thenBuilder=*/insertingBuilder,
                                    /*elseBuilder=*/nonInsertingBuilder)
                      .getResult(0);
    }

    rewriter.replaceAllUsesWith(newWarpOp->getResult(operandNumber), newResult);
    return success();
  }
};

/// Sink scf.if out of WarpExecuteOnLane0Op. This can be done only if
/// the scf.if is the last operation in the region so that it doesn't
/// change the order of execution. This creates a new scf.if after the
/// WarpExecuteOnLane0Op. Each branch of the new scf.if is enclosed in
/// the "inner" WarpExecuteOnLane0Op. Example:
/// ```
/// gpu.warp_execute_on_lane_0(%laneid)[32] {
///   %payload = ... : vector<32xindex>
///   scf.if %pred {
///     vector.store %payload, %buffer[%idx] : memref<128xindex>,
///     vector<32xindex>
///   }
///   gpu.yield
/// }
/// ```
/// %r = gpu.warp_execute_on_lane_0(%laneid)[32] {
///   %payload = ... : vector<32xindex>
///   gpu.yield %payload : vector<32xindex>
/// }
/// scf.if %pred {
///   gpu.warp_execute_on_lane_0(%laneid)[32] args(%r : vector<1xindex>) {
///     ^bb0(%arg1: vector<32xindex>):
///     vector.store %arg1, %buffer[%idx] : memref<128xindex>, vector<32xindex>
///   }
/// }
/// ```
struct WarpOpScfIfOp : public WarpDistributionPattern {
  WarpOpScfIfOp(MLIRContext *ctx, DistributionMapFn fn, PatternBenefit b = 1)
      : WarpDistributionPattern(ctx, b), distributionMapFn(std::move(fn)) {}
  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    gpu::YieldOp warpOpYield = warpOp.getTerminator();
    // Only pick up `IfOp` if it is the last op in the region.
    Operation *lastNode = warpOpYield->getPrevNode();
    auto ifOp = dyn_cast_or_null<scf::IfOp>(lastNode);
    if (!ifOp)
      return failure();

    // The current `WarpOp` can yield two types of values:
    // 1. Not results of `IfOp`:
    //     Preserve them in the new `WarpOp`.
    //     Collect their yield index to remap the usages.
    // 2. Results of `IfOp`:
    //     They are not part of the new `WarpOp` results.
    //     Map current warp's yield operand index to `IfOp` result idx.
    SmallVector<Value> nonIfYieldValues;
    SmallVector<unsigned> nonIfYieldIndices;
    llvm::SmallDenseMap<unsigned, unsigned> ifResultMapping;
    llvm::SmallDenseMap<unsigned, VectorType> ifResultDistTypes;
    for (OpOperand &yieldOperand : warpOpYield->getOpOperands()) {
      const unsigned yieldOperandIdx = yieldOperand.getOperandNumber();
      if (yieldOperand.get().getDefiningOp() != ifOp.getOperation()) {
        nonIfYieldValues.push_back(yieldOperand.get());
        nonIfYieldIndices.push_back(yieldOperandIdx);
        continue;
      }
      OpResult ifResult = cast<OpResult>(yieldOperand.get());
      const unsigned ifResultIdx = ifResult.getResultNumber();
      ifResultMapping[yieldOperandIdx] = ifResultIdx;
      // If this `ifOp` result is vector type and it is yielded by the
      // `WarpOp`, we keep track the distributed type for this result.
      if (!isa<VectorType>(ifResult.getType()))
        continue;
      VectorType distType =
          cast<VectorType>(warpOp.getResult(yieldOperandIdx).getType());
      ifResultDistTypes[ifResultIdx] = distType;
    }

    // Collect `WarpOp`-defined values used in `ifOp`, the new warp op returns
    // them
    auto [escapingValuesThen, escapingValueInputTypesThen,
          escapingValueDistTypesThen] =
        getInnerRegionEscapingValues(warpOp, ifOp.getThenRegion(),
                                     distributionMapFn);
    auto [escapingValuesElse, escapingValueInputTypesElse,
          escapingValueDistTypesElse] =
        getInnerRegionEscapingValues(warpOp, ifOp.getElseRegion(),
                                     distributionMapFn);
    if (llvm::is_contained(escapingValueDistTypesThen, Type{}) ||
        llvm::is_contained(escapingValueDistTypesElse, Type{}))
      return failure();

    // The new `WarpOp` groups yields values in following order:
    // 1. Branch condition
    // 2. Escaping values then branch
    // 3. Escaping values else branch
    // 4. All non-`ifOp` yielded values.
    SmallVector<Value> newWarpOpYieldValues{ifOp.getCondition()};
    newWarpOpYieldValues.append(escapingValuesThen.begin(),
                                escapingValuesThen.end());
    newWarpOpYieldValues.append(escapingValuesElse.begin(),
                                escapingValuesElse.end());
    SmallVector<Type> newWarpOpDistTypes{ifOp.getCondition().getType()};
    newWarpOpDistTypes.append(escapingValueDistTypesThen.begin(),
                              escapingValueDistTypesThen.end());
    newWarpOpDistTypes.append(escapingValueDistTypesElse.begin(),
                              escapingValueDistTypesElse.end());

    llvm::SmallDenseMap<unsigned, unsigned> origToNewYieldIdx;
    for (auto [idx, val] :
         llvm::zip_equal(nonIfYieldIndices, nonIfYieldValues)) {
      origToNewYieldIdx[idx] = newWarpOpYieldValues.size();
      newWarpOpYieldValues.push_back(val);
      newWarpOpDistTypes.push_back(warpOp.getResult(idx).getType());
    }
    // Create the new `WarpOp` with the updated yield values and types.
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
        rewriter, warpOp, newWarpOpYieldValues, newWarpOpDistTypes);
    // `ifOp` returns the result of the inner warp op.
    SmallVector<Type> newIfOpDistResTypes;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults())) {
      Type distType = cast<Value>(res).getType();
      if (auto vecType = dyn_cast<VectorType>(distType)) {
        AffineMap map = distributionMapFn(cast<Value>(res));
        // Fallback to affine map if the dist result was not previously recorded
        distType = ifResultDistTypes.count(i)
                       ? ifResultDistTypes[i]
                       : getDistributedType(vecType, map, warpOp.getWarpSize());
      }
      newIfOpDistResTypes.push_back(distType);
    }
    // Create a new `IfOp` outside the new `WarpOp` region.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newIfOp = scf::IfOp::create(
        rewriter, ifOp.getLoc(), newIfOpDistResTypes, newWarpOp.getResult(0),
        static_cast<bool>(ifOp.thenBlock()),
        static_cast<bool>(ifOp.elseBlock()));
    auto encloseRegionInWarpOp =
        [&](Block *oldIfBranch, Block *newIfBranch,
            llvm::SmallSetVector<Value, 32> &escapingValues,
            SmallVector<Type> &escapingValueInputTypes,
            size_t warpResRangeStart) {
          OpBuilder::InsertionGuard g(rewriter);
          if (!newIfBranch)
            return;
          rewriter.setInsertionPointToStart(newIfBranch);
          llvm::SmallDenseMap<Value, int64_t> escapeValToBlockArgIndex;
          SmallVector<Value> innerWarpInputVals;
          SmallVector<Type> innerWarpInputTypes;
          for (size_t i = 0; i < escapingValues.size();
               ++i, ++warpResRangeStart) {
            innerWarpInputVals.push_back(
                newWarpOp.getResult(warpResRangeStart));
            escapeValToBlockArgIndex[escapingValues[i]] =
                innerWarpInputTypes.size();
            innerWarpInputTypes.push_back(escapingValueInputTypes[i]);
          }
          auto innerWarp = WarpExecuteOnLane0Op::create(
              rewriter, newWarpOp.getLoc(), newIfOp.getResultTypes(),
              newWarpOp.getLaneid(), newWarpOp.getWarpSize(),
              innerWarpInputVals, innerWarpInputTypes);

          innerWarp.getWarpRegion().takeBody(*oldIfBranch->getParent());
          innerWarp.getWarpRegion().addArguments(
              innerWarpInputTypes,
              SmallVector<Location>(innerWarpInputTypes.size(), ifOp.getLoc()));

          SmallVector<Value> yieldOperands;
          for (Value operand : oldIfBranch->getTerminator()->getOperands())
            yieldOperands.push_back(operand);
          rewriter.eraseOp(oldIfBranch->getTerminator());

          rewriter.setInsertionPointToEnd(innerWarp.getBody());
          gpu::YieldOp::create(rewriter, innerWarp.getLoc(), yieldOperands);
          rewriter.setInsertionPointAfter(innerWarp);
          scf::YieldOp::create(rewriter, ifOp.getLoc(), innerWarp.getResults());

          // Update any users of escaping values that were forwarded to the
          // inner `WarpOp`. These values are arguments of the inner `WarpOp`.
          innerWarp.walk([&](Operation *op) {
            for (OpOperand &operand : op->getOpOperands()) {
              auto it = escapeValToBlockArgIndex.find(operand.get());
              if (it == escapeValToBlockArgIndex.end())
                continue;
              operand.set(innerWarp.getBodyRegion().getArgument(it->second));
            }
          });
          mlir::vector::moveScalarUniformCode(innerWarp);
        };
    encloseRegionInWarpOp(&ifOp.getThenRegion().front(),
                          &newIfOp.getThenRegion().front(), escapingValuesThen,
                          escapingValueInputTypesThen, 1);
    if (!ifOp.getElseRegion().empty())
      encloseRegionInWarpOp(&ifOp.getElseRegion().front(),
                            &newIfOp.getElseRegion().front(),
                            escapingValuesElse, escapingValueInputTypesElse,
                            1 + escapingValuesThen.size());
    // Update the users of `<- WarpOp.yield <- IfOp.yield` to use the new `IfOp`
    // result.
    for (auto [origIdx, newIdx] : ifResultMapping)
      rewriter.replaceAllUsesExcept(warpOp.getResult(origIdx),
                                    newIfOp.getResult(newIdx), newIfOp);
    // Similarly, update any users of the `WarpOp` results that were not
    // results of the `IfOp`.
    for (auto [origIdx, newIdx] : origToNewYieldIdx)
      rewriter.replaceAllUsesWith(warpOp.getResult(origIdx),
                                  newWarpOp.getResult(newIdx));
    // Remove the original `WarpOp` and `IfOp`, they should not have any uses
    // at this point.
    rewriter.eraseOp(ifOp);
    rewriter.eraseOp(warpOp);
    return success();
  }

private:
  DistributionMapFn distributionMapFn;
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
    gpu::YieldOp warpOpYield = warpOp.getTerminator();
    // Only pick up `ForOp` if it is the last op in the region.
    Operation *lastNode = warpOpYield->getPrevNode();
    auto forOp = dyn_cast_or_null<scf::ForOp>(lastNode);
    if (!forOp)
      return failure();
    // Collect Values that come from the `WarpOp` but are outside the `ForOp`.
    // Those Values need to be returned by the new warp op.
    auto [escapingValues, escapingValueInputTypes, escapingValueDistTypes] =
        getInnerRegionEscapingValues(warpOp, forOp.getBodyRegion(),
                                     distributionMapFn);
    if (llvm::is_contained(escapingValueDistTypes, Type{}))
      return failure();
    // `WarpOp` can yield two types of values:
    // 1. Values that are not results of the `ForOp`:
    //    These values must also be yielded by the new `WarpOp`. Also, we need
    //    to record the index mapping for these values to replace them later.
    // 2. Values that are results of the `ForOp`:
    //    In this case, we record the index mapping between the `WarpOp` result
    //    index and matching `ForOp` result index.
    // Additionally, we keep track of the distributed types for all `ForOp`
    // vector results.
    SmallVector<Value> nonForYieldedValues;
    SmallVector<unsigned> nonForResultIndices;
    llvm::SmallDenseMap<unsigned, unsigned> forResultMapping;
    llvm::SmallDenseMap<unsigned, VectorType> forResultDistTypes;
    for (OpOperand &yieldOperand : warpOpYield->getOpOperands()) {
      // Yielded value is not a result of the forOp.
      if (yieldOperand.get().getDefiningOp() != forOp.getOperation()) {
        nonForYieldedValues.push_back(yieldOperand.get());
        nonForResultIndices.push_back(yieldOperand.getOperandNumber());
        continue;
      }
      OpResult forResult = cast<OpResult>(yieldOperand.get());
      unsigned int forResultNumber = forResult.getResultNumber();
      forResultMapping[yieldOperand.getOperandNumber()] = forResultNumber;
      // If this `ForOp` result is vector type and it is yielded by the
      // `WarpOp`, we keep track the distributed type for this result.
      if (!isa<VectorType>(forResult.getType()))
        continue;
      VectorType distType = cast<VectorType>(
          warpOp.getResult(yieldOperand.getOperandNumber()).getType());
      forResultDistTypes[forResultNumber] = distType;
    }

    // Newly created `WarpOp` will yield values in following order:
    // 1. All init args of the `ForOp`.
    // 2. All escaping values.
    // 3. All non-`ForOp` yielded values.
    SmallVector<Value> newWarpOpYieldValues;
    SmallVector<Type> newWarpOpDistTypes;
    for (auto [i, initArg] : llvm::enumerate(forOp.getInitArgs())) {
      newWarpOpYieldValues.push_back(initArg);
      // Compute the distributed type for this init arg.
      Type distType = initArg.getType();
      if (auto vecType = dyn_cast<VectorType>(distType)) {
        // If the `ForOp` result corresponds to this init arg is already yielded
        // we can get the distributed type from `forResultDistTypes` map.
        // Otherwise, we compute it using distributionMapFn.
        AffineMap map = distributionMapFn(initArg);
        distType = forResultDistTypes.count(i)
                       ? forResultDistTypes[i]
                       : getDistributedType(vecType, map, warpOp.getWarpSize());
      }
      newWarpOpDistTypes.push_back(distType);
    }
    // Insert escaping values and their distributed types.
    newWarpOpYieldValues.insert(newWarpOpYieldValues.end(),
                                escapingValues.begin(), escapingValues.end());
    newWarpOpDistTypes.insert(newWarpOpDistTypes.end(),
                              escapingValueDistTypes.begin(),
                              escapingValueDistTypes.end());
    // Next, we insert all non-`ForOp` yielded values and their distributed
    // types. We also create a mapping between the non-`ForOp` yielded value
    // index and the corresponding new `WarpOp` yield value index (needed to
    // update users later).
    llvm::SmallDenseMap<unsigned, unsigned> nonForResultMapping;
    for (auto [i, v] :
         llvm::zip_equal(nonForResultIndices, nonForYieldedValues)) {
      nonForResultMapping[i] = newWarpOpYieldValues.size();
      newWarpOpYieldValues.push_back(v);
      newWarpOpDistTypes.push_back(warpOp.getResult(i).getType());
    }
    // Create the new `WarpOp` with the updated yield values and types.
    WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
        rewriter, warpOp, newWarpOpYieldValues, newWarpOpDistTypes);

    // Next, we create a new `ForOp` with the init args yielded by the new
    // `WarpOp`.
    const unsigned escapingValuesStartIdx =
        forOp.getInitArgs().size(); // `ForOp` init args are positioned before
                                    // escaping values in the new `WarpOp`.
    SmallVector<Value> newForOpOperands;
    for (size_t i = 0; i < escapingValuesStartIdx; ++i)
      newForOpOperands.push_back(newWarpOp.getResult(i));

    // Create a new `ForOp` outside the new `WarpOp` region.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newForOp = scf::ForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newForOpOperands, /*bodyBuilder=*/nullptr,
        forOp.getUnsignedCmp());
    // Next, we insert a new `WarpOp` (called inner `WarpOp`) inside the
    // newly created `ForOp`. This `WarpOp` will contain all ops that were
    // contained within the original `ForOp` body.
    rewriter.setInsertionPointToStart(newForOp.getBody());

    SmallVector<Value> innerWarpInput(newForOp.getRegionIterArgs().begin(),
                                      newForOp.getRegionIterArgs().end());
    SmallVector<Type> innerWarpInputType(forOp.getResultTypes().begin(),
                                         forOp.getResultTypes().end());
    // Escaping values are forwarded to the inner `WarpOp` as its (additional)
    // arguments. We keep track of the mapping between these values and their
    // argument index in the inner `WarpOp` (to replace users later).
    llvm::SmallDenseMap<Value, int64_t> argIndexMapping;
    for (size_t i = escapingValuesStartIdx;
         i < escapingValuesStartIdx + escapingValues.size(); ++i) {
      innerWarpInput.push_back(newWarpOp.getResult(i));
      argIndexMapping[escapingValues[i - escapingValuesStartIdx]] =
          innerWarpInputType.size();
      innerWarpInputType.push_back(
          escapingValueInputTypes[i - escapingValuesStartIdx]);
    }
    // Create the inner `WarpOp` with the new input values and types.
    auto innerWarp = WarpExecuteOnLane0Op::create(
        rewriter, newWarpOp.getLoc(), newForOp.getResultTypes(),
        newWarpOp.getLaneid(), newWarpOp.getWarpSize(), innerWarpInput,
        innerWarpInputType);

    // Inline the `ForOp` body into the inner `WarpOp` body.
    SmallVector<Value> argMapping;
    argMapping.push_back(newForOp.getInductionVar());
    for (Value args : innerWarp.getBody()->getArguments())
      argMapping.push_back(args);

    argMapping.resize(forOp.getBody()->getNumArguments());
    SmallVector<Value> yieldOperands;
    for (Value operand : forOp.getBody()->getTerminator()->getOperands())
      yieldOperands.push_back(operand);

    rewriter.eraseOp(forOp.getBody()->getTerminator());
    rewriter.mergeBlocks(forOp.getBody(), innerWarp.getBody(), argMapping);

    // Insert a gpu `YieldOp` at the end of the inner `WarpOp` body that yields
    // original `ForOp` results.
    rewriter.setInsertionPointToEnd(innerWarp.getBody());
    gpu::YieldOp::create(rewriter, innerWarp.getLoc(), yieldOperands);
    rewriter.setInsertionPointAfter(innerWarp);
    // Insert a scf.yield op at the end of the new `ForOp` body that yields
    // the inner `WarpOp` results.
    if (!innerWarp.getResults().empty())
      scf::YieldOp::create(rewriter, forOp.getLoc(), innerWarp.getResults());

    // Update the users of original `WarpOp` results that were coming from the
    // original `ForOp` to the corresponding new `ForOp` result.
    for (auto [origIdx, newIdx] : forResultMapping)
      rewriter.replaceAllUsesExcept(warpOp.getResult(origIdx),
                                    newForOp.getResult(newIdx), newForOp);
    // Similarly, update any users of the `WarpOp` results that were not
    // results of the `ForOp`.
    for (auto [origIdx, newIdx] : nonForResultMapping)
      rewriter.replaceAllUsesWith(warpOp.getResult(origIdx),
                                  newWarpOp.getResult(newIdx));
    // Remove the original `WarpOp` and `ForOp`, they should not have any uses
    // at this point.
    rewriter.eraseOp(forOp);
    rewriter.eraseOp(warpOp);
    // Update any users of escaping values that were forwarded to the
    // inner `WarpOp`. These values are now arguments of the inner `WarpOp`.
    newForOp.walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        auto it = argIndexMapping.find(operand.get());
        if (it == argIndexMapping.end())
          continue;
        operand.set(innerWarp.getBodyRegion().getArgument(it->second));
      }
    });

    // Finally, hoist out any now uniform code from the inner `WarpOp`.
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
  patterns
      .add<WarpOpElementwise, WarpOpDeadResult, WarpOpBroadcast,
           WarpOpShapeCast, WarpOpExtract, WarpOpForwardOperand, WarpOpConstant,
           WarpOpInsertScalar, WarpOpInsert, WarpOpCreateMask,
           WarpOpExtractStridedSlice, WarpOpInsertStridedSlice, WarpOpStep>(
          patterns.getContext(), benefit);
  patterns.add<WarpOpExtractScalar>(patterns.getContext(), warpShuffleFromIdxFn,
                                    benefit);
  patterns.add<WarpOpScfForOp>(patterns.getContext(), distributionMapFn,
                               benefit);
  patterns.add<WarpOpScfIfOp>(patterns.getContext(), distributionMapFn,
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
