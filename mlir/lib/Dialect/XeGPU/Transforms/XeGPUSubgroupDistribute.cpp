//=- XeGPUSubgroupDistribute.cpp - ditribute XeGPU ops to work items *-C++-*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "xegpu-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
bool divisible(APInt lhs, APInt rhs) { return !lhs.urem(rhs); }

/// Clone a create_nd_tdesc feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op. The warp op will
/// still contain the original op that will not be used by the yield op (and
/// should be cleaned up later with dce). The yield op will bypass the
/// create_nd_tdesc's arguments. Tensor descriptor is not distributed because it
/// is a uniform value accorss all work items within the subgroup.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32>) {
///     ...
///     %td = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %td
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %arg0, %dead
///   }
///   %td = xegpu.create_nd_tdesc %r#0[0, 0]: memref<4x8xf32>
///                                 -> !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct SubgroupOpTensorDescOp final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override;
};

/// Sink a store_nd op at the end of enclosing `gpu.warp_execute_on_lane_0`. In
/// case arguments for the store are passed through the warp op interface they
/// would be propagated as returned values. Only the source vector for the store
/// is distributed according to sg_map attribute.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     xegpu.store_nd %arg0, %arg1: vector<4x8xf32>,
///                                 !xegpu.tensor_desc<4x8xf32>
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     gpu.yield %arg0, %arg1: vector<4x8xf32>, !xegpu.tensor_desc<4x8xf32>
///   }
///   xegpu.store_nd %r#0, %r#1: vector<4x1xf32>,
///     !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct SubgroupOpStoreNd final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override;
};

/// Clone a load_nd feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by
/// the yield op (and should be cleaned up later with dce). The yield op will
/// bypass the load's arguments. Only the loaded vector is distributed according
/// to sg_map attribute and, tensor descriptor types is not distributed.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (vector<4x1xf32>) {
///     ...
///     %ld = xegpu.load_nd %arg0, %arg1: !xegpu.tensor_desc<4x8xf32> ->
///       vector<4x8xf32>
///     gpu.yield %ld
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.load_nd %arg0: !xegpu.tensor_desc<4x8xf32> ->
///     vector<4x8xf32> gpu.yield %arg0, %arg1
///   }
///   %ld = xegpu.load_nd %r#0: !xegpu.tensor_desc<4x8xf32> -> vector<4x1xf32>
///
/// ```
struct SubgroupOpLoadNd final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override;
};

/// Returns the distributed vector type for a source vector type according to
/// the sg_map attribute.
FailureOr<VectorType> getDistributedVectorType(VectorType originalT,
                                               xegpu::SGMapAttr sgMap) {
  llvm::SmallVector<int64_t, 2> distributedShape;
  auto layout = sgMap.getWiLayout();
  auto shape = originalT.getShape();
  for (const auto [l, o] : llvm::zip_equal(layout, shape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return failure();
    distributedShape.push_back(o / l);
  }
  auto newVectorType =
      VectorType::get(distributedShape, originalT.getElementType(),
                      originalT.getScalableDims());
  return newVectorType;
}

// Returns the distributed tensor descriptor type for a source tensor descriptor
// type according to the sg_map attribute. Note that tensor descriptor type is
// distributed only for the scattered case. For XeGPU ND operaions
// (create_nd_tdesc, load_nd, store_nd), tensor descriptor is considered uniform
// across all work items within the subgroup and therefore is not distributed.
FailureOr<xegpu::TensorDescType>
getDistributedTensorDescType(xegpu::TensorDescType originalT,
                             xegpu::SGMapAttr sgMap,
                             xegpu::MemorySpace memSpace) {
  llvm::SmallVector<int64_t, 2> distributedShape;
  auto layout = sgMap.getWiLayout();
  auto shape = originalT.getShape();
  for (const auto [l, o] : llvm::zip_equal(layout, shape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return failure();
    // Tensor descriptor is distributed only for the scattered case.
    if (originalT.isScattered())
      distributedShape.push_back(o / l);
    else
      distributedShape.push_back(o);
  }

  return xegpu::TensorDescType::get(
      originalT.getContext(), distributedShape, originalT.getElementType(),
      originalT.getEncoding(), originalT.getSGMapAttr());
}
} // namespace

LogicalResult
SubgroupOpStoreNd::matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                   PatternRewriter &rewriter) const {
  auto yield = cast<gpu::YieldOp>(
      subgroupOp.getBodyRegion().getBlocks().begin()->getTerminator());
  Operation *lastNode = yield->getPrevNode();
  auto storeOp = dyn_cast_or_null<xegpu::StoreNdOp>(lastNode);
  if (!storeOp)
    return failure();

  auto origType = storeOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        storeOp, "the source tensor descriptor lacks sg_map attribute");

  if (storeOp.getTensorDescType().getShape().size() != 2)
    return rewriter.notifyMatchFailure(storeOp, "unsupported shape");

  auto distributedTypeOrFailure =
      getDistributedVectorType(storeOp.getValueType(), sgMap);
  if (failed(distributedTypeOrFailure))
    return rewriter.notifyMatchFailure(storeOp,
                                       "Failed to distribute the type");
  VectorType newVectorType = distributedTypeOrFailure.value();

  auto distributedDescTypeOrFailure = getDistributedTensorDescType(
      storeOp.getTensorDescType(), sgMap,
      storeOp.getTensorDescType().getMemorySpace());
  if (failed(distributedDescTypeOrFailure))
    return rewriter.notifyMatchFailure(storeOp,
                                       "Failed to distribute the desc type");
  xegpu::TensorDescType newTDescType = distributedDescTypeOrFailure.value();

  SmallVector<size_t> newRetIndices;
  gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
      rewriter, subgroupOp,
      ValueRange{storeOp.getTensorDesc(), storeOp.getValue()},
      TypeRange{newTDescType, newVectorType}, newRetIndices);

  rewriter.setInsertionPointAfter(newWarpOp);
  auto newStoreOp =
      cast<xegpu::StoreNdOp>(rewriter.clone(*storeOp.getOperation()));
  rewriter.eraseOp(storeOp);
  newStoreOp.getTensorDescMutable().assign(
      newWarpOp.getResult(newRetIndices[0]));
  newStoreOp.getValueMutable().assign(newWarpOp.getResult(newRetIndices[1]));

  return success();
}

LogicalResult
SubgroupOpLoadNd::matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                  PatternRewriter &rewriter) const {
  OpOperand *operand = getWarpResult(subgroupOp, [](Operation *op) {
    return isa<xegpu::LoadNdOp>(op) && op->hasOneUse();
  });

  if (!operand)
    return rewriter.notifyMatchFailure(subgroupOp,
                                       "warp result is not a xegpu::LoadNd op");

  auto loadOp = operand->get().getDefiningOp<xegpu::LoadNdOp>();

  if (loadOp.getPacked())
    return rewriter.notifyMatchFailure(
        loadOp, "Packed load distribution not supported");

  xegpu::TensorDescType origType = loadOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        loadOp, "the source tensor descriptor lacks sg_map attribute");

  auto origShape = origType.getShape();
  if (origShape.size() != 2)
    return rewriter.notifyMatchFailure(loadOp, "unsupported shape");

  auto distributedTypeOrFailure =
      getDistributedVectorType(loadOp.getType(), sgMap);
  if (failed(distributedTypeOrFailure))
    return rewriter.notifyMatchFailure(loadOp, "Failed to distribute the type");
  VectorType newVectorType = distributedTypeOrFailure.value();

  auto distributedDescTypeOrFailure =
      getDistributedTensorDescType(loadOp.getTensorDescType(), sgMap,
                                   loadOp.getTensorDescType().getMemorySpace());
  if (failed(distributedDescTypeOrFailure))
    return rewriter.notifyMatchFailure(loadOp,
                                       "Failed to distribute the desc type");
  xegpu::TensorDescType newTDescType = distributedDescTypeOrFailure.value();

  unsigned operandIdx = operand->getOperandNumber();

  SmallVector<size_t> newRetIndices;
  gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
      rewriter, subgroupOp, loadOp.getTensorDesc(), TypeRange{newTDescType},
      newRetIndices);

  rewriter.setInsertionPointAfter(newWarpOp);

  auto newLoadOp = rewriter.create<xegpu::LoadNdOp>(
      loadOp.getLoc(), newVectorType, loadOp.getTensorDesc(),
      loadOp.getPackedAttr(), loadOp.getTransposeAttr(), loadOp.getL1HintAttr(),
      loadOp.getL2HintAttr(), loadOp.getL3HintAttr());

  newLoadOp.getTensorDescMutable().assign(
      newWarpOp.getResult(newRetIndices[0]));
  Value distributedVal = newWarpOp.getResult(operandIdx);
  rewriter.replaceAllUsesWith(distributedVal, newLoadOp);

  return success();
}

LogicalResult
SubgroupOpTensorDescOp::matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                        PatternRewriter &rewriter) const {
  OpOperand *operand = getWarpResult(subgroupOp, [](Operation *op) {
    return isa<xegpu::CreateNdDescOp>(op) && op->hasOneUse();
  });

  if (!operand)
    return rewriter.notifyMatchFailure(
        subgroupOp, "warp result is not a xegpu::CreateNdDesc op");
  auto descOp = operand->get().getDefiningOp<xegpu::CreateNdDescOp>();
  assert(descOp && "desc op must be not null");
  unsigned operandIdx = operand->getOperandNumber();

  // TODO: is memref uniform in the region
  rewriter.setInsertionPoint(subgroupOp);
  auto srcTypedVal = dyn_cast<TypedValue<MemRefType>>(descOp.getSource());
  assert(srcTypedVal && "source value must be not null");

  auto descOffsets = descOp.getMixedOffsets();
  if (descOffsets.size() != 2)
    return rewriter.notifyMatchFailure(descOp,
                                       "offsets size is expected to be 2");

  xegpu::SGMapAttr sgMap = descOp.getType().getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        descOp, "the tensor descriptor lacks sg_map attribute");

  auto distributedDescTypeOrFailure = getDistributedTensorDescType(
      descOp.getType(), sgMap, descOp.getType().getMemorySpace());
  if (failed(distributedDescTypeOrFailure))
    return rewriter.notifyMatchFailure(descOp,
                                       "Failed to distribute the desc type");
  xegpu::TensorDescType newTDescType = distributedDescTypeOrFailure.value();
  auto distributedShape = newTDescType.getShape();
  // use the base memref strides
  SmallVector<OpFoldResult> overwriteStrides =
      getAsIndexOpFoldResult(rewriter.getContext(), SmallVector<int64_t>{1, 1});
  SmallVector<OpFoldResult> overwriteSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), distributedShape);

  SmallVector<size_t> newRetIndices;
  gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
      rewriter, subgroupOp, descOp.getSource(), descOp.getSourceType(),
      newRetIndices);

  rewriter.setInsertionPointAfter(newWarpOp);
  auto newDescOp = rewriter.create<xegpu::CreateNdDescOp>(
      newWarpOp.getLoc(), newTDescType,
      dyn_cast<TypedValue<MemRefType>>(newWarpOp.getResult(newRetIndices[0])),
      descOffsets);

  Value distributedVal = newWarpOp.getResult(operandIdx);
  rewriter.replaceAllUsesWith(distributedVal, newDescOp);

  return success();
}

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {
  patterns.add<SubgroupOpTensorDescOp>(patterns.getContext());
  patterns.add<SubgroupOpStoreNd>(patterns.getContext());
  patterns.add<SubgroupOpLoadNd>(patterns.getContext());
}
