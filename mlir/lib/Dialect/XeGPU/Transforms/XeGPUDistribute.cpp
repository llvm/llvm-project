//===- XeGPUDistribute.cpp - XeGPU ditribute ops to work items --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
bool divisible(APInt lhs, APInt rhs) { return !lhs.urem(rhs); }

/// Clone a create_nd_tdesc feeding into vector.yield op for the enclosing
/// `vector.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by the
/// yield op (and should be cleaned up later with dce). The yield op will bypass
/// the create_nd_tdesc's arguments.
/// The rewrite will create a subview of the size used by a single work item and
/// appropriate offset. The distributed create_nd_tdesc points into the subview
/// without offset. The tensor descriptor types is distributed according to
/// sg_map attribute.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = vector.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32>) {
///     ...
///     %td = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %td
///   }
/// ```
/// To
/// ```
///   %r:2 = vector.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %arg0, %dead
///   }
///   %view = memref.subview %r#0[0, %laneid] [4, 1] [1, 1]
///                               : memref<4x8xf32> to memref<4x1xf32>
///   %td = xegpu.create_nd_tdesc %view[0, 0]: memref<4x1xf32>
///                                 -> !xegpu.tensor_desc<4x1xf32>
///
/// ```
struct WarpOpTensorDescOp final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

/// Sink a store_nd feeding into vector.yield op for the enclosing
/// `vector.warp_execute_on_lane_0`. In case arguments for the store are passed
/// through the warp op interface they would be propagated as returned values.
/// Both the stored vector type and tensor descriptor types are distributed
/// according to sg_map attribute.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   vector.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     xegpu.store_nd %arg0, %arg1: vector<4x8xf32>,
///                                 !xegpu.tensor_desc<4x8xf32>
///     vector.yield
///   }
/// ```
/// To
/// ```
///   %r = vector.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     vector.yield
///   }
///   xegpu.store_nd %arg0, %arg1: vector<4x1xf32>, !xegpu.tensor_desc<4x1xf32>
///
/// ```
struct WarpOpStoreNd final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

/// Clone a load_nd feeding into vector.yield op for the enclosing
/// `vector.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by the
/// yield op (and should be cleaned up later with dce). The yield op will bypass
/// the load's arguments.
/// Both the loaded vector type and tensor descriptor types are distributed
/// according to sg_map attribute.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = vector.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32>) {
///     ...
///     %ld = xegpu.load_nd %arg0, %arg1: !xegpu.tensor_desc<4x8xf32>,
///     vector<4x8xf32> vector.yield %ld
///   }
/// ```
/// To
/// ```
///   %r:2 = vector.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.load_nd %arg0, %arg1:
///         !xegpu.tensor_desc<4x8xf32>, vector<4x8xf32>
///     vector.yield %arg0, %arg1
///   }
///   xegpu.store_nd %r#0, %r#1: vector<4x1xf32>, !xegpu.tensor_desc<4x1xf32>
///
/// ```
struct WarpOpLoadNd final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

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
    distributedShape.push_back(o / l);
  }
  xegpu::TensorDescType distributedDescType;
  if (originalT.isScattered()) {

    distributedDescType = xegpu::TensorDescType::get(
        distributedShape, originalT.getElementType(), originalT.getChunkSize(),
        originalT.getMemorySpace(), originalT.getSGMapAttr());
  } else {
    distributedDescType = xegpu::TensorDescType::get(
        distributedShape, originalT.getElementType(),
        originalT.getBoundaryCheck(), originalT.getArrayLength(),
        originalT.getMemorySpace(), originalT.getSGMapAttr());
  }
  return distributedDescType;
}
} // namespace

LogicalResult
WarpOpStoreNd::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                               PatternRewriter &rewriter) const {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
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
  DBGS() << "Matched store_nd: " << storeOp << "\n";

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
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp,
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

LogicalResult WarpOpLoadNd::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                            PatternRewriter &rewriter) const {
  OpOperand *operand = getWarpResult(warpOp, [](Operation *op) {
    return isa<xegpu::LoadNdOp>(op) && op->hasOneUse();
  });

  if (!operand)
    return rewriter.notifyMatchFailure(warpOp,
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
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, loadOp.getTensorDesc(), TypeRange{newTDescType},
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
WarpOpTensorDescOp::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                    PatternRewriter &rewriter) const {
  OpOperand *operand = getWarpResult(warpOp, [](Operation *op) {
    return isa<xegpu::CreateNdDescOp>(op) && op->hasOneUse();
  });

  if (!operand)
    return rewriter.notifyMatchFailure(
        warpOp, "warp result is not a xegpu::CreateNdDesc op");
  auto descOp = operand->get().getDefiningOp<xegpu::CreateNdDescOp>();
  assert(descOp && "desc op must be not null");
  unsigned operandIdx = operand->getOperandNumber();

  // TODO: is memref uniform in the region
  rewriter.setInsertionPoint(warpOp);
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

  auto layout = sgMap.getWiLayout();

  // Calculate the offset within tensor descriptor for the current lane_id. The
  // access to proper element for a work item is done through a lane-specific
  // subview (tdesc offsets are used as base, lane shift is added on top).
  auto laneid = warpOp.getLaneid();
  auto xDim =
      rewriter.create<arith::ConstantIndexOp>(laneid.getLoc(), layout[0]);
  auto shiftx = rewriter.create<arith::RemUIOp>(laneid.getLoc(), laneid, xDim);
  auto shifty = rewriter.create<arith::DivUIOp>(laneid.getLoc(), laneid, xDim);

  auto basex = getValueOrCreateConstantIndexOp(rewriter, laneid.getLoc(),
                                               descOffsets[0]);
  auto basey = getValueOrCreateConstantIndexOp(rewriter, laneid.getLoc(),
                                               descOffsets[1]);
  auto offsetx = rewriter.create<arith::AddIOp>(laneid.getLoc(), shiftx, basex);
  auto offsety = rewriter.create<arith::AddIOp>(laneid.getLoc(), shifty, basey);

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
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, descOp.getSource(), descOp.getSourceType(),
          newRetIndices);

  rewriter.setInsertionPointAfter(newWarpOp);
  auto subview = rewriter.create<memref::SubViewOp>(
      newWarpOp.getLoc(), srcTypedVal, getAsOpFoldResult({offsetx, offsety}),
      overwriteSizes, overwriteStrides);
  subview.getSourceMutable().assign(newWarpOp.getResult(newRetIndices[0]));

  auto zero = rewriter.create<arith::ConstantIndexOp>(laneid.getLoc(), 0);
  auto newDescOp = rewriter.create<xegpu::CreateNdDescOp>(
      newWarpOp.getLoc(), newTDescType, subview,
      getAsOpFoldResult({zero, zero}));

  Value distributedVal = newWarpOp.getResult(operandIdx);
  rewriter.replaceAllUsesWith(distributedVal, newDescOp);

  return success();
}

void xegpu::populateXeGPUDistributePatterns(RewritePatternSet &patterns) {
  patterns.add<WarpOpTensorDescOp>(patterns.getContext());
  patterns.add<WarpOpStoreNd>(patterns.getContext());
  patterns.add<WarpOpLoadNd>(patterns.getContext());
}
