//===- XeGPUSubgroupDistribute.cpp - XeGPU Subgroup Distribute Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/IR/XeGPUTargetInfo.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSUBGROUPDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-subgroup-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

static const char *const resolveSIMTTypeMismatch =
    "resolve_simt_type_mismatch"; // Attribute name for identifying
                                  // UnrelizedConversionCastOp added to resolve
                                  // SIMT type mismatches.

namespace {

//===----------------------------------------------------------------------===//
// SIMT Distribution Patterns
//===----------------------------------------------------------------------===//

/// Helper function to get  distributed vector type for a source vector type
/// according to the lane_layout. We simply divide each dimension of tensor
/// descriptor shape by corresponding lane_layout dimension. If
/// array_length > 1, that is appended to the front of the ditributed shape.
/// NOTE: This is the vector type that will be returned by the
/// gpu.warp_execute_on_lane0 op.
///
/// Examples:
/// | original vector shape | lane_layout | distributed vector shape |
/// |-----------------------|-------------|--------------------------|
/// | 32x16                 | [1, 16]     | 32x1                     |
/// | 32x16                 | [2, 8]      | 16x2                     |
/// | 2x32x16               | [1, 16]     | 2x32x1                   |
static FailureOr<VectorType>
getDistVecTypeBasedOnLaneLayout(xegpu::LayoutAttr layout,
                                VectorType originalType) {
  if (!layout)
    return failure();

  auto laneLayout = layout.getLaneLayout().asArrayRef();
  assert(originalType.getShape().size() >= laneLayout.size() &&
         "Rank of the original vector type should be greater or equal to the "
         "size of the lane layout to distribute the vector type.");
  SmallVector<int64_t> distributedShape(originalType.getShape());
  // Only distribute the last `laneLayout.size()` dimensions. The remaining
  // dimensions are not distributed.
  unsigned distributionStart = originalType.getRank() - laneLayout.size();
  for (auto [i, dim] : llvm::enumerate(originalType.getShape())) {
    if (i < distributionStart)
      continue;

    // Check if the dimension can be distributed evenly.
    if (dim % laneLayout[i - distributionStart] != 0)
      return failure();
    distributedShape[i] = dim / laneLayout[i - distributionStart];
  }
  return VectorType::get(distributedShape, originalType.getElementType());
}

/// Helper function to resolve types if the distributed type out of
/// gpu.warp_execute_on_lane0 is different from the expected xegpu SIMT type.
/// Example 1:
///   distributed type: vector<8x1xf32>
///   expected type: vector<8xf32>
///   resolved using,
///   %0 = vector.shape_cast %1 : vector<8x1xf32> to vector<8xf32>
/// Example 2:
///   distributed type: xegpu.tensor_desc<8x16xf32, #xegpu.layout<...>>
///   expected type: xegpu.tensor_desc<8x16xf32>
///   resolved using,
///   %0 = unrealized_conversion_cast %1 :
///      xegpu.tensor_desc<8x16xf32, #xegpu.layout<..>> ->
///      xegpu.tensor_desc<8x16xf32>
template <typename T>
static Value resolveDistributedTy(Value orig, T expected,
                                  PatternRewriter &rewriter) {
  // If orig and expected types are the same, return orig.
  if (orig.getType() == expected)
    return orig;
  // If orig is a vector type, create a shape cast op to reconcile the types.
  if (isa<VectorType>(orig.getType())) {
    auto castOp =
        vector::ShapeCastOp::create(rewriter, orig.getLoc(), expected, orig);
    return castOp.getResult();
  }
  // If orig is a tensor descriptor type, create an unrealized conversion cast
  // op to reconcile the types.
  if (isa<xegpu::TensorDescType>(orig.getType())) {
    auto castOp = UnrealizedConversionCastOp::create(rewriter, orig.getLoc(),
                                                     expected, orig);
    castOp->setAttr(resolveSIMTTypeMismatch, rewriter.getUnitAttr());
    return castOp.getResult(0);
  }
  llvm_unreachable("Unsupported type for reconciliation");
  return orig;
}

/// Helper function to check if the layout is packed. Layout is packed if it is
/// 2D and lane_data[0] != 1 (data packed from col dimension).
static bool hasPackedLayout(xegpu::LayoutAttr layout) {
  if (layout == xegpu::LayoutAttr())
    return false;
  DenseI32ArrayAttr laneData = layout.getLaneData();
  if (!laneData || laneData.size() != 2)
    return false;
  return laneData.asArrayRef()[0] != 1;
}

/// Given a GPUFuncOp, this pattern creates a new GPUFuncOp and moves the body
/// of the original GPUFuncOp to the new GPUFuncOp such that entire body is
/// contained within a WarpExecuteOnLane0Op.
/// Example:
///
/// ```
///   gpu.func @foo(%arg0: memref<*xf16>) -> vector<8x16xf32> {
///     ...
///     ...
///     gpu.return %result: vector<8x16xf32>
///   }
/// ```
/// To
/// ```
///   gpu.func @foo(%arg0: memref<*xf16>) -> vector<8x16xf32> {
///     %laneid = gpu.lane_id : index
///     %0 = gpu.warp_execute_on_lane_0(%laneid) -> vector<8x16xf32> {
///       ...
///       ...
///       gpu.yield %result: vector<8x16xf32>
///     }
///     return %0
///   }
struct MoveFuncBodyToWarpExecuteOnLane0
    : public OpRewritePattern<gpu::GPUFuncOp> {
  using OpRewritePattern<gpu::GPUFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::GPUFuncOp gpuFuncOp,
                                PatternRewriter &rewriter) const override {
    // If the function only contains a single void return, skip.
    if (llvm::all_of(gpuFuncOp.getBody().getOps(), [](Operation &op) {
          return isa<gpu::ReturnOp>(op) && !op.getNumOperands();
        }))
      return failure();
    // If the function already moved inside a warp_execute_on_lane0, skip.
    if (llvm::any_of(gpuFuncOp.getBody().getOps(), [](Operation &op) {
          return isa<gpu::WarpExecuteOnLane0Op>(op);
        }))
      return failure();
    // Create a new function with the same signature and same attributes.
    SmallVector<Type> workgroupAttributionsTypes =
        llvm::map_to_vector(gpuFuncOp.getWorkgroupAttributions(),
                            [](BlockArgument arg) { return arg.getType(); });
    SmallVector<Type> privateAttributionsTypes =
        llvm::map_to_vector(gpuFuncOp.getPrivateAttributions(),
                            [](BlockArgument arg) { return arg.getType(); });
    auto newGpuFunc = gpu::GPUFuncOp::create(
        rewriter, gpuFuncOp.getLoc(), gpuFuncOp.getName(),
        gpuFuncOp.getFunctionType(), workgroupAttributionsTypes,
        privateAttributionsTypes);
    newGpuFunc->setAttrs(gpuFuncOp->getAttrs());
    // Create a WarpExecuteOnLane0Op with same arguments and results as the
    // original gpuFuncOp.
    rewriter.setInsertionPointToEnd(&newGpuFunc.getFunctionBody().front());
    auto laneId = gpu::LaneIdOp::create(
        rewriter, newGpuFunc.getLoc(), rewriter.getIndexType(),
        /** upperBound = **/ mlir::IntegerAttr());
    ArrayRef<Type> gpuFuncResultType = gpuFuncOp.getFunctionType().getResults();
    auto warpOp = gpu::WarpExecuteOnLane0Op::create(
        rewriter, laneId.getLoc(), gpuFuncResultType, laneId,
        xegpu::targetinfo::subgroupSize, newGpuFunc.getArguments(),
        newGpuFunc.getArgumentTypes());
    Block &warpBodyBlock = warpOp.getBodyRegion().front();
    // Replace the ReturnOp of the original gpu function with a YieldOp.
    auto origRetunOp =
        cast<gpu::ReturnOp>(gpuFuncOp.getBlocks().back().getTerminator());
    rewriter.setInsertionPointAfter(origRetunOp);
    gpu::YieldOp::create(rewriter, origRetunOp.getLoc(),
                         origRetunOp.getOperands());
    rewriter.eraseOp(origRetunOp);
    // Move the original function body to the WarpExecuteOnLane0Op body.
    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), warpOp.getBodyRegion(),
                                warpOp.getBodyRegion().begin());
    rewriter.eraseBlock(&warpBodyBlock);
    // Insert a new ReturnOp after the WarpExecuteOnLane0Op.
    rewriter.setInsertionPointAfter(warpOp);
    gpu::ReturnOp::create(rewriter, newGpuFunc.getLoc(), warpOp.getResults());
    rewriter.replaceOp(gpuFuncOp, newGpuFunc);
    return success();
  }
};

/// Distribute a create_nd_tdesc feeding into vector.yield op of the enclosing
/// `gpu.warp_execute_on_lane_0` region. After the sinking, the warp op will
/// still contain the original op that will not be used by the yield op (and
/// should be cleaned up later). The yield op will bypass the create_nd_tdesc's
/// arguments. Tensor descriptor shape is not distributed because it is a
/// uniform value across all work items within the subgroup. However, the
/// layout information is dropped in the new tensor descriptor type.
///
/// Example:
///
/// ```
///   #layout0 = #xegpu.layout<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32, #layout0>) {
///     ...
///     %td = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32, #layout0>
///     vector.yield %td
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> (...) {
///     ...
///     %dead = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32, #layout0>
///     vector.yield %arg0, %dead
///   }
///   %td = xegpu.create_nd_tdesc %r#0[0, 0]: memref<4x8xf32>
///                                 -> !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct CreateNdDescDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<xegpu::CreateNdDescOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a xegpu::CreateNdDesc op");
    auto descOp = operand->get().getDefiningOp<xegpu::CreateNdDescOp>();
    unsigned operandIdx = operand->getOperandNumber();

    xegpu::LayoutAttr layout = descOp.getType().getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          descOp, "the tensor descriptor lacks layout attribute");

    SmallVector<size_t> newRetIndices;
    rewriter.setInsertionPoint(warpOp);
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, /* new yieled values = */ descOp->getOperands(),
        /* new yielded types = */ descOp.getOperandTypes(), newRetIndices);

    SmallVector<Value> newDescOperands = llvm::map_to_vector(
        newRetIndices, [&](size_t i) { return newWarpOp.getResult(i); });
    rewriter.setInsertionPointAfter(newWarpOp);
    xegpu::TensorDescType distributedTensorDescTy =
        descOp.getType().dropLayouts(); // Distributed tensor descriptor type
                                        // does not contain layout info.
    Value newDescOp = xegpu::CreateNdDescOp::create(
        rewriter, newWarpOp.getLoc(), distributedTensorDescTy, newDescOperands,
        descOp->getAttrs());

    Value distributedVal = newWarpOp.getResult(operandIdx);
    // Resolve the distributed type to the expected type.
    newDescOp =
        resolveDistributedTy(newDescOp, distributedVal.getType(), rewriter);
    rewriter.replaceAllUsesWith(distributedVal, newDescOp);
    return success();
  }
};

/// Distribute a store_nd op at the end of enclosing
/// `gpu.warp_execute_on_lane_0`. In case arguments for the store are passed
/// through the warp op interface they would be propagated as returned values.
/// Source vector is distributed based on lane layout. Appropriate cast ops are
/// inserted if the distributed types does not match expected xegpu SIMT types.
///
/// Example:
///
/// ```
///   #layout0 = #xegpu.layout<wi_layout = [1, 8], wi_data = [1, 1]>
///   gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     xegpu.store_nd %arg0, %arg1: vector<4x8xf32>,
///                                 !xegpu.tensor_desc<4x8xf32, #layout0>
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> (vector<4x1xf32>,
///   !xegpu.tensor_desc<4x8xf32, #layout0>) {
///     gpu.yield %arg0, %arg1: vector<4x8xf32>, !xegpu.tensor_desc<4x8xf32,
///     #layout0>
///   }
///   %0 = vector.shape_cast %r#0: vector<4x1xf32> to vector<4xf32>
///   %1 = unrealized_conversion_cast %r#1: !xegpu.tensor_desc<4x8xf32,
///   #layout0>
///     -> !xegpu.tensor_desc<4x8xf32>
///   xegpu.store_nd %0, %1: vector<4xf32>,
///     !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct StoreNdDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    gpu::YieldOp yield = warpOp.getTerminator();
    Operation *lastNode = yield->getPrevNode();
    auto storeOp = dyn_cast_or_null<xegpu::StoreNdOp>(lastNode);
    if (!storeOp)
      return failure();

    int64_t offsetSize = static_cast<int64_t>(storeOp.getOffsets().size());
    if ((offsetSize != 0) || storeOp.getConstOffsetsAttr())
      return failure();

    xegpu::TensorDescType tensorDescTy = storeOp.getTensorDescType();
    xegpu::LayoutAttr layout = tensorDescTy.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          storeOp, "the source tensor descriptor lacks layout attribute");

    FailureOr<VectorType> distributedTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, storeOp.getValueType());
    if (failed(distributedTypeByWarpOpOrFailure))
      return rewriter.notifyMatchFailure(storeOp,
                                         "Failed to distribute the type");
    VectorType distributedTypeByWarpOp =
        distributedTypeByWarpOpOrFailure.value();

    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp,
        /* new yielded values = */
        ValueRange{storeOp.getValue(), storeOp.getTensorDesc()},
        /* new yielded types = */
        TypeRange{distributedTypeByWarpOp, storeOp.getTensorDescType()},
        newRetIndices);
    // Create a new store op outside the warp op with the distributed vector
    // type. Tensor descriptor is not distributed.
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newStoreOperands;

    // For the value operand, there can be a mismatch between the vector type
    // distributed by the warp op and (xegpu-specific) distributed type
    // supported by the store op. Type mismatch must be resolved using
    // appropriate cast op.
    FailureOr<VectorType> storeNdDistributedValueTyOrFailure =
        xegpu::getDistributedVectorType(storeOp.getTensorDescType());
    if (failed(storeNdDistributedValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          storeOp, "Failed to get distributed vector type for the store op");
    newStoreOperands.push_back(resolveDistributedTy(
        newWarpOp.getResult(newRetIndices[0]),
        storeNdDistributedValueTyOrFailure.value(), rewriter));
    // For the tensor descriptor operand, the layout attribute is dropped after
    // distribution. Types needs to be resolved in this case also.
    xegpu::TensorDescType distributedTensorDescTy =
        storeOp.getTensorDescType().dropLayouts();
    newStoreOperands.push_back(
        resolveDistributedTy(newWarpOp.getResult(newRetIndices[1]),
                             distributedTensorDescTy, rewriter));

    auto newStoreOp =
        xegpu::StoreNdOp::create(rewriter, newWarpOp.getLoc(), TypeRange{},
                                 newStoreOperands, storeOp->getAttrs());
    xegpu::removeLayoutAttrs(newStoreOp);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Distribute a load_nd op feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by
/// the yield op (and should be cleaned up later). The yield op will
/// bypass the load's arguments. Only the loaded vector is distributed
/// according to lane layout and, tensor descriptor types is not
/// distributed. Appropriate cast ops are inserted if the distributed types does
/// not match expected xegpu SIMT types.
///
/// Example:
///
/// ```
///   #layout0 = #xegpu.layout<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (vector<4x1xf32>) {
///     ...
///     %ld = xegpu.load_nd %arg0, %arg1: !xegpu.tensor_desc<4x8xf32, #layout0>
///     ->
///       vector<4x8xf32>
///     gpu.yield %ld
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> (vector<4x1xf32>,
///   !xegpu.tensor_desc<4x8xf32, #layout0>) {
///     ...
///     %dead = xegpu.load_nd %arg0: !xegpu.tensor_desc<4x8xf32, #layout0> ->
///     vector<4x8xf32> gpu.yield %dead, %arg0
///   }
///   %0 = unrealized_conversion_cast %r#1: !xegpu.tensor_desc<4x8xf32,
///        #layout0> -> !xegpu.tensor_desc<4x8xf32>
///   %1 = xegpu.load_nd %0: !xegpu.tensor_desc<4x8xf32> -> vector<4xf32>
///   %2 = vector.shape_cast %r#0: vector<4xf32> to vector<4x1xf32>
///
/// ```
struct LoadNdDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, [&](Operation *op) {
      if (!isa<xegpu::LoadNdOp>(op))
        return false;
      // Make sure the same load op is the last operation in the warp op body.
      // This ensure that load op is not sinked earlier violating any barrier
      // synchronizations.
      gpu::YieldOp yield = warpOp.getTerminator();
      return yield->getPrevNode() == op;
    });

    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a xegpu::LoadNd op");

    auto loadOp = operand->get().getDefiningOp<xegpu::LoadNdOp>();

    int64_t offsetSize = static_cast<int64_t>(loadOp.getOffsets().size());
    if ((offsetSize != 0) || loadOp.getConstOffsetsAttr())
      return failure();

    xegpu::TensorDescType tensorDescTy = loadOp.getTensorDescType();
    xegpu::LayoutAttr layout = tensorDescTy.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          loadOp, "the source tensor descriptor lacks layout attribute");

    unsigned operandIdx = operand->getOperandNumber();
    VectorType distributedTypeByWarpOp =
        cast<VectorType>(warpOp.getResult(operandIdx).getType());

    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp,
        /* new yielded values = */ loadOp.getTensorDesc(),
        /* new yielded types = */ tensorDescTy, newRetIndices);

    // Create a new load op outside the warp op with the distributed vector
    // type.
    rewriter.setInsertionPointAfter(newWarpOp);
    FailureOr<VectorType> loadNdDistValueTyOrFailure =
        xegpu::getDistributedVectorType(loadOp.getTensorDescType());
    if (failed(loadNdDistValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          loadOp, "Failed to get distributed vector type for the load op");
    xegpu::TensorDescType distributedTensorDescTy =
        loadOp.getTensorDescType().dropLayouts(); // Distributed tensor
                                                  // descriptor type does not
                                                  // contain layout info.
    auto newLoadOp = xegpu::LoadNdOp::create(
        rewriter, newWarpOp.getLoc(), loadNdDistValueTyOrFailure.value(),
        resolveDistributedTy(newWarpOp->getResult(newRetIndices[0]),
                             distributedTensorDescTy, rewriter),
        loadOp->getAttrs());
    xegpu::removeLayoutAttrs(newLoadOp);
    // Set the packed attribute if the layout requires it.
    newLoadOp.setPacked(hasPackedLayout(layout));
    Value distributedVal = newWarpOp.getResult(operandIdx);
    // There can be a conflict between the vector type distributed by the
    // warp op and (xegpu-specific) distributed type supported by the load
    // op. Resolve these mismatches by inserting a cast.
    Value tyResolvedVal = resolveDistributedTy(
        newLoadOp.getResult(), distributedTypeByWarpOp, rewriter);
    rewriter.replaceAllUsesWith(distributedVal, tyResolvedVal);
    return success();
  }
};

/// Distribute a dpas op feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by
/// the yield op (and should be cleaned up later). The yield op will
/// bypass the dpas's arguments. Appropriate cast ops are inserted if the
/// distributed types does not match expected xegpu SIMT types.
/// Example:
/// ```
///   #lo_a = #xegpu.layout<wi_layout = [1, 16], wi_data = [1, 1]>
///   #lo_b = #xegpu.layout<wi_layout = [1, 16], wi_data = [2, 1]>
///   #lo_c = #xegpu.layout<wi_layout = [1, 16], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (vector<8x1xf32>) {
///     ...
///     %dpas = xegpu.dpas %arg0, %arg1: vector<8x16xf16>, vector<16x16xf16> ->
///       vector<8x16xf32>
///     gpu.yield %dpas
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> (vector<8x1xf32>,
///   vector<8x1xf16>, vector<16x1xf16>) {
///     ...
///     %dead = xegpu.dpas %arg0, %arg1: vector<8x16xf16>, vector<16x16xf16>
///       -> vector<8x16xf32>
///     gpu.yield %dead, %arg0, %arg1
///   }
///   %0 = vector.shape_cast %r#1: vector<8x1xf16> to vector<8xf16>
///   %1 = vector.shape_cast %r#2: vector<16x1xf16> to vector<16xf16>
///   %2 = xegpu.dpas %0, %1: vector<8xf16>, vector<16xf16> ->
///     vector<8xf32>
///   %dpas = vector.shape_cast %2: vector<8xf32> to vector<8x1xf32>
/// ```
struct DpasDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(warpOp, llvm::IsaPred<xegpu::DpasOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(warpOp,
                                         "warp result is not a xegpu::Dpas op");

    auto dpasOp = operand->get().getDefiningOp<xegpu::DpasOp>();
    unsigned operandIdx = operand->getOperandNumber();
    std::string layoutAName = xegpu::getLayoutName(dpasOp->getOpOperand(0));
    std::string layoutBName = xegpu::getLayoutName(dpasOp->getOpOperand(1));
    std::string layoutCName = xegpu::getLayoutName(dpasOp->getOpResult(0));

    xegpu::LayoutAttr layoutA =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutAName);
    xegpu::LayoutAttr layoutB =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutBName);
    xegpu::LayoutAttr layoutOut =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutCName);
    if (!layoutA || !layoutB || !layoutOut)
      return rewriter.notifyMatchFailure(
          dpasOp,
          "the xegpu::Dpas op lacks layout attribute for A, B or output");

    FailureOr<VectorType> distLhsTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutA, dpasOp.getLhsType());
    FailureOr<VectorType> distRhsTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutB, dpasOp.getRhsType());
    FailureOr<VectorType> distResultTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutOut, dpasOp.getResultType());
    if (failed(distLhsTypeByWarpOpOrFailure) ||
        failed(distRhsTypeByWarpOpOrFailure) ||
        failed(distResultTypeByWarpOpOrFailure))
      return rewriter.notifyMatchFailure(
          dpasOp,
          "Failed to distribute the A, B or output types in xegpu::Dpas op");

    llvm::SmallVector<Value, 3> newYieldValues{dpasOp.getLhs(),
                                               dpasOp.getRhs()};
    llvm::SmallVector<Type, 3> newYieldTypes{
        distLhsTypeByWarpOpOrFailure.value(),
        distRhsTypeByWarpOpOrFailure.value()};
    // Dpas acc operand is optional.
    if (dpasOp.getAcc()) {
      newYieldValues.push_back(dpasOp.getAcc());
      newYieldTypes.push_back(distResultTypeByWarpOpOrFailure.value());
    }
    // Create a new warp op without the dpas.
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, newYieldValues, newYieldTypes, newRetIndices);

    FailureOr<VectorType> expectedDistLhsTyOrFailure =
        xegpu::getDistributedVectorType(dpasOp.getLhsType(), layoutA);
    FailureOr<VectorType> expectedDistRhsTyOrFailure =
        xegpu::getDistributedVectorType(dpasOp.getRhsType(), layoutB);
    FailureOr<VectorType> expectedDistResultTyOrFailure =
        xegpu::getDistributedVectorType(dpasOp.getResultType(), layoutOut);
    if (failed(expectedDistLhsTyOrFailure) ||
        failed(expectedDistRhsTyOrFailure) ||
        failed(expectedDistResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          dpasOp,
          "Failed to get distributed vector type for the dpas operands.");
    // Create a new dpas op outside the warp op.
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newDpasOperands;
    SmallVector<VectorType> newDpasOperandExpectedTypes;

    // Resolve the distributed types with the original types.
    newDpasOperandExpectedTypes.push_back(expectedDistLhsTyOrFailure.value());
    newDpasOperandExpectedTypes.push_back(expectedDistRhsTyOrFailure.value());
    VectorType distributedResultTy = expectedDistResultTyOrFailure.value();
    if (dpasOp.getAcc())
      newDpasOperandExpectedTypes.push_back(distributedResultTy);

    for (unsigned i = 0; i < newRetIndices.size(); i++) {
      newDpasOperands.push_back(
          resolveDistributedTy(newWarpOp.getResult(newRetIndices[i]),
                               newDpasOperandExpectedTypes[i], rewriter));
    }
    auto newDpasOp = xegpu::DpasOp::create(rewriter, newWarpOp->getLoc(),
                                           distributedResultTy, newDpasOperands,
                                           dpasOp->getAttrs());
    xegpu::removeLayoutAttrs(newDpasOp);
    Value distributedVal = newWarpOp.getResult(operandIdx);
    // Resolve the output type.
    Value typeResolved =
        resolveDistributedTy(newDpasOp.getResult(),
                             distResultTypeByWarpOpOrFailure.value(), rewriter);
    rewriter.replaceAllUsesWith(distributedVal, typeResolved);
    return success();
  }
};

/// Sink an update_nd_offset op feeding into yield op of an enclosing
/// `gpu.warp_execute_on_lane_0` region. The warp op will still contain the
/// original op that will not be used by the yield op (and should be cleaned
/// up later). The yield op will bypass the updateOp's arguments. The tensor
/// descriptor type is not distributed. Appropriate cast ops are inserted if
/// the distributed types does not match expected xegpu SIMT types.
/// Example:
/// ```
///   #layout0 = #xegpu.layout<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32, #layout0>) {
///     ...
///     %update = xegpu.update_nd_offset %arg0, [%c32, %c16]:
///       !xegpu.tensor_desc<4x8xf32, #layout0>
///     gpu.yield %update
///   }
///   ...
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> (
///     !xegpu.tensor_desc<4x8xf32, #layout0>,
///     !xegpu.tensor_desc<4x8xf32, #layout0>, index, index) {
///     ...
///     %dead = xegpu.update_nd_offset %arg0, [%c32, %c16]:
///       !xegpu.tensor_desc<4x8xf32, #layout0> gpu.yield %dead, %arg0
///     gpu.yield %dead, %arg0, %c32, %c16
///   }
///   %0 = xegpu.unrealized_conversion_cast %r#1: !xegpu.tensor_desc<4x8xf32,
///        #layout0> -> !xegpu.tensor_desc<4x8xf32>
///   %1 = xegpu.update_nd_offset %0, [%r#2, %r#3]:
///     !xegpu.tensor_desc<4x8xf32>
///   ...
/// ```
struct UpdateNdOffsetDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<xegpu::UpdateNdOffsetOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a xegpu::UpdateNdOffset op");
    auto updateOp = operand->get().getDefiningOp<xegpu::UpdateNdOffsetOp>();
    unsigned operandIdx = operand->getOperandNumber();

    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, updateOp->getOperands(), updateOp.getOperandTypes(),
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    // new update op does not have layout attribute.
    xegpu::TensorDescType distributedTensorDescTy =
        updateOp.getTensorDescType().dropLayouts();
    SmallVector<Value> newUpdateOperands =
        llvm::map_to_vector(newRetIndices, [&](size_t i) {
          // For the tensor descriptor operand, the layout attribute is
          // dropped after distribution. Types needs to be resolved in this
          // case.
          if (isa<xegpu::TensorDescType>(newWarpOp.getResult(i).getType())) {
            return resolveDistributedTy(newWarpOp.getResult(i),
                                        distributedTensorDescTy, rewriter);
          }
          return newWarpOp.getResult(i);
        });
    // Create a new update op outside the warp op.
    auto newUpdateOp = xegpu::UpdateNdOffsetOp::create(
        rewriter, newWarpOp.getLoc(), distributedTensorDescTy,
        newUpdateOperands, updateOp->getAttrs());
    xegpu::removeLayoutAttrs(newUpdateOp);
    Value distributedVal = newWarpOp.getResult(operandIdx);
    // Resolve the distributed type with the original type.
    Value typeResolved = resolveDistributedTy(
        newUpdateOp.getResult(), distributedVal.getType(), rewriter);
    rewriter.replaceAllUsesWith(distributedVal, typeResolved);
    return success();
  }
};

/// Distribute a prefetch_nd op at the end of enclosing
/// `gpu.warp_execute_on_lane_0`. In case arguments for the prefetch are passed
/// through the warp op interface they would be propagated as returned values.
/// Tensor descriptor shape is not distributed because it is a uniform value
/// across all work items within the subgroup. Appropriate cast ops are inserted
/// if the distributed types does not match expected xegpu SIMT types.
///
/// Example:
///
/// ```
///   #layout0 = #xegpu.layout<wi_layout = [1, 8], wi_data = [1, 1]>
///   gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     xegpu.prefetch_nd %arg0 : !xegpu.tensor_desc<4x8xf32, #layout0>
///   }
/// ```
/// To
/// ```
///   %r:1 = gpu.warp_execute_on_lane_0(%laneid) -> (
///    !xegpu.tensor_desc<4x8xf32, #layout0>) {
///     gpu.yield %arg0: !xegpu.tensor_desc<4x8xf32, #layout0>
///   }
///   %1 = unrealized_conversion_cast %r#0: !xegpu.tensor_desc<4x8xf32,
///     #layout0> -> !xegpu.tensor_desc<4x8xf32>
///   xegpu.prefetch_nd %1 : !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct PrefetchNdDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    gpu::YieldOp yield = warpOp.getTerminator();
    Operation *lastNode = yield->getPrevNode();
    auto prefetchOp = dyn_cast_or_null<xegpu::PrefetchNdOp>(lastNode);
    if (!prefetchOp)
      return failure();

    int64_t offsetSize = static_cast<int64_t>(prefetchOp.getOffsets().size());
    if ((offsetSize != 0) || prefetchOp.getConstOffsetsAttr())
      return failure();

    xegpu::LayoutAttr layout = prefetchOp.getTensorDescType().getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          prefetchOp, "the source tensor descriptor lacks layout attribute");

    SmallVector<Value, 1> newYieldValues = {prefetchOp.getTensorDesc()};
    SmallVector<Type, 1> newYieldTypes = {prefetchOp.getTensorDescType()};
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, newYieldValues, newYieldTypes, newRetIndices);
    // Create a new prefetch op outside the warp op with updated tensor
    // descriptor type. Source tensor descriptor require type resolution.
    xegpu::TensorDescType newTensorDescTy =
        prefetchOp.getTensorDescType().dropLayouts();
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newPrefetchOperands = {resolveDistributedTy(
        newWarpOp.getResult(newRetIndices[0]), newTensorDescTy, rewriter)};
    xegpu::PrefetchNdOp::create(rewriter, newWarpOp.getLoc(), TypeRange{},
                                newPrefetchOperands, prefetchOp->getAttrs());
    xegpu::removeLayoutAttrs(prefetchOp);
    rewriter.eraseOp(prefetchOp);
    return success();
  }
};

/// Sink a gpu::BarrierOp at the end of enclosing `gpu.warp_execute_on_lane_0`
/// region. This will simply move the barrier op outside of the warp op.
struct GpuBarrierDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    gpu::YieldOp yield = warpOp.getTerminator();
    Operation *lastNode = yield->getPrevNode();
    // The last node must be a gpu::BarrierOp.
    auto barrierOp = dyn_cast_or_null<gpu::BarrierOp>(lastNode);
    if (!barrierOp)
      return failure();
    // Move the barrier op outside of the warp op.
    rewriter.setInsertionPointAfter(warpOp);
    gpu::BarrierOp::create(rewriter, barrierOp.getLoc(),
                           barrierOp->getResultTypes(),
                           barrierOp->getOperands(), barrierOp->getAttrs());
    rewriter.eraseOp(barrierOp);
    return success();
  }
};

} // namespace

namespace {
struct XeGPUSubgroupDistributePass final
    : public xegpu::impl::XeGPUSubgroupDistributeBase<
          XeGPUSubgroupDistributePass> {
  void runOnOperation() override;
};
} // namespace

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {
  patterns.add<CreateNdDescDistribution, StoreNdDistribution,
               LoadNdDistribution, DpasDistribution, PrefetchNdDistribution,
               UpdateNdOffsetDistribution, GpuBarrierDistribution>(
      patterns.getContext());
}

void XeGPUSubgroupDistributePass::runOnOperation() {
  // Step 1: Attach layouts to op operands.
  // TODO: Following assumptions are made:
  // 1) It is assumed that there are no layout conflicts.
  // 2) Any existing layout attributes attached to the operands are ignored.
  Operation *op = getOperation();
  op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Layouts are needed for vector type only.
      if (!isa<VectorType>(operand.get().getType()))
        continue;

      auto layout =
          xegpu::getDistributeLayoutAttrOfType<xegpu::LayoutAttr>(operand);
      if (!layout) {
        op->emitError("Could not find layout attribute for operand ")
            << operand.getOperandNumber() << " of operation " << op->getName();
        signalPassFailure();
        return;
      }
      xegpu::setDistributeLayoutAttr(operand, layout);
    }
  });
  // Step 2: Move all operations of a GPU function inside
  // gpu.warp_execute_on_lane_0 operation.
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveFuncBodyToWarpExecuteOnLane0>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // At this point, we have moved the entire function body inside the
    // warpOp. Now move any scalar uniform code outside of the warpOp (like
    // GPU index ops, scalar constants, etc.). This will simplify the
    // later lowering and avoid custom patterns for these ops.
    getOperation()->walk([&](Operation *op) {
      if (auto warpOp = dyn_cast<gpu::WarpExecuteOnLane0Op>(op))
        vector::moveScalarUniformCode(warpOp);
    });
  }
  // Step 3: Apply subgroup to workitem distribution patterns.
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUSubgroupDistributePatterns(patterns);
  // distributionFn is used by vector distribution patterns to determine the
  // distributed vector type for a given vector value. In XeGPU subgroup
  // distribution context, we compute this based on lane layout.
  auto distributionFn = [](Value val) {
    VectorType vecType = dyn_cast<VectorType>(val.getType());
    int64_t vecRank = vecType ? vecType.getRank() : 0;
    if (vecRank == 0)
      return AffineMap::get(val.getContext());
    // Get the layout of the vector type.
    // TODO: support more layout types
    auto layout = xegpu::getDistributeLayoutAttrOfType<xegpu::LayoutAttr>(val);
    // If no layout is specified, assume the inner most dimension is distributed
    // for now.
    if (!layout)
      return AffineMap::getMultiDimMapWithTargets(
          vecRank, {static_cast<unsigned int>(vecRank - 1)}, val.getContext());
    SmallVector<unsigned int> distributedDims;
    // Get the distributed dimensions based on the layout.
    ArrayRef<int> laneLayout = layout.getLaneLayout().asArrayRef();
    for (unsigned i = 0; i < laneLayout.size(); ++i) {
      if (laneLayout[i] > 1)
        distributedDims.push_back(i);
    }
    return AffineMap::getMultiDimMapWithTargets(vecRank, distributedDims,
                                                val.getContext());
  };
  // TODO: shuffleFn is not used.
  auto shuffleFn = [](Location loc, OpBuilder &builder, Value val, Value srcIdx,
                      int64_t warpSz) { return Value(); };
  vector::populatePropagateWarpVectorDistributionPatterns(
      patterns, distributionFn, shuffleFn);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Step 4: Finllay, clean up UnrealizedConversionCastOps that were inserted
  // due to tensor desc type mismatches created by using upstream distribution
  // patterns (scf.for)
  getOperation()->walk([&](mlir::UnrealizedConversionCastOp op) {
    // We are only interested in UnrealizedConversionCastOps there were added
    // for resolving SIMT type mismatches.
    if (!op->getAttr(resolveSIMTTypeMismatch))
      return WalkResult::skip();

    Value input = op.getOperand(0);
    Value output = op.getResult(0);

    // Both input and output must have tensor descriptor types.
    xegpu::TensorDescType inputDescType =
        mlir::dyn_cast<xegpu::TensorDescType>(input.getType());
    xegpu::TensorDescType outputDescType =
        mlir::dyn_cast<xegpu::TensorDescType>(output.getType());
    assert(inputDescType && outputDescType &&
           "Unrealized conversion cast must have tensor descriptor types");

    // tensor_desc<shape, layout> -> tensor_desc<shape> Type of conversions.
    // This occurs iside scf.for body to resolve the block argument type to
    // SIMT type.
    if (inputDescType.getLayout()) {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(input);
      if (argument) {
        argument.setType(output.getType());
        output.replaceAllUsesWith(argument);
        if (auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(
                argument.getOwner()->getParentOp())) {
          auto result = loopOp.getTiedLoopResult(argument);
          result.setType(output.getType());
        }
      }
    }

    // tensor_desc<shape> -> tensor_desc<shape, layout> Type of
    // conversions. This occurs at the yield op of scf.for body to go back
    // from SIMT type to original type.
    if (outputDescType.getLayout())
      output.replaceAllUsesWith(input);

    if (op->use_empty())
      op->erase();
    return WalkResult::advance();
  });
}
