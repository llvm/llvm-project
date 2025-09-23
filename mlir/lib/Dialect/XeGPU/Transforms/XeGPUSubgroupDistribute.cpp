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

/// In certain cases, we may need to favor XeGPU specific distribution patterns
/// over generic vector distribution patterns. In such cases, we can assign
/// priorities to patterns.
static constexpr unsigned regularPatternBenefit = 1;
static constexpr unsigned highPatternBenefit = 2;

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
getDistVecTypeBasedOnLaneLayout(xegpu::DistributeLayoutAttr layout,
                                VectorType originalType) {
  if (!layout)
    return failure();
  assert((isa<xegpu::LayoutAttr>(layout) || isa<xegpu::SliceAttr>(layout)) &&
         "Expecting a valid layout.");
  SmallVector<int64_t> effectiveLaneLayout =
      layout.getEffectiveLaneLayoutAsInt();
  assert(static_cast<size_t>(originalType.getRank()) >=
             effectiveLaneLayout.size() &&
         "Rank of the original vector type should be greater or equal to the "
         "size of the lane layout to distribute the vector type.");
  SmallVector<int64_t> distributedShape(originalType.getShape());
  // Only distribute the last `laneLayout.size()` dimensions. The remaining
  // dimensions are not distributed.
  unsigned distributionStart =
      originalType.getRank() - effectiveLaneLayout.size();
  for (auto [i, dim] : llvm::enumerate(originalType.getShape())) {
    if (i < distributionStart)
      continue;

    // Check if the dimension can be distributed evenly.
    if (dim % effectiveLaneLayout[i - distributionStart] != 0)
      return failure();
    distributedShape[i] = dim / effectiveLaneLayout[i - distributionStart];
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
/// TODO: Move to target info.
static bool requirePacked(const xegpu::LayoutAttr layout) {
  if (!layout)
    return false;
  auto laneData = layout.getEffectiveLaneDataAsInt();
  if (laneData.size() != 2)
    return false;
  return laneData[0] != 1;
}

/// Helper function to check if the layout requires a transpose effect.
static bool requireTranspose(const xegpu::LayoutAttr layout,
                             const std::string &chipStr) {
  // Return false for unsupported targets.
  // TODO: Add more support or move to target info.
  if (chipStr != "pvc" && chipStr != "bmg")
    return false;
  if (!layout)
    return false;
  auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
  if (laneLayout.size() != 2)
    return false;
  return laneLayout[0] == xegpu::targetinfo::subgroupSize && laneLayout[1] == 1;
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
    // Chip information is required to decide if the layout requires transpose
    // effect.
    auto chipStr = xegpu::getChipStr(loadOp);
    if (!chipStr)
      return rewriter.notifyMatchFailure(
          loadOp,
          "xegpu::LoadNdOp require chip information to determine transpose "
          "requirement");
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
    newLoadOp.setPacked(requirePacked(layout));
    // Set the transpose attribute if the layout requires it.
    if (requireTranspose(layout, chipStr.value()))
      newLoadOp.setTranspose(
          DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0}));
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

/// Distribute a scattered store op. The offsets argument is required.
/// Both offset and mask vectors must be 1D and have #subgroup_size elements.
/// The layouts are fixed and implicit: one offset/mask per lane.
/// The pass changes the offset/mask vector shapes to a
/// single-element vector, **it is assumed that their producer will also be
/// distributed**. The payload vector also has a fixed distribution:
///   no chunk size -> vector of one element.
///   chunk size    -> vector of the innermost dimension of the SG-payload.
/// Example 1 (no chunk size):
///    %mask = producer_op : vector<16xi1>
///    %offset = producer_op : vector<16xindex>
///    xegpu.store %payload, %src[%offset], %mask : vector<16xf16>,
///     memref<256xf16>, vector<16xindex>, vector<16xi1>
/// To
///    %mask = producer_op : vector<1xi1>
///    %offset = producer_op : vector<1xindex>
///    xegpu.store %payload, %src[%offset], %mask : vector<1xf16>,
///     memref<256xf16>, vector<1xindex>, vector<1xi1>
/// Example 2 (chunk size, same mask and offsets):
///    xegpu.store %payload, %src[%offset], %mask <{chunk_size=8}> :
///     vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
/// To
///    xegpu.store %payload, %src[%offset], %mask <{chunk_size=8}> :
///     vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
struct StoreDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    Operation *lastNode = warpOp.getTerminator()->getPrevNode();
    auto storeScatterOp = dyn_cast_or_null<xegpu::StoreScatterOp>(lastNode);
    if (!storeScatterOp)
      return failure();
    auto offsets = storeScatterOp.getOffsets();
    if (!offsets || !isa<VectorType>(offsets.getType()))
      return rewriter.notifyMatchFailure(
          storeScatterOp, "Store op must have a vector of offsets argument");
    VectorType offsetsTy = cast<VectorType>(offsets.getType());
    VectorType maskTy = cast<VectorType>(storeScatterOp.getMask().getType());
    if (offsetsTy.getRank() != 1 || maskTy.getRank() != 1)
      return rewriter.notifyMatchFailure(storeScatterOp,
                                         "Expected 1D offsets and mask vector");
    VectorType storeVecTy = cast<VectorType>(storeScatterOp.getValueType());
    if (storeVecTy.getRank() > 2)
      return rewriter.notifyMatchFailure(
          storeScatterOp, "Expected at most 2D result at SG level");

    std::string layoutPayloadName =
        xegpu::getLayoutName(storeScatterOp->getOpOperand(0));
    std::string layoutOffsetsName =
        xegpu::getLayoutName(storeScatterOp->getOpOperand(2));
    std::string layoutMaskName =
        xegpu::getLayoutName(storeScatterOp->getOpOperand(3));

    xegpu::LayoutAttr layoutPayload =
        storeScatterOp->getAttrOfType<xegpu::LayoutAttr>(layoutPayloadName);
    xegpu::LayoutAttr layoutOffsets =
        storeScatterOp->getAttrOfType<xegpu::LayoutAttr>(layoutOffsetsName);
    xegpu::LayoutAttr layoutMask =
        storeScatterOp->getAttrOfType<xegpu::LayoutAttr>(layoutMaskName);

    FailureOr<VectorType> distStoreVecByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutPayload, storeVecTy);
    FailureOr<VectorType> distOffsetsByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutOffsets, offsetsTy);
    FailureOr<VectorType> distMaskByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutMask, maskTy);
    if (failed(distStoreVecByWarpOpOrFailure) ||
        failed(distOffsetsByWarpOpOrFailure) ||
        failed(distMaskByWarpOpOrFailure)) {
      return rewriter.notifyMatchFailure(
          storeScatterOp,
          "Some vector operands have no layouts, using defaults instead.");
    }
    VectorType distPayloadTy = distStoreVecByWarpOpOrFailure.value();
    VectorType expectedPayloadTy = VectorType::get(
        {distPayloadTy.getNumElements()}, distPayloadTy.getElementType());

    SmallVector<size_t> newRetIndices;
    SmallVector<Value> operands = storeScatterOp->getOperands();
    SmallVector<Type> operandTypesToYield = {
        expectedPayloadTy, operands[1].getType(),
        distOffsetsByWarpOpOrFailure.value(),
        distMaskByWarpOpOrFailure.value()};

    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, operands, operandTypesToYield, newRetIndices);
    SmallVector<Value> newStoreScatterOpOperands = llvm::map_to_vector(
        newRetIndices, [&](size_t idx) { return newWarpOp.getResult(idx); });

    rewriter.setInsertionPointAfter(newWarpOp);
    xegpu::StoreScatterOp newOp = xegpu::StoreScatterOp::create(
        rewriter, newWarpOp.getLoc(), TypeRange{}, newStoreScatterOpOperands,
        storeScatterOp->getAttrs());
    xegpu::removeLayoutAttrs(newOp);
    rewriter.eraseOp(storeScatterOp);
    return success();
  }
};

/// Distribute a scattered load op. The logic and requirements are the same as
/// for the scattered store distribution. The warpOp's payload vector is
/// expected to be distributed by the load's result consumer.
/// Example 1 (no chunk size):
///    %mask = producer_op : vector<16xi1>
///    %offset = producer_op : vector<16xindex>
///    %0 = xegpu.load %payload, %src[%offset], %mask : memref<256xf16>,
///    vector<16xindex>, vector<16xi1> -> vector<16xf16>
/// To
///    %mask = producer_op : vector<1xi1>
///    %offset = producer_op : vector<1xindex>
///    %0 = xegpu.load %payload, %src[%offset], %mask : memref<256xf16>,
///     vector<1xindex>, vector<1xi1> -> vector<1xf16>
/// Example 2 (chunk size, same mask and offsets):
///    %0 = xegpu.load %payload, %src[%offset], %mask <{chunk_size=8}> :
///     memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
/// To
///    %0 = xegpu.load %payload, %src[%offset], %mask <{chunk_size=8}> :
///     memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
struct LoadDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *producedByLastLoad = getWarpResult(warpOp, [&](Operation *op) {
      // Check if the yield operand that was produced by the *last* scattered
      // load op to avoid sinking it before barriers (maintain memory order).
      return isa<xegpu::LoadGatherOp>(op) &&
             warpOp.getTerminator()->getPrevNode() == op;
    });
    if (!producedByLastLoad)
      return rewriter.notifyMatchFailure(
          warpOp, "The last op is not xegpu::LoadGatherOp");

    auto loadGatherOp =
        producedByLastLoad->get().getDefiningOp<xegpu::LoadGatherOp>();
    auto offsets = loadGatherOp.getOffsets();
    if (!offsets || !isa<VectorType>(offsets.getType()) ||
        !isa<VectorType>(loadGatherOp.getMask().getType()))
      return rewriter.notifyMatchFailure(
          loadGatherOp,
          "Load op must have a vector arguments for offsets and mask");
    VectorType offsetsTy = cast<VectorType>(offsets.getType());
    VectorType maskTy = cast<VectorType>(loadGatherOp.getMask().getType());
    if (offsetsTy.getRank() != 1 || maskTy.getRank() != 1)
      return rewriter.notifyMatchFailure(loadGatherOp,
                                         "Expected 1D offsets and mask vector");
    // Assume offset and mask producers will be distributed as well.
    std::string layoutOffsetsName =
        xegpu::getLayoutName(loadGatherOp->getOpOperand(1));
    std::string layoutMaskName =
        xegpu::getLayoutName(loadGatherOp->getOpOperand(2));

    xegpu::LayoutAttr layoutOffsets =
        loadGatherOp->getAttrOfType<xegpu::LayoutAttr>(layoutOffsetsName);
    xegpu::LayoutAttr layoutMask =
        loadGatherOp->getAttrOfType<xegpu::LayoutAttr>(layoutMaskName);

    FailureOr<VectorType> distOffsetsByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutOffsets, offsetsTy);
    FailureOr<VectorType> distMaskByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutMask, maskTy);
    if (failed(distOffsetsByWarpOpOrFailure) ||
        failed(distMaskByWarpOpOrFailure)) {
      return rewriter.notifyMatchFailure(
          loadGatherOp,
          "Some vector operands have no layouts, using defaults instead.");
    }

    SmallVector<size_t> newRetIndices;
    SmallVector<Value> operands = loadGatherOp->getOperands();
    SmallVector<Type> operandTypesToYield = {
        operands[0].getType(), distOffsetsByWarpOpOrFailure.value(),
        distMaskByWarpOpOrFailure.value()};

    const unsigned operandIdx = producedByLastLoad->getOperandNumber();
    VectorType loadVecTy =
        cast<VectorType>(warpOp.getResult(operandIdx).getType());

    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, operands, operandTypesToYield, newRetIndices);

    SmallVector<Value> newLoadGatherOperands = llvm::map_to_vector(
        newRetIndices, [&](size_t idx) { return newWarpOp.getResult(idx); });

    rewriter.setInsertionPointAfter(newWarpOp);
    xegpu::LoadGatherOp newOp = xegpu::LoadGatherOp::create(
        rewriter, newWarpOp.getLoc(), loadVecTy, newLoadGatherOperands,
        loadGatherOp->getAttrs());
    xegpu::removeLayoutAttrs(newOp);
    Value distributedVal = newWarpOp.getResult(operandIdx);
    rewriter.replaceAllUsesWith(distributedVal, newOp->getResult(0));
    return success();
  }
};

/// Helper to rewrite a 2D VectorMultiReductionOp into a sequence of 1D
/// VectorReductionOps.
static Value lowerToVectorReductions(TypedValue<VectorType> src,
                                     TypedValue<VectorType> acc,
                                     vector::CombiningKind kind,
                                     int64_t reductionDim, Location loc,
                                     PatternRewriter &rewriter) {
  // Expecting a 2D source vector.
  assert(src.getType().getRank() == 2 && "expected a 2D source vector");
  VectorType sourceType = src.getType();
  int64_t sourceH = sourceType.getShape()[0];
  int64_t sourceW = sourceType.getShape()[1];
  int nSlices = (reductionDim == 0) ? sourceW : sourceH;
  // Create a constant vector to hold the result of the reduction.
  TypedAttr zeroAttr = rewriter.getZeroAttr(sourceType.getElementType());
  Value reductionResult = arith::ConstantOp::create(
      rewriter, loc, acc.getType(),
      DenseElementsAttr::get(acc.getType(), zeroAttr));
  // For each slice of the source, extract the slice vector, do a reduction
  // and, insert the reduced value back to the result vector.
  for (int i = 0; i < nSlices; ++i) {
    SmallVector<int64_t, 2> sliceOffsets, sliceSizes;
    if (reductionDim == 1) {
      sliceOffsets = {i, 0};
      sliceSizes = {1, sourceW};
    } else {
      sliceOffsets = {0, i};
      sliceSizes = {sourceH, 1};
    }
    vector::ExtractStridedSliceOp extractOp =
        vector::ExtractStridedSliceOp::create(rewriter, loc, src, sliceOffsets,
                                              sliceSizes, {1, 1});
    int64_t nSliceElements = extractOp.getResult().getType().getNumElements();
    Value slice = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get({nSliceElements}, sourceType.getElementType()),
        extractOp.getResult());
    Value accExtract = vector::ExtractOp::create(rewriter, loc, acc, i);
    Value reduction =
        vector::ReductionOp::create(rewriter, loc, kind, slice, accExtract);
    reductionResult =
        vector::InsertOp::create(rewriter, loc, reduction, reductionResult, i);
  }
  return reductionResult;
}

/// This patterns distribute the `vector.multi_reduction` operation across
/// lanes in a warp. Currently only 2D to 1D reductions are supported. Given
/// layouts for the source and accumulator vectors,
/// * If the reduction dimension is distributed across lanes, the reduction is
///   non-lane-local and the reduction is done using warp shuffles. Here we
///   simply rewrite the MultiDimReductionOp to a sequence of ReductionOps in
///   the warp op body.
/// * If the reduction dimension is not distributed across lanes, the reduction
///   is lane-local. In this case, we yield the source and accumulator vectors
///   from the warp op and perform the lane-local reduction outside the warp op
///   using a sequence of ReductionOps.
/// Example 1 (Reduction is lane-local):
/// ```
/// %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
///   %0 = "some_def"() : () -> (vector<16x32xf32>)
///   %acc = "some_def"() : () -> (vector<32xf32>)
///   %1 = vector.multi_reduction <add>, %0, %acc [0] : vector<16x32xf32> to
///   vector<32xf32> gpu.yield %1 : vector<32xf32>
/// }
/// ```
/// is lowered to:
/// ```
/// %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<16x1xf32>,
/// vector<1xf32>) {
///   %0 = "some_def"() : () -> (vector<16x32xf32>)
///   %acc = "some_def"() : () -> (vector<32xf32>)
///   gpu.yield %0, %acc : vector<16x32xf32>, vector<32xf32>
/// }
/// %c = arith.constant dense<0.0> : vector<1xf32>
/// %1 = vector.shape_cast %r#0 : vector<16x1xf32> to vector<16xf32>
/// %2 = vector.reduction <add>, %1, %r#1 : vector<16xf32> to f32
/// %3 = vector.insert %2, %c[0] : f32 into vector<1xf32>
/// ```
/// Example 2 (Reduction is non-lane-local):
/// ```
/// %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
///   %0 = "some_def"() : () -> (vector<2x32xf32>)
///   %acc = "some_def"() : () -> (vector<2xf32>)
///   %1 = vector.multi_reduction <add>, %0, %acc [1] : vector<2x32xf32> to
///   vector<2xf32>
///   gpu.yield %1 : vector<2xf32>
/// }
/// ```
/// is lowered to:
/// ```
/// %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
///   %0 = "some_def"() : () -> (vector<2x32xf32>)
///   %acc = "some_def"() : () -> (vector<2xf32>)
///   %1 = arith.constant dense<0.0> : vector<2xf32>
///   %2 = vector.extract %0[0] : vector<32xf32> from <vector<2x32xf32>>
///   %3 = ("warp.reduction %2") : f32
///   %4 = vector.insert %3, %1[0] : f32 into vector<2xf32>
///   ... repeat for row 1
///   gpu.yield %1 : vector<2xf32>
/// }
struct VectorMultiReductionDistribution : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<vector::MultiDimReductionOp>);
    if (!yieldOperand)
      return failure();
    auto reductionOp =
        cast<vector::MultiDimReductionOp>(yieldOperand->get().getDefiningOp());
    unsigned operandNumber = yieldOperand->getOperandNumber();
    VectorType sourceType = reductionOp.getSourceVectorType();
    // Only 2D vectors are supported.
    if (sourceType.getRank() != 2)
      return rewriter.notifyMatchFailure(warpOp,
                                         "Only 2D reductions are supported.");
    ArrayRef<int64_t> reductionDims = reductionOp.getReductionDims();
    // Only 1 reduction dimension supported. This also ensures that the result
    // is vector type.
    if (reductionDims.size() != 1)
      return rewriter.notifyMatchFailure(
          warpOp, "Only 1 reduction dimension is supported.");
    int64_t reductionDim = reductionDims[0];
    VectorType distributedResultType =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    VectorType resultType = cast<VectorType>(reductionOp.getType());
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getDistributeLayoutAttr(reductionOp.getSource());

    FailureOr<VectorType> sourceDistTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(sourceLayout, sourceType);
    if (failed(sourceDistTypeOrFailure))
      return rewriter.notifyMatchFailure(
          warpOp, "Failed to distribute the source vector type.");
    VectorType sourceDistType = sourceDistTypeOrFailure.value();
    // Only single dimension distribution is supported.
    bool dim0Distributed =
        sourceDistType.getShape()[0] != sourceType.getShape()[0];
    bool dim1Distributed =
        sourceDistType.getShape()[1] != sourceType.getShape()[1];
    if (dim0Distributed && dim1Distributed)
      return rewriter.notifyMatchFailure(
          warpOp, "Expecting source to be distributed in a single dimension.");
    int64_t sourceDistDim = dim0Distributed ? 0 : (dim1Distributed ? 1 : -1);
    if (sourceDistDim == -1)
      return rewriter.notifyMatchFailure(
          warpOp, "Expecting a distributed source vector.");
    bool resultDistributed =
        distributedResultType.getNumElements() < resultType.getNumElements();
    // If the lane owns all the data required for reduction (i.e. reduction is
    // fully parallel accross lanes), then each lane owns part of the result
    // (i.e. result is distributed). If the reduction require cross-lane
    // shuffling, then the result is shared among all lanes (broadcasted).
    // Therefore we expect following cases:
    //
    // | Source vector        | Reduction dim  | Result vector  |
    // |----------------------|----------------|----------------|
    // |  dim-0 distributed   |       0        | broadcasted    |
    // |  dim-0 distributed   |       1        | distributed    |
    // |  dim-1 distributed   |       0        | distributed    |
    // |  dim-1 distributed   |       1        | broadcasted    |

    bool isReductionLaneLocal = (sourceDistDim == 0 && reductionDim == 1) ||
                                (sourceDistDim == 1 && reductionDim == 0);
    if (isReductionLaneLocal && !resultDistributed)
      return rewriter.notifyMatchFailure(
          warpOp, "Expecting a distributed result for lane-local reduction.");

    if (!isReductionLaneLocal && resultDistributed)
      return rewriter.notifyMatchFailure(
          warpOp,
          "Expecting a broadcasted result for non-lane-local reduction.");

    // Handle lane-local reduction case. In this case we fully distribute the
    // reduction result.
    if (isReductionLaneLocal) {
      // Yield the source and acc vectors from the WarpOp.
      SmallVector<size_t> newRetIndices;
      auto newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, {reductionOp.getSource(), reductionOp.getAcc()},
          {sourceDistType, distributedResultType}, newRetIndices);
      rewriter.setInsertionPointAfter(newWarpOp);
      Value result = lowerToVectorReductions(
          cast<TypedValue<VectorType>>(newWarpOp->getResult(newRetIndices[0])),
          cast<TypedValue<VectorType>>(newWarpOp->getResult(newRetIndices[1])),
          reductionOp.getKind(), reductionDim, reductionOp.getLoc(), rewriter);
      // Replace the warp op result with the final result.
      rewriter.replaceAllUsesWith(reductionOp.getResult(), result);
      return success();
    }
    // For non-lane-local case, we simply rewrite the MultiReductionOp in terms
    // of multiple ReductionOps. Actual distribution is done by the
    // WarpOpReduction pattern.
    rewriter.setInsertionPointAfter(reductionOp);
    Value result = lowerToVectorReductions(
        cast<TypedValue<VectorType>>(reductionOp.getSource()),
        cast<TypedValue<VectorType>>(reductionOp.getAcc()),
        reductionOp.getKind(), reductionDim, reductionOp.getLoc(), rewriter);
    // Replace the warp op result with the final result.
    rewriter.replaceAllUsesWith(reductionOp.getResult(), result);
    return success();
  }
};

/// Distribute a `vector.shape_cast` op feeding into yield op of an enclosing
/// `gpu.warp_execute_on_lane_0` region.
struct VectorShapeCastDistribution : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<vector::ShapeCastOp>);
    if (!yieldOperand)
      return failure();
    auto shapeCastOp =
        cast<vector::ShapeCastOp>(yieldOperand->get().getDefiningOp());
    unsigned operandNumber = yieldOperand->getOperandNumber();
    auto resultDistTy =
        cast<VectorType>(warpOp.getResult(operandNumber).getType());
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getDistributeLayoutAttr(shapeCastOp.getSource());
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getDistributeLayoutAttr(shapeCastOp.getResult());
    if (!sourceLayout || !resultLayout)
      return rewriter.notifyMatchFailure(
          warpOp,
          "the source or result of shape_cast op lacks distribution layout");

    // For rank reducing or increasing shape_cast ops, the lower rank layout
    // must be a slice of higher rank layout.
    int64_t sourceRank = shapeCastOp.getSourceVectorType().getRank();
    int64_t resultRank = shapeCastOp.getResultVectorType().getRank();
    if (sourceRank < resultRank && !sourceLayout.isSliceOf(resultLayout))
      return rewriter.notifyMatchFailure(
          warpOp, "shape_cast is rank reducing but source layout is not a "
                  "slice of result layout");
    if (sourceRank > resultRank && !resultLayout.isSliceOf(sourceLayout))
      return rewriter.notifyMatchFailure(
          warpOp, "shape_cast is rank increasing but result layout is not a "
                  "slice of source layout");

    FailureOr<VectorType> sourceDistTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(sourceLayout,
                                        shapeCastOp.getSourceVectorType());
    if (failed(sourceDistTypeOrFailure))
      return rewriter.notifyMatchFailure(
          warpOp, "failed to get distributed vector type for source");
    VectorType sourceDistType = sourceDistTypeOrFailure.value();
    // Create a new warp op that yields the source of the shape_cast op.
    SmallVector<size_t> newRetIndices;
    auto newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {shapeCastOp.getSource()}, {sourceDistType},
        newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    Value source = newWarpOp.getResult(newRetIndices[0]);
    // Create a new shape_cast op outside the warp op.
    Value newShapeCast = vector::ShapeCastOp::create(
        rewriter, shapeCastOp.getLoc(), resultDistTy, source);
    rewriter.replaceAllUsesWith(newWarpOp.getResult(operandNumber),
                                newShapeCast);
    return success();
  }
};

/// Sink a memref::ExtractAlignedPointerAsIndex op feeding into yield op of an
/// enclosing `gpu.warp_execute_on_lane_0` region. This will simply move the op
/// outside of the warp op.
struct MemrefExtractAlignedPointerAsIndexDistribution final
    : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, llvm::IsaPred<memref::ExtractAlignedPointerAsIndexOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp,
          "warp result is not a memref::MemrefExtractAlignedPointerAsIndex op");
    auto extractOp =
        operand->get().getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
    unsigned operandIdx = operand->getOperandNumber();
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, extractOp.getSource(),
        TypeRange{extractOp.getSource().getType()}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newExtractOp = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, newWarpOp.getLoc(), extractOp.getType(),
        newWarpOp.getResult(newRetIndices[0]));
    Value distributedVal = newWarpOp.getResult(operandIdx);
    rewriter.replaceAllUsesWith(distributedVal, newExtractOp.getResult());
    return success();
  }
};

/// Distribute a vector::BitCastOp feeding into yield op of an enclosing
/// `gpu.warp_execute_on_lane_0` region. Bitcast only impacts the innermost
/// diemension of the source/result vectors. Equivalent vector::BitCastOp is
/// created outside of the warp op with distributed source vector type (computed
/// using assigned layout).
struct VectorBitcastDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::BitCastOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a vector::BitCast op");
    auto bitcastOp = operand->get().getDefiningOp<vector::BitCastOp>();
    unsigned operandIdx = operand->getOperandNumber();
    VectorType distributedSourceType =
        getDistVecTypeBasedOnLaneLayout(
            xegpu::getDistributeLayoutAttr(bitcastOp.getSource()),
            bitcastOp.getSourceVectorType())
            .value_or(VectorType());
    if (!distributedSourceType)
      return rewriter.notifyMatchFailure(
          bitcastOp, "Failed to distribute the source vector type in "
                     "vector::BitCast op");
    VectorType distributedResultType =
        cast<VectorType>(warpOp.getResult(operandIdx).getType());
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, bitcastOp.getSource(),
        TypeRange{distributedSourceType}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newBitcastOp = vector::BitCastOp::create(
        rewriter, newWarpOp.getLoc(), distributedResultType,
        newWarpOp.getResult(newRetIndices[0]));
    Value distributedVal = newWarpOp.getResult(operandIdx);
    rewriter.replaceAllUsesWith(distributedVal, newBitcastOp.getResult());
    return success();
  }
};

/// Distribute a vector::TransposeOp feeding into yield op of an enclosing
/// `gpu.warp_execute_on_lane_0` region. Currently only 2D transposes are
/// supported. In most cases, transpose is a no op because it is entirely
/// handled using the layouts (e.g. 16x1 -> 1x16). However, if each lane owns
/// multiple slices of data after distribution (e.g. 16x2 -> 2x16), a lane-local
/// transpose (i.e. shuffle) is needed. Therefore, we create an equivalent
/// vector::TransposeOp outside of the warp op with distributed source vector
/// type (computed using assigned layout).
struct VectorTransposeDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(warpOp, llvm::IsaPred<vector::TransposeOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          warpOp, "warp result is not a vector::Transpose op");
    auto transposeOp = operand->get().getDefiningOp<vector::TransposeOp>();
    unsigned operandIdx = operand->getOperandNumber();
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getDistributeLayoutAttr(transposeOp.getVector());
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getDistributeLayoutAttr(transposeOp.getResult());
    if (!sourceLayout || !resultLayout)
      return rewriter.notifyMatchFailure(
          transposeOp,
          "the source or result vector of the transpose op lacks layout "
          "attribute");
    int64_t sourceRank = transposeOp.getSourceVectorType().getRank();
    int64_t resultRank = transposeOp.getResultVectorType().getRank();
    // Only 2D transposes are supported for now.
    // TODO: Support nD transposes.
    if (sourceRank != 2 || resultRank != 2)
      return rewriter.notifyMatchFailure(
          transposeOp, "the source or result vector of the transpose op "
                       "does not have 2D layout");
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    // Result layout must be a transpose of source layout.
    if (!resultLayout.isTransposeOf(sourceLayout, perm))
      return rewriter.notifyMatchFailure(
          transposeOp,
          "the source or result vector layouts must be 2D transposes of each "
          "other");
    FailureOr<VectorType> distributedSourceTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(sourceLayout,
                                        transposeOp.getSourceVectorType());
    if (failed(distributedSourceTypeOrFailure))
      return rewriter.notifyMatchFailure(
          transposeOp, "Failed to distribute the source vector type in "
                       "vector::Transpose op");
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, transposeOp.getVector(),
        TypeRange{distributedSourceTypeOrFailure.value()}, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    auto newTransposeOp = vector::TransposeOp::create(
        rewriter, newWarpOp.getLoc(), newWarpOp.getResult(newRetIndices[0]),
        perm);
    Value distributedVal = newWarpOp.getResult(operandIdx);
    rewriter.replaceAllUsesWith(distributedVal, newTransposeOp.getResult());
    return success();
  }
};

} // namespace

namespace {
struct XeGPUSubgroupDistributePass final
    : public xegpu::impl::XeGPUSubgroupDistributeBase<
          XeGPUSubgroupDistributePass> {
  XeGPUSubgroupDistributePass() = default;
  XeGPUSubgroupDistributePass(const XeGPUSubgroupDistributePass &other) =
      default;
  XeGPUSubgroupDistributePass(xegpu::XeGPUSubgroupDistributeOptions options)
      : XeGPUSubgroupDistributeBase(options) {}
  void runOnOperation() override;
};
} // namespace

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<CreateNdDescDistribution, StoreNdDistribution, LoadNdDistribution,
           DpasDistribution, PrefetchNdDistribution, UpdateNdOffsetDistribution,
           GpuBarrierDistribution, VectorMultiReductionDistribution,
           LoadDistribution, StoreDistribution, VectorTransposeDistribution,
           VectorBitcastDistribution,
           MemrefExtractAlignedPointerAsIndexDistribution>(
          patterns.getContext(),
          /*pattern benefit=*/regularPatternBenefit);
  patterns.add<VectorShapeCastDistribution>(
      patterns.getContext(),
      /*pattern benefit=*/highPatternBenefit);
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

      auto layout = xegpu::getDistributeLayoutAttr(operand.get());
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
    xegpu::DistributeLayoutAttr layout = xegpu::getDistributeLayoutAttr(val);
    // If no layout is specified, assume the inner most dimension is distributed
    // for now.
    if (!layout)
      return AffineMap::getMultiDimMapWithTargets(
          vecRank, {static_cast<unsigned int>(vecRank - 1)}, val.getContext());
    SmallVector<unsigned int> distributedDims;
    for (auto [i, v] : llvm::enumerate(layout.getEffectiveLaneLayoutAsInt())) {
      if (v > 1)
        distributedDims.push_back(i);
    }
    return AffineMap::getMultiDimMapWithTargets(vecRank, distributedDims,
                                                val.getContext());
  };
  // TODO: shuffleFn is not used.
  auto shuffleFn = [](Location loc, OpBuilder &builder, Value val, Value srcIdx,
                      int64_t warpSz) { return Value(); };

  auto warpReduction = [](Location loc, OpBuilder &builder, Value input,
                          vector::CombiningKind kind, uint32_t size) {
    // First reduce on a single thread to get per lane reduction value.
    Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
    // Parallel reduction using butterfly shuffles.
    for (uint64_t i = 1; i < size; i <<= 1) {
      Value shuffled =
          builder
              .create<gpu::ShuffleOp>(loc, laneVal, i,
                                      /*width=*/size,
                                      /*mode=*/gpu::ShuffleMode::XOR)
              .getShuffleResult();
      laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
    }
    return laneVal;
  };

  if (enableSGReductions)
    vector::populateDistributeReduction(
        patterns, warpReduction,
        /*pattern benefit=*/regularPatternBenefit);

  vector::populatePropagateWarpVectorDistributionPatterns(
      patterns, distributionFn, shuffleFn,
      /*pattern benefit=*/regularPatternBenefit);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Step 4: Finally, clean up UnrealizedConversionCastOps that were inserted
  // due to tensor desc type mismatches created by using upstream distribution
  // patterns (scf.for). This cleanup should only be done if all the ops are
  // distributed successfully, if some ops are still not distributed and remains
  // inside any WarpExecuteOnLane0Op we avoid this simplication step to avoid
  // breaking the IR.
  bool foundWarpOp = false;
  getOperation()->walk([&](gpu::WarpExecuteOnLane0Op warpOp) {
    // Look for WarpOps that are not trivially dead.
    if (isOpTriviallyDead(warpOp))
      return WalkResult::advance();
    foundWarpOp = true;
    return WalkResult::interrupt();
  });
  if (foundWarpOp)
    return;

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
    // This occurs inside scf.for body to resolve the block argument type to
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
