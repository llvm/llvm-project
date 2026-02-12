//===---- XeGPULayoutImpl.cpp - MLIR Utilities for XeGPUOps
//------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements layout utility functions for XeGPU dialect
// transformation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

void xegpu::recoverTemporaryLayoutsDeprecated(Operation *op) {
  op->walk([&](Operation *nestOp) {
    for (OpOperand &opr : nestOp->getOpOperands()) {
      auto layout = getDistributeLayoutAttr(opr.get());
      setDistributeLayoutAttr(opr, layout);
    }

    for (OpResult result : nestOp->getOpResults()) {
      auto layout = getDistributeLayoutAttr(result);
      setDistributeLayoutAttr(result, layout);
    }
  });
}

SmallVector<NamedAttribute>
xegpu::dropSgLayoutAndDataOnAttrs(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> out;
  out.reserve(attrs.size());

  for (auto attr : attrs) {
    if (auto dist = dyn_cast<xegpu::DistributeLayoutAttr>(attr.getValue())) {
      auto newLayout = dist.dropSgLayoutAndData();
      if (newLayout)
        out.emplace_back(attr.getName(), newLayout);
    } else {
      out.push_back(attr);
    }
  }

  return out;
}

SmallVector<NamedAttribute>
xegpu::dropInstDataOnAttrs(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> out;
  out.reserve(attrs.size());

  for (auto attr : attrs) {
    if (auto dist = dyn_cast<xegpu::DistributeLayoutAttr>(attr.getValue())) {
      auto newLayout = dist.dropInstData();
      if (newLayout)
        out.emplace_back(attr.getName(), newLayout);
    } else {
      out.push_back(attr);
    }
  }

  return out;
}

// Attach layout attributes to all vector-type operands of operations within
// the given operation's region. Reports an error if any vector operand lacks
// a layout attribute.
bool xegpu::recoverTemporaryLayouts(Operation *rootOp) {
  auto result = rootOp->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Layouts are needed for vector type only.
      if (!isa<VectorType>(operand.get().getType()))
        continue;
      // Skip block arguments since they don't have defining ops to attach
      // layout attributes to.
      if (isa<BlockArgument>(operand.get()))
        continue;
      auto layout = xegpu::getDistributeLayoutAttr(operand.get());
      if (!layout) {
        op->emitWarning("Could not find layout attribute for operand ")
            << operand.getOperandNumber() << " of operation " << op->getName();
        continue;
      }
      xegpu::setDistributeLayoutAttr(operand, layout);
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

template <typename T, typename>
void xegpu::removeLayoutAttr(const T &operandOrResult) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getTemporaryLayoutName(operandOrResult);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name))
    owner->removeAttr(name);
}

// Explicit instantiation for OpResult
template void
xegpu::removeLayoutAttr<mlir::OpResult>(const mlir::OpResult &result);

// Explicit instantiation for OpOperand
template void
xegpu::removeLayoutAttr<mlir::OpOperand>(const mlir::OpOperand &operand);

void xegpu::removeLayoutAttrs(Operation *op) {
  op->walk([&](Operation *nestOp) {
    // Remove all attributes of DistributeLayoutAttr type
    SmallVector<StringAttr> attrsToRemove;
    for (auto namedAttr : nestOp->getAttrs()) {
      if (isa<DistributeLayoutAttr>(namedAttr.getValue()))
        attrsToRemove.push_back(namedAttr.getName());
    }
    for (auto attrName : attrsToRemove)
      nestOp->removeAttr(attrName);
  });
}

/// Infers the source layout attribute for a broadcast operation given the
/// result layout attribute, result shape, source shape.
xegpu::DistributeLayoutAttr
xegpu::inferBroadcastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> resShape,
                                  ArrayRef<int64_t> srcShape) {

  SmallVector<int64_t> bcastDims;
  auto returnLayout = resLayout;

  // Handling broadcast from low-rank to high-rank (e.g., 1D to 2D) case.
  int dimDiff = resShape.size() - srcShape.size();

  if (dimDiff > 0) {
    // Adding the missing leading dims
    for (int i = 0; i < dimDiff; i++)
      bcastDims.push_back(i);

    // Create a slice layout for the source
    returnLayout = xegpu::SliceAttr::get(
        resLayout.getContext(), resLayout,
        DenseI64ArrayAttr::get(resLayout.getContext(), bcastDims));
  }
  return returnLayout;
}

/// Infers the source layout attribute for a reduction operation given the
/// result layout attribute and reduced dims.
xegpu::DistributeLayoutAttr
xegpu::inferMultiReductionSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                       SmallVector<int64_t> reduceDims) {

  assert(isa<xegpu::SliceAttr>(resLayout) &&
         "reduction result layout must be slice layout");

  xegpu::SliceAttr sliceLayout = dyn_cast<xegpu::SliceAttr>(resLayout);

  assert((reduceDims == sliceLayout.getDims().asArrayRef()) &&
         "reduction dims must match with slice dims");

  return sliceLayout.getParent();
}

/// Infers the source layout attribute for a bitcast operation given the
/// result layout attribute, result element type bitwidth, and source element
/// type bitwidth.
xegpu::DistributeLayoutAttr
xegpu::inferBitCastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                int resElemTyBitWidth, int srcElemTyBitWidth) {

  SmallVector<int64_t> sgData = resLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = resLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneData = resLayout.getEffectiveLaneDataAsInt();
  size_t sgDataSize = sgData.size();
  size_t instDataSize = instData.size();
  size_t laneDataSize = laneData.size();
  int64_t sgDataValue = -1;
  int64_t instDataValue = -1;
  int64_t laneDataValue = -1;
  int64_t dim = resLayout.getRank() - 1;

  if (srcElemTyBitWidth <= resElemTyBitWidth) {
    int bitWidthRatio = resElemTyBitWidth / srcElemTyBitWidth;
    if (sgDataSize)
      sgDataValue = sgData.back() * bitWidthRatio;
    if (instDataSize)
      instDataValue = instData.back() * bitWidthRatio;
    if (laneDataSize)
      laneDataValue = laneData.back() * bitWidthRatio;
  } else {
    int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
    if (sgDataSize) {
      assert((sgData.back() % bitWidthRatio) == 0 &&
             "sgData not divisible by bitWidthRatio");
      sgDataValue = sgData.back() / bitWidthRatio;
    }
    if (instDataSize) {
      assert((instData.back() % bitWidthRatio) == 0 &&
             "instData not divisible by bitWidthRatio");
      instDataValue = instData.back() / bitWidthRatio;
    }
    if (laneDataSize) {
      assert((laneData.back() % bitWidthRatio) == 0 &&
             "laneData not divisible by bitWidthRatio");
      laneDataValue = laneData.back() / bitWidthRatio;
    }
  }

  xegpu::DistributeLayoutAttr finalSrcLayout;
  finalSrcLayout =
      resLayout.setDimData(dim, sgDataValue, instDataValue, laneDataValue);

  return finalSrcLayout;
}

/// Infers the source layout attribute for an insert strided slice operation
/// given the result layout attribute, result shape, and source shape. Removes
/// leading dimensions from the result layout to match the source shape size.
xegpu::DistributeLayoutAttr xegpu::inferInsertStridedSliceSourceLayout(
    xegpu::DistributeLayoutAttr resLayout, ArrayRef<int64_t> resShape,
    ArrayRef<int64_t> srcShape) {

  int srcShapeSize = srcShape.size();
  int resShapeSize = resShape.size();
  int dimDiff = resShapeSize - srcShapeSize;

  assert(isa<xegpu::LayoutAttr>(resLayout) &&
         "insertStridedSlice result layout must be plain layout");
  auto context = resLayout.getContext();
  auto resInstData = resLayout.getEffectiveInstDataAsInt();
  auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
  auto resLaneData = resLayout.getEffectiveLaneDataAsInt();

  if (resInstData.size() != 0) {
    SmallVector<int> inferredInstData(srcShapeSize);
    for (int i = 0; i < srcShapeSize; i++)
      inferredInstData[i] = resInstData[i + dimDiff];
    return xegpu::LayoutAttr::get(context, inferredInstData);
  }

  if (resLaneLayout.size() != 0) {
    SmallVector<int> inferredLaneLayout(srcShapeSize);
    SmallVector<int> inferredLaneData(srcShapeSize);
    for (int i = 0; i < srcShapeSize; i++) {
      inferredLaneLayout[i] = resLaneLayout[i + dimDiff];
      inferredLaneData[i] = resLaneData[i + dimDiff];
    }
    return xegpu::LayoutAttr::get(context, inferredLaneLayout,
                                  inferredLaneData);
  }
  return nullptr;
}

/// Infers the source layout attribute for a shape cast operation given the
/// result layout attribute, result shape, and source shape.
xegpu::DistributeLayoutAttr
xegpu::inferShapeCastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> resShape,
                                  ArrayRef<int64_t> srcShape) {

  // There are three use cases:
  // 1. expand dims of low-rank dimensions (e.g., 1D to 2D): to set up the
  // tensor before broadcast
  // 2. split dim of a high-rank dimension (e.g., 1D to 2D): to setup tensor
  // for multi-stage reduction
  // 3. combines all dims to a single dim and put in the innermost dim in 2d as
  // [1, combinedData] or [combinedData]. Say, [2, 4, 8] -> [1, 64] or [64]
  // Use cases are only supported after workgroup distribution,
  // like cross-sg reduction saves multidimension data to
  // 1D slm buffer, shapecast inserted by cse/canonicalization passes.

  // Use case 1: Shapes only differ by expanding unit dimensions, for broadcast
  SmallVector<int64_t> expandedUnitDims;

  if (xegpu::matchUnitDimExpansion(srcShape, resShape, expandedUnitDims)) {
    // create a slice layout for the source by removing the expanded unit dims
    auto sliceDimsAttr = DenseI64ArrayAttr::get(
        resLayout.getContext(), ArrayRef<int64_t>(expandedUnitDims));
    auto srcLayout =
        xegpu::SliceAttr::get(resLayout.getContext(), resLayout, sliceDimsAttr);
    return srcLayout;
  }

  // Use case 2: Dim split from source to result, for multi-stage reduction
  SmallVector<SmallVector<int64_t>> splitDimGroups;
  if (xegpu::matchSplitDimExpansion(srcShape, resShape, splitDimGroups)) {
    auto srcLayout = resLayout;
    for (const auto &dimGroup : splitDimGroups)
      srcLayout = srcLayout.collapseDims(dimGroup);

    return srcLayout;
  }

  // Use case 3: Collaspse to innermost dim, for cross-sg reduction to SLM
  auto matchCollapseToInnermostDim = [&](ArrayRef<int64_t> src,
                                         ArrayRef<int64_t> dst) -> bool {
    // only one non-unit dim in dst which is the innermost dim
    if ((dst.size() != 2) && (dst.size() != 1))
      return false;
    int64_t srcSize = std::accumulate(src.begin(), src.end(), 1LL,
                                      std::multiplies<int64_t>());
    if (dst.size() == 1)
      return (dst[0] == srcSize);
    return (dst[0] == 1) && (dst[1] == srcSize);
  };

  if (matchCollapseToInnermostDim(srcShape, resShape)) {
    int srcShapeSize = srcShape.size();
    int resShapeSize = resShape.size();
    auto context = resLayout.getContext();
    auto resInstData = resLayout.getEffectiveInstDataAsInt();
    auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
    auto resLaneData = resLayout.getEffectiveLaneDataAsInt();

    // Extract layout info from result's innermost dimension and apply to
    // source's innermost dimension while setting all other dimensions to 1.
    // The inferred layout is restricted by srcShape to ensure it fits within
    // the source dimensions.
    // Examples 1:
    //   srcShape=[8, 16, 32], resShape=[1, 4096]
    //   resInstData=[1, 16]
    //   -> inferredInstData=[1, 1, min(16, 32)]=[1, 1, 16]
    // Examples 2:
    //   srcShape=[4, 8, 64], resShape=[2048]
    //   resLaneLayout=[16], resLaneData=[2]
    //   -> inferredLaneLayout=[1, 1, 16]
    //   -> inferredLaneData=[1, 1, min(2, 64/16)]=[1, 1, 2]

    if (resInstData.size() != 0) {
      // assert resInstData must be 1 for all but the innermost dim
      for (int i = 0; i < resShapeSize - 1; i++) {
        assert(resInstData[i] == 1 &&
               "only innermost dim can have non-unit instData");
      }
      SmallVector<int> inferredInstData(srcShapeSize, 1);
      inferredInstData[srcShapeSize - 1] =
          std::min(resInstData[resShapeSize - 1], srcShape[srcShapeSize - 1]);
      return xegpu::LayoutAttr::get(context, inferredInstData);
    }

    if (resLaneLayout.size() != 0) {
      for (int i = 0; i < resShapeSize - 1; i++) {
        assert(resLaneData[i] == 1 &&
               "only innermost dim can have non-unit instData");
      }
      assert(srcShape.back() % resLaneLayout.back() == 0 &&
             "source innermost dim must be >= result lane layout");
      SmallVector<int> inferredLaneLayout(srcShapeSize, 1);
      SmallVector<int> inferredLaneData(srcShapeSize, 1);
      inferredLaneLayout.back() = resLaneLayout.back();
      inferredLaneData.back() = std::min(
          resLaneData.back(), srcShape.back() / inferredLaneLayout.back());
      return xegpu::LayoutAttr::get(context, inferredLaneLayout,
                                    inferredLaneData);
    }
  }
  llvm_unreachable("running into unsupported shape cast scenarios");
  return nullptr;
}

/// Sets up layout for reduction operations by creating a SliceAttr for the
/// result.
///
/// Algorithm Overview:
/// This function attempts to construct a source layout that, when sliced along
/// reduction dimensions, produces a result layout compatible with the
/// consumer layout.
///
/// For subgroup layouts, it first tries to align the source layout's subgroup
/// layout and data with the consumer's layout on non-reduction dimensions.
/// Then, it distributes remaining subgroups across reduction dimensions. This
/// avoids subgroup data redistribution overhead between the reduced result and
/// its consumer.
///
/// InstData requries {1, ..., min(maxReduceVectorSize, srcShape),subgroupSize}
/// Lane Layout requires {1, ..., 1, subgroupSize}
/// Lane data requires {1, ..., min(maxReduceVectorSize, srcShape), 1}
///
/// Examples:
///   1. Subgroup layout - Row reduction on 2D tensor:
///      srcShape=[32, 64], reductionDims=[1], resShape=[32], subgroupSize=16,
///      workgroupSize=32
///      Consumer Layout:
///      #xegpu.slice<#xegpu.layout<sg_layout=[4, 8], sg_data=[8, 8]>, dims =
///      [1]>} Result: srcLayout with sgLayout=[4, 8], sgData=[8, 8] (matches
///      consumer on non-reduction dim, minimizing data redistribution on
///      reduction dim)
///   2. Subgroup layout - Same example above but consumer has different layout:
///      sgLayout=[32], sgData=[1]
///      Result: srcLayout with sgLayout=[32,1], sgData=[1, 64]
///      (distributes all subgroups on non reduction dim)
///
///   2. InstData layout - Column reduction:
///      srcShape=[32, 64], reductionDims=[0], subgroupSize=16
///      Result: instData=[1, 16] (maxReduceVectorSize=1, subgroupSize on
///      innermost)
///
///   3. Lane layout - Multi-dimensional reduction:
///      srcShape=[16, 32, 64], reductionDims=[1], subgroupSize=16
///      Result: laneLayout=[1, 1, 16], laneData=[1, 1, 1]
///      (subgroupSize on innermost dim, max vector size on reduction dim)

xegpu::SliceAttr xegpu::setupMultiReductionResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVecTy,
    DistributeLayoutAttr consumerLayout, SmallVector<int64_t> reductionDims,
    const xegpu::uArch::uArch *uArch) {

  auto srcShape = srcVecTy.getShape();
  int srcRank = srcShape.size();
  auto context = consumerLayout.getContext();

  // Reduction layout requires at least 2D tensors
  if (srcRank < 2)
    return nullptr;

  // Helper lambda to convert int64 vectors to int32 DenseArrayAttr
  auto toInt32Attr = [&](ArrayRef<int64_t> vec) {
    SmallVector<int32_t> vec32(vec.begin(), vec.end());
    return DenseI32ArrayAttr::get(context, vec32);
  };

  // Extract original plain layout for workgroup/subgroup size recovery
  xegpu::SliceAttr consumerSliceLayout =
      dyn_cast<xegpu::SliceAttr>(consumerLayout);
  DistributeLayoutAttr plainLayout =
      consumerSliceLayout ? consumerSliceLayout.flatten().getParent()
                          : consumerLayout;

  const int subgroupSize = uArch->getSubgroupSize();
  int64_t maxReduceVectorSize = 1; // could extend to spirv vector Size

  xegpu::DistributeLayoutAttr srcLayout;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    auto sgLayoutVec = plainLayout.getEffectiveSgLayoutAsInt();
    const int workgroupSize = std::accumulate(
        sgLayoutVec.begin(), sgLayoutVec.end(), 1, std::multiplies<int64_t>());
    SmallVector<int64_t> sgLayout(srcRank), sgData(srcRank);
    SmallVector<int64_t> consumerSgLayout =
        consumerLayout.getEffectiveSgLayoutAsInt();
    int remainingSgCount = workgroupSize;
    int consumerIdx = consumerSgLayout.size() - 1;

    // First pass: Match consumer's layout on non-reduction dimensions
    for (int i = srcRank - 1; i >= 0; i--) {
      if (!llvm::is_contained(reductionDims, i) && consumerIdx >= 0) {
        sgLayout[i] = consumerSgLayout[consumerIdx];
        assert((srcShape[i] % sgLayout[i] == 0) &&
               "source shape not divisible by consumer sg_layout");
        sgData[i] = srcShape[i] / sgLayout[i];
        remainingSgCount /= sgLayout[i];
        consumerIdx--;
      }
    }

    // Second pass: Distribute remaining subgroups across reduction dimensions
    for (int i = srcRank - 1; i >= 0; i--) {
      if (llvm::is_contained(reductionDims, i)) {
        sgLayout[i] =
            std::min(srcShape[i], static_cast<int64_t>(remainingSgCount));
        assert((srcShape[i] % sgLayout[i] == 0) &&
               "source shape not divisible by sg_layout");
        sgData[i] = srcShape[i] / sgLayout[i];
        remainingSgCount /= sgLayout[i];
      }
    }

    assert(remainingSgCount == 1 && "not all subgroups distributed");
    srcLayout = xegpu::LayoutAttr::get(
        context, toInt32Attr(sgLayout), toInt32Attr(sgData),
        /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
        /*lane_data =*/nullptr, /*order =*/nullptr);

  } else if (layoutKind == xegpu::LayoutKind::InstData) {

    SmallVector<int64_t> instData(srcRank, 1);
    instData[srcRank - 2] =
        std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    instData[srcRank - 1] = subgroupSize;
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(instData));

  } else if (layoutKind == xegpu::LayoutKind::Lane) {

    SmallVector<int64_t> laneLayout(srcRank, 1), laneData(srcRank, 1);
    laneLayout[srcRank - 1] = subgroupSize;
    laneData[srcRank - 2] =
        std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(laneLayout),
                                       toInt32Attr(laneData),
                                       consumerLayout.getOrder());
  }

  return xegpu::SliceAttr::get(context, srcLayout,
                               DenseI64ArrayAttr::get(context, reductionDims));
}

/// Sets up the result layout for a bitcast operation.
/// When casting to a smaller bitwidth, adjusts the layout dimensions (sgData,
/// instData, or laneData) by multiplying by the bitwidth ratio to ensure the
/// result layout can be correctly divided back to the source layout during
/// inference.
///
/// Examples:
///   1. Casting f32 -> f16 (32-bit to 16-bit, bitWidthRatio = 2):
///      Consumer layout: instData=[1, 16], subgroupSize=16
///      Source shape: [8, 32]
///      Result layout: instData=[1, 32] (16 * 2)
///      The innermost dimension is multiplied by 2 to maintain consistency.
///
///   2. Casting f32 -> i8 (32-bit to 8-bit, bitWidthRatio = 4):
///      Consumer instData=[1, 16], subgroupSize=16
///      Source shape: [4, 128]
///      adjust the instData from [1, 16] to [1, 16 * 4 = 64]
///
///   3. Casting i8 -> i32 (8-bit to 32-bit, bitWidthRatio = 1/4):
///      Consumer layout: laneLayout=[1, 16], laneData=[1, 4]
///      No adjustment needed - returns consumer layout directly.
///
xegpu::DistributeLayoutAttr xegpu::setupBitCastResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVecTy, VectorType resVecTy,
    DistributeLayoutAttr consumerLayout, const xegpu::uArch::uArch *uArch) {

  int srcElemTyBitWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();
  int resElemTyBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  SmallVector<int64_t> sgData = consumerLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneData = consumerLayout.getEffectiveLaneDataAsInt();
  size_t dim = srcShape.size() - 1;
  int64_t sgDataValue = -1;
  int64_t instDataValue = -1;
  int64_t laneDataValue = -1;

  const int subgroupSize = uArch->getSubgroupSize();

  if (srcElemTyBitWidth > resElemTyBitWidth) {
    // When casting to a smaller bitwidth, multiply the result layout
    // accordingly to ensure it can be divided by the ratio back to the
    // source layout.
    int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
    int innermostDimLaneLayout = subgroupSize;
    if (layoutKind == xegpu::LayoutKind::Subgroup) {
      assert(sgData.size() == srcShape.size() &&
             "sgData must be available for all dimensions");
      sgDataValue = sgData[dim];
    } else if (layoutKind == xegpu::LayoutKind::InstData) {
      assert(instData.size() == srcShape.size() &&
             "instData must be available for all dimensions");
      instDataValue = instData[dim];
      // Adjust instDataValue so it still fits within an instruction after
      // dividing by bitWidthRatio
      while ((instDataValue <= srcShape[dim]) &&
             (instDataValue % (innermostDimLaneLayout * bitWidthRatio) != 0))
        instDataValue *= 2;
      assert((srcShape[dim] % instDataValue) == 0 &&
             "srcShape, instData, and lanelayout for innermost must be 2^n !");
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      assert(laneData.size() == srcShape.size() &&
             "laneData must be available for all dimensions");
      laneDataValue = laneData[dim];
      while ((laneDataValue <= srcShape[dim]) &&
             (laneDataValue % bitWidthRatio != 0))
        laneDataValue *= 2;
    }
    // Now set only instData and laneData, preserving sgData
    xegpu::DistributeLayoutAttr resLayout;
    resLayout = consumerLayout.setDimData(dim, sgDataValue, instDataValue,
                                          laneDataValue);
    return resLayout;
  }
  return consumerLayout;
}

/// Sets up the result layout for an insert strided slice operation.
/// Creates a result layout based on the specified layout kind (InstData or
/// Lane).
/// Subgroup layout is currently not supported for this operation.
/// InstData layout is first set to be {1, .., subgroupSize}.
/// Lane layout is first set to be {1, ..., subgroupSize} with lane data {1,
/// ..., 1}. The instData and laneData is then adjusted to contain packed data,
/// by checking if the consumerLayout's innermost dimension.
///
/// Examples:
///   1. InstData layout without packing:
///      resShape=[8, 32], subgroupSize=16, bitwidth=32
///      packingFactor=1, packedDataSize=16
///      consumerLayout: instData=[1, 16]
///      Result: instData=[1, 16]
///
///   2. InstData layout with packing:
///      resShape=[8, 64], subgroupSize=16, bitwidth=8, packingFactor=4
///      consumerLayout: instData=[1, 64]
///      Result: instData=[1, 64] (adjusted for packed data)
///
///   3. Lane layout without packing:
///      resShape=[4, 64], subgroupSize=16, bitwidth=32
///      consumerLayout: laneLayout=[1, 16], laneData=[1, 1]
///      Result: laneLayout=[1, 16], laneData=[1, 1]
///
///   4. Lane layout with packing:
///      resShape=[4, 64], subgroupSize=16, bitwidth=16, packingFactor=2
///      consumerLayout: laneLayout=[1, 16], laneData=[1, 2]
///      Result: laneLayout=[1, 16], laneData=[1, 2] (adjusted for packed data)
xegpu::DistributeLayoutAttr xegpu::setupInsertStridedSliceResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVectorTy,
    VectorType resVectorTy, xegpu::DistributeLayoutAttr consumerLayout,
    const xegpu::uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredResLayout;
  auto subgroupSize = uArch->getSubgroupSize();
  auto context = resVectorTy.getContext();
  auto resShape = resVectorTy.getShape();
  int resShapeSize = resShape.size();
  auto srcShape = srcVectorTy.getShape();
  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();

  SmallVector<int> instData(resShapeSize, 1);
  SmallVector<int> laneLayout(resShapeSize, 1);
  SmallVector<int> laneData(resShapeSize, 1);

  const unsigned packingSize{uArch->getGeneralPackedFormatBitSize()};
  unsigned bitwidth = resVectorTy.getElementType().getIntOrFloatBitWidth();
  int packingFactor = bitwidth < packingSize ? packingSize / bitwidth : 1;
  int packedDataSize = subgroupSize * packingFactor;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(true &&
           "subgroup layout assignment not supported for insertStridedSlice.");
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    assert(srcShape.back() >= subgroupSize &&
           "source innermost dim must be >= subgroupSize");
    instData.back() = subgroupSize;
    if (consumerInstData.back() == packedDataSize &&
        srcShape.back() >= packedDataSize)
      instData.back() = packedDataSize;
    requiredResLayout = xegpu::LayoutAttr::get(context, instData);
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    laneLayout.back() = subgroupSize;
    laneData.back() = 1;
    if (consumerLaneData.back() == packingFactor &&
        srcShape.back() >= packedDataSize)
      laneData.back() = packingFactor;
    requiredResLayout = xegpu::LayoutAttr::get(context, laneLayout, laneData);
  }
  return requiredResLayout;
}

/// Sets up the anchor layout for load gather and load matrix operation.
/// load matrix lowers to load gather and 1d block load. All of them share the
/// same layout setup logic.
/// For Subgroup layout, uses the consumer layout directly.
/// non-chunked loads:
///   InstData = {1, ..., min(consumer, maxLaneLoadSize * subgroupSize)}
///   LaneLayout = {1, ..., subgroupSize}
///   lane_data = {1, ..., min(consumer, maxLaneLoadSize)}
/// chunked loads:
///   InstData = {subgroupSize, min(consumer, maxLaneLoadSize)}
///   LaneLayout = {subgroupSize, 1}
///   lane_data={1,min(consumer, maxLaneLoadSize)}
static xegpu::DistributeLayoutAttr setupGenericLoadAnchorLayout(
    xegpu::LayoutKind layoutKind, mlir::MLIRContext *context,
    xegpu::DistributeLayoutAttr consumerLayout, bool isChunkedLoad,
    int maxChunkSize, int valShapeSize, int subgroupSize) {

  if (layoutKind == xegpu::LayoutKind::Subgroup)
    return consumerLayout;

  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();

  SmallVector<int> instData(valShapeSize, 1);
  SmallVector<int> laneLayout(valShapeSize, 1);
  SmallVector<int> laneData(valShapeSize, 1);

  if (!isChunkedLoad) {
    if (layoutKind == xegpu::LayoutKind::InstData) {
      instData[valShapeSize - 1] =
          std::min(static_cast<int>(consumerInstData[valShapeSize - 1]),
                   maxChunkSize * subgroupSize);
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneLayout.back() = subgroupSize;
      laneData.back() =
          std::min(static_cast<int>(consumerLaneData.back()), maxChunkSize);
      return xegpu::LayoutAttr::get(context, laneLayout, laneData);
    }
  } else {
    assert(valShapeSize == 2 && "Chunked Store must access 2D tensor tile.");
    if (layoutKind == xegpu::LayoutKind::InstData) {
      instData[0] = subgroupSize;
      instData[1] =
          std::min(static_cast<int>(consumerInstData[1]), maxChunkSize);
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneLayout[0] = subgroupSize;
      laneData[1] =
          std::min(static_cast<int>(consumerLaneData[1]), maxChunkSize);
      return xegpu::LayoutAttr::get(context, laneLayout, laneData);
    }
  }
  return nullptr;
}

/// Sets up the anchor layout for a load gather operation.
xegpu::DistributeLayoutAttr xegpu::setupLoadGatherAnchorLayout(
    xegpu::LayoutKind layoutKind, VectorType resVecTy, int chunkSize,
    xegpu::DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  int resShapeSize = resVecTy.getShape().size();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SpirvLoadGatherInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize = uArchInstruction->getMaxLaneLoadSize(elemBitWidth);

  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      (chunkSize > 1), maxChunkSize,
                                      resShapeSize, subgroupSize);
}

/// Sets up the anchor layout for load matrix operation.
/// TODO: enhance load matrix to indicate lowering to chunked load or not.
xegpu::DistributeLayoutAttr
xegpu::setupLoadMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                   VectorType resVecTy,
                                   xegpu::DistributeLayoutAttr consumerLayout,
                                   const xegpu::uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  int resShapeSize = resVecTy.getShape().size();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction = dyn_cast<xegpu::uArch::LoadMatrixInstruction>(
      uArch->getInstruction(xegpu::uArch::InstructionKind::LoadMatrix));
  int maxChunkSize = uArchInstruction->getMaxLaneLoadSize(elemBitWidth);
  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      false, maxChunkSize, resShapeSize,
                                      subgroupSize);
}

/// Sets up the anchor layout for store scatter and store matrix operation.
/// store matrix lowers to store scatter and 1d block store. All of them share
/// the same layout setup logic. For Subgroup layout, not support yet.
/// non-chunked stores:
///   InstData = {1, ..., subgroupSize}
///   LaneLayout = {1, ..., subgroupSize}
///   lane_data = {1, ..., 1}
/// chunked stores:
///   InstData = {subgroupSize, min(srcVec, maxLaneStoreSize)}
///   LaneLayout = {subgroupSize, 1}
///   lane_data={1,min(srcVec, maxLaneStoreSize)}
static xegpu::DistributeLayoutAttr
setupGenericStoreAnchorLayout(xegpu::LayoutKind layoutKind,
                              mlir::MLIRContext *context, bool isChunkedStore,
                              int maxChunkSize, ArrayRef<int64_t> srcShape,
                              int subgroupSize) {

  int srcShapeSize = srcShape.size();
  SmallVector<int> instData(srcShapeSize, 1);
  SmallVector<int> laneLayout(srcShapeSize, 1);
  SmallVector<int> laneData(srcShapeSize, 1);

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(true &&
           "subgroup layout assignment not supported for storeScatter.");
    return nullptr;
  }

  if (!isChunkedStore) {
    if (layoutKind == xegpu::LayoutKind::InstData) {
      instData[srcShapeSize - 1] = subgroupSize;
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneLayout[srcShapeSize - 1] = subgroupSize;
      return xegpu::LayoutAttr::get(context, laneLayout, laneData);
    }
  } else {
    assert(srcShapeSize == 2 && "Chunked Store must access 2D tensor tile.");
    if (layoutKind == xegpu::LayoutKind::InstData) {
      instData[0] = subgroupSize;
      instData[1] = std::min(static_cast<int>(srcShape[1]), maxChunkSize);
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneLayout[0] = subgroupSize;
      laneData[1] = std::min(static_cast<int>(srcShape[1]), maxChunkSize);
      return xegpu::LayoutAttr::get(context, laneLayout, laneData);
    }
  }
  return nullptr;
}

/// Sets up the anchor layout for a store scatter operation.
xegpu::DistributeLayoutAttr
xegpu::setupStoreScatterAnchorLayout(xegpu::LayoutKind layoutKind,
                                     VectorType srcVecTy, int chunkSize,
                                     const uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  auto context = srcVecTy.getContext();
  auto elemBitWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SpirvStoreScatterInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize = uArchInstruction->getMaxLaneStoreSize(elemBitWidth);
  return setupGenericStoreAnchorLayout(layoutKind, context, (chunkSize > 1),
                                       maxChunkSize, srcShape, subgroupSize);
}

/// Sets up the anchor layout for a store matrix operation.
xegpu::DistributeLayoutAttr
xegpu::setupStoreMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                    VectorType srcVecTy,
                                    const xegpu::uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  auto context = srcVecTy.getContext();
  auto elemBitWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction = dyn_cast<xegpu::uArch::StoreMatrixInstruction>(
      uArch->getInstruction(xegpu::uArch::InstructionKind::StoreMatrix));
  int maxChunkSize = uArchInstruction->getMaxLaneStoreSize(elemBitWidth);

  return setupGenericStoreAnchorLayout(layoutKind, context, false, maxChunkSize,
                                       srcShape, subgroupSize);
}

// This function returns the default lane layout for a given vector type.
// - `packingSize` means multiple consecutive elements can be accessed together
// as a single unit.
// - `vnni` means data packing is column-wise (i.e., 2x1xf16 with vnni vs.
// 1x2xf16 w/o vnni).
template <typename RankedTy>
static xegpu::LayoutAttr getDefaultLaneLayout2DBlockIo(
    RankedTy ty, const xegpu::uArch::uArch *uArch,
    std::optional<unsigned> packingSize = std::nullopt, bool vnni = false) {
  // Expecting a 1D or 2D vector.
  assert(((ty.getRank() == 1 && !vnni) || ty.getRank() == 2) &&
         "Expected 1D non-vnni or 2D vector.");
  // Expecting int or float element type.
  assert(ty.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");

  auto context = ty.getContext();
  auto rank = ty.getRank();
  SmallVector<int> laneLayout(rank, 1);
  SmallVector<int> laneData(rank, 1);
  if (packingSize.has_value()) {
    unsigned bitwidth = ty.getElementType().getIntOrFloatBitWidth();
    int &laneDataPos = vnni ? laneData[rank - 2] : laneData.back();
    laneDataPos = bitwidth < *packingSize ? *packingSize / bitwidth : 1;
  }
  laneLayout.back() = uArch->getSubgroupSize();
  return xegpu::LayoutAttr::get(context, laneLayout, laneData);
}

// This function returns all layouts for the given sgCount, whose sgData:
// 1. Evenly divides the wgShape.
// 2. Is a multiple of instData.
// Example:
//   wgShape = [128, 64], instData = [8, 16], sgCount = 32
// Returns layouts:
//   [(8,4), (16,2)], which correspond to sgData [16,16] and [8,32].
using LayoutRepresentation = std::pair<int64_t, int64_t>;
static SmallVector<LayoutRepresentation>
getValidLayouts(ArrayRef<int64_t> wgShape, ArrayRef<int64_t> instData,
                int64_t sgCount) {
  SmallVector<LayoutRepresentation> candidates;
  for (int sgLayout0 = 1; sgLayout0 <= sgCount; ++sgLayout0) {
    if (sgCount % sgLayout0)
      continue;
    int64_t sgLayout1 = sgCount / sgLayout0;
    int64_t sgData0 = wgShape[0] / sgLayout0;
    int64_t sgData1 = wgShape[1] / sgLayout1;
    if ((wgShape[0] % sgLayout0 || wgShape[1] % sgLayout1) ||
        (sgData0 % instData[0] || sgData1 % instData[1]))
      continue;
    candidates.emplace_back(sgLayout0, sgLayout1);
  }
  // Sort primarily by how balanced they are
  // (i.e., minimize the absolute difference between the two dimensions), and
  // secondarily by the first dimension in ascending order.
  llvm::sort(candidates, [](const LayoutRepresentation &lhs,
                            const LayoutRepresentation &rhs) {
    int diffLhs = std::abs(lhs.first - lhs.second);
    int diffRhs = std::abs(rhs.first - rhs.second);
    if (diffLhs != diffRhs)
      return diffLhs < diffRhs;
    return lhs.first < rhs.first;
  });
  return candidates;
}

/// Sets up the anchor layouts for dpas operands (A, B, and C/D).
/// The numSg and consumerLayout (optional) are only used by sg layout creation.
std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
xegpu::setupDpasLayout(xegpu::LayoutKind layoutKind, VectorType aTy,
                       VectorType bTy, VectorType cdTy,
                       xegpu::DistributeLayoutAttr consumerLayout,
                       const xegpu::uArch::uArch *uArch, int numSg) {
  auto context = aTy.getContext();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
          xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));

  auto getInstDataVectors = [&]()
      -> std::optional<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                                  SmallVector<int64_t>>> {
    const int subgroupSize = uArch->getSubgroupSize();
    const unsigned dataALen = aTy.getShape().front();
    auto supportedALen = uArchInstruction->getSupportedM(aTy.getElementType());
    const int maxALen =
        xegpu::getLargestDivisor(dataALen, ArrayRef<unsigned>(supportedALen));

    const unsigned dataBLen = bTy.getShape().back();
    auto supportedBLen = uArchInstruction->getSupportedN(bTy.getElementType());
    const int maxBLen =
        xegpu::getLargestDivisor(dataBLen, ArrayRef<unsigned>(supportedBLen));

    auto supportedCLen = uArchInstruction->getSupportedN(cdTy.getElementType());
    const int maxCLen =
        xegpu::getLargestDivisor(dataBLen, ArrayRef<unsigned>(supportedCLen));
    if (maxALen == -1 || maxBLen == -1 || maxCLen == -1)
      return std::nullopt;

    SmallVector<int64_t> instDataA(aTy.getRank(), 1);
    instDataA[aTy.getRank() - 2] = maxALen;
    instDataA[aTy.getRank() - 1] = subgroupSize;
    SmallVector<int64_t> instDataB(bTy.getRank(), 1);
    instDataB[bTy.getRank() - 2] = subgroupSize;
    instDataB[bTy.getRank() - 1] = maxBLen;
    SmallVector<int64_t> instDataCD(cdTy.getRank(), 1);
    instDataCD[cdTy.getRank() - 2] = maxALen;
    instDataCD[cdTy.getRank() - 1] = maxCLen;
    return std::make_tuple(instDataA, instDataB, instDataCD);
  };

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto instDataVecs = getInstDataVectors();
    if (!instDataVecs)
      return std::nullopt;
    auto [instDataA, instDataB, instDataCD] = *instDataVecs;
    assert(instDataA.size() == 2 && instDataB.size() == 2 &&
           instDataCD.size() == 2 &&
           "Sg layout creation expects valid 2D inst data");

    std::optional<LayoutRepresentation> consumerSgLayout = std::nullopt;
    if (consumerLayout && consumerLayout.isForWorkgroup()) {
      SmallVector<int64_t> sgLayoutD =
          consumerLayout.getEffectiveSgLayoutAsInt();
      consumerSgLayout = std::make_pair(sgLayoutD[0], sgLayoutD[1]);
    }

    // Step 1. Get all valid layouts for A, B and C/D operands.
    // Order them from most balanced to least balanced.
    auto layoutsA = getValidLayouts(aTy.getShape(), instDataA, numSg);
    auto layoutsB = getValidLayouts(bTy.getShape(), instDataB, numSg);
    auto layoutsCD = getValidLayouts(cdTy.getShape(), instDataCD, numSg);
    if (layoutsA.empty() || layoutsB.empty() || layoutsCD.empty())
      return std::nullopt;

    // Step 2. If the consumer layout can be reused for all operands, that
    // layout is chosen. Otherwise, pick the most balanced subgroup layout
    // that is valid for A, B and C (if present) operands
    llvm::DenseSet<LayoutRepresentation> setA(layoutsA.begin(), layoutsA.end());
    llvm::DenseSet<LayoutRepresentation> setCD(layoutsCD.begin(),
                                               layoutsCD.end());
    std::optional<LayoutRepresentation> bestPick;
    for (auto &sgLayout : layoutsB) {
      if (setA.contains(sgLayout) && setCD.contains(sgLayout)) {
        // Is in (A and B and CD) and matches consumer -> best pick
        if (consumerSgLayout.has_value() && sgLayout == *consumerSgLayout) {
          bestPick = sgLayout;
          break;
        }
        // Is in (A and B and CD) layoutsB is ordered from most
        // balanced to least. So the first one we see is the most balanced one,
        // remember it and later only update if there is one that matches the
        // consumer.
        if (!bestPick)
          bestPick = sgLayout;
      }
    }
    // Step 3. If there is no subgroup layout compatible with A, B and C (if
    // present) operands, we fail.
    if (!bestPick)
      return std::nullopt;
    SmallVector<int> sgLayout = {static_cast<int>(bestPick->first),
                                 static_cast<int>(bestPick->second)};
    SmallVector<int> sgDataA = {
        static_cast<int>(aTy.getShape()[0] / sgLayout[0]),
        static_cast<int>(aTy.getShape()[1] / sgLayout[1])};
    SmallVector<int> sgDataB = {
        static_cast<int>(bTy.getShape()[0] / sgLayout[0]),
        static_cast<int>(bTy.getShape()[1] / sgLayout[1])};
    SmallVector<int> sgDataCD = {
        static_cast<int>(cdTy.getShape()[0] / sgLayout[0]),
        static_cast<int>(cdTy.getShape()[1] / sgLayout[1])};

    auto dpasALayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, sgLayout),
        DenseI32ArrayAttr::get(context, sgDataA),
        /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
        /*lane_data =*/nullptr, /*order =*/nullptr);

    auto dpasBLayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, sgLayout),
        DenseI32ArrayAttr::get(context, sgDataB),
        /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
        /*lane_data =*/nullptr, /*order =*/nullptr);

    auto dpasCDLayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, sgLayout),
        DenseI32ArrayAttr::get(context, sgDataCD),
        /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
        /*lane_data =*/nullptr, /*order =*/nullptr);
    return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout);
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    auto instDataVecs = getInstDataVectors();
    if (!instDataVecs)
      return std::nullopt;
    auto [instDataA, instDataB, instDataCD] = *instDataVecs;
    return std::make_tuple(
        xegpu::LayoutAttr::get(
            context, SmallVector<int>(instDataA.begin(), instDataA.end())),
        xegpu::LayoutAttr::get(
            context, SmallVector<int>(instDataB.begin(), instDataB.end())),
        xegpu::LayoutAttr::get(
            context, SmallVector<int>(instDataCD.begin(), instDataCD.end())));
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    auto aLayout = getDefaultLaneLayout2DBlockIo(
        aTy, uArch, uArchInstruction->getPackedFormatBitSizeA());
    auto bLayout = getDefaultLaneLayout2DBlockIo(
        bTy, uArch, uArchInstruction->getPackedFormatBitSizeB(), true);
    auto cdLayout = getDefaultLaneLayout2DBlockIo(
        cdTy, uArch, uArchInstruction->getPackedFormatBitSizeB());
    return std::make_tuple(aLayout, bLayout, cdLayout);
  }
  return std::nullopt;
}
