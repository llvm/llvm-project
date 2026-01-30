//===---- XeGPUUtils.cpp - MLIR Utilities for XeGPUOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpls.h"
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
      auto layout = xegpu::getDistributeLayoutAttr(operand.get());
      if (!layout) {
        op->emitError("Could not find layout attribute for operand ")
            << operand.getOperandNumber() << " of operation " << op->getName();
        return WalkResult::interrupt();
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
    for (OpOperand &opr : nestOp->getOpOperands())
      removeLayoutAttr(opr);
    for (OpResult result : nestOp->getOpResults())
      removeLayoutAttr(result);
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout"))
      op->removeAttr("layout");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_a"))
      op->removeAttr("layout_a");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_b"))
      op->removeAttr("layout_b");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_cd"))
      op->removeAttr("layout_cd");
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

  // Hanlding broadcast from low-rank to high-rank (e.g., 1D to 2D) case.
  int dimDiff = resShape.size() - srcShape.size();

  if (dimDiff > 0) {
    // adding the missing leading dims
    for (int i = 0; i < dimDiff; i++)
      bcastDims.push_back(i);

    // create a slice layout for the source
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

  //  assert the resLayout must be slice layout
  assert(isa<xegpu::SliceAttr>(resLayout) &&
         "reduction result layout must be slice layout");

  // assert that the reduceDims must match with the slice dims of resLayout
  xegpu::SliceAttr sliceLayout = dyn_cast<xegpu::SliceAttr>(resLayout);
  auto sliceDims = sliceLayout.getDims().asArrayRef();
  assert(reduceDims == sliceDims &&
         "reduction dims must match with slice dims");

  //  then return the parent layout of sliceLayout
  return sliceLayout.getParent();
}

/// Infers the source layout attribute for a bitcast operation given the
/// result layout attribute, result element type bitwidth, and source element
/// type bitwidth.
xegpu::DistributeLayoutAttr
xegpu::inferBitCastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                int resElemTyBitWidth, int srcElemTyBitWidth) {
  // the result and source layout must be the same
  // only adjust the sg_data, inst_data, lane_data accordingly
  // based on the bitwidth ratio between source and result element type

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
      sgDataValue = sgData[sgDataSize - 1] * bitWidthRatio;
    if (instDataSize)
      instDataValue = instData[instDataSize - 1] * bitWidthRatio;
    if (laneDataSize)
      laneDataValue = laneData[laneDataSize - 1] * bitWidthRatio;
  } else {
    int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
    if (sgDataSize) {
      assert((sgData[sgDataSize - 1] % bitWidthRatio) == 0 &&
             "sgData not divisible by bitWidthRatio");
      sgDataValue = sgData[sgDataSize - 1] / bitWidthRatio;
    }
    if (instDataSize) {
      assert((instData[instDataSize - 1] % bitWidthRatio) == 0 &&
             "instData not divisible by bitWidthRatio");
      instDataValue = instData[instDataSize - 1] / bitWidthRatio;
    }
    if (laneDataSize) {
      assert((laneData[laneDataSize - 1] % bitWidthRatio) == 0 &&
             "laneData not divisible by bitWidthRatio");
      laneDataValue = laneData[laneDataSize - 1] / bitWidthRatio;
    }
  }

  // Now set only instData and laneData, preserving sgData
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

  // assert resLayout must be a plain layout
  assert(isa<xegpu::LayoutAttr>(resLayout) &&
         "insertStridedSlice result layout must be plain layout");
  auto context = resLayout.getContext();
  auto resInstData = resLayout.getEffectiveInstDataAsInt();
  auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
  auto resLaneData = resLayout.getEffectiveLaneDataAsInt();

  if (resInstData.size() != 0) {
    SmallVector<int> inferredInstData(srcShapeSize);
    // remove the initial dims in resInstData to match srcShapeSize
    for (int i = 0; i < srcShapeSize; i++)
      inferredInstData[i] = resInstData[i + dimDiff];
    return xegpu::LayoutAttr::get(context, inferredInstData);
  }

  if (resLaneLayout.size() != 0) {
    // construct source lane_layout like [1, ..., 1, subgroupSize]
    SmallVector<int> inferredLaneLayout(srcShapeSize);
    SmallVector<int> inferredLaneData(srcShapeSize);
    // remove the initial dims in resInstData to match srcShapeSize
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

  // there are three use cases:
  // 1. expand dims of low-rank dimensions (e.g., 1D to 2D): to set up the
  // tensor before broadcast
  // 2. split dim of a high-rank dimension (e.g., 1D to 2D): to setup tensor
  // for multi-stage reduction
  // 3. combines all dims to a single dim and put in the innermost dim in 2d as
  // [1, combinedData] or [combinedData]. Only used after workgroup
  // distribution. Example like cross-sg reduction saves multidimension data to
  // 1D slm buffer, shapecast inserted by cse/canonicalization passes.

  // Use case 1: Check if shapes only differ by expanding unit dimensions (like
  // expand_dims)
  SmallVector<int64_t> expandedUnitDims;
  auto checkOnlyExpandUnitDims = [&](ArrayRef<int64_t> src,
                                     ArrayRef<int64_t> dst) -> bool {
    // All unit dimensions in dst that don't appear in src are the expanded
    // unit dimensions
    size_t srcIdx = 0;
    for (size_t dstIdx = 0; dstIdx < dst.size(); ++dstIdx)
      if (srcIdx < src.size() && src[srcIdx] == dst[dstIdx])
        srcIdx++;
      else if (dst[dstIdx] == 1)
        expandedUnitDims.push_back(dstIdx);
      else
        return false;
    return srcIdx == src.size();
  };

  if (checkOnlyExpandUnitDims(srcShape, resShape)) {
    // create a slice layout for the source by removing the expanded unit dims
    auto sliceDimsAttr = DenseI64ArrayAttr::get(
        resLayout.getContext(), ArrayRef<int64_t>(expandedUnitDims));
    auto srcLayout =
        xegpu::SliceAttr::get(resLayout.getContext(), resLayout, sliceDimsAttr);
    return srcLayout;
  }

  // Maps each source dimension to the range of destination dimensions it splits
  // into
  SmallVector<SmallVector<int64_t>> splitDimGroups;

  auto checkSplitDims = [&](ArrayRef<int64_t> src,
                            ArrayRef<int64_t> dst) -> bool {
    // each dim in src can be mapped to one or more dims in dst whose product
    // equals to the src dim
    splitDimGroups.clear();
    size_t srcIdx = 0;
    int64_t accumulatedSize = 1;
    SmallVector<int64_t> currentDstDims;

    for (size_t dstIdx = 0; dstIdx < dst.size(); ++dstIdx) {
      if (srcIdx >= src.size())
        return false;
      accumulatedSize *= dst[dstIdx];
      currentDstDims.push_back(dstIdx);

      if (accumulatedSize == src[srcIdx]) {
        // Record the mapping: srcIdx -> currentDstDims
        splitDimGroups.push_back(currentDstDims);
        // move to next src dim
        srcIdx++;
        accumulatedSize = 1;
        currentDstDims.clear();
      } else if (accumulatedSize > src[srcIdx]) {
        return false;
      }
    }
    return srcIdx == src.size();
  };

  if (checkSplitDims(srcShape, resShape)) {
    return resLayout.collapseDims(splitDimGroups);
  }

  auto checkCombineToInnerMostDim = [&](ArrayRef<int64_t> src,
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

  if (checkCombineToInnerMostDim(srcShape, resShape)) {
    int srcShapeSize = srcShape.size();
    int resShapeSize = resShape.size();
    auto context = resLayout.getContext();
    auto resInstData = resLayout.getEffectiveInstDataAsInt();
    auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
    auto resLaneData = resLayout.getEffectiveLaneDataAsInt();

    // get the layout info from the innermost dim of result layout
    if (resInstData.size() != 0) {
      SmallVector<int> inferredInstData(srcShapeSize, 1);
      assert((resShapeSize == 1 || resInstData[0] == 1) &&
             "only innermost dim can have data and instData layout");
      inferredInstData[srcShapeSize - 1] = resInstData[resShapeSize - 1];
      return xegpu::LayoutAttr::get(context, inferredInstData);
    }

    if (resLaneLayout.size() != 0) {
      SmallVector<int> inferredLaneLayout(srcShapeSize, 1);
      SmallVector<int> inferredLaneData(srcShapeSize, 1);
      assert((resShapeSize == 1 || resLaneLayout[0] == 1) &&
             "only innermost dim can have data and lane layout");
      inferredLaneLayout[srcShapeSize - 1] = resLaneLayout[resShapeSize - 1];
      inferredLaneData[srcShapeSize - 1] = resLaneData[resShapeSize - 1];
      return xegpu::LayoutAttr::get(context, inferredLaneLayout,
                                    inferredLaneData);
    }
  }
  assert("running into unsupported shape cast scenarios");
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
/// avoid subgroup data redistribution overhead between the reduced result and
/// its consumer.
///
/// InstData requries {1, ..., min(maxReduceVectorSize, srcShape),subgroupSize}
/// Lane Layout requires {1, ..., 1, subgroupSize}
/// Lane data requires {1, ..., min(maxReduceVectorSize, srcShape), 1}

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

  auto sgLayoutVec = plainLayout.getEffectiveSgLayoutAsInt();
  const int workgroupSize = std::accumulate(
      sgLayoutVec.begin(), sgLayoutVec.end(), 1, std::multiplies<int64_t>());
  const int subgroupSize = uArch->getSubgroupSize();
  int64_t maxReduceVectorSize = 1; // could extend to spirv vector Size

  xegpu::DistributeLayoutAttr srcLayout;

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup: {
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
        sgLayout[i] = std::min(srcShape[i] / subgroupSize,
                               static_cast<int64_t>(remainingSgCount));
        assert((srcShape[i] % sgLayout[i] == 0) &&
               "source shape not divisible by sg_layout");
        sgData[i] = srcShape[i] / sgLayout[i];
        remainingSgCount /= sgLayout[i];
      }
    }

    assert(remainingSgCount == 1 && "not all subgroups distributed");
    srcLayout =
        xegpu::LayoutAttr::get(context, toInt32Attr(sgLayout),
                               toInt32Attr(sgData), consumerLayout.getOrder());
    break;
  }

  case xegpu::LayoutKind::InstData: {
    SmallVector<int64_t> instData(srcRank, 1);
    instData[srcRank - 2] =
        std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    instData[srcRank - 1] = subgroupSize;
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(instData));
    break;
  }

  case xegpu::LayoutKind::Lane: {
    SmallVector<int64_t> laneLayout(srcRank, 1), laneData(srcRank, 1);
    laneLayout[srcRank - 1] = subgroupSize;
    laneData[srcRank - 2] =
        std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(laneLayout),
                                       toInt32Attr(laneData),
                                       consumerLayout.getOrder());
    break;
  }

  default:
    llvm_unreachable("unsupported layout kind");
  }

  return xegpu::SliceAttr::get(context, srcLayout,
                               DenseI64ArrayAttr::get(context, reductionDims));
}

/// Sets up the result layout for a bitcast operation.
/// When casting to a smaller bitwidth, adjusts the layout dimensions (sgData,
/// instData, or laneData) by multiplying by the bitwidth ratio to ensure the
/// result layout can be correctly divided back to the source layout during
/// inference.
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
    switch (layoutKind) {
    case xegpu::LayoutKind::Subgroup:
      assert(sgData.size() == srcShape.size() &&
             "sgData must be available for all dimensions");
      sgDataValue = sgData[dim];
      break;
    case xegpu::LayoutKind::InstData:
      assert(instData.size() == srcShape.size() &&
             "instData must be available for all dimensions");
      instDataValue = instData[dim];
      // adjust instDataValue so it still fits within an instruction after
      // dividing by bitWidthRatio
      while ((instDataValue <= srcShape[dim]) &&
             (instDataValue % (innermostDimLaneLayout * bitWidthRatio) != 0))
        instDataValue *= 2;
      assert((srcShape[dim] % instDataValue) == 0 &&
             "srcShape, instData, and lanelayout for innermost must be 2^n !");
      break;
    case xegpu::LayoutKind::Lane:
      assert(laneData.size() == srcShape.size() &&
             "laneData must be available for all dimensions");
      laneDataValue = laneData[dim];
      while ((laneDataValue <= srcShape[dim]) &&
             (laneDataValue % bitWidthRatio != 0))
        laneDataValue *= 2;
      break;
    default:
      llvm_unreachable("unsupported layout kind");
    }
    // Now set only instData and laneData, preserving sgData
    xegpu::DistributeLayoutAttr resLayout;
    resLayout = consumerLayout.setDimData(dim, sgDataValue, instDataValue,
                                          laneDataValue);
    return resLayout;
  }
  return consumerLayout;
}

xegpu::DistributeLayoutAttr
xegpu::setupLoadMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                   VectorType resVectorTy,
                                   xegpu::DistributeLayoutAttr consumerLayout,
                                   const xegpu::uArch::uArch *uArch) {
  xegpu::DistributeLayoutAttr requiredLayout;
  auto subgroupSize = uArch->getSubgroupSize();
  SmallVector<int> defaultInstData = {1, subgroupSize};
  SmallVector<int> defaultLaneLayout = {1, subgroupSize};
  SmallVector<int> defaultLaneData = {1, 1};
  auto context = resVectorTy.getContext();

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup:
    requiredLayout = consumerLayout;
    break;
  case xegpu::LayoutKind::InstData:
    requiredLayout = xegpu::LayoutAttr::get(context, defaultInstData);
    break;
  case xegpu::LayoutKind::Lane:
    requiredLayout =
        xegpu::LayoutAttr::get(context, defaultLaneLayout, defaultLaneData);
    break;
  default:
    llvm_unreachable("unsupported layout kind");
  }
  return requiredLayout;
}

xegpu::DistributeLayoutAttr
xegpu::setupStoreMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                    VectorType srcVectorTy,
                                    const xegpu::uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredLayout;
  auto subgroupSize = uArch->getSubgroupSize();
  SmallVector<int> defaultInstData = {1, subgroupSize};
  SmallVector<int> defaultLaneLayout = {1, subgroupSize};
  SmallVector<int> defaultLaneData = {1, 1};
  auto context = srcVectorTy.getContext();

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup:
    assert(true &&
           "subgroup layout assignment not supported yet for storeMatrix.");
    break;
  case xegpu::LayoutKind::InstData:
    requiredLayout = xegpu::LayoutAttr::get(context, defaultInstData);
    break;
  case xegpu::LayoutKind::Lane:
    requiredLayout =
        xegpu::LayoutAttr::get(context, defaultLaneLayout, defaultLaneData);

    break;
  default:
    llvm_unreachable("unsupported layout kind");
  }
  return requiredLayout;
}

/// Sets up the result layout for an insert strided slice operation.
/// Creates a default layout based on the specified layout kind (InstData or
/// Lane).
/// Subgroup layout is currently not supported for this operation.
/// InstData layout requires {1, .., subgroupSize} by default.
/// Lane layout requires {1, ..., subgroupSize} with lane data {1, ..., 1}.
/// The instData and laneData is adjusted to contain packed data, by checking if
/// the consumerLayout's innermost dimension.
xegpu::DistributeLayoutAttr xegpu::setupInsertStridedSliceResultLayout(
    xegpu::LayoutKind layoutKind, VectorType resVectorTy,
    xegpu::DistributeLayoutAttr consumerLayout,
    const xegpu::uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredResLayout;
  auto subgroupSize = uArch->getSubgroupSize();
  auto context = resVectorTy.getContext();
  auto resShape = resVectorTy.getShape();
  int resShapeSize = resShape.size();
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

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup:
    assert(true &&
           "subgroup layout assignment not supported for insertStridedSlice.");
    break;
  case xegpu::LayoutKind::InstData:
    instData[resShapeSize - 1] = subgroupSize;
    if (consumerInstData[resShapeSize - 1] == packedDataSize)
      instData[resShapeSize - 1] = packedDataSize;
    requiredResLayout = xegpu::LayoutAttr::get(context, instData);
    break;
  case xegpu::LayoutKind::Lane:
    laneLayout[resShapeSize - 1] = subgroupSize;
    laneData[resShapeSize - 1] = 1;
    if (consumerLaneData[resShapeSize - 1] == packingFactor)
      laneData[resShapeSize - 1] = packingFactor;
    requiredResLayout = xegpu::LayoutAttr::get(context, laneLayout, laneData);
    break;
  default:
    llvm_unreachable("unsupported layout kind");
  }
  return requiredResLayout;
}

static xegpu::DistributeLayoutAttr
getDefaultLaneLayoutAttr(mlir::MLIRContext *ctx, unsigned rank,
                         const xegpu::uArch::uArch *uArch) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1) {
    return xegpu::LayoutAttr::get(ctx, {uArch->getSubgroupSize()}, {1});
  }
  return xegpu::LayoutAttr::get(ctx, {1, uArch->getSubgroupSize()}, {1, 1});
}

xegpu::DistributeLayoutAttr xegpu::setupLoadGatherAnchorLayout(
    LayoutKind layoutKind, VectorType resVecTy, int chunkSize,
    DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredLayout;
  const int subgroupSize = uArch->getSubgroupSize();

  auto resShape = resVecTy.getShape();
  int resShapeSize = resShape.size();
  SmallVector<int> instData(resShapeSize);
  auto context = resVecTy.getContext();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));

  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int32_t> instData32;

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup:
    requiredLayout = consumerLayout;
    break;
  case xegpu::LayoutKind::InstData:
    if (resVecTy.getRank() == 1) {
      instData[0] = subgroupSize;
    } else {
      assert((resVecTy.getRank() == 2) && "StoreScatterOp can access 2D tensor "
                                          "tile at maximum at subgroup level.");
      if (chunkSize == 1) {
        instData[0] = 1;
        instData[1] = subgroupSize;
      } else {
        instData[0] = subgroupSize;
        instData[1] = std::min(static_cast<int>(resShape[1]),
                               uArchInstruction->getMaxLaneLoadStoreSize());
        instData[1] =
            std::min(instData[1], static_cast<int>(consumerInstData[1]));
      }
    }
    requiredLayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, instData));
    break;
  case xegpu::LayoutKind::Lane:
    if (chunkSize == 1)
      requiredLayout =
          getDefaultLaneLayoutAttr(context, resVecTy.getRank(), uArch);
    else {
      assert((resVecTy.getRank() <= 2) && "StoreScatterOp can access 2D tensor "
                                          "tile at maximum at subgroup level.");
      assert(resShape[1] <= static_cast<int64_t>(
                                uArchInstruction->getMaxLaneLoadStoreSize()) &&
             "StoreScatterOp lane size exceeds max lane load/store size.");
      requiredLayout = xegpu::LayoutAttr::get(
          context, {subgroupSize, 1}, {1, static_cast<int>(resShape[1])});
    }
    break;
  default:
    llvm_unreachable("unsupported layout kind");
  }
  return requiredLayout;
}

xegpu::DistributeLayoutAttr
xegpu::setupStoreScatterAnchorLayout(LayoutKind layoutKind, VectorType srcVecTy,
                                     int chunkSize, const uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredLayout;
  const int subgroupSize = uArch->getSubgroupSize();

  auto srcShape = srcVecTy.getShape();
  int srcShapeSize = srcShape.size();
  SmallVector<int> instData(srcShapeSize);

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  auto context = srcVecTy.getContext();

  switch (layoutKind) {
  case xegpu::LayoutKind::Subgroup:
    assert(
        true &&
        "subgroup layout assignment not supported yet for store scatter op.");
    break;
  case xegpu::LayoutKind::InstData:
    if (srcVecTy.getRank() == 1) {
      instData[0] = subgroupSize;
    } else {
      assert((srcVecTy.getRank() <= 2) && "StoreScatterOp can access 2D tensor "
                                          "tile at maximum at subgroup level.");
      if (chunkSize == 1) {
        instData[0] = 1;
        instData[1] = subgroupSize;
      } else {
        instData[0] = subgroupSize;
        instData[1] = std::min(static_cast<int>(srcShape[1]),
                               uArchInstruction->getMaxLaneLoadStoreSize());
      }
    }
    requiredLayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, instData));
    break;
  case xegpu::LayoutKind::Lane:
    if (chunkSize == 1)
      requiredLayout =
          getDefaultLaneLayoutAttr(context, srcVecTy.getRank(), uArch);
    else {
      assert((srcVecTy.getRank() <= 2) && "StoreScatterOp can access 2D tensor "
                                          "tile at maximum at subgroup level.");
      assert(srcShape[1] <= static_cast<int64_t>(
                                uArchInstruction->getMaxLaneLoadStoreSize()) &&
             "StoreScatterOp lane size exceeds max lane load/store size.");
      requiredLayout = xegpu::LayoutAttr::get(
          context, {subgroupSize, 1}, {1, static_cast<int>(srcShape[1])});
    }
    break;
  default:
    llvm_unreachable("unsupported layout kind");
  }
  return requiredLayout;
}