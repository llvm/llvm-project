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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

std::string xegpu::getTemporaryLayoutName(const OpOperand &operand) {
  const StringRef prefix("layout_operand_");
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();
  return llvm::formatv("{0}{1}", prefix, idx).str();
}

std::string xegpu::getTemporaryLayoutName(const OpResult result) {
  const StringRef prefix = "layout_result_";
  return llvm::formatv("{0}{1}", prefix, result.getResultNumber()).str();
}

xegpu::DistributeLayoutAttr xegpu::getDistributeLayoutAttr(const Value value) {
  if (!value)
    return nullptr;

  if (auto tdescTy =
          dyn_cast_if_present<xegpu::TensorDescType>(value.getType()))
    return tdescTy.getLayoutAttr();

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *defOp = result.getDefiningOp();
    assert(defOp && "result must have a defining op");

    if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(defOp)) {
      auto layout = anchorOp.getAnchorLayout();
      return layout;
    }

    std::string layoutName = getTemporaryLayoutName(result);
    if (defOp->hasAttr(layoutName)) {
      auto layout =
          defOp->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
      return layout;
    }
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      if (tiedInit)
        return getDistributeLayoutAttr(tiedInit->get());
    }
  }

  return nullptr;
}
xegpu::DistributeLayoutAttr
xegpu::getDistributeLayoutAttr(const OpOperand &opr) {
  Operation *op = opr.getOwner();
  unsigned idx = const_cast<OpOperand &>(opr).getOperandNumber();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(op)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op)) {
      if (idx == 0) {
        return dpasOp.getLayoutAAttr();
      } else if (idx == 1) {
        return dpasOp.getLayoutBAttr();
      } else if (idx == 2) {
        return dpasOp.getLayoutCdAttr();
      }
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(op)) {
      return convertOp.getInputLayoutAttr();
    }
    auto layout = anchorOp.getAnchorLayout();

    if (idx == 0)
      return layout;

    // For store operations (StoreScatterOp, StoreNdOp, StoreMatrixOp),
    // the layout is valid for the first two operands: value and memref/tdesc.
    // For other operations, the layout applies to the first operand only.
    if (isa<xegpu::StoreScatterOp, xegpu::StoreNdOp, xegpu::StoreMatrixOp>(
            op) &&
        (idx < 2))
      return layout;
  }

  std::string layoutName = xegpu::getTemporaryLayoutName(opr);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  auto layout = getDistributeLayoutAttr(opr.get());
  return layout;
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout) {
  Operation *owner = result.getOwner();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (anchorOp.getAnchorLayout() == layout)
      return;
    anchorOp.setAnchorLayout(layout);
    return;
  }

  std::string name = xegpu::getTemporaryLayoutName(result);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(const OpOperand &operand,
                                    const DistributeLayoutAttr layout) {
  Operation *owner = operand.getOwner();
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();

  if (!layout) {
    return;
  }
  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(owner)) {
      if (idx == 0) {
        return dpasOp.setLayoutAAttr(layout);
      } else if (idx == 1) {
        return dpasOp.setLayoutBAttr(layout);
      } else if (idx == 2) {
        return dpasOp.setLayoutCdAttr(layout);
      }
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(owner)) {
      return convertOp.setInputLayoutAttr(layout);
    }

    // For store operations (StoreScatterOp, StoreNdOp, StoreMatrixOp),
    // the layout is valid for the first two operands: value and memref/tdesc.
    // For other operations, the layout applies to the first operand only.
    if (isa<xegpu::StoreScatterOp, xegpu::StoreNdOp, xegpu::StoreMatrixOp>(
            owner)) {
      if (idx < 2) {
        anchorOp.setAnchorLayout(layout);
      }
    } else {
      if (idx == 0) {
        anchorOp.setAnchorLayout(layout);
      }
    }
  }

  std::string name = xegpu::getTemporaryLayoutName(operand);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template <typename T, typename>
xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout(const T &operandOrResult) {
  Operation *op = operandOrResult.getOwner();

  std::string layoutName = xegpu::getTemporaryLayoutName(operandOrResult);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  return nullptr;
}

template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpResult>(const OpResult &result);
template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpOperand>(const OpOperand &operand);

template <typename T, typename>
void xegpu::setTemporaryLayout(const T &operandOrResult,
                               const xegpu::DistributeLayoutAttr layout) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getTemporaryLayoutName(operandOrResult);
  if (owner->hasAttrOfType<xegpu::DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template void xegpu::setTemporaryLayout<mlir::OpResult>(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout);

template void xegpu::setTemporaryLayout<mlir::OpOperand>(
    const mlir::OpOperand &operand,
    const mlir::xegpu::DistributeLayoutAttr layout);

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

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's region. Reports an error if any vector operand lacks
/// a layout attribute.
// bool xegpu::recoverTemporaryLayouts(Operation *rootOp) {
//   auto result = rootOp->walk([&](Operation *op) {
//     for (OpOperand &operand : op->getOpOperands()) {
//       // Layouts are needed for vector type only.
//       if (!isa<VectorType>(operand.get().getType()))
//         continue;
//       auto layout = xegpu::getDistributeLayoutAttr(operand.get());
//       if (!layout) {
//         op->emitError("Could not find layout attribute for operand ")
//             << operand.getOperandNumber() << " of operation " <<
//             op->getName();
//         return WalkResult::interrupt();
//       }
//       xegpu::setDistributeLayoutAttr(operand, layout);
//     }
//     return WalkResult::advance();
//   });
//   return !result.wasInterrupted();
// }

// Prerequisite for Layout Recovery
// It relies on the following invariant:
// 1. there is no layout conflict between different uses of the same definition.
// 2. each definition has a well-defined layout requirement at its use point.
//     - Every definition must have at least one use that appears after it in
//     topological order.
//     - If a definition has no such use (e.g., a loop result or region output),
//     an explicit convert_layout operation is inserted to create a use.
//     - Only the result of convert_layout is permitted to have no subsequent
//     use.

// The recover proceeds by scanning the operation in reverse topological order
// as follows:
//    For regular operations: First the result layouts are propagated from uses.
//      Then the result layouts are propagated to uses (operands).
//
//    For region operations (e.g., loops):
//       - When backward propagation reaches a region op, it sets the layout of
//       the region op’s results according to use points like regular ops.
//       - Then, the result layouts (such as a loop output) are propagated to
//       thiers corresponding operands in the yield.
//       - When backward propagation reaches the first operation inside the
//       region, the pass examines the region op’s initialization list,
//       propagating from region arguments to the corresponding initialization
//       operands.
//       - This ensures that layout constraints are consistently propagated
//       across region boundaries
//        while preserving a single well-defined use for each definition at the
//        region-op level.

// Forward declarations
static void walkRegionBackward(Region &region,
                               llvm::function_ref<void(Operation *)> visit);
static void propagateResultsToRegularOperands(Operation *op);
static void propagateRegionResultsToYieldOperands(
    mlir::RegionBranchTerminatorOpInterface yieldOp);
static void propagateRegionArgsToInits(mlir::RegionBranchOpInterface *regionOp);

// the inner function for recoverTemporaryLayouts is a recursive function
// the input rootOp is the function operation, which is also a region op.
// it recursivley process the region op in reverse topological order.
bool xegpu::recoverTemporaryLayouts(Operation *rootOp) {
  rootOp->walk([&](func::FuncOp func) {
    walkRegionBackward(func.getBody(), [&](Operation *op) {
      if (auto regionOp = dyn_cast<mlir::RegionBranchOpInterface>(op)) {
        // hit the region op after visiting inside region
        propagateRegionArgsToInits(&regionOp);
      } else if (auto yieldOp =
                     dyn_cast<mlir::RegionBranchTerminatorOpInterface>(op)) {
        // yield op inside region op
        propagateRegionResultsToYieldOperands(yieldOp);
      } else {
        // if the op is regular op, calling propagateResultsToRegularOperands
        propagateResultsToRegularOperands(op);
      }
    });
  });
}

static void walkRegionBackward(Region &region,
                               llvm::function_ref<void(Operation *)> visit) {
  // blocks: back -> front
  for (Block &block : llvm::reverse(region)) {
    // ops: back -> front, early-inc so visit() may erase current op safely
    for (Operation &op : llvm::reverse(block)) {
      // make sure we first visit inside the region op (so yield op first)
      // and then move to region op itself
      for (Region &nested : llvm::reverse(op.getRegions()))
        walkRegionBackward(nested, visit);

      visit(&op);
    }
  }
}

// For regular operations: First the result layouts are propagated from uses.
// Then the result layouts are propagated to uses (operands).
static void propagateResultsToRegularOperands(Operation *op) {
  OpResult result = op->getOpResults()[0];
  auto resLayout = xegpu::getDistributeLayoutAttr(result);
  assert(resLayout &&
         "result layout must be defined before propagating to uses");

  // if op is reduction op, call inferReductionSourceLayout
  if (auto reduceOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
    SmallVector<int64_t> reduceDims(reduceOp.getReductionDims().begin(),
                                    reduceOp.getReductionDims().end());
    auto srcLayout = xegpu::inferReductionSourceLayout(resLayout, reduceDims);
    // set the layout to the operand
    xegpu::setTemporaryLayout(reduceOp->getOpOperand(0), srcLayout);
    xegpu::setTemporaryLayout(reduceOp->getOpOperand(1), resLayout);
    return;
  }

  // if op is broadcast op, call inferBroadcastSourceLayout
  if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
    ArrayRef<int64_t> resShape =
        llvm::cast<VectorType>(broadcastOp.getResult().getType()).getShape();
    ArrayRef<int64_t> srcShape =
        llvm::cast<VectorType>(broadcastOp.getSource().getType()).getShape();
    auto srcLayout =
        xegpu::inferBroadcastSourceLayout(resLayout, resShape, srcShape);
    // set the layout to the operand
    xegpu::setTemporaryLayout(broadcastOp->getOpOperand(0), srcLayout);
    return;
  }

  // if op is bitcast op, call inferBitCastSourceLayout
  if (auto bitcastOp = dyn_cast<vector::BitCastOp>(op)) {
    int resElemTyBitWidth =
        llvm::cast<VectorType>(bitcastOp.getResult().getType())
            .getElementTypeBitWidth();
    int srcElemTyBitWidth =
        llvm::cast<VectorType>(bitcastOp.getSource().getType())
            .getElementTypeBitWidth();
    auto srcLayout = xegpu::inferBitCastSourceLayout(
        resLayout, resElemTyBitWidth, srcElemTyBitWidth);
    // set the layout to the operand
    xegpu::setTemporaryLayout(bitcastOp->getOpOperand(0), srcLayout);
    return;
  }

  // if op is shape_cast op, call inferShapecastSourceLayout
  if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(op)) {
    ArrayRef<int64_t> resShape =
        llvm::cast<VectorType>(shapeCastOp.getResult().getType()).getShape();
    ArrayRef<int64_t> srcShape =
        llvm::cast<VectorType>(shapeCastOp.getSource().getType()).getShape();
    auto srcLayout =
        xegpu::inferShapecastSourceLayout(resLayout, resShape, srcShape);
    // set the layout to the operand
    xegpu::setTemporaryLayout(shapeCastOp->getOpOperand(0), srcLayout);
    return;
  }

  // if op is a anchor op, no need to do anything
  if (isa<xegpu::AnchorLayoutInterface>(op)) {
    return;
  }

  // for other regular ops, propagate the result layout to all vector operands
  for (OpOperand &opr : op->getOpOperands()) {
    // Layouts are needed for vector type only.
    if (!isa<VectorType>(opr.get().getType()))
      continue;
    xegpu::setTemporaryLayout(opr, resLayout);
  }
}

static void propagateRegionResultsToYieldOperands(
    mlir::RegionBranchTerminatorOpInterface yieldOp) {
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(yieldOp->getNumOperands(),
                                              nullptr);
  yieldOp.getSuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    // find out the successor which is the parent operation
    if (!successor.isParent())
      continue;
    // For parent successor, the region arguments of the current region
    // correspond to the results of the parent operation
    Operation *parentOp = yieldOp->getParentOp();
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      Value parentResult = parentOp->getResult(i);
      auto layout = xegpu::getDistributeLayoutAttr(parentResult);
      assert(
          layout &&
          "region result layout must be defined before propagating to yield");
      xegpu::setTemporaryLayout(yieldOp->getOpOperand(i), layout);
    }
  }
}

void propagateRegionArgsToInits(mlir::RegionBranchOpInterface *regionOp) {

  // Get entry successors (regions that can be entered initially)
  SmallVector<RegionSuccessor> successors;
  regionOp->getEntrySuccessorRegions(/*operands=*/ArrayRef<Attribute>(),
                                     successors);

  // For each possible entry region, get the operands forwarded to it
  for (RegionSuccessor &successor : successors) {
    if (successor.isParent())
      continue;
    OperandRange initOperands = regionOp->getEntrySuccessorOperands(successor);
    // initOperands are the initialization arguments for this successor
    // iterate the region arguments
    Region *successorRegion = successor.getSuccessor();
    for (unsigned i = 0; i < successorRegion->getNumArguments(); ++i) {
      Value regionArg = successorRegion->getArgument(i);
      auto layout = xegpu::getDistributeLayoutAttr(regionArg);
      if (!layout)
        continue;
      // The initOperands are a subset of the parent operation's operands
      // We need to find which operand index this corresponds to
      Value initOperand = initOperands[i];
      // Find the operand index in the parent operation
      for (OpOperand &operand : (*regionOp)->getOpOperands()) {
        if (operand.get() == initOperand) {
          xegpu::setTemporaryLayout(operand, layout);
          break;
        }
      }
    }
  }
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
xegpu::inferReductionSourceLayout(xegpu::DistributeLayoutAttr resLayout,
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
  size_t dim = sgData.size() - 1;
  int64_t sgDataValue, instDataValue, laneDataValue;

  if (srcElemTyBitWidth >= resElemTyBitWidth) {
    int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
    sgDataValue = (dim < sgData.size()) ? sgData[dim] * bitWidthRatio : -1;
    instDataValue =
        (dim < instData.size()) ? instData[dim] * bitWidthRatio : -1;
    laneDataValue =
        (dim < laneData.size()) ? laneData[dim] * bitWidthRatio : -1;
  } else {
    int bitWidthRatio = resElemTyBitWidth / srcElemTyBitWidth;
    assert((laneData[dim] % bitWidthRatio) == 0 &&
           "laneData not divisible by bitWidthRatio");
    sgDataValue = (dim < sgData.size()) ? sgData[dim] / bitWidthRatio : -1;
    instDataValue =
        (dim < instData.size()) ? instData[dim] / bitWidthRatio : -1;
    laneDataValue =
        (dim < laneData.size()) ? laneData[dim] / bitWidthRatio : -1;
  }

  // Now set only instData and laneData, preserving sgData
  xegpu::DistributeLayoutAttr finalSrcLayout;
  finalSrcLayout =
      resLayout.setDimData(dim, sgDataValue, instDataValue, laneDataValue);

  return finalSrcLayout;
}

/// Infers the source layout attribute for a shape cast operation given the
/// result layout attribute, result shape, and source shape.
xegpu::DistributeLayoutAttr
xegpu::inferShapecastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> resShape,
                                  ArrayRef<int64_t> srcShape) {

  // there are three use cases:
  // 1. expand dims of low-rank dimensions (e.g., 1D to 2D): to set up the
  // tensor before broadcast
  // 2. split dim of a high-rank dimension (e.g., 1D to 2D): to setup tensor
  // for multi-stage reduction
  // 3. combines all dims to a single dim and put in the innermost dim in 2d as
  // [1, combinedData]. only used after workgroup distribution to save
  // multidimension data to 1D slm buffer so no need to handle sg_layout and
  // sg_data.

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
    assert((dst.size() == 2) && "dst shape must be 2D");
    int64_t srcSize = std::accumulate(src.begin(), src.end(), 1LL,
                                      std::multiplies<int64_t>());
    return (dst[0] == 1) && (dst[1] == srcSize);
  };

  if (checkCombineToInnerMostDim(srcShape, resShape)) {
    const int subgroupSize = 16; // assuming 16 lanes per subgroup
    const int vectorSize = 8;    // assuming 8 elements per vector lane
    int srcShapeSize = srcShape.size();

    SmallVector<int64_t> instData(srcShapeSize, 1);
    instData[srcShapeSize - 1] = subgroupSize;
    instData[srcShapeSize - 2] =
        vectorSize; // assuming 8 elements per instruction as starting point

    // construct a vector layout with lane_layout = [1, ..., 1, subgroupSize]
    SmallVector<int64_t> laneLayout(srcShapeSize, 1);
    laneLayout[srcShapeSize - 1] = subgroupSize;
    // construct a vector layout with lane_data = [1, ..., 1]
    SmallVector<int64_t> laneData(srcShapeSize, 1);
  }

  // TODO: Complete implementation for other shape cast scenarios
  return nullptr;
}

/// Sets up layout for reduction operations by creating a SliceAttr for the
/// result.
///
/// Algorithm Overview:
/// This function attempts to construct a source layout that, when sliced along
/// reduction dimensions, produces a result layout compatible with the
/// consumer's preferred layout. This minimizes data redistribution overhead
/// between the reduction operation and its consumer.
///
/// Strategy:
/// 1. First, check if the consumer's preferred layout is already a SliceAttr
///    with matching reduction dimensions. If so, use its parent layout directly
///    and adjust the sg_data/inst_data acccording to source shape.
/// 2. If step 1 fails, construct a new layout by distributing
///    workgroup/subgroup resources across dimensions. It will try to align
///    with the consumer's sg_layout for non-reduction dimensions, so that the
///    reduced result stays with the same subgroup distribution as expected by
///    the consumer.

xegpu::SliceAttr
xegpu::reductionLayoutSetupRule(ArrayRef<int64_t> srcShape,
                                SmallVector<int64_t> reductionDims,
                                DistributeLayoutAttr consumerLayout) {

  xegpu::SliceAttr consumerSliceLayout =
      dyn_cast<xegpu::SliceAttr>(consumerLayout);

  // Hardware constraints (TODO: these should ideally be queried from device
  // capabilities)
  const int workgroupSize = 16; // Total number of subgroups in a workgroup
  const int subgroupSize = 16;  // Number of SIMD lanes per subgroup
  const int vectorSize = 8;     // Elements processed per vector instruction
  int srcShapeSize = srcShape.size();
  xegpu::DistributeLayoutAttr proposedSrcLayout;
  auto context = consumerLayout.getContext();
  // Reduction layout requires at least 2D tensors
  if (srcShapeSize < 2)
    return nullptr;

  SmallVector<int64_t> sgLayout(srcShapeSize);
  SmallVector<int64_t> sgData(srcShapeSize);

  SmallVector<int64_t> instData(srcShapeSize, 1);
  instData[srcShapeSize - 1] = subgroupSize;
  instData[srcShapeSize - 2] =
      vectorSize; // This will be adjusted based on actual data distribution

  SmallVector<int64_t> laneLayout(srcShapeSize, 1);
  laneLayout[srcShapeSize - 1] = subgroupSize;
  SmallVector<int64_t> laneData(srcShapeSize, 1);

  // Strategy 1: Try to preserve the consumer's slice layout structure
  // If the consumer already expects a slice layout with the same reduction
  // dims, we can directly use its parent layout as our source layout. This
  // ensures best alignment and avoids any data movement across subgroups
  // and lanes.

  auto canPreserveSliceLayout =
      [&](ArrayRef<int64_t> srcShape, SmallVector<int64_t> reductionDims,
          DistributeLayoutAttr consumerLayout) -> bool {
    // Verify that the consumer layout can be adapted to the source shape:
    // For each dimension, check if srcShape[i] is divisible by the parent's
    // sg_layout[i] and lane_layout[i]. If so, sg_layout and lane_layout can be
    // reused.

    if (!consumerSliceLayout)
      return false;
    if (!consumerSliceLayout.getDims().asArrayRef().equals(reductionDims))
      return false;
    xegpu::DistributeLayoutAttr parentLayout = consumerSliceLayout.getParent();
    if (!parentLayout.getRank() == srcShapeSize)
      return false;

    SmallVector<int64_t> parentSgLayout =
        parentLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> parentLaneLayout =
        parentLayout.getEffectiveLaneLayoutAsInt();

    if (parentSgLayout.size() != static_cast<size_t>(srcShapeSize))
      return false;
    if (parentLaneLayout.size() != static_cast<size_t>(srcShapeSize))
      return false;
    for (int i = 0; i < srcShapeSize; i++) {
      if (srcShape[i] % parentSgLayout[i] != 0)
        return false;
      if (instData[i] % parentLaneLayout[i] != 0)
        return false;
    }
    return true;
  };

  if (canPreserveSliceLayout(srcShape, reductionDims, consumerLayout)) {
    // for each slice dim in source shape, if the dim size is different than the
    // result shape, try to adjust the sg_data/inst_data accordingly.
    SmallVector<int64_t> parentSgLayout =
        consumerSliceLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> parentLaneLayout =
        consumerSliceLayout.getEffectiveLaneLayoutAsInt();
    SmallVector<int64_t> parentLaneData =
        consumerSliceLayout.getEffectiveLaneDataAsInt();

    for (int i = 0; i < srcShapeSize; i++) {
      sgLayout[i] = parentSgLayout[i];
      sgData[i] = srcShape[i] / sgLayout[i];
      instData[i] = std::min(instData[i], sgData[i]);
      laneLayout[i] = parentLaneLayout[i];
      laneData[i] = parentLaneData[i];
    }

  } else {
    // Strategy 2: Construct a new layout aligned with consumer's sg_layout for
    // the result (non-reduction dims) then distribute remaining subgroups
    // across reduced dimensions

    SmallVector<int64_t> cplSgLayout =
        consumerLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> cplSgData = consumerLayout.getEffectiveSgDataAsInt();
    SmallVector<int64_t> cplInstData =
        consumerLayout.getEffectiveInstDataAsInt();
    int remainingSgCount = workgroupSize;
    SmallVector<int64_t> remainingDims;
    int cplId = cplSgLayout.size() - 1;
    // For non-reduction dimensions, try to match consumer's sg_layout
    // This ensures the result after reduction has the expected distribution
    for (int i = srcShapeSize - 1; i >= 0; i--) {
      if (!llvm::is_contained(reductionDims, i) && cplId >= 0) {
        assert((srcShape[i] % cplSgLayout[cplId] == 0) &&
               "source shape not divisible by consumer sg_layout");
        sgLayout[i] = cplSgLayout[cplId];
        sgData[i] = srcShape[i] / sgLayout[i];
        instData[i] = std::min(cplInstData[cplId], sgData[i]);
        remainingSgCount /= sgLayout[i];
        cplId--;
      }
    }

    // Second pass: Distribute remaining subgroups across unhandled dimensions
    // This handles reduction dimensions and dimensions that didn't align with
    // consumer
    for (int i = srcShapeSize - 1; i >= 0; i--) {
      if (llvm::is_contained(reductionDims, i)) {
        sgLayout[i] = std::min((srcShape[i] / laneLayout[i]),
                               static_cast<int64_t>(remainingSgCount));
        sgData[i] = srcShape[i] / sgLayout[i];
        instData[i] = std::min(instData[i], sgData[i]);
        remainingSgCount /= sgLayout[i];
      }
    }
    assert(remainingSgCount == 1 && "not all subgroups have been distributed");
  }

  SmallVector<int32_t> sgLayout32(sgLayout.begin(), sgLayout.end());
  SmallVector<int32_t> sgData32(sgData.begin(), sgData.end());
  SmallVector<int32_t> instData32(instData.begin(), instData.end());
  SmallVector<int32_t> laneLayout32(laneLayout.begin(), laneLayout.end());
  SmallVector<int32_t> laneData32(laneData.begin(), laneData.end());
  proposedSrcLayout = xegpu::LayoutAttr::get(
      context, DenseI32ArrayAttr::get(context, sgLayout32),
      DenseI32ArrayAttr::get(context, sgData32),
      DenseI32ArrayAttr::get(context, instData32),
      DenseI32ArrayAttr::get(context, laneLayout32),
      DenseI32ArrayAttr::get(context, laneData32), consumerLayout.getOrder());
  xegpu::SliceAttr reductionResLayout =
      xegpu::SliceAttr::get(context, proposedSrcLayout,
                            DenseI64ArrayAttr::get(context, reductionDims));
  return reductionResLayout;
}

xegpu::DistributeLayoutAttr
xegpu::bitCastLayoutSetupRule(xegpu::DistributeLayoutAttr resLayout,
                              int resElemTyBitWidth, int srcElemTyBitWidth) {
  SmallVector<int64_t> sgData = resLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = resLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneData = resLayout.getEffectiveLaneDataAsInt();
  size_t dim = sgData.size() - 1;
  int64_t sgDataValue, instDataValue, laneDataValue;
  if (srcElemTyBitWidth < resElemTyBitWidth) {
    int bitWidthRatio = resElemTyBitWidth / srcElemTyBitWidth;
    sgDataValue = (dim < sgData.size()) ? sgData[dim] * bitWidthRatio : -1;
    instDataValue =
        (dim < instData.size()) ? instData[dim] * bitWidthRatio : -1;
    laneDataValue =
        (dim < laneData.size()) ? laneData[dim] * bitWidthRatio : -1;
  }
  // Now set only instData and laneData, preserving sgData
  xegpu::DistributeLayoutAttr finalResLayout;
  finalResLayout =
      resLayout.setDimData(dim, sgDataValue, instDataValue, laneDataValue);
  return finalResLayout;
}