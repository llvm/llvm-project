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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

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

// Sets the layout on a TensorDesc value by updating its type to include
// the given layout, if the type does not already have a layout attached.
static void setTensorDescLayout(Value val, xegpu::DistributeLayoutAttr layout) {
  auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(val.getType());
  if (!tensorDescTy || tensorDescTy.getLayoutAttr())
    return;
  auto typeWithLayout = xegpu::TensorDescType::get(
      tensorDescTy.getContext(), tensorDescTy.getShape(),
      tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
  val.setType(typeWithLayout);
}

// the walkRegionBackward() is a recursive function
// the input rootOp is the function operation, which is also a region op.
// it recursively processes the region op in reverse topological order.
static void walkRegionBackward(Region &region,
                               llvm::function_ref<void(Operation *)> visit) {

  // Use post-order traversal to process blocks in reverse topological order.
  // This ensures that use blocks are visited before def blocks, which is
  // required for backward layout propagation.
  if (region.empty())
    return;
  llvm::ReversePostOrderTraversal<Region *> rpot(&region);
  SmallVector<Block *> blocks(rpot.begin(), rpot.end());
  for (Block *block : llvm::reverse(blocks)) {
    // ops: back -> front
    for (Operation &op : llvm::reverse(*block)) {
      // make sure we first visit inside the region op (so yield op first)
      // and then move to region op itself
      // Regions are iterated in forward order so that for multi-region ops
      // like scf.while, earlier regions (e.g., "before/cond") are processed
      // first. This ensures that when a later region's terminator (e.g., "do"
      // yield) needs the layout of an earlier region's block args, those
      // layouts are already available from use points.
      for (Region &nested : op.getRegions())
        walkRegionBackward(nested, visit);

      visit(&op);
    }
  }
}

static xegpu::DistributeLayoutAttr getLayoutFromUsePoints(Value result) {
  xegpu::DistributeLayoutAttr layout = nullptr;
  for (OpOperand &use : result.getUses()) {
    if (auto tmpLayout = xegpu::getDistributeLayoutAttr(use)) {
      if (!layout)
        layout = tmpLayout;
      break;
    }
  }
  return layout;
}

// For regular operations: First the result layouts are propagated from uses.
// Then the result layouts are propagated to uses (operands).
static void propagateResultsToRegularOperands(Operation *op) {
  if (op->getNumResults() == 0)
    return;
  if (op->getNumResults() > 1 && !isa<vector::DeinterleaveOp>(op))
    return;
  OpResult result = op->getResult(0);
  xegpu::DistributeLayoutAttr resLayout = getLayoutFromUsePoints(result);
  Type resultType = result.getType();

  if (!resLayout)
    return;

  // Recover layout for TensorDesc type results by updating the type to include
  // the layout. For vector type
  if (isa<xegpu::TensorDescType>(resultType))
    setTensorDescLayout(result, resLayout);

  // Recover layout for vector type results, or for multi-reduction ops which
  // may reduce to a scalar that still needs a layout.
  if (isa<VectorType>(resultType) || isa<vector::MultiDimReductionOp>(op))
    xegpu::setTemporaryLayout(result, resLayout);

  for (OpOperand &opr : op->getOpOperands()) {
    xegpu::DistributeLayoutAttr operandLayout =
        xegpu::inferSourceLayoutFromResult(opr, resLayout);
    // Recover layout for vector operands
    if (isa<VectorType>(opr.get().getType()) && operandLayout)
      xegpu::setTemporaryLayout(opr, operandLayout);
  }
}

// Propagate layout from region op results and sibling region block args
// to yield/condition operands. For each successor of this terminator:
// - Parent successor: propagate from parent op's result layouts (use points).
// - Region successor: propagate from target region's block arg layouts (use
//   points), e.g., scf.yield in "after/do" region propagates to "before/cond"
//   block args.
static void propagateRegionResultsToYieldOperands(
    mlir::RegionBranchTerminatorOpInterface yieldOp) {
  auto regionBranchOp =
      dyn_cast<RegionBranchOpInterface>(yieldOp->getParentOp());
  if (!regionBranchOp)
    return;

  SmallVector<RegionSuccessor> successors;
  SmallVector<Attribute> operandAttrs(yieldOp->getNumOperands(), nullptr);
  yieldOp.getSuccessorRegions(operandAttrs, successors);

  for (const RegionSuccessor &successor : successors) {
    OperandRange succOps = yieldOp.getSuccessorOperands(successor);
    if (succOps.empty())
      continue;
    unsigned beginIdx = succOps.getBeginOperandIndex();
    ValueRange successorInputs = regionBranchOp.getSuccessorInputs(successor);
    unsigned count = std::min<unsigned>(succOps.size(), successorInputs.size());

    for (unsigned i = 0; i < count; ++i) {
      xegpu::DistributeLayoutAttr layout;
      if (successor.isParent()) {
        // For parent successor, get layout from external use points of the
        // parent op's results.
        auto regionResult = regionBranchOp->getResult(i);
        layout = getLayoutFromUsePoints(regionResult);
        if (layout) {
          // set layout for the region op, like scf.loop
          xegpu::setTemporaryLayout(regionResult, layout);
          if (isa<xegpu::TensorDescType>(regionResult.getType()))
            setTensorDescLayout(regionResult, layout);
        }
      } else {
        // For region successor, get layout from the target region's block
        // arg use points (e.g., "before/cond" region args for scf.while
        // "after/do" yield).
        layout = getLayoutFromUsePoints(successorInputs[i]);
      }
      if (!layout)
        continue;
      auto operandType = succOps[i].getType();
      if (isa<VectorType>(operandType) ||
          dyn_cast<xegpu::TensorDescType>(operandType))
        // recover layout for yield op operands
        xegpu::setTemporaryLayout(yieldOp->getOpOperand(beginIdx + i), layout);
    }
  }
}

// Propagate layout from region arguments to region op's init operands. This
// sets the temporary layout for region arguments and init operands.
static void propagateRegionArgsToInits(mlir::RegionBranchOpInterface regionOp) {
  // Iterate all regions of the region op. For each block argument that has a
  // layout (determined from its use points), trace back to find the
  // corresponding init operand of the regionOp and set the layout on it.
  // This works generically for scf.for, scf.while, and other
  // RegionBranchOpInterface ops.
  for (Region &region : regionOp->getRegions()) {
    RegionSuccessor regionSuccessor(&region);
    // Use getSuccessorInputs to get the block arguments that correspond to
    // predecessor operands. This correctly handles ops like scf.for where
    // the induction variable is a block arg but not a successor input.
    ValueRange successorInputs = regionOp.getSuccessorInputs(regionSuccessor);
    for (auto [inputIdx, regionArg] : llvm::enumerate(successorInputs)) {
      auto layout = getLayoutFromUsePoints(regionArg);
      if (!layout)
        continue;

      // Recover layout for tensor_desc block args by updating the type.
      if (isa<xegpu::TensorDescType>(regionArg.getType()))
        setTensorDescLayout(regionArg, layout);

      // Recover layout for region op operands, like scf.for's init operands.
      // Find all predecessor values that flow into this block argument.
      SmallVector<Value> predValues;
      regionOp.getPredecessorValues(regionSuccessor, inputIdx, predValues);
      for (Value predVal : predValues) {
        // Match predecessor value to an operand of the regionOp.
        for (OpOperand &operand : regionOp->getOpOperands()) {
          if (operand.get() == predVal)
            xegpu::setTemporaryLayout(operand, layout);
        }
      }
    }
  }
}

// Prerequisite for Layout Recovery
// It relies on the following invariant:
// 1. there is no layout conflict between different uses of the same definition.
// 2. each definition has a well-defined layout requirement at its use point.
//     - Every definition must have at least one use that appears after it in
//     topological order.
//     - TODO: If a definition has no such use (e.g., a loop result or region
//     output), an explicit convert_layout operation is inserted to create a
//     use.
//     - Only the result of convert_layout is permitted to have no subsequent
//     use.
//
// The recovery proceeds by scanning the operation in reverse topological order
// as follows:
//    For regular operations: First the result layouts are propagated from uses.
//      Then the result layouts are propagated to operands.
//
//    For region operations (e.g., loops):
//       - When backward propagation reaches a region op, it sets the layout of
//       the region op’s results according to use points like regular ops.
//       - Then, the result layouts (such as a loop output) are propagated to
//       their corresponding operands in the yield.
//       - When backward propagation reaches the first operation inside the
//       region, the pass examines the region op’s initialization list,
//       propagating from region arguments to the corresponding initialization
//       operands.
//       - This ensures that layouts are consistently propagated
//       across region boundaries while preserving a single well-defined use for
//       each definition at the region-op level.
bool xegpu::recoverTemporaryLayouts(Operation *rootOp) {
  auto processFunc = [&](Region &body, StringRef funcName) {
    walkRegionBackward(body, [&](Operation *op) {
      if (auto regionOp = dyn_cast<mlir::RegionBranchOpInterface>(op)) {
        propagateRegionArgsToInits(regionOp);
      } else if (auto yieldOp =
                     dyn_cast<mlir::RegionBranchTerminatorOpInterface>(op)) {
        propagateRegionResultsToYieldOperands(yieldOp);
      } else if (!dyn_cast<xegpu::AnchorLayoutInterface>(op)) {
        propagateResultsToRegularOperands(op);
      }
    });
  };
  removeTemporaryLayoutAttrs(rootOp);
  rootOp->walk([&](func::FuncOp func) {
    processFunc(func.getBody(), func.getSymName());
  });
  rootOp->walk([&](gpu::GPUFuncOp func) {
    processFunc(func.getBody(), func.getName());
  });

  return true;
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

void xegpu::removeTemporaryLayoutAttrs(Operation *op) {
  op->walk([&](Operation *nestOp) {
    SmallVector<StringAttr> attrsToRemove;
    for (auto namedAttr : nestOp->getDiscardableAttrs()) {
      if (isa<xegpu::DistributeLayoutAttr>(namedAttr.getValue()))
        attrsToRemove.push_back(namedAttr.getName());
    }
    for (auto attrName : attrsToRemove)
      nestOp->removeDiscardableAttr(attrName);
  });
}

/// Infers the source layout attribute for a broadcast operation given the
/// result layout attribute, result shape, source shape.
xegpu::DistributeLayoutAttr
xegpu::inferBroadcastSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> resShape,
                                  ArrayRef<int64_t> srcShape) {

  SmallVector<int64_t> bcastDims;
  size_t dimDiff = resShape.size() - srcShape.size();
  auto bcastSourceLayout = resLayout;
  for (size_t i = dimDiff; i < resShape.size(); i++) {
    if ((srcShape[i - dimDiff] == 1) && (resShape[i] != 1))
      bcastDims.push_back(i);
  }

  // the sg_layout and lane_layout for unit dimensions are preserved so it can
  // be propagate to producer op so potentially used by the multi-reduction op.
  if (!bcastDims.empty())
    bcastSourceLayout = bcastSourceLayout.setUnitDimData(bcastDims);

  if (dimDiff > 0) {
    SmallVector<int64_t> sliceDims;
    for (size_t i = 0; i < dimDiff; i++)
      sliceDims.push_back(i);
    bcastSourceLayout = xegpu::SliceAttr::get(
        resLayout.getContext(), bcastSourceLayout,
        DenseI64ArrayAttr::get(resLayout.getContext(), sliceDims));
  }
  return bcastSourceLayout;
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

xegpu::DistributeLayoutAttr
xegpu::inferReductionSourceLayout(xegpu::DistributeLayoutAttr resLayout) {
  return xegpu::inferMultiReductionSourceLayout(resLayout, {0});
}

/// Infers the source layout attribute for a transpose operation given the
/// result layout attribute and permutation.
xegpu::DistributeLayoutAttr
xegpu::inferTransposeSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> permutation) {
  return resLayout.transposeDims(permutation);
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

/// Infers the source layout attribute for an interleave operation given the
/// result layout attribute. Interleave doubles the size of the innermost
/// dimension, so the layout inference is similar to bitcast where the source
/// element type is larger than the result element type (ratio = 2).
xegpu::DistributeLayoutAttr
xegpu::inferInterleaveSourceLayout(xegpu::DistributeLayoutAttr resLayout) {

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

  // Interleave doubles the innermost dimension, so we need to halve the
  // layout values (similar to bitcast with ratio = 2)
  constexpr int ratio = 2;
  if (sgDataSize) {
    assert((sgData.back() % ratio) == 0 &&
           "sgData not divisible by interleave ratio");
    sgDataValue = sgData.back() / ratio;
  }
  if (instDataSize) {
    assert((instData.back() % ratio) == 0 &&
           "instData not divisible by interleave ratio");
    instDataValue = instData.back() / ratio;
  }
  if (laneDataSize) {
    assert((laneData.back() % ratio) == 0 &&
           "laneData not divisible by interleave ratio");
    laneDataValue = laneData.back() / ratio;
  }

  return resLayout.setDimData(dim, sgDataValue, instDataValue, laneDataValue);
}

/// Infers the source layout attribute for a deinterleave operation given the
/// result layout attribute. Deinterleave halves the size of the innermost
/// dimension, so the layout inference is similar to bitcast where the source
/// element type is smaller than the result element type (ratio = 2).
xegpu::DistributeLayoutAttr
xegpu::inferDeinterleaveSourceLayout(xegpu::DistributeLayoutAttr resLayout) {

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

  // Deinterleave halves the innermost dimension, so we need to double the
  // layout values (similar to bitcast with ratio = 2)
  constexpr int ratio = 2;
  if (sgDataSize)
    sgDataValue = sgData.back() * ratio;
  if (instDataSize)
    instDataValue = instData.back() * ratio;
  if (laneDataSize)
    laneDataValue = laneData.back() * ratio;

  return resLayout.setDimData(dim, sgDataValue, instDataValue, laneDataValue);
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

  if (dimDiff > 0) {
    // assert that the leading dimensions being sliced off are not distributed
    // (i.e. sg_layout and lane_layout for those dimensions are all 1)
    auto resSgLayout = resLayout.getEffectiveSgLayoutAsInt();
    auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
    for (int i = 0; i < dimDiff; i++) {
      assert((resSgLayout.size() == 0 || resSgLayout[i] == 1) &&
             (resLaneLayout.size() == 0 || resLaneLayout[i] == 1) &&
             "Leading dimensions being sliced off must not be distributed");
    }
    return resLayout.dropDims(llvm::to_vector(llvm::seq<int64_t>(0, dimDiff)));
  }
  return resLayout;
}

/// Infers the source layout attribute for an insert operation
/// given the result layout attribute, result shape, and source shape. Removes
/// leading dimensions from the result layout to match the source shape size.
// TODO: add propagation support for insert op
xegpu::DistributeLayoutAttr
xegpu::inferInsertSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                               ArrayRef<int64_t> resShape,
                               ArrayRef<int64_t> srcShape) {

  int srcShapeSize = srcShape.size();
  int resShapeSize = resShape.size();
  int dimDiff = resShapeSize - srcShapeSize;

  if (dimDiff > 0) {
    // assert that the leading dimensions being sliced off are not distributed
    // (i.e. sg_layout and lane_layout for those dimensions are all 1)
    auto resSgLayout = resLayout.getEffectiveSgLayoutAsInt();
    auto resLaneLayout = resLayout.getEffectiveLaneLayoutAsInt();
    for (int i = 0; i < dimDiff; i++) {
      assert((resSgLayout.size() == 0 || resSgLayout[i] == 1) &&
             (resLaneLayout.size() == 0 || resLaneLayout[i] == 1) &&
             "Leading dimensions being sliced off must not be distributed");
    }
    return resLayout.dropDims(llvm::to_vector(llvm::seq<int64_t>(0, dimDiff)));
  }
  return resLayout;
}

/// Infers the source layout attribute for extract operation
/// given the result layout attribute, result shape, and source shape. Adds
/// leading dimensions to the source layout to match the source shape size.
// TODO: add layout attribute interface: expandDims() and use it here.
// TODO: add propagation support for extract op
xegpu::DistributeLayoutAttr
xegpu::inferExtractSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                ArrayRef<int64_t> resShape,
                                ArrayRef<int64_t> srcShape) {

  int srcShapeSize = srcShape.size();
  int resShapeSize = resShape.size();
  int dimDiff = srcShapeSize - resShapeSize;
  auto context = resLayout.getContext();
  // construct the source layout by adding unit dimensions to the front of
  // result layout
  if (dimDiff > 0) {
    auto sgLayout = resLayout.getEffectiveSgLayoutAsInt();
    auto sgData = resLayout.getEffectiveSgDataAsInt();
    auto instData = resLayout.getEffectiveInstDataAsInt();
    auto laneLayout = resLayout.getEffectiveLaneLayoutAsInt();
    auto laneData = resLayout.getEffectiveLaneDataAsInt();
    auto order = resLayout.getEffectiveOrderAsInt();

    // Example: result shape is 3D with order [1, 2, 0], source shape is 5D
    // (adding 2 leading dimensions). Expected source order: [3, 4, 2, 1, 0]
    // Step 1: shift existing order by dimDiff: [1, 2, 0] -> [3, 4, 2]
    // Step 2: append new leading dims in reverse (slowest first): [3, 4, 2, 1,
    // 0]

    // Shift existing dimension indices in order by dimDiff to account for the
    // new leading dimensions being added to the source shape
    for (auto &o : order)
      o += dimDiff;

    // Add unit dimensions to the front of non-empty layout vectors and append
    // the new dimension indices to the order array in reverse (slowest
    // dimension has the lowest index and appears last in the order array)
    for (int i = 0; i < dimDiff; i++) {
      if (!sgLayout.empty())
        sgLayout.insert(sgLayout.begin(), 1);
      if (!sgData.empty())
        sgData.insert(sgData.begin(), 1);
      if (!instData.empty())
        instData.insert(instData.begin(), 1);
      if (!laneLayout.empty())
        laneLayout.insert(laneLayout.begin(), 1);
      if (!laneData.empty())
        laneData.insert(laneData.begin(), 1);
      order.push_back(dimDiff - 1 - i);
    }

    DenseI32ArrayAttr orderAttr = resLayout ? resLayout.getOrder() : nullptr;
    auto toAttr = [&](ArrayRef<int64_t> v) -> DenseI32ArrayAttr {
      if (v.empty())
        return DenseI32ArrayAttr();
      SmallVector<int32_t> v32(v.begin(), v.end());
      return DenseI32ArrayAttr::get(context, v32);
    };
    auto srcLayout = xegpu::LayoutAttr::get(
        context, sgLayout.empty() ? nullptr : toAttr(sgLayout),
        sgData.empty() ? nullptr : toAttr(sgData),
        instData.empty() ? nullptr : toAttr(instData),
        laneLayout.empty() ? nullptr : toAttr(laneLayout),
        laneData.empty() ? nullptr : toAttr(laneData),
        (!orderAttr || orderAttr.empty()) ? nullptr : toAttr(order));
    return srcLayout;
  }
  return resLayout;
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

/// Infers the layout attribute for mask and offset operand for Chunked load
/// and store, given the anchor layout attribute for the value being load/store.
xegpu::DistributeLayoutAttr xegpu::inferMaskOffsetLayoutForScatterIO(
    xegpu::DistributeLayoutAttr payloadLayout, int chunkSize) {
  auto rank = payloadLayout.getRank();
  if (chunkSize > 1)
    return payloadLayout.dropDims(
        llvm::to_vector(llvm::seq<int64_t>(rank - 1, rank)));
  return payloadLayout;
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
/// its consumer. When the consumer layout is a slice layout, it attempts to
/// reuse the slice layout's parent layout for the source to further minimize
/// potential data redistribution.
///
/// InstData requries {1, ..., min(maxReduceVectorSize, srcShape),subgroupSize}
/// Lane Layout requires {1, ..., 1, subgroupSize}
/// Lane data requires {1, ..., min(maxReduceVectorSize, srcShape), 1}
///
/// Examples:
///   1. Subgroup layout - Row reduction on 2D tensor:
///      srcShape=[32, 128], reductionDims=[1], resShape=[32], subgroupSize=16,
///      NumSg=32
///      * Consumer Layout:
///        #xegpu.slice<#xegpu.layout<sg_layout=[4, 8], sg_data=[8, 8]>, dims =
///        [1]>}
////     * Result Layout:
///        #xegpu.slice<#xegpu.layout<sg_layout=[4, 8],sg_data=[8, 16]>, dims =
///        [1]>}
///      Note that the sg_layout is reused but sg_data needs to be adjusted to
///      evenly distribute the source tensor tile among the reduction dim.
///
///   2. Subgroup layout - Same example above but consumer doesn't have a
///   reusable slice layout.
///      * Consumer Layout:
///        #xegpu.layout<sgLayout=[32], sgData=[1]>
///      * Result Layout:
///        #xegpu.slice<#xegpu.layout<sgLayout=[32,1], sgData=[1, 64]>, dims =
///        [1]>}
///      * Consumer Layout:
///        #xegpu.slice<#xegpu.layout<sgLayout=[8, 2, 4], sgData=[4, 64, 32]>,
///      dims = [1, 2]>}
///      * Result Layout:
///        #xegpu.slice<#xegpu.layout<sgLayout=[8,4], sgData=[4, 32]>, dims =
///        [1]>}
///      Note that the consumer's layout can't be directly reused as is.
///      So the algorithm distributes all subgroups on non reduction dimensions
///      first and then distribute remaining subgroups on the reduction
///      dimension.
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
    int numSg, const xegpu::uArch::uArch *uArch) {

  auto srcShape = srcVecTy.getShape();
  int srcRank = srcShape.size();
  auto context = srcVecTy.getContext();

  // Helper lambda to convert int64 vectors to int32 DenseArrayAttr
  auto toInt32Attr = [&](ArrayRef<int64_t> vec) {
    SmallVector<int32_t> vec32(vec.begin(), vec.end());
    return DenseI32ArrayAttr::get(context, vec32);
  };

  const int subgroupSize = uArch->getSubgroupSize();
  int64_t maxReduceVectorSize = 1; // could extend to spirv vector Size
  xegpu::DistributeLayoutAttr srcLayout;
  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    xegpu::SliceAttr consumerSliceLayout =
        dyn_cast_if_present<xegpu::SliceAttr>(consumerLayout);
    if (consumerSliceLayout &&
        consumerSliceLayout.getDims().asArrayRef().equals(reductionDims)) {
      srcLayout = consumerSliceLayout.getParent();
      SmallVector<int64_t> sgLayoutFromConsumer =
          srcLayout.getEffectiveSgLayoutAsInt();
      auto srcSgData = computeShapeRatio(srcShape, sgLayoutFromConsumer);
      if (srcSgData)
        for (int dim = 0; dim < srcRank; dim++) {
          if (llvm::is_contained(reductionDims, dim))
            srcLayout =
                srcLayout.setDimData(dim, srcSgData.value()[dim], -1, -1);
        }
    } else {
      SmallVector<int64_t> consumerSgLayout =
          consumerLayout ? consumerLayout.getEffectiveSgLayoutAsInt()
                         : SmallVector<int64_t>();
      SmallVector<int64_t> consumerSgData =
          consumerLayout ? consumerLayout.getEffectiveSgDataAsInt()
                         : SmallVector<int64_t>();
      SmallVector<int64_t> consumerOrder =
          consumerLayout ? consumerLayout.getEffectiveOrderAsInt()
                         : SmallVector<int64_t>();
      DenseI32ArrayAttr orderAttr =
          consumerLayout ? consumerLayout.getOrder() : nullptr;
      SmallVector<int64_t> sgLayout(srcRank), sgData(srcRank), order(srcRank);
      int remainingSgCount =
          consumerLayout ? consumerLayout.getNumSubgroups() : numSg;
      int consumerIdx = 0;

      // First pass: Match consumer's layout on non-reduction dimensions
      for (int i = 0; i < srcRank; i++) {
        if (!llvm::is_contained(reductionDims, i) &&
            consumerIdx < static_cast<int>(consumerSgLayout.size())) {
          sgLayout[i] = consumerSgLayout[consumerIdx];
          sgData[i] = consumerSgData[consumerIdx];
          remainingSgCount /= sgLayout[i];
          order[i] = consumerOrder[consumerIdx];
          consumerIdx++;
        }
      }

      // Second pass: Distribute remaining subgroups across reduction dimensions
      // the reduction to scalar case is handled only by this loop
      int64_t remainOrder = consumerSgLayout.size();
      for (int i = 0; i < srcRank; i++) {
        if (llvm::is_contained(reductionDims, i)) {
          sgLayout[i] =
              std::min(srcShape[i], static_cast<int64_t>(remainingSgCount));
          assert((srcShape[i] % sgLayout[i] == 0) &&
                 "source shape not divisible by sg_layout");
          sgData[i] = srcShape[i] / sgLayout[i];
          remainingSgCount /= sgLayout[i];
          order[i] = remainOrder++;
        }
      }

      assert(remainingSgCount == 1 && "not all subgroups distributed");
      srcLayout = xegpu::LayoutAttr::get(
          context, toInt32Attr(sgLayout), toInt32Attr(sgData),
          /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
          /*lane_data =*/nullptr, /*order =*/
          (!orderAttr || orderAttr.empty()) ? nullptr : toInt32Attr(order));
    }
  } else if (layoutKind == xegpu::LayoutKind::InstData) {

    SmallVector<int64_t> instData(srcRank, 1);
    if (srcRank >= 2)
      instData[srcRank - 2] =
          std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    instData[srcRank - 1] =
        std::min(static_cast<int64_t>(subgroupSize), srcShape[srcRank - 1]);
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(instData));
  } else if (layoutKind == xegpu::LayoutKind::Lane) {

    SmallVector<int64_t> laneLayout(srcRank, 1), laneData(srcRank, 1);
    laneLayout[srcRank - 1] =
        std::min(static_cast<int64_t>(subgroupSize), srcShape[srcRank - 1]);
    if (srcRank >= 2)
      laneData[srcRank - 2] =
          std::min(maxReduceVectorSize, srcShape[srcRank - 2]);
    srcLayout = xegpu::LayoutAttr::get(context, toInt32Attr(laneLayout),
                                       toInt32Attr(laneData));
  }

  return xegpu::SliceAttr::get(context, srcLayout,
                               DenseI64ArrayAttr::get(context, reductionDims));
}

/// Sets up layout for Reduction operations by creating a SliceAttr for the
/// result.
xegpu::SliceAttr
xegpu::setupReductionResultLayout(xegpu::LayoutKind layoutKind,
                                  VectorType srcVecTy,
                                  const xegpu::uArch::uArch *uArch) {

  auto srcShape = srcVecTy.getShape();
  auto context = srcVecTy.getContext();
  auto subgroupSize = uArch->getSubgroupSize();
  xegpu::LayoutAttr srcLayout;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(true && "subgroup layout assignment not supported for reduction (op "
                   "is not expected at this level).");
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    assert(true && "instData layout assignment not supported for reduction (op "
                   "is not expected at this level).");
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    SmallVector<int32_t> laneLayout(1), laneData(1);
    laneLayout[0] = std::min(subgroupSize, static_cast<int32_t>(srcShape[0]));
    laneData[0] = 1;
    srcLayout = xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, laneLayout),
        DenseI32ArrayAttr::get(context, laneData));
  }

  auto result = xegpu::SliceAttr::get(context, srcLayout,
                                      DenseI64ArrayAttr::get(context, 0));
  return result;
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
  assert(consumerLayout.getRank() == static_cast<int64_t>(srcShape.size()) &&
         "laneData must be available for all dimensions");
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
      sgDataValue = sgData[dim];
    } else if (layoutKind == xegpu::LayoutKind::InstData) {
      instDataValue = instData[dim];
      // Adjust instDataValue so it still fits within an instruction after
      // dividing by bitWidthRatio
      while ((instDataValue <= srcShape[dim]) &&
             (instDataValue % (innermostDimLaneLayout * bitWidthRatio) != 0))
        instDataValue *= 2;
      assert((srcShape[dim] % instDataValue) == 0 &&
             "srcShape, instData, and lanelayout for innermost must be 2^n !");
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
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

/// Sets up the result layout for an interleave operation to ensure the source
/// layout can be safely derived. Interleave doubles the innermost dimension,
/// so the result layout must ensure that laneData is a multiple
/// of 2, and instData must be divisible by innermostDimLaneLayout * 2.
///
/// Example:
///   Interleave: vector<128x256xf4> -> vector<128x512xf4>
///   Consumer layout: laneLayout=[1, 16], laneData=[1, 4], instData=[1, 64]
///   Result layout adjustment to ensure source can be safely inferred:
///     - laneData must be >= 2 and multiple of 2 (so source = laneData/2 is
///     valid)
///     - instData must be divisible by (16 * 2 = 32) (so source = instData/2 is
///     valid)
///     - Adjusted instData: ensure (instData % 32 == 0)
///
xegpu::DistributeLayoutAttr xegpu::setupInterleaveResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVecTy, VectorType resVecTy,
    DistributeLayoutAttr consumerLayout, const xegpu::uArch::uArch *uArch) {

  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  SmallVector<int64_t> sgData = consumerLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneData = consumerLayout.getEffectiveLaneDataAsInt();

  assert(consumerLayout.getRank() == static_cast<int64_t>(srcShape.size()) &&
         "consumer layout rank must match source shape rank");
  const size_t innerMostDim = srcShape.size() - 1;
  int64_t sgDataValue = -1;
  int64_t instDataValue = -1;
  int64_t laneDataValue = -1;

  // Interleave doubles the innermost dimension (ratio = 2)
  constexpr int ratio = 2;
  int innermostDimLaneLayout = uArch->getSubgroupSize();

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    sgDataValue = sgData[innerMostDim];
    // Ensure sgDataValue is divisible by ratio so source sgData can be inferred
    while ((sgDataValue <= srcShape[innerMostDim]) &&
           (sgDataValue % ratio != 0))
      sgDataValue *= ratio;
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    instDataValue = instData[innerMostDim];
    // Adjust instDataValue so it can be divided by (innermostDimLaneLayout *
    // ratio) when inferring the source layout
    while ((instDataValue <= srcShape[innerMostDim]) &&
           (instDataValue % (innermostDimLaneLayout * ratio) != 0))
      instDataValue *= ratio;
    assert((srcShape[innerMostDim] % instDataValue) == 0 &&
           "srcShape, instData, and laneLayout for innermost must be 2^n!");
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    laneDataValue = laneData[innerMostDim];
    // Ensure laneDataValue is at least 2 and divisible by ratio
    // so that source laneData = laneDataValue/2 is valid
    while ((laneDataValue <= srcShape[innerMostDim]) &&
           (laneDataValue % ratio != 0))
      laneDataValue *= ratio;
  }

  return consumerLayout.setDimData(innerMostDim, sgDataValue, instDataValue,
                                   laneDataValue);
}

/// Sets up the result layout for an insert strided slice operation.
/// Creates a result layout based on the specified layout kind (InstData or
/// Lane).
xegpu::DistributeLayoutAttr xegpu::setupInsertStridedSliceResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVectorTy,
    VectorType resVectorTy, xegpu::DistributeLayoutAttr consumerLayout,
    const xegpu::uArch::uArch *uArch) {

  xegpu::DistributeLayoutAttr requiredResLayout;
  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();
  SmallVector<int64_t> consumerLaneLayout =
      consumerLayout.getEffectiveLaneLayoutAsInt();
  ArrayRef<int64_t> srcShape = srcVectorTy.getShape();
  int64_t instDataValue = -1;
  int64_t laneDataValue = -1;

  requiredResLayout = consumerLayout;
  int srcRank = srcShape.size();

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(true &&
           "subgroup layout assignment not supported for insertStridedSlice.");
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    for (int dim = 0; dim < srcRank; dim++) {
      instDataValue = std::min(srcShape[dim], consumerInstData[dim]);
      requiredResLayout =
          requiredResLayout.setDimData(dim, -1, instDataValue, -1);
    }
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    for (int dim = 0; dim < srcRank; dim++) {
      assert(srcShape[dim] % consumerLaneLayout[dim] == 0 &&
             "srcShape must be divisible by laneLayout for all dimensions");
      laneDataValue = std::min(srcShape[dim] / consumerLaneLayout[dim],
                               consumerLaneData[dim]);
      requiredResLayout =
          requiredResLayout.setDimData(dim, -1, -1, laneDataValue);
    }
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
    int maxChunkSize, ArrayRef<int64_t> resShape, int subgroupSize) {

  if (layoutKind == xegpu::LayoutKind::Subgroup)
    return consumerLayout;

  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();

  SmallVector<int> instData(resShape.size(), 1);
  SmallVector<int> laneLayout(resShape.size(), 1);
  SmallVector<int> laneData(resShape.size(), 1);

  if (!isChunkedLoad) {
    if (layoutKind == xegpu::LayoutKind::InstData) {
      instData.back() = std::min(static_cast<int>(consumerInstData.back()),
                                 maxChunkSize * subgroupSize);
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneData.back() =
          std::min(static_cast<int>(consumerLaneData.back()), maxChunkSize);
      laneLayout.back() = std::min(static_cast<int64_t>(subgroupSize),
                                   resShape.back() / laneData.back());
      return xegpu::LayoutAttr::get(context, laneLayout, laneData);
    }
  } else {
    assert(resShape.size() == 2 && "Chunked Store must access 2D tensor tile.");
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
  ArrayRef<int64_t> resShape = resVecTy.getShape();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::LoadGatherInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize = uArchInstruction->getMaxLaneLoadSize(elemBitWidth);

  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      (chunkSize > 1), maxChunkSize, resShape,
                                      subgroupSize);
}

/// Sets up the anchor layout for load matrix operation.
/// TODO: enhance load matrix to indicate lowering to chunked load or not.
xegpu::DistributeLayoutAttr
xegpu::setupLoadMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                   VectorType resVecTy,
                                   xegpu::DistributeLayoutAttr consumerLayout,
                                   const xegpu::uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> resShape = resVecTy.getShape();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::LoadGatherInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize = uArchInstruction->getMaxLaneLoadSize(elemBitWidth);
  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      false, maxChunkSize, resShape,
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
      instData[srcShapeSize - 1] =
          std::min(subgroupSize, static_cast<int>(srcShape.back()));
      return xegpu::LayoutAttr::get(context, instData);
    } else if (layoutKind == xegpu::LayoutKind::Lane) {
      laneLayout[srcShapeSize - 1] =
          std::min(subgroupSize, static_cast<int>(srcShape.back()));
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
      dyn_cast<xegpu::uArch::StoreScatterInstructionInterface>(
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

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize = uArchInstruction->getMaxLaneStoreSize(elemBitWidth);

  return setupGenericStoreAnchorLayout(layoutKind, context, false, maxChunkSize,
                                       srcShape, subgroupSize);
}

// This function returns the default lane layout for a given vector type.
// - `packingSize` means multiple consecutive elements can be accessed
// together as a single unit.
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

/// Helper function to compute inst_data vectors for DPAS operands A, B, and
/// C/D.
static std::optional<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                                SmallVector<int64_t>>>
getDpasInstDataVectors(VectorType aTy, VectorType bTy, VectorType cdTy,
                       const xegpu::uArch::uArch *uArch,
                       bool isDpasMx = false) {
  const int subgroupSize = uArch->getSubgroupSize();

  const xegpu::uArch::MMAInstructionInterface *uArchInstruction;
  if (isDpasMx)
    uArchInstruction = dyn_cast<xegpu::uArch::SubgroupScaledMatrixMultiplyAcc>(
        uArch->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupScaledMatrixMultiplyAcc));
  else
    uArchInstruction =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));

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

  // For DPAS_MX, use getSupportedK to get the scaled K dimension.
  // assume single element in the returned vector.
  int kDimSize = subgroupSize;
  if (isDpasMx) {
    auto supportedKLen = uArchInstruction->getSupportedK(aTy.getElementType());
    kDimSize = supportedKLen[0];
  }

  SmallVector<int64_t> instDataA(aTy.getRank(), 1);
  instDataA[aTy.getRank() - 2] = maxALen;
  instDataA[aTy.getRank() - 1] = kDimSize;
  SmallVector<int64_t> instDataB(bTy.getRank(), 1);
  instDataB[bTy.getRank() - 2] = kDimSize;
  instDataB[bTy.getRank() - 1] = maxBLen;
  SmallVector<int64_t> instDataCD(cdTy.getRank(), 1);
  instDataCD[cdTy.getRank() - 2] = maxALen;
  instDataCD[cdTy.getRank() - 1] = maxCLen;
  return std::make_tuple(instDataA, instDataB, instDataCD);
}

/// Helper function to set up subgroup layouts for DPAS operands A, B, and C/D.
/// Returns the three layouts if successful, nullopt otherwise.
static std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
getupDpasSubgroupLayouts(mlir::MLIRContext *context, VectorType aTy,
                         VectorType bTy, VectorType cdTy,
                         xegpu::DistributeLayoutAttr consumerLayout, int numSg,
                         const xegpu::uArch::uArch *uArch) {
  auto instDataVecs = getDpasInstDataVectors(aTy, bTy, cdTy, uArch);
  if (!instDataVecs)
    return std::nullopt;
  auto [instDataA, instDataB, instDataCD] = *instDataVecs;
  assert(instDataA.size() == 2 && instDataB.size() == 2 &&
         instDataCD.size() == 2 &&
         "Sg layout creation expects valid 2D inst data");

  std::optional<LayoutRepresentation> consumerSgLayout = std::nullopt;
  if (consumerLayout && consumerLayout.isForWorkgroup()) {
    SmallVector<int64_t> sgLayoutD = consumerLayout.getEffectiveSgLayoutAsInt();
    consumerSgLayout = std::make_pair(sgLayoutD[0], sgLayoutD[1]);
  }

  // Get all valid layouts for A, B and C/D operands
  auto layoutsA = getValidLayouts(aTy.getShape(), instDataA, numSg);
  auto layoutsB = getValidLayouts(bTy.getShape(), instDataB, numSg);
  auto layoutsCD = getValidLayouts(cdTy.getShape(), instDataCD, numSg);
  if (layoutsA.empty() || layoutsB.empty() || layoutsCD.empty())
    return std::nullopt;

  // Pick the best subgroup layout
  llvm::DenseSet<LayoutRepresentation> setA(layoutsA.begin(), layoutsA.end());
  llvm::DenseSet<LayoutRepresentation> setCD(layoutsCD.begin(),
                                             layoutsCD.end());
  std::optional<LayoutRepresentation> bestPick;
  auto checkAlignedSgDataAB = [&](LayoutRepresentation sgLayout) {
    return aTy.getShape().back() / sgLayout.second ==
           bTy.getShape().front() / sgLayout.first;
  };
  for (auto &sgLayout : layoutsB) {
    if (setA.contains(sgLayout) && setCD.contains(sgLayout)) {
      if (!checkAlignedSgDataAB(sgLayout))
        continue;
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
  if (!bestPick)
    return std::nullopt;

  SmallVector<int> sgLayout = {static_cast<int>(bestPick->first),
                               static_cast<int>(bestPick->second)};
  SmallVector<int> sgDataA = {static_cast<int>(aTy.getShape()[0] / sgLayout[0]),
                              static_cast<int>(aTy.getShape()[1])};
  SmallVector<int> sgDataB = {
      static_cast<int>(bTy.getShape()[0]),
      static_cast<int>(bTy.getShape()[1] / sgLayout[1])};
  SmallVector<int> sgDataCD = {
      static_cast<int>(cdTy.getShape()[0] / sgLayout[0]),
      static_cast<int>(cdTy.getShape()[1] / sgLayout[1])};

  auto dpasALayout =
      xegpu::LayoutAttr::get(context, DenseI32ArrayAttr::get(context, sgLayout),
                             DenseI32ArrayAttr::get(context, sgDataA), nullptr,
                             nullptr, nullptr, nullptr);
  auto dpasBLayout =
      xegpu::LayoutAttr::get(context, DenseI32ArrayAttr::get(context, sgLayout),
                             DenseI32ArrayAttr::get(context, sgDataB), nullptr,
                             nullptr, nullptr, nullptr);
  auto dpasCDLayout =
      xegpu::LayoutAttr::get(context, DenseI32ArrayAttr::get(context, sgLayout),
                             DenseI32ArrayAttr::get(context, sgDataCD), nullptr,
                             nullptr, nullptr, nullptr);

  return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout);
}

/// Sets up the anchor layouts for dpas operands (A, B, and C/D).
/// The numSg and consumerLayout (optional) are only used by sg layout
/// creation.
std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
xegpu::setupDpasLayout(xegpu::LayoutKind layoutKind, VectorType aTy,
                       VectorType bTy, VectorType cdTy,
                       xegpu::DistributeLayoutAttr consumerLayout, int numSg,
                       const xegpu::uArch::uArch *uArch) {
  auto context = aTy.getContext();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
          xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    return getupDpasSubgroupLayouts(context, aTy, bTy, cdTy, consumerLayout,
                                    numSg, uArch);
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    auto instDataVecs = getDpasInstDataVectors(aTy, bTy, cdTy, uArch);
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
        cdTy, uArch /*, packingSize = std::nullopt */);
    return std::make_tuple(aLayout, bLayout, cdLayout);
  }
  return std::nullopt;
}

/// Helper to create a scale layout derived from a matrix operand layout.
/// The scale layout is computed by mapping each dimension of the matrix layout
/// to the corresponding scale tensor dimension using the ratio between the
/// matrix and scale shapes.
static xegpu::DistributeLayoutAttr
createScaleLayout(mlir::MLIRContext *context, VectorType matrixTy,
                  VectorType scaleTy, xegpu::DistributeLayoutAttr matrixLayout,
                  bool isBScale, const xegpu::uArch::uArch *uArch) {
  if (!scaleTy || !matrixLayout)
    return nullptr;

  // Calculate scaling factor by dividing matrix shape by scale shape
  ArrayRef<int64_t> matrixShape = matrixTy.getShape();
  ArrayRef<int64_t> scaleShape = scaleTy.getShape();

  // Scale shapes can be 1D or 2D, handle both cases
  if (scaleShape.empty())
    return nullptr;

  auto uArchInstruction =
      dyn_cast<xegpu::uArch::SubgroupScaledMatrixMultiplyAcc>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::SubgroupScaledMatrixMultiplyAcc));

  int64_t rank = matrixLayout.getRank();
  assert(rank == 2 && "dpas layouts must be two dimensions");

  SmallVector<int64_t> sgLayout = matrixLayout.getEffectiveSgLayoutAsInt();
  SmallVector<int64_t> sgData = matrixLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = matrixLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneLayout = matrixLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> laneData = matrixLayout.getEffectiveLaneDataAsInt();
  auto order = matrixLayout.getOrder();

  SmallVector<int> scaleSgLayout;
  SmallVector<int> scaleSgData;
  if (!sgLayout.empty() && !sgData.empty()) {
    scaleSgLayout.assign(sgLayout.begin(), sgLayout.end());
    scaleSgData.assign(sgData.begin(), sgData.end());
    scaleSgData[rank - 2] = std::max<int64_t>(
        scaleShape[rank - 2] / (matrixShape[rank - 2] / sgData[rank - 2]), 1);
    scaleSgData[rank - 1] = std::max<int64_t>(
        scaleShape[rank - 1] / (matrixShape[rank - 1] / sgData[rank - 1]), 1);
  }

  // For DPAS_MX scales: if matrix has inst_data, scale needs adjusted
  // inst_data. Scale inst_data is derived from matrix inst_data divided by
  // scale factor.
  SmallVector<int> scaleInstData;
  if (!instData.empty()) {
    scaleInstData.assign(instData.begin(), instData.end());
    if (isBScale)
      scaleInstData[rank - 2] = std::max<int64_t>(
          scaleShape[rank - 2] / (matrixShape[rank - 2] / instData[rank - 2]),
          1);
    else
      scaleInstData[rank - 1] = std::max<int64_t>(
          scaleShape[rank - 1] / (matrixShape[rank - 1] / instData[rank - 1]),
          1);
  }

  SmallVector<int> scaleLaneLayout;
  SmallVector<int> scaleLaneData;
  if (!laneLayout.empty() && !laneData.empty()) {
    scaleLaneLayout.assign(laneLayout.begin(), laneLayout.end());
    scaleLaneData.assign(laneData.begin(), laneData.end());
    bool isRowMajor = uArchInstruction->isLaneLayoutRowMajorOrder();
    if (isBScale ^ isRowMajor) {
      std::swap(scaleLaneLayout[rank - 2], scaleLaneLayout[rank - 1]);
      scaleLaneLayout[rank - 2] =
          std::min<int64_t>(scaleShape[rank - 2], scaleLaneLayout[rank - 2]);
    }
    scaleLaneData[rank - 2] =
        std::max<int64_t>(scaleShape[rank - 2] / scaleLaneLayout[rank - 2], 1);
    scaleLaneData[rank - 1] =
        std::max<int64_t>(scaleShape[rank - 1] / scaleLaneLayout[rank - 1], 1);
  }
  return xegpu::LayoutAttr::get(
      context,
      scaleSgLayout.empty() ? nullptr
                            : DenseI32ArrayAttr::get(context, scaleSgLayout),
      scaleSgData.empty() ? nullptr
                          : DenseI32ArrayAttr::get(context, scaleSgData),
      scaleInstData.empty() ? nullptr
                            : DenseI32ArrayAttr::get(context, scaleInstData),
      scaleLaneLayout.empty()
          ? nullptr
          : DenseI32ArrayAttr::get(context, scaleLaneLayout),
      scaleLaneData.empty() ? nullptr
                            : DenseI32ArrayAttr::get(context, scaleLaneData),
      order);
}

/// Sets up the anchor layouts for dpas_mx operands (A, B, C/D, A_scale, and
/// B_scale). The numSg and consumerLayout (optional) are only used by sg layout
/// creation.
std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
xegpu::setupDpasMxLayout(xegpu::LayoutKind layoutKind, VectorType aTy,
                         VectorType bTy, VectorType cdTy, VectorType aScaleTy,
                         VectorType bScaleTy,
                         xegpu::DistributeLayoutAttr consumerLayout, int numSg,
                         const xegpu::uArch::uArch *uArch) {
  auto context = aTy.getContext();

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto dpasLayouts = getupDpasSubgroupLayouts(context, aTy, bTy, cdTy,
                                                consumerLayout, numSg, uArch);
    if (!dpasLayouts)
      return std::nullopt;

    auto [dpasALayout, dpasBLayout, dpasCDLayout] = *dpasLayouts;

    // Create scale layouts
    auto aScaleLayout =
        createScaleLayout(context, aTy, aScaleTy, dpasALayout, false, uArch);

    auto bScaleLayout =
        createScaleLayout(context, bTy, bScaleTy, dpasBLayout, true, uArch);

    return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout, aScaleLayout,
                           bScaleLayout);
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    auto instDataVecs =
        getDpasInstDataVectors(aTy, bTy, cdTy, uArch, /*isDpasMx=*/true);
    if (!instDataVecs)
      return std::nullopt;
    auto [instDataA, instDataB, instDataCD] = *instDataVecs;

    auto dpasALayout = xegpu::LayoutAttr::get(
        context, SmallVector<int>(instDataA.begin(), instDataA.end()));
    auto dpasBLayout = xegpu::LayoutAttr::get(
        context, SmallVector<int>(instDataB.begin(), instDataB.end()));
    auto dpasCDLayout = xegpu::LayoutAttr::get(
        context, SmallVector<int>(instDataCD.begin(), instDataCD.end()));

    // Create scale layouts
    auto aScaleLayout =
        createScaleLayout(context, aTy, aScaleTy, dpasALayout, false, uArch);
    auto bScaleLayout =
        createScaleLayout(context, bTy, bScaleTy, dpasBLayout, true, uArch);

    return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout, aScaleLayout,
                           bScaleLayout);
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    const auto *uArchInstruction =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    auto aLayout = getDefaultLaneLayout2DBlockIo(
        aTy, uArch, uArchInstruction->getPackedFormatBitSizeA());
    auto bLayout = getDefaultLaneLayout2DBlockIo(
        bTy, uArch, uArchInstruction->getPackedFormatBitSizeB(), true);
    auto cdLayout = getDefaultLaneLayout2DBlockIo(cdTy, uArch);

    // Create scale layouts
    auto aScaleLayout =
        createScaleLayout(context, aTy, aScaleTy, aLayout, false, uArch);
    auto bScaleLayout =
        createScaleLayout(context, bTy, bScaleTy, bLayout, true, uArch);

    return std::make_tuple(aLayout, bLayout, cdLayout, aScaleLayout,
                           bScaleLayout);
  }
  return std::nullopt;
}

xegpu::DistributeLayoutAttr
xegpu::inferSourceLayoutFromResult(OpOperand &operand,
                                   xegpu::DistributeLayoutAttr resLayout) {
  if (!resLayout)
    return nullptr;
  Operation *op = operand.getOwner();
  unsigned idx = operand.getOperandNumber();

  // For vector::BroadcastOp, infer the source layout from the result layout.
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
    auto srcTy = dyn_cast<VectorType>(broadcast.getSourceType());
    if (!srcTy)
      return nullptr;
    return xegpu::inferBroadcastSourceLayout(
        resLayout, broadcast.getResultVectorType().getShape(),
        srcTy.getShape());
  }

  // For vector::MultiDimReductionOp, infer source layout from result layout
  // using reduction dims. Acc operand is expected to have the same layout as
  // the result.
  if (auto reduction = dyn_cast<vector::MultiDimReductionOp>(op)) {
    if (idx == 0) {
      SmallVector<int64_t> reductionDims(reduction.getReductionDims());
      return xegpu::inferMultiReductionSourceLayout(resLayout, reductionDims);
    }
    if (idx == 1)
      return resLayout;
  }

  if (auto reduction = dyn_cast<vector::ReductionOp>(op))
    return xegpu::inferReductionSourceLayout(resLayout);

  // For vector::BitCastOp, infer source layout from result layout using
  // element type bitwidths.
  if (auto bitcast = dyn_cast<vector::BitCastOp>(op)) {
    int resElemBitWidth =
        bitcast.getResultVectorType().getElementType().getIntOrFloatBitWidth();
    int srcElemBitWidth =
        bitcast.getSourceVectorType().getElementType().getIntOrFloatBitWidth();
    return xegpu::inferBitCastSourceLayout(resLayout, resElemBitWidth,
                                           srcElemBitWidth);
  }

  // For vector::ShapeCastOp, infer source layout from result layout using
  // shapes.
  if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(op)) {
    return xegpu::inferShapeCastSourceLayout(
        resLayout, shapeCast.getResultVectorType().getShape(),
        shapeCast.getSourceVectorType().getShape());
  }

  // For vector::InsertStridedSliceOp, infer source layout from result layout.
  // Dest vector must have the same layout as the result.
  if (auto insertSlice = dyn_cast<vector::InsertStridedSliceOp>(op)) {
    if (idx == 0) {
      return xegpu::inferInsertStridedSliceSourceLayout(
          resLayout, insertSlice.getDestVectorType().getShape(),
          insertSlice.getSourceVectorType().getShape());
    }
    if (idx == 1)
      return resLayout;
  }

  // For vector::Insert Op, infer source layout from result layout using
  // shapes.
  if (auto insert = dyn_cast<vector::InsertOp>(op)) {
    VectorType resVecTy = dyn_cast<VectorType>(insert.getResult().getType());
    VectorType valueToStoreTy =
        dyn_cast<VectorType>(insert.getValueToStore().getType());

    if ((idx == 0) && valueToStoreTy) {
      return xegpu::inferInsertSourceLayout(resLayout, resVecTy.getShape(),
                                            valueToStoreTy.getShape());
    }
    if (idx == 1)
      return resLayout;
  }

  // For vector::Extract Op, infer source layout from result layout using
  // shapes.
  if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
    VectorType srcVecTy = dyn_cast<VectorType>(extract.getSource().getType());
    VectorType resVecTy = dyn_cast<VectorType>(extract.getResult().getType());
    if (!srcVecTy || !resVecTy)
      return nullptr;
    return xegpu::inferExtractSourceLayout(resLayout, resVecTy.getShape(),
                                           srcVecTy.getShape());
  }

  // For vector::TransposeOp, infer source layout from result layout using
  // permutation.
  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    return xegpu::inferTransposeSourceLayout(resLayout,
                                             transpose.getPermutation());
  }

  // For vector::BitCastOp, infer source layout from result layout using
  // element type bitwidths.
  if (auto bitcast = dyn_cast<vector::BitCastOp>(op)) {
    int resElemBitWidth =
        bitcast.getResultVectorType().getElementType().getIntOrFloatBitWidth();
    int srcElemBitWidth =
        bitcast.getSourceVectorType().getElementType().getIntOrFloatBitWidth();
    return xegpu::inferBitCastSourceLayout(resLayout, resElemBitWidth,
                                           srcElemBitWidth);
  }

  // for vector::interleave
  if (auto interleave = dyn_cast<vector::InterleaveOp>(op)) {
    return xegpu::inferInterleaveSourceLayout(resLayout);
  }

  // for vector::deinterleave
  if (auto deinterleave = dyn_cast<vector::DeinterleaveOp>(op)) {
    return xegpu::inferDeinterleaveSourceLayout(resLayout);
  }

  // For vector::ExtractStridedSliceOp, simply return result layout
  if (dyn_cast<vector::ExtractStridedSliceOp>(op))
    return resLayout;
  // For elementwise operations, all operands must have the same layout as the
  // result.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return resLayout;

  return nullptr;
}

xegpu::DistributeLayoutAttr xegpu::getConsumerLayoutAt(OpOperand &operand) {
  Operation *op = operand.getOwner();
  xegpu::DistributeLayoutAttr resLayout;
  if (op->getNumResults() == 1)
    resLayout = xegpu::getDistributeLayoutAttr(op->getResult(0));
  auto inferredOperandLayout = inferSourceLayoutFromResult(operand, resLayout);
  if (inferredOperandLayout)
    return inferredOperandLayout;
  // By default, assume no layout conflict and return the current layout of
  // the operand.
  return xegpu::getDistributeLayoutAttr(operand.get());
}
