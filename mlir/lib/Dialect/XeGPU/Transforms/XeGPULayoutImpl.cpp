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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

// Returns true if `op` is safe and cheap to clone (no side effects, no
// regions, and all operands are themselves trivially rematerializable, e.g.
// block-arg-free pure value generators such as `vector.step`, splat
// `arith.constant`, or `vector.create_mask` whose operands are constants).
bool xegpu::isTriviallyRematerializable(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  if (!isMemoryEffectFree(op))
    return false;
  for (Value v : op->getOperands()) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return false;
    if (!isTriviallyRematerializable(defOp))
      return false;
  }
  return true;
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

  if (isa<vector::DeinterleaveOp>(op))
    xegpu::setTemporaryLayout(op->getResult(1), resLayout);

  for (OpOperand &opr : op->getOpOperands()) {
    xegpu::DistributeLayoutAttr operandLayout =
        xegpu::inferSourceLayoutFromResultForNonAnchorOp(opr, resLayout);
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
LogicalResult
xegpu::propagateRegionArgsToInits(mlir::RegionBranchOpInterface regionOp,
                                  xegpu::GetLayoutFnTy getLayoutOfValue) {
  // Iterate all regions of the region op. For each block argument that has a
  // layout (obtained via `getLayoutOfValue`), trace back to find the
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
      auto layout = getLayoutOfValue(regionArg);
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
  return success();
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
        (void)xegpu::propagateRegionArgsToInits(regionOp,
                                                getLayoutFromUsePoints);
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
///
/// vector.transpose semantics is `result[i] = source[permutation[i]]`, so
/// `result_layout[i] = source_layout[permutation[i]]`. To recover the source
/// layout from the result layout we must apply the inverse permutation.
xegpu::DistributeLayoutAttr
xegpu::inferTransposeSourceLayout(xegpu::DistributeLayoutAttr resLayout,
                                  ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inversePermutation =
      invertPermutationVector(permutation);
  return resLayout.transposeDims(inversePermutation);
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
// TODO: add layout attribute interface: expandDim() and use it here.
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

  // Use case 3: General dim collapse, for cross-sg reduction to SLM and other
  // shape casts where consecutive src dims fold into a single dst dim.
  //
  // Mirrors use case 2's elegant shape: walk the dst-side groups and call
  // a single layout-attribute primitive per group. Here the primitive is
  // `expandDim(dim, targetShape)`, the inverse of `collapseDims`. It applies
  // the per-field distribution policy required for a no-data-movement collapse
  // (sg_layout/lane_layout spread outer-to-inner; sg_data/lane_data/inst_data
  // fill innermost-first; inst_data is seeded from lane_layout * lane_data).
  // See LayoutAttr::expandDim for the full policy.
  //
  // Iteration goes innermost-first (reverse dst order) so that each
  // expandDim/dropDims call only mutates dst positions whose indices are
  // unaffected by earlier calls.
  SmallVector<SmallVector<int64_t>> collapseDims;
  if (xegpu::matchDimCollapse(srcShape, resShape, collapseDims)) {
    auto srcLayout = resLayout;
    for (int64_t dstIdx = static_cast<int64_t>(collapseDims.size()) - 1;
         dstIdx >= 0; --dstIdx) {
      ArrayRef<int64_t> srcDims = collapseDims[dstIdx];
      if (srcDims.empty()) {
        // Unit dst dim with no backing src dim: drop it.
        srcLayout = srcLayout.dropDims({dstIdx});
        continue;
      }
      if (srcDims.size() == 1)
        // 1:1 mapping, nothing to do for this dim.
        continue;
      SmallVector<int64_t> targetShape;
      targetShape.reserve(srcDims.size());
      for (int64_t d : srcDims)
        targetShape.push_back(srcShape[d]);
      srcLayout = srcLayout.expandDim(dstIdx, targetShape);
    }
    return srcLayout;
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

/// Returns true if every dimension of `shape` except the innermost
/// `numInnerDims` is a unit (size-1) dimension.
///
/// Several reduction layout-setup paths (InstData, Lane) only distribute the
/// innermost one or two dimensions and rely on all the leading dimensions
/// being degenerate. This helper makes that assumption explicit and checkable
/// instead of silently leaving leading dimensions undistributed.
static bool leadingDimsAreUnit(ArrayRef<int64_t> shape, int numInnerDims) {
  int numLeading = static_cast<int>(shape.size()) - numInnerDims;
  if (numLeading <= 0)
    return true;
  return llvm::all_of(shape.take_front(numLeading),
                      [](int64_t dim) { return dim == 1; });
}

/// Builds a LayoutAttr carrying inst_data, lane_layout, and lane_data (no
/// sg_layout / sg_data / order). Used by InstData-kind setup paths so the
/// result layout can later be distributed without re-deriving the lane
/// layout. `instData`, `laneLayout`, and `laneData` may have different
/// element types; they are normalized to int32 entries.
static xegpu::LayoutAttr buildInstDataLayoutWithLane(
    mlir::MLIRContext *context, ArrayRef<int64_t> instData,
    ArrayRef<int64_t> laneLayout, ArrayRef<int64_t> laneData,
    DenseI32ArrayAttr orderAttr = nullptr) {
  auto toI32Attr = [&](auto range) {
    SmallVector<int32_t> v(range.begin(), range.end());
    return DenseI32ArrayAttr::get(context, v);
  };
  return xegpu::LayoutAttr::get(context, /*sg_layout=*/nullptr,
                                /*sg_data=*/nullptr,
                                /*inst_data=*/toI32Attr(instData),
                                /*lane_layout=*/toI32Attr(laneLayout),
                                /*lane_data=*/toI32Attr(laneData),
                                /*order=*/orderAttr);
}

static xegpu::LayoutAttr
buildLaneLayout(mlir::MLIRContext *context, ArrayRef<int64_t> laneLayout,
                ArrayRef<int64_t> laneData,
                DenseI32ArrayAttr orderAttr = nullptr) {
  auto toI32Attr = [&](auto range) {
    SmallVector<int32_t> v(range.begin(), range.end());
    return DenseI32ArrayAttr::get(context, v);
  };
  return xegpu::LayoutAttr::get(context, /*sg_layout=*/nullptr,
                                /*sg_data=*/nullptr,
                                /*inst_data=*/nullptr,
                                /*lane_layout=*/toI32Attr(laneLayout),
                                /*lane_data=*/toI32Attr(laneData),
                                /*order=*/orderAttr);
}

/// Computes the lane_layout and lane_data for a multi-reduction's source
/// layout. Only the innermost two dimensions are distributed; all leading
/// dimensions are assumed to be unit (the caller verifies this via
/// `leadingDimsAreUnit`).
///
/// The layout is chosen to minimize cross-lane reduction: whenever possible a
/// reduction dimension is reduced *within* a lane (lane_layout == 1, with up to
/// `maxReduceVectorSize` elements packed into lane_data), and the subgroup's
/// lanes are spread across a non-reduction dimension instead.
///
///   - Exactly one of the innermost two dims is a reduction dim: place
///     `subgroupSize` lanes on the non-reduction dim and keep lane_layout == 1
///     on the reduction dim, packing up to `maxReduceVectorSize` reduced
///     elements into lane_data along that reduction dim.
///   - Both innermost dims are reduction dims (or the source is rank 1): fall
///     back to the default of `subgroupSize` lanes on the innermost dim,
///     packing `maxReduceVectorSize` elements on the second-to-innermost dim.
///
/// Returns the (lane_layout, lane_data) pair. The corresponding inst_data is
/// simply the element-wise product lane_layout * lane_data.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
computeReductionLaneLayoutAndData(ArrayRef<int64_t> srcShape,
                                  ArrayRef<int64_t> reductionDims,
                                  int subgroupSize,
                                  int64_t maxReduceVectorSize) {
  int srcRank = srcShape.size();
  SmallVector<int64_t> laneLayout(srcRank, 1), laneData(srcRank, 1);

  int innermost = srcRank - 1;
  int secondInnermost = srcRank - 2;

  // `laneDim` carries the subgroupSize lanes; `vectorDim` packs the reduced
  // elements into lane_data. Default: lanes on the innermost dim, reduced
  // vector on the second-to-innermost dim.
  int laneDim = innermost;
  int vectorDim = secondInnermost; // negative for rank 1

  laneLayout[laneDim] =
      std::min(static_cast<int64_t>(subgroupSize), srcShape[laneDim]);
  if (vectorDim >= 0)
    laneData[vectorDim] = std::min(maxReduceVectorSize, srcShape[vectorDim]);

  return {laneLayout, laneData};
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
/// For the InstData and Lane layout kinds only the innermost two dimensions
/// are distributed; all leading dimensions are assumed to be unit dimensions.
/// This assumption is checked via `leadingDimsAreUnit`. The lane_layout and
/// lane_data are computed by `computeReductionLaneLayoutAndData`, which picks
/// a layout that minimizes cross-lane reduction (reducing within a lane when
/// only one of the innermost two dims is a reduction dim). The inst_data is
/// simply the element-wise product lane_layout * lane_data.
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
    xegpu::SliceAttr consumerSliceLayout =
        dyn_cast_if_present<xegpu::SliceAttr>(consumerLayout);
    auto reductionDimsOverrideConsumer =
        consumerSliceLayout
            ? SmallVector<int64_t>(consumerSliceLayout.getDims().asArrayRef())
            : reductionDims;
    auto [laneLayout, laneData] = computeReductionLaneLayoutAndData(
        srcShape, reductionDimsOverrideConsumer, subgroupSize,
        maxReduceVectorSize);
    // inst_data is the per-instruction data, i.e. the element-wise product of
    // lane_layout and lane_data.
    SmallVector<int64_t> instData(srcRank);
    for (int i = 0; i < srcRank; i++)
      instData[i] = laneLayout[i] * laneData[i];
    srcLayout =
        buildInstDataLayoutWithLane(context, instData, laneLayout, laneData);
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    // Only the innermost two dimensions are distributed; all leading dimensions
    // are assumed to be unit dimensions.
    assert(leadingDimsAreUnit(srcShape, /*numInnerDims=*/2) &&
           "Lane reduction layout assumes all leading (non-innermost-two) "
           "dimensions are unit dimensions");
    xegpu::SliceAttr consumerSliceLayout =
        dyn_cast_if_present<xegpu::SliceAttr>(consumerLayout);
    auto reductionDimsOverrideConsumer =
        consumerSliceLayout
            ? SmallVector<int64_t>(consumerSliceLayout.getDims().asArrayRef())
            : reductionDims;
    auto [laneLayout, laneData] = computeReductionLaneLayoutAndData(
        srcShape, reductionDimsOverrideConsumer, subgroupSize,
        maxReduceVectorSize);
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

/// Adjusts `consumerLayout`'s innermost-dim data field selected by
/// `layoutKind` so that the source layout can be safely inferred by dividing
/// that value by `ratio`. Doubles the value until the divisibility constraint
/// is met, bounded above by `bound` like result-shape.
///
/// Used by ops whose source relates to the result by a fixed factor along the
/// innermost dim (e.g., bitcast: bitwidth ratio; interleave: 2x).
///
/// Divisibility constraints per LayoutKind:
///   - Subgroup: sgData[innermost] % ratio == 0
///   - InstData: instData[innermost] % (laneLayout[innermost] * ratio) == 0
///               (laneLayout falls back to subgroupSize if absent)
///   - Lane:     laneData[innermost] % ratio == 0
static xegpu::DistributeLayoutAttr
adjustInnermostDimForDivisibility(xegpu::DistributeLayoutAttr consumerLayout,
                                  xegpu::LayoutKind layoutKind,
                                  size_t innerMostDim, int ratio, int64_t bound,
                                  const xegpu::uArch::uArch *uArch) {
  SmallVector<int64_t> sgData = consumerLayout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> instData = consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> laneData = consumerLayout.getEffectiveLaneDataAsInt();
  SmallVector<int64_t> laneLayout =
      consumerLayout.getEffectiveLaneLayoutAsInt();

  int64_t sgDataValue = -1;
  int64_t instDataValue = -1;
  int64_t laneDataValue = -1;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    sgDataValue = sgData[innerMostDim];
    while ((sgDataValue <= bound) && (sgDataValue % ratio) != 0)
      sgDataValue *= 2;
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    instDataValue = instData[innerMostDim];
    const int innermostDimLaneLayout = laneLayout.empty()
                                           ? uArch->getSubgroupSize()
                                           : laneLayout[innerMostDim];
    while ((instDataValue <= bound) &&
           (instDataValue % (innermostDimLaneLayout * ratio) != 0))
      instDataValue *= 2;
    assert((bound % instDataValue) == 0 &&
           "bound, instData, and laneLayout for innermost must be 2^n!");
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    laneDataValue = laneData[innerMostDim];
    while ((laneDataValue <= bound) && (laneDataValue % ratio) != 0)
      laneDataValue *= 2;
  }

  return consumerLayout.setDimData(innerMostDim, sgDataValue, instDataValue,
                                   laneDataValue);
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
  ArrayRef<int64_t> resShape = resVecTy.getShape();

  assert(consumerLayout.getRank() == static_cast<int64_t>(srcShape.size()) &&
         "laneData must be available for all dimensions");

  // Casting to same/larger element type: result has fewer (or equal) elements
  // along the innermost dim, no adjustment needed.
  if (srcElemTyBitWidth <= resElemTyBitWidth)
    return consumerLayout;

  // Casting to smaller element type: result has more elements along innermost
  // dim. Adjust the innermost data field upward so the source layout can be
  // recovered by dividing by bitWidthRatio.
  size_t innerMostDim = srcShape.size() - 1;
  int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
  return adjustInnermostDimForDivisibility(consumerLayout, layoutKind,
                                           innerMostDim, bitWidthRatio,
                                           resShape[innerMostDim], uArch);
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

  ArrayRef<int64_t> resShape = resVecTy.getShape();
  assert(consumerLayout.getRank() == static_cast<int64_t>(resShape.size()) &&
         "consumer layout rank must match source shape rank");

  // Interleave doubles the innermost dimension (ratio = 2). Adjust the
  // innermost data field so the source layout can be recovered by dividing
  // by 2.
  const size_t innerMostDim = resShape.size() - 1;
  constexpr int ratio = 2;
  return adjustInnermostDimForDivisibility(consumerLayout, layoutKind,
                                           innerMostDim, ratio,
                                           resShape[innerMostDim], uArch);
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
  int64_t laneDataValue = -1;

  requiredResLayout = consumerLayout;
  int srcRank = srcShape.size();

  if (layoutKind == xegpu::LayoutKind::Subgroup ||
      layoutKind == xegpu::LayoutKind::InstData) {
    assert(true &&
           "subgroup layout assignment not supported for insertStridedSlice.");
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

/// Computes lane_layout and lane_data for scatter-style store anchor layouts
/// (store scatter, store matrix). Lanes and the per-lane vector both live on
/// the innermost dim:
///   - laneLayout[innermost] = min(subgroupSize, srcShape[innermost])
///   - laneData[innermost]   = min(srcShape[innermost] / laneLayout[innermost],
///                                 maxChunkSize)
/// All other entries are 1.
static std::pair<SmallVector<int>, SmallVector<int>>
computeScatterStoreLaneLayoutAndData(ArrayRef<int64_t> srcShape,
                                     int subgroupSize, int64_t maxChunkSize) {
  int rank = srcShape.size();
  SmallVector<int> laneLayout(rank, 1), laneData(rank, 1);
  int innermost = rank - 1;
  laneLayout[innermost] = std::min(static_cast<int>(subgroupSize),
                                   static_cast<int>(srcShape[innermost]));
  laneData[innermost] =
      std::min(static_cast<int>(srcShape[innermost] / laneLayout[innermost]),
               static_cast<int>(maxChunkSize));
  return {laneLayout, laneData};
}

/// Sets up the anchor layout for load gather and load matrix operation.
/// load matrix lowers to load gather and 1d block load. All of them share the
/// same layout setup logic.
///
/// For Subgroup layout, uses the consumer layout directly.
///
/// For InstData layout, takes consumer's inst_data as-is. lane_layout and
/// lane_data are taken from the consumer when present; otherwise the helper
/// derives the standard scatter-style default (subgroupSize lanes on the
/// innermost dim, per-lane vector capped by maxChunkSize).
///
/// For Lane layout, lane_layout/lane_data are taken from the consumer when
/// present; otherwise derived from the same default.
static xegpu::DistributeLayoutAttr setupGenericLoadAnchorLayout(
    xegpu::LayoutKind layoutKind, mlir::MLIRContext *context,
    xegpu::DistributeLayoutAttr consumerLayout, int maxChunkSize,
    ArrayRef<int64_t> resShape, int subgroupSize) {

  if (layoutKind == xegpu::LayoutKind::Subgroup)
    return consumerLayout;

  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneLayout =
      consumerLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();

  // Pick lane_layout / lane_data: prefer consumer's, fall back to the
  // scatter-store default (subgroupSize lanes on innermost dim, per-lane
  // vector capped by maxChunkSize).
  SmallVector<int64_t> laneLayout;
  SmallVector<int64_t> laneData;
  if (!consumerLaneLayout.empty() && !consumerLaneData.empty()) {
    laneLayout.assign(consumerLaneLayout.begin(), consumerLaneLayout.end());
    laneData.assign(consumerLaneData.begin(), consumerLaneData.end());
  } else {
    auto [defLaneLayout, defLaneData] = computeScatterStoreLaneLayoutAndData(
        resShape, subgroupSize, maxChunkSize);
    laneLayout.assign(defLaneLayout.begin(), defLaneLayout.end());
    laneData.assign(defLaneData.begin(), defLaneData.end());
  }

  if (layoutKind == xegpu::LayoutKind::InstData) {
    // Take consumer's inst_data as-is. If the consumer doesn't have one,
    // fall back to lane_layout * lane_data per dim.
    SmallVector<int64_t> instData;
    if (!consumerInstData.empty()) {
      instData.assign(consumerInstData.begin(), consumerInstData.end());
    } else {
      instData.resize(resShape.size());
      for (size_t i = 0; i < resShape.size(); ++i)
        instData[i] = laneLayout[i] * laneData[i];
    }
    return buildInstDataLayoutWithLane(context, instData, laneLayout, laneData);
  }
  if (layoutKind == xegpu::LayoutKind::Lane)
    return buildLaneLayout(context, laneLayout, laneData);
  return nullptr;
}

/// Sets up the anchor layout for a load gather operation.
xegpu::DistributeLayoutAttr xegpu::setupLoadGatherAnchorLayout(
    xegpu::LayoutKind layoutKind, VectorType resVecTy, int contigChunkSize,
    xegpu::DistributeLayoutAttr consumerLayout, const uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> resShape = resVecTy.getShape();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::LoadGatherInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize = std::min(
      uArchInstruction->getMaxLaneLoadSize(elemBitWidth), contigChunkSize);

  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      maxChunkSize, resShape, subgroupSize);
}

/// Sets up the anchor layout for load matrix operation.
/// TODO: enhance load matrix to indicate lowering to chunked load or not.
xegpu::DistributeLayoutAttr
xegpu::setupLoadMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                   VectorType resVecTy, int contigChunkSize,
                                   xegpu::DistributeLayoutAttr consumerLayout,
                                   const xegpu::uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> resShape = resVecTy.getShape();
  auto context = resVecTy.getContext();
  auto elemBitWidth = resVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::LoadGatherInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize = std::min(
      uArchInstruction->getMaxLaneLoadSize(elemBitWidth), contigChunkSize);
  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      maxChunkSize, resShape, subgroupSize);
}

/// Sets up the anchor layout for store scatter and store matrix operation.
/// store matrix lowers to store scatter and 1d block store. All of them
/// share the same layout setup logic. For Subgroup layout, not supported
/// yet.
///
/// Lane layout is derived first via `computeScatterStoreLaneLayoutAndData`;
/// inst_data is then the element-wise product lane_layout * lane_data.
static xegpu::DistributeLayoutAttr
setupGenericStoreAnchorLayout(xegpu::LayoutKind layoutKind,
                              mlir::MLIRContext *context, int maxChunkSize,
                              ArrayRef<int64_t> srcShape, int subgroupSize) {

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(true &&
           "subgroup layout assignment not supported for storeScatter.");
    return nullptr;
  }

  auto [laneLayout, laneData] = computeScatterStoreLaneLayoutAndData(
      srcShape, subgroupSize, maxChunkSize);

  if (layoutKind == xegpu::LayoutKind::InstData) {
    SmallVector<int> instData(srcShape.size());
    for (size_t i = 0; i < srcShape.size(); ++i)
      instData[i] = static_cast<int>(laneLayout[i] * laneData[i]);
    return xegpu::LayoutAttr::get(
        context, /*sg_layout=*/nullptr,
        /*sg_data=*/nullptr,
        /*inst_data=*/DenseI32ArrayAttr::get(context, instData),
        /*lane_layout=*/DenseI32ArrayAttr::get(context, laneLayout),
        /*lane_data=*/DenseI32ArrayAttr::get(context, laneData),
        /*order=*/nullptr);
  }
  if (layoutKind == xegpu::LayoutKind::Lane) {
    return xegpu::LayoutAttr::get(context, laneLayout, laneData);
  }
  return nullptr;
}

/// Sets up the anchor layout for a store scatter operation.
xegpu::DistributeLayoutAttr
xegpu::setupStoreScatterAnchorLayout(xegpu::LayoutKind layoutKind,
                                     VectorType srcVecTy, int contigChunkSize,
                                     const uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  auto context = srcVecTy.getContext();
  auto elemBitWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize = std::min(
      uArchInstruction->getMaxLaneStoreSize(elemBitWidth), contigChunkSize);
  return setupGenericStoreAnchorLayout(layoutKind, context, maxChunkSize,
                                       srcShape, subgroupSize);
}

/// Sets up the anchor layout for a store matrix operation.
xegpu::DistributeLayoutAttr
xegpu::setupStoreMatrixAnchorLayout(xegpu::LayoutKind layoutKind,
                                    VectorType srcVecTy, int contigChunkSize,
                                    const xegpu::uArch::uArch *uArch) {

  const int subgroupSize = uArch->getSubgroupSize();
  ArrayRef<int64_t> srcShape = srcVecTy.getShape();
  auto context = srcVecTy.getContext();
  auto elemBitWidth = srcVecTy.getElementType().getIntOrFloatBitWidth();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize = std::min(
      uArchInstruction->getMaxLaneStoreSize(elemBitWidth), contigChunkSize);

  return setupGenericStoreAnchorLayout(layoutKind, context, maxChunkSize,
                                       srcShape, subgroupSize);
}

/// If `consumerLayout` has inst_data set but no lane_layout/lane_data,
/// derive a lane factorization by re-running the load-side Lane setup with
/// inst_data as the destination shape, and merge the result back so the
/// returned LayoutAttr carries inst_data + lane_layout + lane_data. This
/// guarantees the lane factorization the downstream load setup will see is
/// the same one its own Lane-kind setup would produce.
xegpu::DistributeLayoutAttr xegpu::completeLoadGatherLayoutFromInstData(
    xegpu::DistributeLayoutAttr consumerLayout, Type elemTy,
    const xegpu::uArch::uArch *uArch) {
  if (!consumerLayout)
    return consumerLayout;
  SmallVector<int64_t> instData = consumerLayout.getEffectiveInstDataAsInt();
  if (instData.empty())
    return consumerLayout;
  if (!consumerLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !consumerLayout.getEffectiveLaneDataAsInt().empty())
    return consumerLayout;

  // Reuse the load-side setup with inst_data as the destination shape.
  const int subgroupSize = uArch->getSubgroupSize();
  auto *context = consumerLayout.getContext();
  auto elemBitWidth = elemTy.getIntOrFloatBitWidth();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::LoadGatherInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  if (!uArchInstruction)
    return consumerLayout;
  int maxChunkSize = uArchInstruction->getMaxLaneLoadSize(elemBitWidth);

  auto laneOnly = setupGenericLoadAnchorLayout(
      xegpu::LayoutKind::Lane, context, /*consumerLayout=*/nullptr,
      maxChunkSize, instData, subgroupSize);
  if (!laneOnly)
    return consumerLayout;

  SmallVector<int64_t> laneLayout = laneOnly.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> laneData = laneOnly.getEffectiveLaneDataAsInt();
  return buildInstDataLayoutWithLane(context, instData, laneLayout, laneData);
}

/// If `consumerLayout` has inst_data set but no lane_layout/lane_data,
/// derive a lane factorization by re-running the store-side Lane setup with
/// inst_data as the destination shape, and merge the result back. Returned
/// LayoutAttr carries inst_data + lane_layout + lane_data.
xegpu::DistributeLayoutAttr xegpu::completeStoreScatterLayoutFromInstData(
    xegpu::DistributeLayoutAttr consumerLayout, Type elemTy,
    const xegpu::uArch::uArch *uArch) {
  if (!consumerLayout)
    return consumerLayout;
  SmallVector<int64_t> instData = consumerLayout.getEffectiveInstDataAsInt();
  if (instData.empty())
    return consumerLayout;
  if (!consumerLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !consumerLayout.getEffectiveLaneDataAsInt().empty())
    return consumerLayout;

  const int subgroupSize = uArch->getSubgroupSize();
  auto *context = consumerLayout.getContext();
  auto elemBitWidth = elemTy.getIntOrFloatBitWidth();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstructionInterface>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  if (!uArchInstruction)
    return consumerLayout;
  int maxChunkSize = uArchInstruction->getMaxLaneStoreSize(elemBitWidth);

  auto laneOnly = setupGenericStoreAnchorLayout(
      xegpu::LayoutKind::Lane, context, maxChunkSize, instData, subgroupSize);
  if (!laneOnly)
    return consumerLayout;

  SmallVector<int64_t> laneLayout = laneOnly.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> laneData = laneOnly.getEffectiveLaneDataAsInt();
  return buildInstDataLayoutWithLane(context, instData, laneLayout, laneData);
}

// Forward declaration: defined later in the file.
using LayoutRepresentation = std::pair<int64_t, int64_t>;
static SmallVector<LayoutRepresentation>
getValidLayouts(ArrayRef<int64_t> wgShape, ArrayRef<int64_t> instData,
                int64_t sgCount);

/// Validates whether `instData` is a hardware-viable inst_data for an ND op
/// with the given block params and lane factor. Specifically:
///  - leading batch dims must be 1
///  - innermost dim must be a divisor of `dataShape.back()`, a multiple of
///    `bWidths.front()` (smallest supported block width), and ≤ a small
///    multiple of the largest supported block width
///  - second-to-innermost dim must be in the supported heights for rank >= 2
///  - each dim must be a multiple of `lane_layout[dim] * lane_data[dim]`
static bool isValidNdInstData(ArrayRef<int64_t> instData,
                              ArrayRef<int64_t> dataShape,
                              ArrayRef<int> bWidths, ArrayRef<int> bHeights,
                              ArrayRef<int64_t> laneLayout,
                              ArrayRef<int64_t> laneData) {
  int rank = dataShape.size();
  if (static_cast<int>(instData.size()) != rank)
    return false;

  for (int dim = 0; dim < rank - 2; ++dim)
    if (instData[dim] != 1)
      return false;

  int64_t inner = instData.back();
  if (inner <= 0 || dataShape.back() % inner != 0)
    return false;
  int minWidth = *llvm::min_element(bWidths);
  int maxWidth = *llvm::max_element(bWidths);
  if (inner % minWidth != 0 || inner > maxWidth * /*maxBlockCount*/ 4)
    return false;

  if (rank >= 2) {
    int64_t height = instData[rank - 2];
    if (!llvm::is_contained(bHeights, static_cast<int>(height)))
      return false;
  }

  for (int dim = 0; dim < rank; ++dim) {
    int64_t laneProduct = laneLayout[dim] * laneData[dim];
    if (laneProduct == 0 || instData[dim] % laneProduct != 0)
      return false;
  }
  return true;
}

/// Generic anchor-layout setup for ND ops (load_nd, store_nd, prefetch_nd).
///
/// Given hardware-supported block widths/heights, picks the largest divisor
/// of the trailing two dims of `dataShape` as the default `inst_data`. The
/// lane layout is the standard 2D-block-IO default (subgroupSize lanes on the
/// innermost dim, optional packing on the innermost dim).
///
/// `consumerLayout` (optional) is honored when its parameters are valid w.r.t.
/// the uArch constraints; otherwise the helper falls back to defaults.
///
/// For Lane kind: returns just the lane layout / lane data (consumer ignored;
///   lane layout is fully determined by hardware).
/// For InstData kind: returns inst_data + lane_layout/lane_data (Category A:
///   inst_data = k * lane_layout * lane_data, k >= 1). Honors consumer's
///   inst_data when it is uArch-valid.
/// For Subgroup kind: if the consumer specifies a workgroup-level layout,
///   reuses it directly; otherwise picks the most balanced sg_layout via
///   `getValidLayouts` (requires `numSg`).
static xegpu::DistributeLayoutAttr setupGenericNdAnchorLayout(
    xegpu::LayoutKind layoutKind, mlir::MLIRContext *context,
    ArrayRef<int64_t> dataShape, Type elemTy, ArrayRef<int> bWidths,
    ArrayRef<int> bHeights, unsigned packingSize,
    xegpu::DistributeLayoutAttr consumerLayout, int numSg,
    const xegpu::uArch::uArch *uArch) {
  int rank = dataShape.size();
  assert(rank >= 1 && "Expected at least 1D shape for ND op");

  // Compute the default 2D block IO lane layout / lane data. Cap each by
  // the innermost shape so the product `lane_layout * lane_data` doesn't
  // exceed it (e.g. ui8 with shape innermost=16 cannot use the full
  // packing factor of 2 because subgroupSize * 2 = 32 > 16).
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  int packingFactor = bitwidth < packingSize ? packingSize / bitwidth : 1;
  SmallVector<int64_t> laneLayout(rank, 1);
  SmallVector<int64_t> laneData(rank, 1);
  int64_t innermostShape = dataShape.back();
  int64_t lanesOnInnermost =
      std::min<int64_t>(uArch->getSubgroupSize(), innermostShape);
  laneLayout.back() = lanesOnInnermost;
  if (lanesOnInnermost > 0)
    laneData.back() =
        std::min<int64_t>(packingFactor, innermostShape / lanesOnInnermost);
  else
    laneData.back() = 1;

  // Honor the consumer's lane info when valid. The consumer (e.g. dpas) may
  // require VNNI-style packing on a non-innermost dim that the load's own
  // packing default doesn't capture.
  auto honorConsumerLane = [&]() {
    if (!consumerLayout)
      return;
    SmallVector<int64_t> consumerLaneLayout =
        consumerLayout.getEffectiveLaneLayoutAsInt();
    SmallVector<int64_t> consumerLaneData =
        consumerLayout.getEffectiveLaneDataAsInt();
    if (consumerLaneLayout.size() != static_cast<size_t>(rank) ||
        consumerLaneData.size() != static_cast<size_t>(rank))
      return;
    // Validate: lane_layout * lane_data must divide each dim of dataShape.
    for (int dim = 0; dim < rank; ++dim) {
      int64_t product = consumerLaneLayout[dim] * consumerLaneData[dim];
      if (product == 0 || dataShape[dim] % product != 0)
        return;
    }
    laneLayout.assign(consumerLaneLayout.begin(), consumerLaneLayout.end());
    laneData.assign(consumerLaneData.begin(), consumerLaneData.end());
  };
  honorConsumerLane();

  // If the consumer carries an explicit `order`, propagate it through.
  DenseI32ArrayAttr orderAttr =
      consumerLayout ? consumerLayout.getOrder() : nullptr;

  if (layoutKind == xegpu::LayoutKind::Lane)
    return buildLaneLayout(context, laneLayout, laneData, orderAttr);

  // Subgroup-kind fast path: if the consumer already specifies a
  // workgroup-level layout, reuse it directly. Skip the inst_data
  // computation, which can fail for very small shapes (e.g. dpas_mx scale
  // operands like 128x16 where no supported block width divides 16).
  if (layoutKind == xegpu::LayoutKind::Subgroup && consumerLayout &&
      consumerLayout.isForWorkgroup())
    return consumerLayout;

  // Compute inst_data from hardware block params. For Nd ops, the lane
  // factorization above (laneLayout / laneData) is rigid; inst_data must be
  // a multiple of lane_layout * lane_data on each dim (Category A
  // invariant). If block params are unavailable for this element type
  // (e.g. sub-byte floats with no uArch entry), fall back to
  // lane_layout * lane_data (k = 1).
  SmallVector<int64_t> instData(rank, 1);
  int instWidth =
      xegpu::getLargestDivisor(static_cast<int>(dataShape.back()), bWidths);
  if (instWidth == -1)
    instData.back() = laneLayout.back() * laneData.back();
  else
    instData.back() = instWidth;
  if (rank >= 2) {
    int instHeight = xegpu::getLargestDivisor(
        static_cast<int>(dataShape[rank - 2]), bHeights);
    if (instHeight == -1)
      instData[rank - 2] = laneLayout[rank - 2] * laneData[rank - 2];
    else
      instData[rank - 2] = instHeight;
  }

  // Honor the consumer's inst_data if it is uArch-valid.
  if (consumerLayout) {
    SmallVector<int64_t> consumerInstData =
        consumerLayout.getEffectiveInstDataAsInt();
    if (!consumerInstData.empty() &&
        isValidNdInstData(consumerInstData, dataShape, bWidths, bHeights,
                          laneLayout, laneData))
      instData.assign(consumerInstData.begin(), consumerInstData.end());
  }

  // Category A invariant: inst_data is a multiple of lane_layout * lane_data.
  for (int dim = 0; dim < rank; ++dim) {
    int64_t laneProduct = laneLayout[dim] * laneData[dim];
    assert(instData[dim] % laneProduct == 0 &&
           "inst_data must be a multiple of lane_layout * lane_data for ND op");
    (void)laneProduct;
  }

  if (layoutKind == xegpu::LayoutKind::InstData)
    return buildInstDataLayoutWithLane(context, instData, laneLayout, laneData,
                                       orderAttr);

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    if (rank != 2)
      return nullptr;
    // The consumer-fast-path above already returned for the case where the
    // consumer carries a workgroup-level layout, so here numSg must be set.
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto sgLayouts = getValidLayouts(dataShape, instData, numSg);
    if (sgLayouts.empty())
      return nullptr;
    SmallVector<int> sgLayout = {static_cast<int>(sgLayouts[0].first),
                                 static_cast<int>(sgLayouts[0].second)};
    SmallVector<int> sgData = {static_cast<int>(dataShape[0]) / sgLayout[0],
                               static_cast<int>(dataShape[1]) / sgLayout[1]};
    return xegpu::LayoutAttr::get(
        context, DenseI32ArrayAttr::get(context, sgLayout),
        DenseI32ArrayAttr::get(context, sgData),
        /*inst_data=*/nullptr, /*lane_layout=*/nullptr,
        /*lane_data=*/nullptr, /*order=*/nullptr);
  }

  return nullptr;
}

/// Sets up the anchor layout for a store_nd operation. StoreNd picks its
/// own layout based on uArch block parameters (it does not take a consumer
/// layout, since it is a data sink).
xegpu::DistributeLayoutAttr
xegpu::setupStoreNdAnchorLayout(xegpu::LayoutKind layoutKind,
                                VectorType srcVecTy, int numSg,
                                const xegpu::uArch::uArch *uArch) {
  auto context = srcVecTy.getContext();
  Type elemTy = srcVecTy.getElementType();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockStoreInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockStore));
  if (!uArchInstruction)
    return nullptr;
  auto blockWHC = uArchInstruction->getBlockWidthHeightCount(elemTy);
  if (!blockWHC)
    return nullptr;
  auto [bWidths, bHeights, bCounts] = blockWHC.value();
  unsigned packingSize = uArchInstruction->getPackedFormatBitSize();

  return setupGenericNdAnchorLayout(layoutKind, context, srcVecTy.getShape(),
                                    elemTy, bWidths, bHeights, packingSize,
                                    /*consumerLayout=*/nullptr, numSg, uArch);
}

/// Sets up the anchor layout for a prefetch_nd operation. PrefetchNd has no
/// consumer (it produces no value), so it picks its own layout from uArch
/// block parameters.
xegpu::DistributeLayoutAttr
xegpu::setupPrefetchNdAnchorLayout(xegpu::LayoutKind layoutKind,
                                   xegpu::TensorDescType tdescTy, int numSg,
                                   const xegpu::uArch::uArch *uArch) {
  auto context = tdescTy.getContext();
  Type elemTy = tdescTy.getElementType();

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockPrefetchInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockPrefetch));
  if (!uArchInstruction)
    return nullptr;
  auto blockWHC = uArchInstruction->getBlockWidthHeightCount(elemTy);
  if (!blockWHC)
    return nullptr;
  auto [bWidths, bHeights, bCounts] = blockWHC.value();
  unsigned packingSize = uArchInstruction->getPackedFormatBitSize();

  return setupGenericNdAnchorLayout(layoutKind, context, tdescTy.getShape(),
                                    elemTy, bWidths, bHeights, packingSize,
                                    /*consumerLayout=*/nullptr, numSg, uArch);
}

/// Sets up the anchor layout for a load_nd operation. LoadNd takes a
/// consumer layout (from its result's downstream uses) and validates it
/// against uArch constraints; if valid, the consumer's `inst_data` /
/// `sg_layout` are honored. Otherwise the helper falls back to defaults
/// derived from uArch block parameters.
xegpu::DistributeLayoutAttr
xegpu::setupLoadNdAnchorLayout(xegpu::LayoutKind layoutKind,
                               VectorType resVecTy,
                               xegpu::DistributeLayoutAttr consumerLayout,
                               int numSg, const xegpu::uArch::uArch *uArch) {
  auto context = resVecTy.getContext();
  Type elemTy = resVecTy.getElementType();

  // Subgroup-kind fast path: if the consumer already specifies a complete
  // workgroup-level layout, reuse it directly. We don't need the uArch block
  // params at all (which may be unavailable for unusual element types like
  // sub-byte floats used in dpas_mx scale operands).
  if (layoutKind == xegpu::LayoutKind::Subgroup && consumerLayout &&
      consumerLayout.isForWorkgroup())
    return consumerLayout;

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockLoadInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockLoad));
  if (!uArchInstruction)
    return nullptr;
  unsigned packingSize = uArchInstruction->getPackedFormatBitSize();

  // Lane kind only needs subgroupSize + packingSize. Skip the block-WHC
  // lookup, which can fail for element types without a uArch entry (e.g.
  // sub-byte floats like f4E2M1FN), and let the generic helper produce a
  // default lane layout from packingSize alone.
  if (layoutKind == xegpu::LayoutKind::Lane)
    return setupGenericNdAnchorLayout(
        layoutKind, context, resVecTy.getShape(), elemTy,
        /*bWidths=*/{}, /*bHeights=*/{}, packingSize, consumerLayout, numSg,
        uArch);

  // InstData / Subgroup kinds need block params. Transform / transpose /
  // upConv are lane-level concerns; treat them as no-op at the propagation
  // stage (consistent with visitLoadNdOp, which warns on transpose).
  auto blockWHC = uArchInstruction->getBlockWidthHeightCount(
      elemTy, /*hasTransform=*/false, /*hasTranspose=*/false,
      /*upConv=*/false);
  if (!blockWHC)
    return nullptr;
  auto [bWidths, bHeights, bCounts] = blockWHC.value();

  return setupGenericNdAnchorLayout(layoutKind, context, resVecTy.getShape(),
                                    elemTy, bWidths, bHeights, packingSize,
                                    consumerLayout, numSg, uArch);
}

// Returns the default (lane_layout, lane_data) pair for a given 1D/2D vector
// type used by 2D block IO ops.
// - `packingSize` means multiple consecutive elements can be accessed
//   together as a single unit.
// - `vnni` means data packing is column-wise (i.e., 2x1xf16 with vnni vs.
//   1x2xf16 w/o vnni).
template <typename RankedTy>
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
getDefaultLaneLayoutAndData2DBlockIo(
    RankedTy ty, const xegpu::uArch::uArch *uArch,
    std::optional<unsigned> packingSize = std::nullopt, bool vnni = false) {
  // Expecting at least 1D vector. For rank > 2, leading dims are batch dims.
  assert(((ty.getRank() >= 1 && !vnni) || ty.getRank() >= 2) &&
         "Expected at least 1D non-vnni or 2D vector.");
  // Expecting int or float element type.
  assert(ty.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");

  auto rank = ty.getRank();
  SmallVector<int64_t> laneLayout(rank, 1);
  SmallVector<int64_t> laneData(rank, 1);
  if (packingSize.has_value()) {
    unsigned bitwidth = ty.getElementType().getIntOrFloatBitWidth();
    int64_t &laneDataPos = vnni ? laneData[rank - 2] : laneData.back();
    laneDataPos = bitwidth < *packingSize ? *packingSize / bitwidth : 1;
  }
  laneLayout.back() = uArch->getSubgroupSize();
  return {laneLayout, laneData};
}

// Convenience wrapper: returns a LayoutAttr carrying only the default lane
// layout / lane data for a 2D block IO vector type.
template <typename RankedTy>
static xegpu::LayoutAttr getDefaultLaneLayout2DBlockIo(
    RankedTy ty, const xegpu::uArch::uArch *uArch,
    std::optional<unsigned> packingSize = std::nullopt, bool vnni = false) {
  auto [laneLayout, laneData] =
      getDefaultLaneLayoutAndData2DBlockIo(ty, uArch, packingSize, vnni);
  return buildLaneLayout(ty.getContext(), laneLayout, laneData);
}

// This function returns all layouts for the given sgCount, whose sgData:
// 1. Evenly divides the wgShape.
// 2. Is a multiple of instData.
// Example:
//   wgShape = [128, 64], instData = [8, 16], sgCount = 32
// Returns layouts:
//   [(8,4), (16,2)], which correspond to sgData [16,16] and [8,32].
// Definition (forward-declared above).
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

  // M dimension is the second-to-last dim of A (handles batch dims).
  const unsigned dataALen = aTy.getShape()[aTy.getRank() - 2];
  auto supportedALen = uArchInstruction->getSupportedM(aTy.getElementType());
  const int maxALen =
      xegpu::getLargestDivisor(dataALen, ArrayRef<unsigned>(supportedALen));

  // N dimension is the last dim of B.
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
    if (supportedKLen.empty())
      return std::nullopt;
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

/// Helper function to set up subgroup layouts for DPAS operands A, B, and
/// C/D. Returns the three layouts if successful, nullopt otherwise.
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
  auto [laneLayoutA, laneDataA] = getDefaultLaneLayoutAndData2DBlockIo(
      aTy, uArch, uArchInstruction->getPackedFormatBitSizeA());
  auto [laneLayoutB, laneDataB] = getDefaultLaneLayoutAndData2DBlockIo(
      bTy, uArch, uArchInstruction->getPackedFormatBitSizeB(), true);
  auto [laneLayoutCD, laneDataCD] =
      getDefaultLaneLayoutAndData2DBlockIo(cdTy, uArch);
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
        buildInstDataLayoutWithLane(context, instDataA, laneLayoutA, laneDataA),
        buildInstDataLayoutWithLane(context, instDataB, laneLayoutB, laneDataB),
        buildInstDataLayoutWithLane(context, instDataCD, laneLayoutCD,
                                    laneDataCD));
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    auto aLayout = buildLaneLayout(context, laneLayoutA, laneDataA);
    auto bLayout = buildLaneLayout(context, laneLayoutB, laneDataB);
    auto cdLayout = buildLaneLayout(context, laneLayoutCD, laneDataCD);
    return std::make_tuple(aLayout, bLayout, cdLayout);
  }
  return std::nullopt;
}

/// Helper to create a scale layout derived from a matrix operand layout.
/// The scale layout is computed by mapping each dimension of the matrix
/// layout to the corresponding scale tensor dimension using the ratio
/// between the matrix and scale shapes.
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
  assert(rank >= 2 && "dpas layouts must be at least two dimensions");

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
    if (isBScale ^ isRowMajor)
      std::swap(scaleLaneLayout[rank - 2], scaleLaneLayout[rank - 1]);
    // Cap lane_layout by the per-instruction tile (inst_data) on each dim.
    // Then derive lane_data = inst_data / lane_layout so the Category A
    // invariant inst_data = lane_layout * lane_data * k (with k = 1) holds
    // for the scale operand's load_nd consumer.
    if (!scaleInstData.empty()) {
      for (int64_t d = rank - 2; d < rank; ++d) {
        scaleLaneLayout[d] =
            std::min<int64_t>(scaleInstData[d], scaleLaneLayout[d]);
        scaleLaneData[d] = std::max<int64_t>(
            scaleInstData[d] / std::max<int64_t>(scaleLaneLayout[d], 1), 1);
      }
    } else {
      // No inst_data on the matrix layout; fall back to capping by scale
      // shape and deriving lane_data from it (legacy behavior).
      scaleLaneLayout[rank - 2] =
          std::min<int64_t>(scaleShape[rank - 2], scaleLaneLayout[rank - 2]);
      scaleLaneData[rank - 2] = std::max<int64_t>(
          scaleShape[rank - 2] / scaleLaneLayout[rank - 2], 1);
      scaleLaneData[rank - 1] = std::max<int64_t>(
          scaleShape[rank - 1] / scaleLaneLayout[rank - 1], 1);
    }
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
/// B_scale). The numSg and consumerLayout (optional) are only used by sg
/// layout creation.
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

    const auto *uArchInstruction =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    auto [laneLayoutA, laneDataA] = getDefaultLaneLayoutAndData2DBlockIo(
        aTy, uArch, uArchInstruction->getPackedFormatBitSizeA());
    auto [laneLayoutB, laneDataB] = getDefaultLaneLayoutAndData2DBlockIo(
        bTy, uArch, uArchInstruction->getPackedFormatBitSizeB(), true);
    auto [laneLayoutCD, laneDataCD] =
        getDefaultLaneLayoutAndData2DBlockIo(cdTy, uArch);
    auto dpasALayout =
        buildInstDataLayoutWithLane(context, instDataA, laneLayoutA, laneDataA);
    auto dpasBLayout =
        buildInstDataLayoutWithLane(context, instDataB, laneLayoutB, laneDataB);
    auto dpasCDLayout = buildInstDataLayoutWithLane(context, instDataCD,
                                                    laneLayoutCD, laneDataCD);

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

xegpu::DistributeLayoutAttr xegpu::inferSourceLayoutFromResultForNonAnchorOp(
    OpOperand &operand, xegpu::DistributeLayoutAttr resLayout) {
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

  // For vector::InsertStridedSliceOp, infer source layout from result
  // layout. Dest vector must have the same layout as the result.
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

  // For elementwise operations, all operands must have the same layout as
  // the result.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return resLayout;

  return nullptr;
}

xegpu::DistributeLayoutAttr xegpu::getConsumerLayoutAt(OpOperand &operand) {
  Operation *op = operand.getOwner();
  // Anchor ops declare the layout they
  // require on each operand. Trust that declaration directly so that
  // ResolveLayoutConflicts compares producer-vs-declared
  if (isa<xegpu::AnchorLayoutInterface>(op))
    return xegpu::getDistributeLayoutAttr(operand);
  // For non-anchor ops, derive the operand layout from the op's result
  // layout via op-specific semantics.
  xegpu::DistributeLayoutAttr resLayout;
  if (op->getNumResults() == 1 || isa<vector::DeinterleaveOp>(op))
    resLayout = xegpu::getDistributeLayoutAttr(op->getResult(0));
  return inferSourceLayoutFromResultForNonAnchorOp(operand, resLayout);
}
