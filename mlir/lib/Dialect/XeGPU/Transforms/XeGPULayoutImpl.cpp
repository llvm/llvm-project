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
      if (successor.isOperation()) {
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

/// Returns true if every dimension of `shape` except the innermost
/// `numInnerDims` is a unit (size-1) dimension.
[[maybe_unused]] static bool leadingDimsAreUnit(ArrayRef<int64_t> shape,
                                                int numInnerDims) {
  int numLeading = static_cast<int>(shape.size()) - numInnerDims;
  if (numLeading <= 0)
    return true;
  return llvm::all_of(shape.take_front(numLeading),
                      [](int64_t dim) { return dim == 1; });
}

static xegpu::LayoutAttr buildInstDataLayoutWithLane(
    mlir::MLIRContext *context, ArrayRef<int64_t> instData,
    ArrayRef<int64_t> laneLayout, ArrayRef<int64_t> laneData,
    DenseI32ArrayAttr orderAttr = nullptr) {
  auto toI32Attr = [&](auto range) {
    SmallVector<int32_t> v(range.begin(), range.end());
    return DenseI32ArrayAttr::get(context, v);
  };
  return xegpu::LayoutAttr::get(context, /*sg_layout=*/nullptr,
                                /*sg_data=*/nullptr, toI32Attr(instData),
                                toI32Attr(laneLayout), toI32Attr(laneData),
                                orderAttr);
}

static bool isValidLaneLayout(ArrayRef<int64_t> dataShape,
                              ArrayRef<int64_t> laneLayout,
                              ArrayRef<int64_t> laneData) {
  return !llvm::any_of(llvm::seq<int>(0, dataShape.size()), [&](int dim) {
    return dataShape[dim] % (laneLayout[dim] * laneData[dim]) != 0;
  });
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
                                /*inst_data=*/nullptr, toI32Attr(laneLayout),
                                toI32Attr(laneData), orderAttr);
}

static xegpu::LayoutAttr
buildLayout(mlir::MLIRContext *context, ArrayRef<int64_t> sgLayout,
            ArrayRef<int64_t> sgData, ArrayRef<int64_t> instData,
            ArrayRef<int64_t> laneLayout, ArrayRef<int64_t> laneData,
            DenseI32ArrayAttr orderAttr = nullptr) {
  auto toI32Attr = [&](auto range) {
    SmallVector<int32_t> v(range.begin(), range.end());
    return DenseI32ArrayAttr::get(context, v);
  };
  return xegpu::LayoutAttr::get(
      context, sgLayout.empty() ? nullptr : toI32Attr(sgLayout),
      sgData.empty() ? nullptr : toI32Attr(sgData),
      instData.empty() ? nullptr : toI32Attr(instData),
      laneLayout.empty() ? nullptr : toI32Attr(laneLayout),
      laneData.empty() ? nullptr : toI32Attr(laneData), orderAttr);
}

static xegpu::LayoutAttr buildSgLayout(mlir::MLIRContext *context,
                                       ArrayRef<int64_t> wgTileShape,
                                       ArrayRef<int64_t> sgLayout,
                                       int dimK = -1,
                                       DenseI32ArrayAttr orderAttr = nullptr) {
  SmallVector<int64_t> sgData(sgLayout.size());
  for (int dim = 0; dim < (int)sgLayout.size(); ++dim) {
    if (dim == dimK)
      sgData[dim] = wgTileShape[dim];
    else
      sgData[dim] = wgTileShape[dim] / sgLayout[dim];
  }
  return buildLayout(context, sgLayout, sgData,
                     /*inst_data=*/{}, /*lane_layout=*/{},
                     /*lane_data=*/{}, /*order=*/nullptr);
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

  // Right-aligned source in result, look for stretched unit dims.
  for (size_t i = dimDiff; i < resShape.size(); i++) {
    if ((srcShape[i - dimDiff] == 1) && (resShape[i] != 1))
      bcastDims.push_back(i);
  }

  // Case UnitDimStretch (e.g., 1x4 -> 4x4): the source layout data field must
  // be 1.
  if (!bcastDims.empty())
    bcastSourceLayout = bcastSourceLayout.setUnitDimData(bcastDims);

  // Case RankDiff:
  if (dimDiff) {
    SmallVector<int64_t> sliceDims;
    bool isOuterDimDiffUnitDims = llvm::all_of(
        resShape.take_front(dimDiff), [&](int64_t dim) { return dim == 1; });
    if (dimDiff && bcastDims.size() == dimDiff && isOuterDimDiffUnitDims) {
      // Case RankDiffInnerDims (e.g., 1x4 -> 1x16x4):
      //  slice the expanded inner dims
      sliceDims.assign(bcastDims.begin(), bcastDims.end());
    } else {
      // Case RankDiffOuterDims (e.g., 1x4 -> 1x1x4):
      //  slice the outer dims
      llvm::append_range(sliceDims, llvm::seq<int64_t>(0, dimDiff));
    }
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

    DenseI32ArrayAttr orderAttr = DenseI32ArrayAttr::get(
        context, SmallVector<int32_t>(order.begin(), order.end()));
    if (!resLayout.getOrder())
      orderAttr = nullptr;

    return buildLayout(context, sgLayout, sgData, instData, laneLayout,
                       laneData, orderAttr);
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
  SmallVector<SmallVector<int64_t>> collapseDims;
  if (xegpu::matchDimCollapse(srcShape, resShape, collapseDims)) {
    auto srcLayout = resLayout;
    for (int64_t dstIdx = static_cast<int64_t>(collapseDims.size()) - 1;
         dstIdx >= 0; --dstIdx) {
      ArrayRef<int64_t> srcDims = collapseDims[dstIdx];
      if (srcDims.empty()) {
        srcLayout = srcLayout.dropDims({dstIdx});
        continue;
      }
      if (srcDims.size() == 1)
        continue;
      SmallVector<int64_t> targetShape;
      targetShape.reserve(srcDims.size());
      for (int64_t d : srcDims)
        targetShape.push_back(srcShape[d]);
      srcLayout = srcLayout.expandDim(dstIdx, targetShape);
    }
    return srcLayout;
  }
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

//===----------------------------------------------------------------------===//
// Layout derivation helpers: factorize sgCount into
// sg_layout candidates, then
// compute per-subgroup (sgData) and per-lane
// (lane_layout/lane_data/inst_data).
//===----------------------------------------------------------------------===//

using LayoutRepresentation = SmallVector<int64_t>;

/// Enumerates all ways to split `total` into `rank` factors whose product
/// equals `total`. Returns the list of all such factorizations.
static SmallVector<LayoutRepresentation> enumerateFactorizations(int64_t total,
                                                                 int64_t rank) {
  SmallVector<LayoutRepresentation> results;
  SmallVector<int64_t> current(rank, 0);

  // Returns all divisors of `n` in ascending order.
  auto getDivisors = [](int64_t n) {
    SmallVector<int64_t> divs;
    for (int64_t i = 1; i * i <= n; ++i) {
      if (n % i == 0) {
        divs.push_back(i);
        if (i != n / i)
          divs.push_back(n / i);
      }
    }
    llvm::sort(divs);
    return divs;
  };

  std::function<void(int64_t, int64_t)> generate = [&](int64_t dim,
                                                       int64_t remaining) {
    if (dim == rank - 1) {
      current[dim] = remaining;
      results.push_back(LayoutRepresentation(current));
      return;
    }
    for (int64_t factor : getDivisors(remaining)) {
      current[dim] = factor;
      generate(dim + 1, remaining / factor);
    }
  };

  generate(0, total);
  return results;
}

// Computes all valid N-dimensional sg_layout candidates for the given
// sgCount, whose sgData (= wgShape / sgLayout):
//   1. Evenly divides wgShape (i.e., wgShape[d] % sgLayout[d] == 0).
//   2. Is a multiple of instData (i.e., sgData[d] % instData[d] == 0).
// Results are sorted by balance (smallest max-min spread first), with
// lexicographic order as a tiebreaker.
//
// Example (2D):
//   wgShape = [128, 64], instData = [8, 16], sgCount = 32
//   Returns: [[8,4], [16,2]], corresponding to sgData [16,16] and [8,32].
static SmallVector<LayoutRepresentation>
getSgLayoutCandidates(ArrayRef<int64_t> wgShape, ArrayRef<int64_t> instData,
                      int64_t sgCount) {
  int64_t rank = wgShape.size();
  assert(rank > 0 && "wgShape must be non-empty");
  assert(static_cast<int64_t>(instData.size()) == rank &&
         "instData rank must match wgShape rank");

  // Step 1: Get all N-D factorizations of sgCount.
  auto allFactorizations = enumerateFactorizations(sgCount, rank);

  // Step 2: Filter to keep only valid candidates.
  SmallVector<LayoutRepresentation> candidates;
  for (const auto &sgLayout : allFactorizations) {
    bool valid = true;
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (wgShape[dim] % sgLayout[dim] != 0) {
        valid = false;
        break;
      }
      int64_t sgData = wgShape[dim] / sgLayout[dim];
      if (sgData % instData[dim] != 0) {
        valid = false;
        break;
      }
    }
    if (valid)
      candidates.push_back(sgLayout);
  }

  // Step 3: Sort by balance (smallest max-min spread), then lexicographic.
  llvm::sort(candidates, [](const LayoutRepresentation &lhs,
                            const LayoutRepresentation &rhs) {
    int64_t spreadLhs = *llvm::max_element(lhs) - *llvm::min_element(lhs);
    int64_t spreadRhs = *llvm::max_element(rhs) - *llvm::min_element(rhs);
    if (spreadLhs != spreadRhs)
      return spreadLhs < spreadRhs;
    return lhs < rhs;
  });
  return candidates;
}

/// Helper function to compute inst_data vectors for DPAS operands A, B, and
/// C/D.
static std::optional<SmallVector<int64_t>> get2DBlockIOInstDataLayout(
    ArrayRef<int64_t> dataShape, Type elemTy,
    const xegpu::uArch::BlockIOInstructionInterface *uArchInstruction,
    bool transform = false, bool transpose = false) {
  int rank = dataShape.size();
  auto blockWHC =
      uArchInstruction->getBlockWidthHeightCount(elemTy, transform, transpose);
  if (!blockWHC)
    return std::nullopt;
  auto [bWidths, bHeights, bCounts] = blockWHC.value();
  // Compute inst_data from hardware block params. For Nd ops, the lane
  // factorization above (laneLayout / laneData) is rigid; inst_data must be
  // a multiple of lane_layout * lane_data on each dim (Category A
  // invariant).
  SmallVector<int64_t> instData(rank, 1);
  assert(rank >= 2 && "dataShape must be at least 2D for 2D-block IO");
  int instWidth =
      xegpu::getLargestDivisor(static_cast<int>(dataShape.back()), bWidths);
  int instHeight =
      xegpu::getLargestDivisor(static_cast<int>(dataShape[rank - 2]), bHeights);
  instData.back() = instWidth;
  instData[rank - 2] = instHeight;

  return instData;
}

/// Helper function to compute inst_data vectors for DPAS operands A, B, and
/// C/D. Look up the uArch table and search for the largest supported block size
/// that divides the data shape
static std::optional<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                                SmallVector<int64_t>>>
getDpasInstDataLayouts(
    VectorType aTy, VectorType bTy, VectorType cdTy,
    const xegpu::uArch::MMAInstructionInterface *uArchInstruction) {

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

  auto supportedKLen = uArchInstruction->getSupportedK(aTy.getElementType());
  if (supportedKLen.empty())
    return std::nullopt;
  auto kDimSize = supportedKLen[0];

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

/// Computes lane_layout and lane_data for scatter-style store anchor layouts
/// (store scatter, store matrix). Lanes and the per-lane vector both live on
/// the innermost dim:
///   - laneLayout[innermost] = min(subgroupSize, srcShape[innermost])
///   - laneData[innermost]   = min(srcShape[innermost] / laneLayout[innermost],
///                                 maxChunkSize)
/// All other entries are 1.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
computeScatterIOLaneLayoutAndData(ArrayRef<int64_t> instShape,
                                  int64_t subgroupSize, int64_t maxChunkSize) {
  int64_t rank = instShape.size();
  SmallVector<int64_t> laneLayout(rank, 1), laneData(rank, 1);
  int64_t innermost = rank - 1;
  laneLayout[innermost] = std::min(subgroupSize, instShape[innermost]);
  laneData[innermost] =
      std::min(instShape[innermost] / laneLayout[innermost], maxChunkSize);
  return {laneLayout, laneData};
}

// Computes the per-lane layout and data for a 2D block load/store/prefetch:
// lanes are spread across the subgroup along the last dim (or rank-2 if
// transposed), and laneData packs sub-bitwidth elements along the packing dim.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
compute2DBlockIOLaneLayoutAndData(ArrayRef<int64_t> instShape,
                                  int64_t subgroupSize, int64_t bitwidth,
                                  int64_t packingSize, bool transform = false) {
  int64_t rank = instShape.size();
  SmallVector<int64_t> laneLayout(rank, 1), laneData(rank, 1);
  int kDim = transform ? rank - 2 : rank - 1;
  unsigned vnniFactor = packingSize / bitwidth;
  laneData[kDim] = bitwidth < packingSize ? vnniFactor : 1;
  laneLayout.back() =
      std::min(subgroupSize, instShape.back() / laneData.back());

  // assert that the lane layout and data fit in the inst shape
  for (int64_t i = 0; i < rank; ++i) {
    int64_t laneProduct = laneLayout[i] * laneData[i];
    assert(instShape[i] % laneProduct == 0 &&
           "lane_layout * lane_data must evenly divide the inst shape");
    (void)laneProduct;
  }
  return {laneLayout, laneData};
}

/// Computes the (lane_layout, lane_data) for a multi-reduction's source layout.
/// Only the innermost two dims are distributed; leading dims are assumed unit.
/// `subgroupSize` lanes go on one dim; up to `maxReduceVectorSize` elements are
/// packed into lane_data on the other. To minimize cross-lane reduction, lanes
/// are spread across a non-reduction dim when possible so the reduction happens
/// within a lane. inst_data is the element-wise product lane_layout *
/// lane_data.
///
/// e.g. with srcShape=[32, 128], subgroupSize=16, maxReduceVectorSize=2:
///   - Switch: reductionDims=[1] and consumerReductionDims=[] -> lanes move
///     to the non-reduction dim 0: lane_layout=[16, 1], lane_data=[1, 2].
///   - Default: reductionDims=[0, 1] (both reduced) -> lanes stay on the
///     innermost dim: lane_layout=[1, 16], lane_data=[2, 1].
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
computeReductionLaneLayoutAndData(ArrayRef<int64_t> srcShape,
                                  ArrayRef<int64_t> reductionDims,
                                  int subgroupSize, int64_t maxReduceVectorSize,
                                  bool verticalLaneLayout = false) {
  int srcRank = srcShape.size();
  SmallVector<int64_t> laneLayout(srcRank, 1), laneData(srcRank, 1);

  int innermost = srcRank - 1;
  int secondInnermost = srcRank - 2;

  if (verticalLaneLayout && secondInnermost >= 0) {
    std::swap(innermost, secondInnermost);
  }
  int laneDim = innermost;
  int vectorDim = secondInnermost; // negative for rank 1

  laneLayout[laneDim] =
      std::min(static_cast<int64_t>(subgroupSize), srcShape[laneDim]);
  if (vectorDim >= 0)
    laneData[vectorDim] = std::min(maxReduceVectorSize, srcShape[vectorDim]);

  return {laneLayout, laneData};
}

//===----------------------------------------------------------------------===//
// Result/anchor-layout setup. Each op category derives lane_layout/lane_data
// (and inst_data / sgData) differently. Two things vary across ops:
//
//   * Consumer dependence: consumer-driven ops prefer the layout requested by
//     their downstream uses and fall back to uArch defaults only when it is
//     absent/invalid; sinks (StoreNd, PrefetchNd) have no consumer and always
//     pick their own layout from uArch.
//
//   * Derivation direction between inst_data and lane_layout/lane_data. Both
//     obey the invariant inst_data = k * lane_layout * lane_data, where `k` is
//     a per-dim integer >= 1 giving how many times each lane repeats its
//     access to cover one instruction's data tile (k == 1 means one lane
//     position per element; k > 1 means the instruction loads/stores several
//     elements per lane along that dim). Ops solve this invariant from
//     opposite ends:
//       - Rigid-lane ops (Nd block IO, DPAS): hardware fixes lane_layout /
//         lane_data first, then inst_data is built as a multiple of their
//         product (using get2DBlockIOInstDataLayout / getDpasInstDataLayouts).
//       - inst_data-first ops (scatter load): take inst_data from the consumer
//         and derive lane_layout/lane_data underneath it.
//
//   - DPAS (+DPAS_MX)   : rigid lanes — inst_data from HW block dims; A/B/C/D
//                         lanes/data follow each operand's matmul role; DPAS_MX
//                         additionally lays out the scale operand.
//   - LoadNd            : consumer-driven, rigid lanes — honors the consumer's
//                         inst_data / lane / sg_layout (incl. transpose & VNNI
//                         packing) when it satisfies uArch block constraints,
//                         else falls back to the default 2D-block scheme (lanes
//                         on the last dim, rank-2 if transposed). The fallback
//                         picks the LARGEST uArch block that divides the data
//                         shape, so the resulting inst_data block can be bigger
//                         than what the consumer asked for (fewer, wider
//                         loads).
//   - StoreNd/PrefetchNd: data sinks, no consumer, rigid lanes — pick the
//                         2D-block layout directly from uArch (no VNNI
//                         packing).
//   - Load  (scatter)   : load_gather / load_matrix, consumer-driven,
//                         inst_data-first — reuse the consumer's inst_data and
//                         derive lane_layout/lane_data, else default to lanes +
//                         per-lane chunk on the innermost dim (chunk capped by
//                         maxChunkSize).
//   - Store (scatter)   : store_scatter / store_matrix — same scatter scheme,
//                         but always self-derived from the scatter default.
//   - Reduction         : (multi_)reduction, consumer-driven — distribute the
//                         inner two dims, with lanes on the innermost dim by
//                         default (reducing across lanes) and switched to a
//                         non-reduction dim only when that keeps the reduction
//                         within a lane. Reuses the consumer's slice layout
//                         when it slices exactly the reduction dims, otherwise
//                         re-derives. See setupMultiReductionResultLayout for
//                         the exact switch condition and worked examples.
//   - BitCast/Interleave: scale the innermost data field by the bitwidth /
//                         interleave ratio so the source layout divides back
//                         out.
//   - InsertStridedSlice: clamp lane_data per dim to fit the inserted slice
//                         (Lane kind only; sg/inst layouts unsupported).
//===----------------------------------------------------------------------===//

/// Helper function to set up subgroup layouts for DPAS operands A, B, and
/// C/D. Compute subgroup layout candidates based on wgtile and instData, and
/// then pick the best one that satisfies all operands and the consumer (if
/// specified).
static std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
getDpasSubgroupLayouts(
    mlir::MLIRContext *context, VectorType aTy, VectorType bTy, VectorType cdTy,
    xegpu::DistributeLayoutAttr consumerLayout, int numSg,
    std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>>
        instDataVecs) {
  auto [instDataA, instDataB, instDataCD] = instDataVecs;

  std::optional<LayoutRepresentation> consumerSgLayout = std::nullopt;
  if (consumerLayout && consumerLayout.isForWorkgroup()) {
    consumerSgLayout = consumerLayout.getEffectiveSgLayoutAsInt();
  }

  // Get all valid layouts for A, B and C/D operands
  auto layoutsA = getSgLayoutCandidates(aTy.getShape(), instDataA, numSg);
  auto layoutsB = getSgLayoutCandidates(bTy.getShape(), instDataB, numSg);
  auto layoutsCD = getSgLayoutCandidates(cdTy.getShape(), instDataCD, numSg);
  if (layoutsA.empty() || layoutsB.empty() || layoutsCD.empty())
    return std::nullopt;

  // Pick the best subgroup layout
  std::optional<LayoutRepresentation> bestPick;
  auto checkAlignedSgDataAB = [&](const LayoutRepresentation &sgLayout) {
    return aTy.getShape().back() / sgLayout[1] ==
           bTy.getShape().front() / sgLayout[0];
  };
  for (auto &sgLayout : layoutsB) {
    if (llvm::is_contained(layoutsA, sgLayout) &&
        llvm::is_contained(layoutsCD, sgLayout)) {
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

  const auto &picked = *bestPick;

  auto dpasALayout = buildSgLayout(context, aTy.getShape(), picked,
                                   /*dimK=*/aTy.getRank() - 1);
  auto dpasBLayout = buildSgLayout(context, bTy.getShape(), picked,
                                   /*dimK=*/bTy.getRank() - 2);
  auto dpasCDLayout = buildSgLayout(context, cdTy.getShape(), picked);
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
  if (!uArchInstruction)
    return std::nullopt;
  auto subgroupSize = uArch->getSubgroupSize();

  auto [laneLayoutA, laneDataA] = compute2DBlockIOLaneLayoutAndData(
      aTy.getShape(), subgroupSize,
      aTy.getElementType().getIntOrFloatBitWidth(),
      uArchInstruction->getPackedFormatBitSizeA());
  auto [laneLayoutB, laneDataB] = compute2DBlockIOLaneLayoutAndData(
      bTy.getShape(), subgroupSize,
      bTy.getElementType().getIntOrFloatBitWidth(),
      uArchInstruction->getPackedFormatBitSizeB(), /*vnni=*/true);
  auto [laneLayoutCD, laneDataCD] = compute2DBlockIOLaneLayoutAndData(
      cdTy.getShape(), subgroupSize,
      cdTy.getElementType().getIntOrFloatBitWidth(),
      cdTy.getElementType().getIntOrFloatBitWidth());

  auto instDataVecs = getDpasInstDataLayouts(aTy, bTy, cdTy, uArchInstruction);
  if (!instDataVecs)
    return std::nullopt;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    return getDpasSubgroupLayouts(context, aTy, bTy, cdTy, consumerLayout,
                                  numSg, *instDataVecs);
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
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

  SmallVector<int64_t> scaleSgLayout;
  SmallVector<int64_t> scaleSgData;
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
  SmallVector<int64_t> scaleInstData;
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

  SmallVector<int64_t> scaleLaneLayout;
  SmallVector<int64_t> scaleLaneData;
  if (!laneLayout.empty() && !laneData.empty()) {
    scaleLaneLayout.assign(laneLayout.begin(), laneLayout.end());
    scaleLaneData.assign(laneData.size(), 1);

    bool isRowMajor = uArchInstruction->isLaneLayoutRowMajorOrder();
    if (isBScale ^ isRowMajor)
      std::swap(scaleLaneLayout[rank - 2], scaleLaneLayout[rank - 1]);
    // Cap lane_layout by the per-instruction tile (inst_data) on each dim.
    // Then derive lane_data = inst_data / lane_layout so the Category A
    // invariant inst_data = lane_layout * lane_data * k (with k = 1) holds
    // for the scale operand's load_nd consumer.
    auto layoutCap = scaleInstData.empty() ? scaleShape : scaleInstData;
    for (int64_t d = rank - 2; d < rank; ++d)
      scaleLaneLayout[d] = std::min<int64_t>(layoutCap[d], scaleLaneLayout[d]);
  }
  return buildLayout(context, scaleSgLayout, scaleSgData, scaleInstData,
                     scaleLaneLayout, scaleLaneData, order);
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
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
          xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
  if (!uArchInstruction)
    return std::nullopt;
  auto subgroupSize = uArch->getSubgroupSize();

  auto [laneLayoutA, laneDataA] = compute2DBlockIOLaneLayoutAndData(
      aTy.getShape(), subgroupSize,
      aTy.getElementType().getIntOrFloatBitWidth(),
      uArchInstruction->getPackedFormatBitSizeA());
  auto [laneLayoutB, laneDataB] = compute2DBlockIOLaneLayoutAndData(
      bTy.getShape(), subgroupSize,
      bTy.getElementType().getIntOrFloatBitWidth(),
      uArchInstruction->getPackedFormatBitSizeB(), /*vnni=*/true);
  auto [laneLayoutCD, laneDataCD] = compute2DBlockIOLaneLayoutAndData(
      cdTy.getShape(), subgroupSize,
      cdTy.getElementType().getIntOrFloatBitWidth(),
      cdTy.getElementType().getIntOrFloatBitWidth());
  auto instDataVecs = getDpasInstDataLayouts(aTy, bTy, cdTy, uArchInstruction);
  if (!instDataVecs)
    return std::nullopt;

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto dpasLayouts = getDpasSubgroupLayouts(
        context, aTy, bTy, cdTy, consumerLayout, numSg, *instDataVecs);
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

    auto [instDataA, instDataB, instDataCD] = *instDataVecs;

    auto dpasALayout =
        buildInstDataLayoutWithLane(context, instDataA, laneLayoutA, laneDataA);
    auto dpasBLayout =
        buildInstDataLayoutWithLane(context, instDataB, laneLayoutB, laneDataB);
    auto dpasCDLayout = buildInstDataLayoutWithLane(context, instDataCD,
                                                    laneLayoutCD, laneDataCD);

    auto aScaleLayout =
        createScaleLayout(context, aTy, aScaleTy, dpasALayout, false, uArch);
    auto bScaleLayout =
        createScaleLayout(context, bTy, bScaleTy, dpasBLayout, true, uArch);

    return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout, aScaleLayout,
                           bScaleLayout);
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    auto dpasALayout = buildLaneLayout(context, laneLayoutA, laneDataA);
    auto dpasBLayout = buildLaneLayout(context, laneLayoutB, laneDataB);
    auto dpasCDLayout = buildLaneLayout(context, laneLayoutCD, laneDataCD);

    auto aScaleLayout =
        createScaleLayout(context, aTy, aScaleTy, dpasALayout, false, uArch);
    auto bScaleLayout =
        createScaleLayout(context, bTy, bScaleTy, dpasBLayout, true, uArch);

    return std::make_tuple(dpasALayout, dpasBLayout, dpasCDLayout, aScaleLayout,
                           bScaleLayout);
  }
  return std::nullopt;
}

/// Sets up the anchor layout for a store_nd operation. StoreNd picks its
/// own layout based on uArch block parameters (it does not take a consumer
/// layout, since it is a data sink).
xegpu::DistributeLayoutAttr
xegpu::setupStoreNdAnchorLayout(xegpu::LayoutKind layoutKind,
                                VectorType srcVecTy, int numSg,
                                const xegpu::uArch::uArch *uArch) {
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockStoreInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockStore));
  if (!uArchInstruction)
    return nullptr;

  auto context = srcVecTy.getContext();
  Type elemTy = srcVecTy.getElementType();
  auto subgroupSize = uArch->getSubgroupSize();
  auto dataShape = srcVecTy.getShape();
  [[maybe_unused]] int rank = srcVecTy.getRank();
  assert(rank >= 2 && "Expected at least 2D shape for ND op");

  // Compute the default 2D block IO lane layout / lane data.
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  auto [laneLayout, laneData] = compute2DBlockIOLaneLayoutAndData(
      dataShape, subgroupSize, bitwidth,
      uArchInstruction->getPackedFormatBitSize());

  if (layoutKind == xegpu::LayoutKind::Lane)
    return buildLaneLayout(context, laneLayout, laneData);

  auto instData =
      get2DBlockIOInstDataLayout(dataShape, elemTy, uArchInstruction);

  if (layoutKind == xegpu::LayoutKind::InstData) {
    assert(instData && isValidLaneLayout(*instData, laneLayout, laneData) &&
           "Expected the store layout to satisfy uArch block constraints");
    return buildInstDataLayoutWithLane(context, *instData, laneLayout,
                                       laneData);
  }

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto sgLayouts = getSgLayoutCandidates(dataShape, *instData, numSg);
    if (sgLayouts.empty())
      return nullptr;
    return buildSgLayout(context, dataShape, sgLayouts.front(), /*dimK=*/-1);
  }

  return nullptr;
}

/// Sets up the anchor layout for a prefetch_nd operation. PrefetchNd has no
/// consumer (it produces no value), so it picks its own layout from uArch
/// block parameters.
xegpu::DistributeLayoutAttr
xegpu::setupPrefetchNdAnchorLayout(xegpu::LayoutKind layoutKind,
                                   xegpu::TensorDescType tdescTy, int numSg,
                                   const xegpu::uArch::uArch *uArch) {

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockPrefetchInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockPrefetch));
  if (!uArchInstruction)
    return nullptr;

  auto context = tdescTy.getContext();
  Type elemTy = tdescTy.getElementType();
  auto subgroupSize = uArch->getSubgroupSize();
  auto dataShape = tdescTy.getShape();
  [[maybe_unused]] int rank = tdescTy.getRank();
  assert(rank >= 2 && "Expected at least 2D shape for ND op");

  // Compute the default 2D block IO lane layout / lane data.
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  auto [laneLayout, laneData] = compute2DBlockIOLaneLayoutAndData(
      dataShape, subgroupSize, bitwidth,
      uArchInstruction->getPackedFormatBitSize());

  if (layoutKind == xegpu::LayoutKind::Lane)
    return buildLaneLayout(context, laneLayout, laneData);

  auto instData =
      get2DBlockIOInstDataLayout(dataShape, elemTy, uArchInstruction);

  if (layoutKind == xegpu::LayoutKind::InstData) {
    assert(instData && isValidLaneLayout(*instData, laneLayout, laneData) &&
           "Expected the prefetch layout to satisfy uArch block constraints");
    return buildInstDataLayoutWithLane(context, *instData, laneLayout,
                                       laneData);
  }

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(numSg > 0 &&
           "Number of subgroups must be provided for sg layout creation.");
    auto sgLayouts = getSgLayoutCandidates(dataShape, *instData, numSg);
    if (sgLayouts.empty())
      return nullptr;
    return buildSgLayout(context, dataShape, sgLayouts.front(), /*dimK=*/-1);
  }

  return nullptr;
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

  assert(consumerLayout && "Expected a valid consumer layout");
  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(consumerLayout.isForWorkgroup() &&
           "Expected consumer layout to be a complete workgroup-level layout");
    return consumerLayout;
  }

  auto context = resVecTy.getContext();
  Type elemTy = resVecTy.getElementType();
  auto subgroupSize = uArch->getSubgroupSize();
  auto dataShape = resVecTy.getShape();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::Subgroup2DBlockLoadInstruction>(
          uArch->getInstruction(
              xegpu::uArch::InstructionKind::Subgroup2DBlockLoad));
  if (!uArchInstruction)
    return nullptr;

  int rank = resVecTy.getRank();
  SmallVector<int64_t> consumerInstData =
      consumerLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> consumerLaneLayout =
      consumerLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();
  auto consumerOrderAttr = consumerLayout.getOrder();

  assert(!consumerLaneLayout.empty() && !consumerLaneData.empty() &&
         "Expected consumer layout to have lane_layout and lane_data");

  // vertical lane layout means that the blockload must be transposed
  // note scaleA on PVC has vertical lane layout even without transposed order
  // attr
  bool hasTranspose =
      consumerLaneLayout[rank - 2] > 1 && consumerLaneLayout[rank - 1] == 1;
  bool hasTransform = !hasTranspose && consumerLaneData[rank - 2] > 1 &&
                      consumerLaneData[rank - 1] == 1;
  assert((consumerLaneData[rank - 2] == 1 || consumerLaneData[rank - 1] == 1) &&
         "Expected consumer lane data to have at most one non-unit dim");

  if (layoutKind == xegpu::LayoutKind::InstData) {
    auto blockWHC = uArchInstruction->getBlockWidthHeightCount(
        elemTy, hasTransform, hasTranspose,
        /*upConv=*/false);
    if (!blockWHC)
      return nullptr;
    auto [bWidths, bHeights, bCounts] = blockWHC.value();

    SmallVector<int64_t> laneLayout;
    // set the laneLayout to use consumer's LaneLayout as base, but adjust its
    // size to match the subgroupsize in case its original value is larger than
    // 1
    for (int i = 0; i < rank; i++) {
      if (consumerLaneLayout[i] > 1)
        laneLayout.push_back(std::max(static_cast<int64_t>(subgroupSize),
                                      consumerLaneLayout[i]));
      else
        laneLayout.push_back(1);
    }

    // See whether the consumer's inst_data satisfies the block constraints.
    int64_t height = consumerInstData[rank - 2];
    int64_t width = consumerInstData[rank - 1];
    auto maxBlockCount = *llvm::max_element(bCounts);
    auto maxWidth = *llvm::max_element(bWidths);
    if (llvm::is_contained(bWidths, static_cast<int>(width)) ||
        (width % maxWidth == 0 && width / maxWidth < maxBlockCount)) {
      if (llvm::is_contained(bHeights, static_cast<int>(height))) {
        return buildInstDataLayoutWithLane(context, consumerInstData,
                                           laneLayout, consumerLaneData,
                                           consumerOrderAttr);
      }
    }

    // if consumer instData size too small, try the larger one. like DPAS_MX's
    // scale is smaller than block load
    auto instData = get2DBlockIOInstDataLayout(
        dataShape, elemTy, uArchInstruction, hasTransform, hasTranspose);
    // assert instData is valid against consumer layout since
    // transform/transpose attribute are derived from consumer layout
    assert(instData &&
           isValidLaneLayout(*instData, laneLayout, consumerLaneData) &&
           "Expected the load layout to satisfy uArch block constraints");
    return buildInstDataLayoutWithLane(context, *instData, laneLayout,
                                       consumerLaneData, consumerOrderAttr);
  }
  if (layoutKind == xegpu::LayoutKind::Lane) {
    assert(isValidLaneLayout(dataShape, consumerLaneLayout, consumerLaneData) &&
           "Expected the lane layout to satisfy uArch block constraints");
    return consumerLayout;
  }
  return nullptr;
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
  assert(!consumerLaneLayout.empty() && !consumerLaneData.empty() &&
         "Expected consumer layout to have lane_layout and lane_data");
  laneLayout.assign(consumerLaneLayout.begin(), consumerLaneLayout.end());
  laneData.assign(consumerLaneData.begin(), consumerLaneData.end());

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

  const auto *uArchInstruction = dyn_cast<xegpu::uArch::LoadGatherInstruction>(
      uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize =
      std::min(uArchInstruction->getMaxLaneAccessSizeBytes(), contigChunkSize);

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

  const auto *uArchInstruction = dyn_cast<xegpu::uArch::LoadGatherInstruction>(
      uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
  int maxChunkSize =
      std::min(uArchInstruction->getMaxLaneAccessSizeBytes(), contigChunkSize);
  return setupGenericLoadAnchorLayout(layoutKind, context, consumerLayout,
                                      maxChunkSize, resShape, subgroupSize);
}

/// Sets up the anchor layout for store scatter and store matrix operation.
/// store matrix lowers to store scatter and 1d block store. All of them
/// share the same layout setup logic. For Subgroup layout, not supported
/// yet.
///
/// Lane layout is derived first via `computeScatterIOLaneLayoutAndData`;
/// inst_data is then the element-wise product lane_layout * lane_data.
static xegpu::DistributeLayoutAttr
setupGenericStoreAnchorLayout(xegpu::LayoutKind layoutKind,
                              mlir::MLIRContext *context, int maxChunkSize,
                              ArrayRef<int64_t> srcShape, int subgroupSize) {

  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    assert(false &&
           "subgroup layout assignment not supported for storeScatter.");
    return nullptr;
  }

  auto [laneLayout, laneData] =
      computeScatterIOLaneLayoutAndData(srcShape, subgroupSize, maxChunkSize);

  if (layoutKind == xegpu::LayoutKind::InstData) {
    SmallVector<int64_t> instData(srcShape.size());
    for (size_t i = 0; i < srcShape.size(); ++i)
      instData[i] = laneLayout[i] * laneData[i];
    return buildInstDataLayoutWithLane(context, instData, laneLayout, laneData);
  }
  if (layoutKind == xegpu::LayoutKind::Lane) {
    return buildLaneLayout(context, laneLayout, laneData);
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

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize =
      std::min(uArchInstruction->getMaxLaneAccessSizeBytes(), contigChunkSize);
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

  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::StoreScatterInstruction>(
          uArch->getInstruction(xegpu::uArch::InstructionKind::StoreScatter));
  int maxChunkSize =
      std::min(uArchInstruction->getMaxLaneAccessSizeBytes(), contigChunkSize);

  return setupGenericStoreAnchorLayout(layoutKind, context, maxChunkSize,
                                       srcShape, subgroupSize);
}

/// Completes a scatter IO layout by deriving lane_layout and lane_data from
/// `specifiedLayout`'s inst_data when they are missing. The layout is returned
/// unchanged if `specifiedLayout` is null, carries no inst_data, or already has
/// both lane_layout and lane_data.
///
/// When lane info is absent, inst_data is treated as the effective shape and
/// the lane factorization is filled in as follows:
///   - If `consumerLayout` is present and its lane_layout / lane_data are a
///     valid factorization of inst_data, that consumer lane info is reused so
///     the completed layout matches the consumer (avoiding a relayout).
///   - Otherwise a standard scatter-style factorization is computed via
///     `computeScatterIOLaneLayoutAndData`, bounded by `maxChunkSize` — the
///     per-lane load width reported by the uArch's LoadGather instruction
///     (`getMaxLaneAccessSizeBytes`).
///
std::optional<xegpu::DistributeLayoutAttr>
xegpu::completeScatterLoadLaneLayoutFromInstData(
    xegpu::DistributeLayoutAttr specifiedLayout,
    xegpu::DistributeLayoutAttr consumerLayout, Type elemTy,
    const xegpu::uArch::LoadGatherInstruction *uArchInstruction,
    const int subgroupSize) {
  if (!specifiedLayout)
    return specifiedLayout;
  SmallVector<int64_t> specifiedInstData =
      specifiedLayout.getEffectiveInstDataAsInt();
  if (specifiedInstData.empty())
    return specifiedLayout;
  if (!specifiedLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !specifiedLayout.getEffectiveLaneDataAsInt().empty())
    return specifiedLayout;

  // Reuse the load-side setup with inst_data as the destination shape.
  auto *context = specifiedLayout.getContext();
  int maxChunkSize = uArchInstruction->getMaxLaneAccessSizeBytes();
  if (consumerLayout) {
    auto consumerLaneLayout = consumerLayout.getEffectiveLaneLayoutAsInt();
    auto consumerLaneData = consumerLayout.getEffectiveLaneDataAsInt();
    if (!consumerLaneLayout.empty() && !consumerLaneData.empty() &&
        isValidLaneLayout(specifiedInstData, consumerLaneLayout,
                          consumerLaneData))
      return buildInstDataLayoutWithLane(context, specifiedInstData,
                                         consumerLaneLayout, consumerLaneData);
  }
  auto [defLaneLayout, defLaneData] = computeScatterIOLaneLayoutAndData(
      specifiedInstData, subgroupSize, maxChunkSize);
  if (!isValidLaneLayout(specifiedInstData, defLaneLayout, defLaneData))
    return std::nullopt;
  return buildInstDataLayoutWithLane(context, specifiedInstData, defLaneLayout,
                                     defLaneData);
}

/// Like completeScatterLoadLaneLayoutFromInstData, but for scatter stores. A
/// store is a data sink, so lane info is derived purely from inst_data (bounded
/// by the uArch's per-lane store width); there is no consumer layout to reuse.
std::optional<xegpu::DistributeLayoutAttr>
xegpu::completeScatterStoreLaneLayoutFromInstData(
    xegpu::DistributeLayoutAttr specifiedLayout, Type elemTy,
    const xegpu::uArch::StoreScatterInstruction *uArchInstruction,
    const int subgroupSize) {
  if (!specifiedLayout)
    return specifiedLayout;
  SmallVector<int64_t> specifiedInstData =
      specifiedLayout.getEffectiveInstDataAsInt();
  if (specifiedInstData.empty())
    return specifiedLayout;
  if (!specifiedLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !specifiedLayout.getEffectiveLaneDataAsInt().empty())
    return specifiedLayout;

  // Reuse the store-side setup with inst_data as the source shape.
  auto *context = specifiedLayout.getContext();
  int maxChunkSize = uArchInstruction->getMaxLaneAccessSizeBytes();
  auto [defLaneLayout, defLaneData] = computeScatterIOLaneLayoutAndData(
      specifiedInstData, subgroupSize, maxChunkSize);
  if (!isValidLaneLayout(specifiedInstData, defLaneLayout, defLaneData))
    return std::nullopt;
  return buildInstDataLayoutWithLane(context, specifiedInstData, defLaneLayout,
                                     defLaneData);
}

/// Completes a 2D-block store/prefetch layout from its inst_data. store_nd and
/// prefetch_nd are data sinks, so lane info is derived purely from inst_data
/// (no consumer to reuse). One helper serves both via
/// BlockIOInstructionInterface.
std::optional<xegpu::DistributeLayoutAttr>
xegpu::completeBlockStoreLaneLayoutFromInstData(
    xegpu::DistributeLayoutAttr specifiedLayout, Type elemTy,
    const xegpu::uArch::BlockIOInstructionInterface *uArchInstruction,
    const int subgroupSize) {
  if (!specifiedLayout)
    return specifiedLayout;
  SmallVector<int64_t> specifiedInstData =
      specifiedLayout.getEffectiveInstDataAsInt();
  if (specifiedInstData.empty())
    return specifiedLayout;
  if (!specifiedLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !specifiedLayout.getEffectiveLaneDataAsInt().empty())
    return specifiedLayout;

  auto *context = specifiedLayout.getContext();
  auto [laneLayout, laneData] = compute2DBlockIOLaneLayoutAndData(
      specifiedInstData, subgroupSize, elemTy.getIntOrFloatBitWidth(),
      uArchInstruction->getPackedFormatBitSize());
  if (!isValidLaneLayout(specifiedInstData, laneLayout, laneData))
    return std::nullopt;
  return buildInstDataLayoutWithLane(context, specifiedInstData, laneLayout,
                                     laneData);
}

/// Like completeBlockStoreLaneLayoutFromInstData, but for load_nd. The
/// consumer's lane_data and order are reused as-is; lane_layout is rebuilt from
/// the consumer's lane_layout, bumping every non-unit dim up to the subgroup
/// size. The user-provided inst_data is preserved.
std::optional<xegpu::DistributeLayoutAttr>
xegpu::completeBlockLoadLaneLayoutFromInstData(
    xegpu::DistributeLayoutAttr specifiedLayout,
    xegpu::DistributeLayoutAttr consumerLayout, Type elemTy,
    const xegpu::uArch::BlockIOInstructionInterface *uArchInstruction,
    const int subgroupSize) {
  if (!specifiedLayout)
    return specifiedLayout;
  SmallVector<int64_t> specifiedInstData =
      specifiedLayout.getEffectiveInstDataAsInt();
  if (specifiedInstData.empty())
    return specifiedLayout;
  if (!specifiedLayout.getEffectiveLaneLayoutAsInt().empty() &&
      !specifiedLayout.getEffectiveLaneDataAsInt().empty())
    return specifiedLayout;
  if (!consumerLayout)
    return specifiedLayout;
  SmallVector<int64_t> consumerLaneLayout =
      consumerLayout.getEffectiveLaneLayoutAsInt();
  SmallVector<int64_t> consumerLaneData =
      consumerLayout.getEffectiveLaneDataAsInt();
  if (consumerLaneLayout.empty() || consumerLaneData.empty())
    return specifiedLayout;

  auto *context = specifiedLayout.getContext();
  int rank = specifiedInstData.size();

  SmallVector<int64_t> laneLayout;
  // set the laneLayout to use consumer's LaneLayout as base, but adjust its
  // size to match the subgroupsize in case its original value is larger than 1
  for (int i = 0; i < rank; i++) {
    if (consumerLaneLayout[i] > 1) {
      laneLayout.push_back(
          std::max(static_cast<int64_t>(subgroupSize), consumerLaneLayout[i]));
    } else {
      laneLayout.push_back(1);
    }
  }

  if (!isValidLaneLayout(specifiedInstData, laneLayout, consumerLaneData))
    return std::nullopt;
  return buildInstDataLayoutWithLane(context, specifiedInstData, laneLayout,
                                     consumerLaneData,
                                     consumerLayout.getOrder());
}

/// Completes user-provided DPAS A/B/C-D anchors that carry only inst_data by
/// filling in lane_layout / lane_data. The lane factorization mirrors the
/// InstData branch of `setupDpasLayout` (derived from each operand's shape and
/// matmul role, B using VNNI packing); the user's inst_data is preserved.
std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
xegpu::completeDpasLaneLayoutFromInstData(xegpu::DistributeLayoutAttr aLayout,
                                          xegpu::DistributeLayoutAttr bLayout,
                                          xegpu::DistributeLayoutAttr cdLayout,
                                          VectorType aTy, VectorType bTy,
                                          VectorType cdTy,
                                          const xegpu::uArch::uArch *uArch) {
  auto context = aTy.getContext();
  const auto *uArchInstruction =
      dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(uArch->getInstruction(
          xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
  if (!uArchInstruction)
    return std::nullopt;
  auto subgroupSize = uArch->getSubgroupSize();
  llvm::SmallVector<int64_t> laneLayoutA, laneDataA, laneLayoutB, laneDataB,
      laneLayoutCD, laneDataCD;
  SmallVector<int64_t> instDataA = aLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> instDataB = bLayout.getEffectiveInstDataAsInt();
  SmallVector<int64_t> instDataCD = cdLayout.getEffectiveInstDataAsInt();

  if (isa<xegpu::uArch::Xe2, xegpu::uArch::Xe3>(uArch)) {
    std::tie(laneLayoutA, laneDataA) = compute2DBlockIOLaneLayoutAndData(
        aTy.getShape(), subgroupSize,
        aTy.getElementType().getIntOrFloatBitWidth(),
        uArchInstruction->getPackedFormatBitSizeA());
    std::tie(laneLayoutB, laneDataB) = compute2DBlockIOLaneLayoutAndData(
        bTy.getShape(), subgroupSize,
        bTy.getElementType().getIntOrFloatBitWidth(),
        uArchInstruction->getPackedFormatBitSizeB(), /*vnni=*/true);
    std::tie(laneLayoutCD, laneDataCD) = compute2DBlockIOLaneLayoutAndData(
        cdTy.getShape(), subgroupSize,
        cdTy.getElementType().getIntOrFloatBitWidth(),
        cdTy.getElementType().getIntOrFloatBitWidth());
  } else {
    assert(false && "Unsupported uArch for DPAS lane layout completion");
  }

  if (!isValidLaneLayout(instDataA, laneLayoutA, laneDataA) ||
      !isValidLaneLayout(instDataB, laneLayoutB, laneDataB) ||
      !isValidLaneLayout(instDataCD, laneLayoutCD, laneDataCD))
    return std::nullopt;
  return std::make_tuple(
      buildInstDataLayoutWithLane(context, instDataA, laneLayoutA, laneDataA,
                                  aLayout.getOrder()),
      buildInstDataLayoutWithLane(context, instDataB, laneLayoutB, laneDataB,
                                  bLayout.getOrder()),
      buildInstDataLayoutWithLane(context, instDataCD, laneLayoutCD, laneDataCD,
                                  cdLayout.getOrder()));
}

/// Like completeDpasLaneLayoutFromInstData, but for dpas_mx: also re-derives
/// the A_scale / B_scale layouts from the completed A / B layouts via
/// `createScaleLayout`, matching the default path of `setupDpasMxLayout`.
std::optional<
    std::tuple<xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr, xegpu::DistributeLayoutAttr,
               xegpu::DistributeLayoutAttr>>
xegpu::completeDpasMxLaneLayoutFromInstData(
    xegpu::DistributeLayoutAttr aLayout, xegpu::DistributeLayoutAttr bLayout,
    xegpu::DistributeLayoutAttr cdLayout, VectorType aTy, VectorType bTy,
    VectorType cdTy, VectorType aScaleTy, VectorType bScaleTy,
    const xegpu::uArch::uArch *uArch) {
  auto completed = completeDpasLaneLayoutFromInstData(
      aLayout, bLayout, cdLayout, aTy, bTy, cdTy, uArch);
  if (!completed)
    return std::nullopt;
  auto context = aTy.getContext();
  auto [completedA, completedB, completedCD] = *completed;

  auto aScaleLayout =
      createScaleLayout(context, aTy, aScaleTy, completedA, false, uArch);
  auto bScaleLayout =
      createScaleLayout(context, bTy, bScaleTy, completedB, true, uArch);

  return std::make_tuple(completedA, completedB, completedCD, aScaleLayout,
                         bScaleLayout);
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
/// This is a best-effort alignment, not a hard constraint: the goal is only to
/// pick a *legal* source layout that minimizes redistribution against the
/// (single, first-arriving) consumer layout. There is no failure path - when
/// the consumer's slice layout cannot be reused as-is (example 2 below), the
/// function falls back to distributing all subgroups on the non-reduction
/// dimensions first and the remainder on the reduction dimensions, which always
/// yields a valid source layout. If the resulting source layout still differs
/// from what some consumer expects (e.g. a second, inconsistent consumer), that
/// mismatch is reconciled later by the layout conflict resolution process
/// (`ResolveLayoutConflicts`), which inserts a `convert_layout` op - this
/// function never has to give up.
///
/// For the InstData and Lane layout kinds only the innermost two dimensions
/// are distributed; all leading dimensions are assumed to be unit dimensions.
/// This assumption is checked via `leadingDimsAreUnit`. The lane_layout and
/// lane_data are computed by `computeReductionLaneLayoutAndData`, which picks
/// a layout that minimizes cross-lane reduction (reducing within a lane when
/// only one of the innermost two dims is a reduction dim). The inst_data is
/// simply the element-wise product lane_layout * lane_data.
///
/// The function returns the *result* layout (the SliceAttr). The *source*
/// layout it decides on is the parent of that slice; both are listed below so
/// the relationship is explicit.
///
/// Examples:
///   1. Subgroup layout - Row reduction on 2D tensor:
///      srcShape=[32, 128], reductionDims=[1], resShape=[32], subgroupSize=16,
///      NumSg=32
///      * Consumer Layout:
///        #xegpu.slice<#xegpu.layout<sg_layout=[4, 8], sg_data=[8, 8]>, dims =
///        [1]>}
///      * Source Layout (decided by this function):
///        #xegpu.layout<sg_layout=[4, 8], sg_data=[8, 16]>
///      * Result Layout (returned):
///        #xegpu.slice<#xegpu.layout<sg_layout=[4, 8], sg_data=[8, 16]>, dims =
///        [1]>}
///      The consumer slices exactly the reduction dim, so its parent layout is
///      reused for the source: sg_layout is kept, but the source's sg_data on
///      the reduction dim is grown from 8 to 16 (= srcShape[1] / sg_layout[1] =
///      128 / 8) so the source tile is evenly distributed over the reduction
///      dim. Slicing that source over dim 1 reproduces the consumer.
///
///   2. Subgroup layout - Same shapes as above but consumer doesn't have a
///   reusable slice layout, so the algorithm distributes all subgroups on the
///   non-reduction dims first and the remainder on the reduction dims.
///      2a. * Consumer Layout:
///            #xegpu.layout<sg_layout=[32], sg_data=[1]>
///          * Source Layout (decided by this function):
///            #xegpu.layout<sg_layout=[32, 1], sg_data=[1, 128]>
///          * Result Layout (returned):
///            #xegpu.slice<#xegpu.layout<sg_layout=[32, 1], sg_data=[1, 128]>,
///            dims = [1]>}
///          All 32 subgroups land on the non-reduction dim 0; the reduction dim
///          1 gets the leftover (sg_layout=1, so the whole length 128 lives in
///          one subgroup's sg_data).
///      2b. * Consumer Layout:
///            #xegpu.slice<#xegpu.layout<sg_layout=[8, 2, 4], sg_data=[4, 64,
///            32]>, dims = [1, 2]>}
///          * Source Layout (decided by this function):
///            #xegpu.layout<sg_layout=[8, 4], sg_data=[4, 32]>
///          * Result Layout (returned):
///            #xegpu.slice<#xegpu.layout<sg_layout=[8, 4], sg_data=[4, 32]>,
///            dims = [1]>}
///          The consumer slices dims [1, 2] which do not match this op's
///          reductionDims, so it can't be reused as-is; subgroups are
///          re-distributed (non-reduction dim first, then reduction dim).
///
///   3. Lane layout - Default (lanes on innermost dim):
///      srcShape=[32, 64], reductionDims=[0], subgroupSize=16
///      * Source Layout (decided by this function):
///        laneLayout=[1, 16], laneData=[1, 1] (returned sliced over dim 0).
///      The innermost dim is not reduced, so lanes stay on it.
///
///   4. Lane layout - Switch (lanes moved off the reduction dim):
///      srcShape=[32, 64], reductionDims=[1], subgroupSize=16
///      * Source Layout (decided by this function):
///        laneLayout=[16, 1], laneData=[1, 1] (returned sliced over dim 1).
///      The innermost dim is the sole reduction dim, so lanes move to the
///      non-reduction dim to reduce within a lane. This switch only happens
///      when the consumer has no reduction dims to broadcast the result back
///      along (i.e. the consumer layout is not a slice over this reduction);
///      otherwise the default (example 3) is used.
///
///   5. Lane layout - No switch when both inner dims are reduced (reduction to
///   scalar):
///      srcShape=[32, 64], reductionDims=[0, 1], subgroupSize=16
///      * Source Layout (decided by this function):
///        laneLayout=[1, 16], laneData=[1, 1] (returned sliced over dims
///        [0,1]).
///      Both dims are reduced, so this is not a *sole* innermost reduction; the
///      switch condition (example 4) does not apply and lanes stay on the
///      innermost dim. The cross-lane reduction here is unavoidable.
///
///   6. Lane layout - No switch when the consumer slices the reduction dim:
///      srcShape=[32, 64], reductionDims=[1], subgroupSize=16
///      * Consumer Layout:
///        #xegpu.slice<#xegpu.layout<laneLayout=[1, 16], laneData=[1, 1]>,
///        dims = [1]>}
///      * Source Layout (decided by this function):
///        #xegpu.layout<laneLayout=[1, 16], laneData=[1, 1]> (the consumer
///        slice's parent, reused directly; returned sliced over dim 1).
///      Same shape/reductionDims as example 4, but here the consumer is a slice
///      over the reduction dim, so it can broadcast the result back along that
///      dim. The slice's parent layout is reused as the source (no switch, no
///      re-derivation); the inst_data propagation step has already inserted a
///      convert_layout if needed, so the lane-level layout can be reused as-is.

xegpu::SliceAttr xegpu::setupMultiReductionResultLayout(
    xegpu::LayoutKind layoutKind, VectorType srcVecTy,
    DistributeLayoutAttr consumerLayout, SmallVector<int64_t> reductionDims,
    int numSg, const xegpu::uArch::uArch *uArch) {

  auto srcShape = srcVecTy.getShape();
  int srcRank = srcShape.size();
  auto context = srcVecTy.getContext();

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
      DenseI32ArrayAttr resOrderAttr = DenseI32ArrayAttr::get(
          context, SmallVector<int32_t>(order.begin(), order.end()));
      if (!orderAttr || orderAttr.empty())
        resOrderAttr = nullptr;
      assert(remainingSgCount == 1 && "not all subgroups distributed");
      srcLayout = buildLayout(context, sgLayout, sgData,
                              /*instData=*/{}, /*laneLayout=*/{},
                              /*laneData=*/{}, resOrderAttr);
    }
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    xegpu::SliceAttr consumerSliceLayout =
        dyn_cast_if_present<xegpu::SliceAttr>(consumerLayout);
    auto consumerReductionDims =
        consumerSliceLayout
            ? SmallVector<int64_t>(consumerSliceLayout.getDims().asArrayRef())
            : SmallVector<int64_t>({});
    // A[i] reduced from A[i, j] is stored out directly, use vertical Lane
    // layout like [16, 1]
    bool verticalLaneLayout = consumerReductionDims.empty() &&
                              reductionDims.size() == 1 &&
                              reductionDims[0] == (srcRank - 1);
    auto [laneLayout, laneData] = computeReductionLaneLayoutAndData(
        srcShape, reductionDims, subgroupSize, maxReduceVectorSize,
        verticalLaneLayout);
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
    auto consumerReductionDims =
        consumerSliceLayout
            ? SmallVector<int64_t>(consumerSliceLayout.getDims().asArrayRef())
            : SmallVector<int64_t>({});
    if (consumerSliceLayout &&
        consumerSliceLayout.getDims().asArrayRef().equals(reductionDims)) {
      // at the lane level, the consumerSliceLayout can be directly reused
      // since the inst_data propagation already insert convert_layout if
      // the layout is not consistent
      srcLayout = consumerSliceLayout.getParent();
    } else {
      bool verticalLaneLayout = consumerReductionDims.empty() &&
                                reductionDims.size() == 1 &&
                                reductionDims[0] == (srcRank - 1);
      auto [laneLayout, laneData] = computeReductionLaneLayoutAndData(
          srcShape, reductionDims, subgroupSize, maxReduceVectorSize,
          verticalLaneLayout);
      srcLayout = buildLaneLayout(context, laneLayout, laneData);
    }
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
    assert(false &&
           "subgroup layout assignment not supported for reduction (op "
           "is not expected at this level).");
  } else if (layoutKind == xegpu::LayoutKind::InstData) {
    assert(false &&
           "instData layout assignment not supported for reduction (op "
           "is not expected at this level).");
  } else if (layoutKind == xegpu::LayoutKind::Lane) {
    SmallVector<int64_t> laneLayout(1), laneData(1);
    laneLayout[0] = std::min(static_cast<int64_t>(subgroupSize), srcShape[0]);
    laneData[0] = 1;
    srcLayout = buildLaneLayout(context, laneLayout, laneData);
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
    assert(false && "subgroup/instData layout assignment not supported for "
                    "insertStridedSlice.");
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

/// Back-propagates a known result layout to the layout required on `operand`
/// for a non-anchor (layout-propagating) vector op. Dispatches on the op kind —
/// broadcast, (multi)reduction, bitcast, shape/transpose, insert/extract,
/// interleave, etc. — applying the shape/permutation/bitwidth transform to
/// derive the source layout; elementwise and pass-through ops reuse resLayout
/// as-is. Returns nullptr for unknown ops or an absent result layout.
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

/// Returns the layout required on `operand`: anchor ops report their declared
/// per-operand layout directly; non-anchor ops back-derive it from their result
/// layout via inferSourceLayoutFromResultForNonAnchorOp.
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
