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

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's region. Reports an error if any vector operand lacks
/// a layout attribute.
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

// The recover proceeds by scanning the operation in reverse topological orderas
// follows: Across operations: layouts are propagated from uses to definitions.
// Within an operation: layouts are propagated from definitions (result) to uses
// (operands).
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
xegpu::DistributeLayoutAttr xegpu::inferBroadCastSourceLayout(
    MLIRContext *context, xegpu::DistributeLayoutAttr resLayout,
    ArrayRef<int64_t> resShape, ArrayRef<int64_t> srcShape) {

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
        context, resLayout, DenseI64ArrayAttr::get(context, bcastDims));
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

// /// Infers the source layout attribute for a bitcast operation given the
// /// result layout attribute, result element type bitwidth, and source element
// /// type bitwidth.
// xegpu::DistributeLayoutAttr
// xegpu::inferBitCastSourceLayout(MLIRContext *context,
//                                 xegpu::DistributeLayoutAttr resLayout,
//                                 int resElemTyBitWidth, int
//                                 srcElemTyBitWidth){
//   // the result and source layout must be the same
//   // if resLayout is SliceAttr, we need to first get its root layout
//   xegpu::DistributeLayoutAttr resRootLayout = resLayout;
//   while (auto sliceLayout = dyn_cast<xegpu::SliceAttr>(resRootLayout)) {
//     resRootLayout = sliceLayout.getParent();
//   }
//   // change the laneData of resRootLayout according to the bitwidth ratio
//   xegpu::LayoutAttr resRootPlainLayout =
//   dyn_cast<xegpu::LayoutAttr>(resRootLayout); SmallVector<int64_t> laneData =
//   resRootPlainLayout.getEffectiveLaneDataAsInt();

//   if (srcElemTyBitWidth >= resElemTyBitWidth) {
//     int bitWidthRatio = srcElemTyBitWidth / resElemTyBitWidth;
//     laneData[laneData.size()-1] = laneData[laneData.size()-1] *
//     bitWidthRatio;
//   } else {
//     int bitWidthRatio = resElemTyBitWidth / srcElemTyBitWidth;
//     assert((laneData[laneData.size()-2] % bitWidthRatio) == 0 &&
//            "laneData not divisible by bitWidthRatio");
//     laneData[laneData.size()-1] = laneData[laneData.size()-1] /
//     bitWidthRatio;
//   }

//   // now reconstruct the source layout with updated laneData
//   // by updating the root layout and going throught the slice layers
//   SmallVector<int32_t> laneData32(laneData.begin(), laneData.end());
//   xegpu::LayoutAttr proposedSrcLayout = xegpu::LayoutAttr::get(
//           context,
//           resRootPlainLayout.getSgLayout(),
//           resRootPlainLayout.getSgData(),
//           resRootPlainLayout.getInstData(),
//           resRootPlainLayout.getLaneLayout(),
//           DenseI32ArrayAttr::get(context, laneData32),
//           resRootPlainLayout.getOrder());

//   // reconstruct slice layers if any
//   // First collect all slice layers from innermost to outermost
//   SmallVector<DenseI64ArrayAttr> sliceDims;
//   xegpu::DistributeLayoutAttr currentLayout = resLayout;
//   while (auto sliceLayout = dyn_cast<xegpu::SliceAttr>(currentLayout)) {
//     sliceDims.push_back(sliceLayout.getDims());
//     currentLayout = sliceLayout.getParent();
//   }

//   // Now rebuild from outermost to innermost (reverse order)
//   xegpu::DistributeLayoutAttr finalSrcLayout = proposedSrcLayout;
//   for (auto it = sliceDims.rbegin(); it != sliceDims.rend(); ++it) {
//     finalSrcLayout = xegpu::SliceAttr::get(context, finalSrcLayout, *it);
//   }
//   return finalSrcLayout;
// }

// /// Infers the source layout attribute for a shape cast operation given the
// /// result layout attribute, result shape, and source shape.
// xegpu::DistributeLayoutAttr xegpu::inferShapeCastSourceLayout(
//     MLIRContext *context, xegpu::DistributeLayoutAttr resLayout,
//     ArrayRef<int64_t> resShape, ArrayRef<int64_t> srcShape){

// // there are two use cases:
// // 1. expand dims of low-rank dimensions (e.g., 1D to 2D): to set up the
// tensor before broadcast
// // 2. split dim of a high-rank dimension (e.g., 1D to 2D): to setup tensor
// for multi-stage reduction

//   SmallVector<int64_t> shapeCastDims;
//   auto returnLayout = resLayout;

//   int resRank = resShape.size();
//   int srcRank = srcShape.size();

//   if (srcRank < resRank) {
//     // Case 1: expand dims of low-rank dimensions (e.g., 1D to 2D)
//     int dimDiff = resRank - srcRank;
//     // adding the missing leading dims
//     for (int i = 0; i < dimDiff; i++)
//       shapeCastDims.push_back(i);

//     // create a slice layout for the source
//     returnLayout = xegpu::SliceAttr::get(
//         context, resLayout, DenseI64ArrayAttr::get(context, shapeCastDims));
//   } else if (srcRank > resRank) {
//     // Case 2: split dim of a high-rank dimension (e.g., 1D to 2D)
//     // find the split dims by comparing srcShape and resShape
//     int srcIdx = 0;
//     int resIdx = 0;
//     while (srcIdx < srcRank && resIdx < resRank) {
//       if (srcShape[srcIdx] == resShape[resIdx]) {
//         srcIdx++;
//         resIdx++;
//       } else if (srcShape[srcIdx] < resShape[resIdx]) {
//         shapeCastDims.push_back(srcIdx);
//         srcIdx++;
//       } else {
//         // this should not happen in valid shape cast
//         assert(false && "Invalid shape cast: source shape dimension smaller
//         than result shape dimension");
//       }
//     }
//     // handle remaining src dims
//     while (srcIdx < srcRank) {
//       shapeCastDims.push_back(srcIdx);
//       srcIdx++;
//     }

//     // create a slice layout for the source
//     returnLayout = xegpu::SliceAttr::get(
//         context, resLayout, DenseI64ArrayAttr::get(context, shapeCastDims));
//   }
//   return returnLayout;

// }

xegpu::SliceAttr
xegpu::reductionLayoutSetupRule(ArrayRef<int64_t> srcShape,
                                SmallVector<int64_t> reductionDims,
                                DistributeLayoutAttr consumerPreferredLayout) {

  xegpu::SliceAttr sliceCPL =
      dyn_cast<xegpu::SliceAttr>(consumerPreferredLayout);

  // try to align wiht customer's preferred layout so that the slice layout
  // structure is preserved, and thus avoid potential data movement acorss sg or
  // lanes.

  const int workgroupSize = 16; // assuming 16 subgroups for now
  const int subgroupSize = 16;  // assuming 16 lanes per subgroup
  const int vectorSize = 8;     // assuming 8 elements per vector lane
  int srcShapeSize = srcShape.size();
  xegpu::DistributeLayoutAttr proposedSrcLayout;
  auto context = consumerPreferredLayout.getContext();
  // if srcShapeSize is less than 2, we cannot proceed
  if (srcShapeSize < 2)
    return nullptr;

  llvm::errs() << "DEBUG: Entering \n";

  SmallVector<int64_t> sgLayout(srcShapeSize);
  SmallVector<int64_t> sgData(srcShapeSize);

  SmallVector<int64_t> instData(srcShapeSize, 1);
  instData[srcShapeSize - 1] = subgroupSize;
  instData[srcShapeSize - 2] =
      vectorSize; // assuming 8 elements per instruction as starting point
  llvm::errs() << "DEBUG: Initial instData = [";
  for (size_t i = 0; i < instData.size(); i++) {
    llvm::errs() << instData[i];
    if (i < instData.size() - 1)
      llvm::errs() << ", ";
  }
  llvm::errs() << "]\n";
  // construct a vector layout with lane_layout = [1, ..., 1, subgroupSize]
  SmallVector<int64_t> laneLayout(srcShapeSize, 1);
  laneLayout[srcShapeSize - 1] = subgroupSize;
  llvm::errs() << "DEBUG: laneLayout = [";
  for (size_t i = 0; i < laneLayout.size(); i++) {
    llvm::errs() << laneLayout[i];
    if (i < laneLayout.size() - 1)
      llvm::errs() << ", ";
  }
  llvm::errs() << "]\n";
  // construct a vector layout with lane_data = [1, ..., 1]
  SmallVector<int64_t> laneData(srcShapeSize, 1);

  bool failToAlignSliceStruct = false;
  if (sliceCPL && sliceCPL.getDims().asArrayRef().equals(reductionDims)) {

    xegpu::DistributeLayoutAttr parentCPL = sliceCPL.getParent();

    // for each slice dim in source shape, if the dim size is differnt than the
    // result shape, try to adjust the sg_data/inst_data accordingly.
    SmallVector<int64_t> pcplSgLayout = parentCPL.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> pcplLaneLayout =
        parentCPL.getEffectiveLaneLayoutAsInt();
    SmallVector<int64_t> pcplLaneData = parentCPL.getEffectiveLaneDataAsInt();

    assert(srcShapeSize == parentCPL.getRank() &&
           "parent layout rank must match source shape rank");

    proposedSrcLayout = parentCPL;

    llvm::errs() << "DEBUG: srcShapeSize = " << srcShapeSize << "\n";
    llvm::errs() << "DEBUG: parentCPL rank = " << parentCPL.getRank() << "\n";
    llvm::errs() << "DEBUG: srcShape = [";
    for (int i = 0; i < srcShapeSize; i++) {
      llvm::errs() << srcShape[i];
      if (i < srcShapeSize - 1)
        llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    if (pcplSgLayout.size() == static_cast<size_t>(srcShapeSize)) {
      for (int i = 0; i < srcShapeSize; i++) {
        if (srcShape[i] % pcplSgLayout[i] == 0) {
          sgLayout[i] = pcplSgLayout[i];
          sgData[i] = srcShape[i] / sgLayout[i];
          instData[i] = std::min(instData[i], sgData[i]);
        } else {
          failToAlignSliceStruct = true;
          break;
        }
      }
    }

    if (pcplLaneLayout.size() == static_cast<size_t>(srcShapeSize)) {
      for (int i = 0; i < srcShapeSize; i++) {
        if (instData[i] % pcplLaneLayout[i] == 0) {
          laneLayout[i] = pcplLaneLayout[i];
          laneData[i] = pcplLaneData[i];
        } else {
          failToAlignSliceStruct = true;
          break;
        }
      }
    }
  } else {
    failToAlignSliceStruct = true;
  }

  if (failToAlignSliceStruct) {

    // try to align the sg layout
    SmallVector<int64_t> cplSgLayout =
        consumerPreferredLayout.getEffectiveSgLayoutAsInt();
    llvm::errs() << "DEBUG: cplSgLayout size = " << cplSgLayout.size() << "\n";
    // if sg layout doesn't cover all the sg ids, distribute rest to
    // non-reduction dims
    int remainingSgCount = workgroupSize;

    SmallVector<int64_t> remainingDims;
    // print debug info for consumerPreferredLayout and cplSgLayout
    llvm::errs() << "DEBUG: consumerPreferredLayout sgLayout = [";
    auto cplSgLayoutFull = consumerPreferredLayout.getEffectiveSgLayoutAsInt();
    for (size_t i = 0; i < cplSgLayoutFull.size(); i++) {
      llvm::errs() << cplSgLayoutFull[i];
      if (i < cplSgLayoutFull.size() - 1)
        llvm::errs() << ", ";
    }
    // if cplSgLayout is not empty, try to align the sg layout first
    int cplId = cplSgLayout.size() - 1;
    llvm::errs() << "DEBUG: Starting first loop, cplId = " << cplId << "\n";
    for (int i = srcShapeSize - 1; i >= 0; i--) {
      llvm::errs() << "DEBUG: Loop 1, i = " << i << ", is_reduction_dim = "
                   << llvm::is_contained(reductionDims, i) << "\n";
      // try to align with cplSgLayout first for non-reduction dims
      if (!llvm::is_contained(reductionDims, i) && cplId >= 0) {
        if (srcShape[i] % cplSgLayout[cplId] == 0) {
          sgLayout[i] = cplSgLayout[cplId];
          sgData[i] = srcShape[i] / sgLayout[i];
          instData[i] = std::min(instData[i], sgData[i]);
          remainingSgCount /= sgLayout[i];
          llvm::errs() << "DEBUG: Set sgLayout[" << i << "] = " << sgLayout[i]
                       << ", sgData[" << i << "] = " << sgData[i]
                       << ", remainingSgCount = " << remainingSgCount << "\n";
          cplId--;
          continue;
        }
      }
      remainingDims.push_back(i);
      llvm::errs() << "DEBUG: Added i = " << i << " to remainingDims\n";
    }

    llvm::errs() << "DEBUG: Starting second loop\n";
    for (int i = srcShapeSize - 1; i >= 0; i--) {
      llvm::errs() << "DEBUG: Loop 2, i = " << i << ", is_remaining_dim = "
                   << llvm::is_contained(remainingDims, i) << "\n";
      if (llvm::is_contained(remainingDims, i)) {

        llvm::errs() << "DEBUG: Before Set sgLayout[" << i
                     << "] = " << sgLayout[i] << ", sgData[" << i
                     << "] = " << sgData[i]
                     << ", remainingSgCount = " << remainingSgCount << "\n";

        sgLayout[i] = std::min((srcShape[i] / laneLayout[i]),
                               static_cast<int64_t>(remainingSgCount));
        sgData[i] = srcShape[i] / sgLayout[i];
        instData[i] = std::min(instData[i], sgData[i]);
        remainingSgCount /= sgLayout[i];

        llvm::errs() << "DEBUG: After Set sgLayout[" << i
                     << "] = " << sgLayout[i] << ", sgData[" << i
                     << "] = " << sgData[i]
                     << ", remainingSgCount = " << remainingSgCount << "\n";

        if (remainingSgCount == 1) {
          llvm::errs() << "DEBUG: Breaking from loop 2, remainingSgCount = 1\n";
          break;
        }
      }
    }
  }
  // Convert int64_t vectors to int32_t for DenseI32ArrayAttr
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
      DenseI32ArrayAttr::get(context, laneData32),
      consumerPreferredLayout.getOrder());

  // finally, create the slice layout for reduction source
  xegpu::SliceAttr reductionSrcLayout =
      xegpu::SliceAttr::get(context, proposedSrcLayout,
                            DenseI64ArrayAttr::get(context, reductionDims));

  return reductionSrcLayout;
}
