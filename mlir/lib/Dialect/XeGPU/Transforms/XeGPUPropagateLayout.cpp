//===- XeGPUPropagateLayout.cpp - XeGPU Layout Propagation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/uArchCommon.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUPROPAGATELAYOUT
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// LayoutPropagation
//===----------------------------------------------------------------------===//

/// Assigns a distribution layout to the vector and tensor_desc values with a
/// single backward scan of the IR, writing each decision directly into the IR
/// (anchor layouts, temporary layouts and tensor_desc types).
///
/// Ops are visited backward by xegpu::walkBackward, shared with the layout
/// recovery: in reverse program order, nested regions before their parent op.
/// So an op is reached only after every consumer of its results, except a
/// consumer reached through a loop back edge (scf.for's yield -> iter_arg) or,
/// for scf.while, whose regions are walked against its control flow (before
/// region first), through scf.condition -> after region.
/// Each op therefore makes a one-time decision:
///   - each result takes the layout demanded by its nearest consumer in
///     program order. Consumers are visited from the farthest to the nearest,
///     and each overwrites the layout recorded for the value, so the recorded
///     layout is the nearest consumer's demand when the definition is reached;
///   - the op's transfer function (`visit*Op`) turns the result layouts into
///     the layouts it demands of its operands, fixing the anchor layout of an
///     anchor op on the way.
/// Region ops are handled at their boundaries:
///   - a terminator operand can feed both a region argument (the next
///     iteration or the next region) and a region result (on exit). It takes
///     the region argument's layout when that is known, i.e. when the
///     argument's region has already been walked (e.g. scf.while's "after"
///     region yield, feeding the "before" region). Otherwise it falls back to
///     the region result's layout (e.g. scf.for's yield, whose iter_arg is
///     reached only through the back edge, or scf.condition, whose "after"
///     region has not been walked yet);
///   - a region op's init operand takes the layout of the region argument it
///     feeds.
///
/// Values left without a layout (e.g. a cross-iteration reduction, which is
/// constrained only through the back edge) are filled afterwards by the
/// forward walk in `xegpu::propagateLayouts`; remaining producer/consumer
/// mismatches are resolved by ResolveLayoutConflicts.
class LayoutPropagation final : public xegpu::BackwardLayoutVisitor {
public:
  LayoutPropagation(Operation *scopeRoot, xegpu::LayoutKind layoutKind)
      : scopeRoot(scopeRoot), layoutKind(layoutKind) {}

  /// Runs the backward scan over every function under `scopeRoot`.
  LogicalResult run();

  /// Returns the layout decided for `val`, or null if none was decided.
  xegpu::DistributeLayoutAttr getLayout(Value val) const {
    return valueLayouts.lookup(val);
  }

  /// Prints the layouts decided for function arguments and op results.
  void print(llvm::raw_ostream &os);

private:
  // The op this propagation runs on.
  Operation *scopeRoot;
  xegpu::LayoutKind layoutKind;

  // Layout of each value (op result or block argument): the demand of the
  // nearest consumer visited so far, which is the decided layout once the
  // value's definition has been reached.
  DenseMap<Value, xegpu::DistributeLayoutAttr> valueLayouts;
  // Set when an anchor op cannot be assigned a layout.
  bool propagationFailed = false;

  //===--------------------------------------------------------------------===//
  // Backward scan
  //===--------------------------------------------------------------------===//

  // Records the layout a consumer demands of `val`, overriding the demand of
  // any consumer visited earlier (i.e. farther from the definition).
  void recordDemand(Value val, xegpu::DistributeLayoutAttr layout) {
    if (layout)
      valueLayouts[val] = layout;
  }
  void visitRegularOp(Operation *op) override;
  void visitTerminator(RegionBranchTerminatorOpInterface terminator) override;
  void visitRegionOp(RegionBranchOpInterface regionOp) override;
  void visitFunction(FunctionOpInterface funcOp) override;

  //===--------------------------------------------------------------------===//
  // Transfer functions: given the layouts the consumers demand of an op's
  // results, compute the layouts the op demands of its operands and the result
  // layouts it produces.
  //===--------------------------------------------------------------------===//

  // `operands` receives the layout the op demands of each operand. `results`
  // holds the consumers' layout of each result on entry; a transfer function
  // that requires a different result layout (e.g. a reduction) overwrites it.
  using LayoutsMutRef = MutableArrayRef<xegpu::DistributeLayoutAttr>;

  void visitOperation(Operation *op, LayoutsMutRef operands,
                      LayoutsMutRef results);

  void visitDpasOp(xegpu::DpasOp dpas, LayoutsMutRef operands,
                   LayoutsMutRef results);
  void visitDpasMxOp(xegpu::DpasMxOp dpasMx, LayoutsMutRef operands,
                     LayoutsMutRef results);
  void visitStoreNdOp(xegpu::StoreNdOp store, LayoutsMutRef operands,
                      LayoutsMutRef results);
  void visitStoreScatterOp(xegpu::StoreScatterOp storeScatter,
                           LayoutsMutRef operands, LayoutsMutRef results);
  void visitLoadNdOp(xegpu::LoadNdOp load, LayoutsMutRef operands,
                     LayoutsMutRef results);
  void visitLoadGatherOp(xegpu::LoadGatherOp load, LayoutsMutRef operands,
                         LayoutsMutRef results);
  void visitPrefetchNdOp(xegpu::PrefetchNdOp prefetch, LayoutsMutRef operands,
                         LayoutsMutRef results);
  void visitLoadMatrixOp(xegpu::LoadMatrixOp load, LayoutsMutRef operands,
                         LayoutsMutRef results);
  void visitStoreMatrixOp(xegpu::StoreMatrixOp store, LayoutsMutRef operands,
                          LayoutsMutRef results);
  void visitConvertLayoutOp(xegpu::ConvertLayoutOp convertLayout,
                            LayoutsMutRef operands, LayoutsMutRef results);
  void visitTransposeOp(vector::TransposeOp transpose, LayoutsMutRef operands,
                        LayoutsMutRef results);
  void visitVectorBitcastOp(vector::BitCastOp bitcast, LayoutsMutRef operands,
                            LayoutsMutRef results);
  void visitVectorInterleaveOp(vector::InterleaveOp interleave,
                               LayoutsMutRef operands, LayoutsMutRef results);
  void visitVectorDeinterleaveOp(vector::DeinterleaveOp deinterleave,
                                 LayoutsMutRef operands, LayoutsMutRef results);
  void visitVectorMultiReductionOp(vector::MultiDimReductionOp reduction,
                                   LayoutsMutRef operands,
                                   LayoutsMutRef results);
  void visitVectorReductionOp(vector::ReductionOp reduction,
                              LayoutsMutRef operands, LayoutsMutRef results);
  void visitVectorBroadCastOp(vector::BroadcastOp broadcast,
                              LayoutsMutRef operands, LayoutsMutRef results);
  void visitShapeCastOp(vector::ShapeCastOp shapeCast, LayoutsMutRef operands,
                        LayoutsMutRef results);
  void
  visitInsertStridedSliceOp(vector::InsertStridedSliceOp insertStridedSlice,
                            LayoutsMutRef operands, LayoutsMutRef results);

  bool hasParamsOfLayoutKind(xegpu::DistributeLayoutAttr anchorLayout);

  // Number of subgroups for `op`. Missing count is fatal in subgroup mode.
  FailureOr<int64_t> getNumSgOrFail(Operation *op, int sgSize,
                                    xegpu::DistributeLayoutAttr consumerLayout);

  // Reserved for the anchor ops that are the sources of the propagation
  // (store/dpas), whose layout must be correct.
  void markFailure(Operation *op, const llvm::Twine &message) {
    op->emitError(message);
    propagationFailed = true;
  }
};
} // namespace

void LayoutPropagation::visitOperation(Operation *op, LayoutsMutRef operands,
                                       LayoutsMutRef results) {
  TypeSwitch<Operation *>(op)
      .Case([&](xegpu::DpasOp op) { visitDpasOp(op, operands, results); })
      .Case([&](xegpu::DpasMxOp op) { visitDpasMxOp(op, operands, results); })
      .Case([&](xegpu::StoreNdOp op) { visitStoreNdOp(op, operands, results); })
      .Case([&](xegpu::StoreScatterOp op) {
        visitStoreScatterOp(op, operands, results);
      })
      .Case([&](xegpu::LoadNdOp op) { visitLoadNdOp(op, operands, results); })
      .Case([&](xegpu::LoadGatherOp op) {
        visitLoadGatherOp(op, operands, results);
      })
      .Case([&](xegpu::PrefetchNdOp op) {
        visitPrefetchNdOp(op, operands, results);
      })
      .Case([&](vector::TransposeOp op) {
        visitTransposeOp(op, operands, results);
      })
      .Case([&](vector::BitCastOp op) {
        visitVectorBitcastOp(op, operands, results);
      })
      .Case([&](vector::InterleaveOp op) {
        visitVectorInterleaveOp(op, operands, results);
      })
      .Case([&](vector::DeinterleaveOp op) {
        visitVectorDeinterleaveOp(op, operands, results);
      })
      .Case([&](vector::MultiDimReductionOp op) {
        visitVectorMultiReductionOp(op, operands, results);
      })
      .Case([&](vector::ReductionOp op) {
        visitVectorReductionOp(op, operands, results);
      })
      .Case([&](vector::BroadcastOp op) {
        visitVectorBroadCastOp(op, operands, results);
      })
      .Case([&](vector::ShapeCastOp op) {
        visitShapeCastOp(op, operands, results);
      })
      .Case([&](vector::InsertStridedSliceOp op) {
        visitInsertStridedSliceOp(op, operands, results);
      })
      .Case([&](xegpu::LoadMatrixOp op) {
        visitLoadMatrixOp(op, operands, results);
      })
      .Case([&](xegpu::StoreMatrixOp op) {
        visitStoreMatrixOp(op, operands, results);
      })
      .Case([&](xegpu::ConvertLayoutOp op) {
        visitConvertLayoutOp(op, operands, results);
      })
      // All other ops: every vector / tensor_desc operand takes the layout of
      // the first result that has one.
      .Default([&](Operation *op) {
        auto it = llvm::find_if(results, [](auto layout) { return !!layout; });
        if (it == results.end())
          return;
        for (auto [operandLayout, operand] :
             llvm::zip_equal(operands, op->getOperands()))
          if (isa<xegpu::TensorDescType, VectorType>(operand.getType()))
            operandLayout = *it;
      });
}

bool LayoutPropagation::hasParamsOfLayoutKind(
    xegpu::DistributeLayoutAttr anchorLayout) {
  if (anchorLayout == nullptr) {
    return false;
  }
  if (layoutKind == xegpu::LayoutKind::InstData) {
    return !(anchorLayout.getEffectiveInstDataAsInt().empty());
  }
  if (layoutKind == xegpu::LayoutKind::Lane) {
    return !(anchorLayout.getEffectiveLaneLayoutAsInt().empty() ||
             anchorLayout.getEffectiveLaneDataAsInt().empty());
  }
  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    return !(anchorLayout.getEffectiveSgLayoutAsInt().empty() ||
             anchorLayout.getEffectiveSgDataAsInt().empty());
  }
  return false;
}

FailureOr<int64_t>
LayoutPropagation::getNumSgOrFail(Operation *op, int sgSize,
                                  xegpu::DistributeLayoutAttr consumerLayout) {
  // The consumer's sg_layout, when present, dictates the count.
  if (consumerLayout) {
    auto sgLayout = consumerLayout.getEffectiveSgLayoutAsInt();
    if (!sgLayout.empty())
      return llvm::product_of(sgLayout);
  }
  // Otherwise fall back to the kernel's known_block_size.
  if (FailureOr<int64_t> numSg =
          xegpu::getNumSubgroupsFromBlockSize(op, sgSize);
      succeeded(numSg))
    return *numSg;
  // Only subgroup mode needs the count; elsewhere a missing one is benign.
  if (layoutKind == xegpu::LayoutKind::Subgroup) {
    markFailure(op, "Unable to determine the number of subgroups for the "
                    "operation. Please check @known_block_size is properly "
                    "attached as kernel attributes, with power-of-two "
                    "dimensions covering at least one subgroup.");
    return failure();
  }
  return int64_t{0};
}

void LayoutPropagation::visitPrefetchNdOp(xegpu::PrefetchNdOp prefetch,
                                          LayoutsMutRef operands,
                                          LayoutsMutRef results) {

  xegpu::DistributeLayoutAttr prefetchLayout;
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(prefetch).value_or(""));
  if (!uArch)
    return;
  xegpu::DistributeLayoutAttr anchorLayout = prefetch.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    prefetchLayout = anchorLayout;
    if (layoutKind == xegpu::LayoutKind::InstData) {
      const auto *uArchInstruction =
          dyn_cast<xegpu::uArch::Subgroup2DBlockPrefetchInstruction>(
              uArch->getInstruction(
                  xegpu::uArch::InstructionKind::Subgroup2DBlockPrefetch));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeBlockStoreLaneLayoutFromInstData(
          anchorLayout, prefetch.getTensorDescType().getElementType(),
          uArchInstruction, uArch->getSubgroupSize());
      if (!completed) {
        prefetch.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      prefetch.setLayoutAttr(*completed);
      prefetchLayout = *completed;
    }
  } else {
    auto tdescTy = prefetch.getTensorDescType();
    auto numSgOrErr =
        getNumSgOrFail(prefetch, uArch->getSubgroupSize(), nullptr);
    if (failed(numSgOrErr))
      return;

    auto layoutAttr = xegpu::setupPrefetchNdAnchorLayout(
        layoutKind, tdescTy, numSgOrErr.value_or(0), uArch);
    if (!layoutAttr) {
      prefetch.emitWarning(
          "Failed to determine required layout for prefetch_nd.");
      return;
    }
    prefetchLayout = layoutAttr;
    prefetch.setLayoutAttr(layoutAttr);
  }
  // Propagate the layout to the source tensor descriptor.
  operands[0] = prefetchLayout;
}

void LayoutPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction, LayoutsMutRef operands,
    LayoutsMutRef results) {
  Type resultTy = reduction.getDestType();
  // The layout of a vector result must be present; a scalar result has none.
  xegpu::DistributeLayoutAttr consumerLayoutAttr;
  if (!resultTy.isIntOrFloat()) {
    consumerLayoutAttr = results[0];
    if (!consumerLayoutAttr)
      return;
  }

  VectorType sourceTy = reduction.getSourceVectorType();
  SmallVector<int64_t> reductionDims(reduction.getReductionDims());

  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(reduction).value_or(""));
  if (!uArch)
    return;

  auto numSgOrErr =
      getNumSgOrFail(reduction, uArch->getSubgroupSize(), consumerLayoutAttr);
  if (failed(numSgOrErr))
    return;

  // The result layout the operation requires. It may differ from the layout
  // its consumers demand; the conflict is resolved later by converting the
  // required result layout to the consumer layout.
  auto requiredResLayoutAttr = xegpu::setupMultiReductionResultLayout(
      layoutKind, sourceTy, consumerLayoutAttr, reductionDims,
      numSgOrErr.value_or(0), uArch);

  results[0] = requiredResLayoutAttr;

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferMultiReductionSourceLayout(
      requiredResLayoutAttr, reductionDims);

  operands[0] = srcLayoutAttr;
  // Accumulator should have the same layout as the result.
  operands[1] = requiredResLayoutAttr;
}

void LayoutPropagation::visitVectorReductionOp(vector::ReductionOp reduction,
                                               LayoutsMutRef operands,
                                               LayoutsMutRef results) {

  VectorType sourceTy = reduction.getSourceVectorType();
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(reduction).value_or(""));
  if (!uArch)
    return;

  auto requiredResLayoutAttr =
      xegpu::setupReductionResultLayout(layoutKind, sourceTy, uArch);
  results[0] = requiredResLayoutAttr;

  auto srcLayoutAttr = xegpu::inferReductionSourceLayout(requiredResLayoutAttr);
  operands[0] = srcLayoutAttr;
  if (reduction.getAcc())
    operands[1] = requiredResLayoutAttr;
}

void LayoutPropagation::visitVectorBroadCastOp(vector::BroadcastOp broadcast,
                                               LayoutsMutRef operands,
                                               LayoutsMutRef results) {
  // The layout of the result must be present.
  xegpu::DistributeLayoutAttr resultLayoutAttr = results[0];
  if (!resultLayoutAttr)
    return;

  // Only consider vector to vector broadcasts for now.
  VectorType resultTy = broadcast.getResultVectorType();
  VectorType sourceTy = dyn_cast<VectorType>(broadcast.getSourceType());
  // skip layout propagation for non-vector source operand.
  if (!sourceTy)
    return;

  auto srcShape = sourceTy.getShape();
  auto resShape = resultTy.getShape();

  xegpu::DistributeLayoutAttr srcLayoutAttr =
      xegpu::inferBroadcastSourceLayout(resultLayoutAttr, resShape, srcShape);

  operands[0] = srcLayoutAttr;
}

void LayoutPropagation::visitShapeCastOp(vector::ShapeCastOp shapeCast,
                                         LayoutsMutRef operands,
                                         LayoutsMutRef results) {
  // The layout of the result must be present.
  xegpu::DistributeLayoutAttr resultLayoutAttr = results[0];
  if (!resultLayoutAttr)
    return;
  ArrayRef<int64_t> resShape = shapeCast.getResultVectorType().getShape();
  ArrayRef<int64_t> srcShape = shapeCast.getSourceVectorType().getShape();

  auto requiredResLayoutAttr = xegpu::setupShapeCastResultLayout(
      layoutKind, shapeCast.getSourceVectorType(),
      shapeCast.getResultVectorType(), resultLayoutAttr);
  // The consumer layout cannot be expressed on the source: no lane_data makes
  // each lane's data a contiguous run of the collapsed source dim. shape_cast
  // is not an anchor op, so warn and leave the value un-laid-out instead of
  // propagating a layout that would move data between lanes.
  if (!requiredResLayoutAttr) {
    shapeCast.emitWarning("Failed to infer source layout for shape_cast; the "
                          "result layout required by its consumers cannot be "
                          "collapsed onto the source shape.");
    return;
  }
  results[0] = requiredResLayoutAttr;

  xegpu::DistributeLayoutAttr srcLayoutAttr = xegpu::inferShapeCastSourceLayout(
      requiredResLayoutAttr, resShape, srcShape);
  // shape_cast is not an anchor op: another consumer of the source value may
  // still supply a valid layout, so warn instead of stopping the propagation.
  if (!srcLayoutAttr) {
    shapeCast.emitWarning("Failed to infer source layout for shape_cast; "
                          "unsupported shape-cast pattern.");
    return;
  }

  operands[0] = srcLayoutAttr;
}

/// Set the layouts for DPAS A, B, and C operands.
void LayoutPropagation::visitDpasOp(xegpu::DpasOp dpas, LayoutsMutRef operands,
                                    LayoutsMutRef results) {
  xegpu::DistributeLayoutAttr dpasALayout;
  xegpu::DistributeLayoutAttr dpasBLayout;
  xegpu::DistributeLayoutAttr dpasCDLayout;

  const auto *uArch = xegpu::uArch::getUArch(getChipStr(dpas).value_or(""));
  if (!uArch)
    return;
  VectorType aTy = dpas.getLhsType();
  VectorType bTy = dpas.getRhsType();
  VectorType cdTy = dpas.getResultType();

  xegpu::DistributeLayoutAttr anchorLayoutCD = dpas.getLayoutCdAttr();
  if (hasParamsOfLayoutKind(anchorLayoutCD)) {
    xegpu::DistributeLayoutAttr anchorLayoutA = dpas.getLayoutAAttr();
    xegpu::DistributeLayoutAttr anchorLayoutB = dpas.getLayoutBAttr();
    assert(hasParamsOfLayoutKind(anchorLayoutA) &&
           "Expected anchor layout for DPAS A operand.");
    assert(hasParamsOfLayoutKind(anchorLayoutB) &&
           "Expected anchor layout for DPAS B operand.");
    dpasALayout = anchorLayoutA;
    dpasBLayout = anchorLayoutB;
    dpasCDLayout = anchorLayoutCD;
    if (layoutKind == xegpu::LayoutKind::InstData) {
      auto completed = xegpu::completeDpasLaneLayoutFromInstData(
          anchorLayoutA, anchorLayoutB, anchorLayoutCD, aTy, bTy, cdTy, uArch);
      if (!completed) {
        markFailure(
            dpas,
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      auto [completedA, completedB, completedCD] = *completed;
      dpas.setLayoutAAttr(completedA);
      dpas.setLayoutBAttr(completedB);
      dpas.setLayoutCdAttr(completedCD);
      dpasALayout = completedA;
      dpasBLayout = completedB;
      dpasCDLayout = completedCD;
    }
  } else {
    xegpu::DistributeLayoutAttr requiredCDLayoutAttr, requiredALayout,
        requiredBLayout;

    xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
    if (!consumerLayoutAttr)
      return;

    auto numSgOrErr =
        getNumSgOrFail(dpas, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (failed(numSgOrErr))
      return;

    auto layouts =
        xegpu::setupDpasLayout(layoutKind, aTy, bTy, cdTy, consumerLayoutAttr,
                               numSgOrErr.value_or(0), uArch);
    if (!layouts.has_value()) {
      markFailure(dpas,
                  "Failed to determine required layouts for DPAS operands.");
      return;
    }

    std::tie(requiredALayout, requiredBLayout, requiredCDLayoutAttr) = *layouts;

    dpas.setLayoutAAttr(requiredALayout);
    dpas.setLayoutBAttr(requiredBLayout);
    dpas.setLayoutCdAttr(requiredCDLayoutAttr);
    dpasALayout = requiredALayout;
    dpasBLayout = requiredBLayout;
    dpasCDLayout = requiredCDLayoutAttr;
  }
  operands[0] = dpasALayout;
  operands[1] = dpasBLayout;
  if (operands.size() > 2)
    operands[2] = dpasCDLayout;
}

/// Propagate layout for DpasMxOp operands using the layout attributes.
/// DpasMxOp has operands: a, b, acc (optional), scale_a (optional), scale_b
/// (optional)
void LayoutPropagation::visitDpasMxOp(xegpu::DpasMxOp dpasMx,
                                      LayoutsMutRef operands,
                                      LayoutsMutRef results) {

  // Initialize layout variables
  xegpu::DistributeLayoutAttr dpasMxALayout, dpasMxBLayout, dpasMxCDLayout;
  xegpu::DistributeLayoutAttr dpasMxAScaleLayout, dpasMxBScaleLayout;

  // Get existing layout attributes from the operation
  xegpu::DistributeLayoutAttr anchorLayoutA = dpasMx.getLayoutAAttr();
  xegpu::DistributeLayoutAttr anchorLayoutB = dpasMx.getLayoutBAttr();
  xegpu::DistributeLayoutAttr anchorLayoutCD = dpasMx.getLayoutCdAttr();

  const auto *uArch = xegpu::uArch::getUArch(getChipStr(dpasMx).value_or(""));
  if (!uArch)
    return;

  VectorType aTy = dpasMx.getAType();
  VectorType bTy = dpasMx.getBType();
  VectorType cdTy = dpasMx.getResultType();

  // Get scale types if present
  VectorType aScaleTy;
  VectorType bScaleTy;
  Value scaleA = dpasMx.getScaleA();
  Value scaleB = dpasMx.getScaleB();
  if (scaleA)
    aScaleTy = dyn_cast<VectorType>(scaleA.getType());
  if (scaleB)
    bScaleTy = dyn_cast<VectorType>(scaleB.getType());

  // Check if all layouts are already set
  if (anchorLayoutA && anchorLayoutB && anchorLayoutCD &&
      hasParamsOfLayoutKind(anchorLayoutA) &&
      hasParamsOfLayoutKind(anchorLayoutB) &&
      hasParamsOfLayoutKind(anchorLayoutCD)) {
    dpasMxALayout = anchorLayoutA;
    dpasMxBLayout = anchorLayoutB;
    dpasMxCDLayout = anchorLayoutCD;

    // Get scale layouts if available
    xegpu::DistributeLayoutAttr anchorLayoutAScale =
        dpasMx.getLayoutAScaleAttr();
    xegpu::DistributeLayoutAttr anchorLayoutBScale =
        dpasMx.getLayoutBScaleAttr();
    if (anchorLayoutAScale)
      dpasMxAScaleLayout = anchorLayoutAScale;
    if (anchorLayoutBScale)
      dpasMxBScaleLayout = anchorLayoutBScale;

    if (layoutKind == xegpu::LayoutKind::InstData) {
      auto completed = xegpu::completeDpasMxLaneLayoutFromInstData(
          anchorLayoutA, anchorLayoutB, anchorLayoutCD, aTy, bTy, cdTy,
          aScaleTy, bScaleTy, uArch);
      if (!completed) {
        markFailure(
            dpasMx,
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      auto [completedA, completedB, completedCD, completedAScale,
            completedBScale] = *completed;
      dpasMx.setLayoutAAttr(completedA);
      dpasMx.setLayoutBAttr(completedB);
      dpasMx.setLayoutCdAttr(completedCD);
      dpasMxALayout = completedA;
      dpasMxBLayout = completedB;
      dpasMxCDLayout = completedCD;
      if (completedAScale) {
        dpasMx.setLayoutAScaleAttr(completedAScale);
        dpasMxAScaleLayout = completedAScale;
      }
      if (completedBScale) {
        dpasMx.setLayoutBScaleAttr(completedBScale);
        dpasMxBScaleLayout = completedBScale;
      }
    }
  } else {
    xegpu::DistributeLayoutAttr requiredCDLayoutAttr, requiredALayout,
        requiredBLayout, requiredAScaleLayout, requiredBScaleLayout;

    xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
    if (!consumerLayoutAttr)
      return;

    auto numSgOrErr =
        getNumSgOrFail(dpasMx, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (failed(numSgOrErr))
      return;

    auto layouts = xegpu::setupDpasMxLayout(
        layoutKind, aTy, bTy, cdTy, aScaleTy, bScaleTy, consumerLayoutAttr,
        numSgOrErr.value_or(0), uArch);
    if (!layouts.has_value()) {
      markFailure(dpasMx,
                  "Failed to determine required layouts for DPAS_MX operands.");
      return;
    }

    std::tie(requiredALayout, requiredBLayout, requiredCDLayoutAttr,
             requiredAScaleLayout, requiredBScaleLayout) = *layouts;

    dpasMx.setLayoutAAttr(requiredALayout);
    dpasMx.setLayoutBAttr(requiredBLayout);
    dpasMx.setLayoutCdAttr(requiredCDLayoutAttr);
    if (requiredAScaleLayout)
      dpasMx.setLayoutAScaleAttr(requiredAScaleLayout);
    if (requiredBScaleLayout)
      dpasMx.setLayoutBScaleAttr(requiredBScaleLayout);

    dpasMxALayout = requiredALayout;
    dpasMxBLayout = requiredBLayout;
    dpasMxCDLayout = requiredCDLayoutAttr;
    if (requiredAScaleLayout)
      dpasMxAScaleLayout = requiredAScaleLayout;
    if (requiredBScaleLayout)
      dpasMxBScaleLayout = requiredBScaleLayout;
  }

  // Propagate layouts to operands. Because acc, scale_a, scale_b are all
  // optional (AttrSizedOperandSegments), the index of each present operand in
  // `operands` depends on which optionals are actually supplied. Use the
  // op's accessors to determine the correct positional index.
  operands[0] = dpasMxALayout;
  operands[1] = dpasMxBLayout;
  unsigned idx = 2;
  if (dpasMx.getAcc()) {
    operands[idx] = dpasMxCDLayout;
    ++idx;
  }
  if (dpasMx.getScaleA()) {
    if (dpasMxAScaleLayout)
      operands[idx] = dpasMxAScaleLayout;
    ++idx;
  }
  if (dpasMx.getScaleB()) {
    if (dpasMxBScaleLayout)
      operands[idx] = dpasMxBScaleLayout;
    ++idx;
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutPropagation::visitStoreNdOp(xegpu::StoreNdOp store,
                                       LayoutsMutRef operands,
                                       LayoutsMutRef results) {
  xegpu::DistributeLayoutAttr storeLayout;
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(store).value_or(""));
  if (!uArch)
    return;
  xegpu::DistributeLayoutAttr anchorLayout = store.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    storeLayout = anchorLayout;
    if (layoutKind == xegpu::LayoutKind::InstData) {

      const auto *uArchInstruction =
          dyn_cast<xegpu::uArch::Subgroup2DBlockStoreInstruction>(
              uArch->getInstruction(
                  xegpu::uArch::InstructionKind::Subgroup2DBlockStore));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeBlockStoreLaneLayoutFromInstData(
          anchorLayout, store.getValueType().getElementType(), uArchInstruction,
          uArch->getSubgroupSize());
      if (!completed) {
        markFailure(
            store,
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      store.setLayoutAttr(*completed);
      storeLayout = *completed;
    }
  } else {
    auto numSgOrErr = getNumSgOrFail(store, uArch->getSubgroupSize(), nullptr);
    if (failed(numSgOrErr))
      return;

    auto layoutAttr = xegpu::setupStoreNdAnchorLayout(
        layoutKind, store.getValueType(), numSgOrErr.value_or(0), uArch);
    if (!layoutAttr) {
      markFailure(store, "Failed to determine required layout for store_nd.");
      return;
    }
    storeLayout = layoutAttr;
    store.setLayoutAttr(layoutAttr);
  }
  // Propagate the layout to the value operand.
  // Both operands should have the same layout
  for (xegpu::DistributeLayoutAttr &operand : operands)
    operand = storeLayout;
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// LoadNdOp.
void LayoutPropagation::visitLoadNdOp(xegpu::LoadNdOp load,
                                      LayoutsMutRef operands,
                                      LayoutsMutRef results) {
  xegpu::DistributeLayoutAttr loadLayout;

  const auto *uArch = xegpu::uArch::getUArch(getChipStr(load).value_or(""));
  if (!uArch)
    return;
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  xegpu::DistributeLayoutAttr anchorLayout = load.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    loadLayout = anchorLayout;
    if (layoutKind == xegpu::LayoutKind::InstData && consumerLayoutAttr &&
        !consumerLayoutAttr.getEffectiveLaneLayoutAsInt().empty()) {
      const auto *uArchInstruction =
          dyn_cast<xegpu::uArch::Subgroup2DBlockLoadInstruction>(
              uArch->getInstruction(
                  xegpu::uArch::InstructionKind::Subgroup2DBlockLoad));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeBlockLoadLaneLayoutFromInstData(
          anchorLayout, consumerLayoutAttr, load.getType().getElementType(),
          uArchInstruction, uArch->getSubgroupSize());
      if (!completed) {
        load.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      load.setLayoutAttr(*completed);
      loadLayout = *completed;
    }
  } else {
    auto numSgOrErr =
        getNumSgOrFail(load, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (failed(numSgOrErr))
      return;
    auto layoutAttr = xegpu::setupLoadNdAnchorLayout(
        layoutKind, load.getType(), consumerLayoutAttr, numSgOrErr.value_or(0),
        uArch);
    if (!layoutAttr) {
      load.emitWarning("Failed to determine required layout for load_nd.");
      return;
    }
    loadLayout = layoutAttr;
    load.setLayoutAttr(layoutAttr);
  }
  // Propagate the new layout to the tensor descriptor operand.
  operands[0] = loadLayout;
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// ConvertLayoutOp.
void LayoutPropagation::visitConvertLayoutOp(xegpu::ConvertLayoutOp convert,
                                             LayoutsMutRef operands,
                                             LayoutsMutRef results) {

  // TODO: fix if one of the layouts is a slice layout
  auto targetLayoutAttr =
      dyn_cast<xegpu::LayoutAttr>(convert.getTargetLayoutAttr());
  // input_layout is optional, so it may be null.
  auto inputLayoutAttr =
      dyn_cast_if_present<xegpu::LayoutAttr>(convert.getInputLayoutAttr());

  // The result's propagated layout is authoritative for the converted value.
  // Fill the lane_layout / lane_data / order parameters the target_layout is
  // missing from it (sg_layout / sg_data / inst_data are left as-is), so the
  // target stays consistent with what is actually propagated downstream.
  auto resultLayoutAttr = dyn_cast_if_present<xegpu::LayoutAttr>(results[0]);
  if (resultLayoutAttr && targetLayoutAttr) {
    if (layoutKind == xegpu::LayoutKind::InstData &&
        !targetLayoutAttr.getLaneLayout()) {
      targetLayoutAttr = xegpu::LayoutAttr::get(
          convert.getContext(), targetLayoutAttr.getSgLayout(),
          targetLayoutAttr.getSgData(), targetLayoutAttr.getInstData(),
          resultLayoutAttr.getLaneLayout(), resultLayoutAttr.getLaneData(),
          resultLayoutAttr.getOrder());
      convert.setTargetLayoutAttr(targetLayoutAttr);
    }
  }

  // Fill only the lane_layout / lane_data / order parameters the input_layout
  // is missing from the target_layout (sg_layout / sg_data / inst_data are left
  // as-is), so the producer side receives a fully-populated lane layout.
  if (inputLayoutAttr && targetLayoutAttr) {
    if (layoutKind == xegpu::LayoutKind::InstData &&
        !inputLayoutAttr.getLaneLayout()) {
      auto merged = xegpu::LayoutAttr::get(
          convert.getContext(), inputLayoutAttr.getSgLayout(),
          inputLayoutAttr.getSgData(), inputLayoutAttr.getInstData(),
          targetLayoutAttr.getLaneLayout(), targetLayoutAttr.getLaneData(),
          targetLayoutAttr.getOrder());
      convert.setInputLayoutAttr(merged);
    }
  }

  // Propagate the effective input layout to the source operand.
  operands[0] = convert.getEffectiveInputLayout();
}

/// For vector::TransposeOp, the layout of the result is transposed and
/// propagated to the operand.
void LayoutPropagation::visitTransposeOp(vector::TransposeOp transpose,
                                         LayoutsMutRef operands,
                                         LayoutsMutRef results) {
  // Need the layout of transpose result to propagate to the operands.
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (!consumerLayoutAttr)
    return;
  auto srcLayoutAttr = xegpu::inferTransposeSourceLayout(
      consumerLayoutAttr, transpose.getPermutation());

  // Propagate the new layout to the vector operand.
  operands[0] = srcLayoutAttr;
}

/// For vector::BitCastOp, the lane_data of the source layout is changed based
/// on the bit width of the source and result types.
void LayoutPropagation::visitVectorBitcastOp(vector::BitCastOp bitcast,
                                             LayoutsMutRef operands,
                                             LayoutsMutRef results) {
  // Need the layout of bitcast result to propagate to the operands.
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (!consumerLayoutAttr)
    return;

  auto srcVecType = bitcast.getSourceVectorType();
  auto resVecType = bitcast.getResultVectorType();
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(bitcast).value_or(""));
  if (!uArch)
    return;
  auto requiredResLayoutAttr = setupBitCastResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  results[0] = requiredResLayoutAttr;

  int inElemTyBitWidth = srcVecType.getElementType().getIntOrFloatBitWidth();
  int outElemTyBitWidth = resVecType.getElementType().getIntOrFloatBitWidth();

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferBitCastSourceLayout(
      requiredResLayoutAttr, outElemTyBitWidth, inElemTyBitWidth);

  operands[0] = srcLayoutAttr;
}

/// For vector::InterleaveOp, the result has double the innermost dimension
/// size compared to each source operand. The layout is propagated from result
/// to sources, adjusting for the 2x size increase.
void LayoutPropagation::visitVectorInterleaveOp(vector::InterleaveOp interleave,
                                                LayoutsMutRef operands,
                                                LayoutsMutRef results) {
  // Need the layout of interleave result to propagate to the operands.
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (!consumerLayoutAttr)
    return;

  auto srcVecType = interleave.getSourceVectorType();
  auto resVecType = interleave.getResultVectorType();
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(interleave).value_or(""));
  if (!uArch)
    return;

  // Setup the result layout to ensure the source layout can be safely derived
  auto requiredResLayoutAttr = setupInterleaveResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  results[0] = requiredResLayoutAttr;

  // Derive the source layout from the result layout (halve the innermost dim)
  auto srcLayoutAttr =
      xegpu::inferInterleaveSourceLayout(requiredResLayoutAttr);

  // Both operands (lhs and rhs) get the same source layout
  operands[0] = srcLayoutAttr;
  operands[1] = srcLayoutAttr;
}

/// For vector::DeinterleaveOp, the source has double the innermost dimension
/// size compared to each result. The layout is propagated from results to
/// source, adjusting for the 2x size decrease in results.
void LayoutPropagation::visitVectorDeinterleaveOp(
    vector::DeinterleaveOp deinterleave, LayoutsMutRef operands,
    LayoutsMutRef results) {
  // Need the layout of deinterleave results to propagate to the operand.
  // Use the first result's layout (both results should have the same layout)
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (!consumerLayoutAttr)
    return;

  // Derive the source layout from the result layout (double the innermost
  // dim) No setup function needed - just infer directly
  auto srcLayoutAttr = xegpu::inferDeinterleaveSourceLayout(consumerLayoutAttr);

  operands[0] = srcLayoutAttr;
}

void LayoutPropagation::visitInsertStridedSliceOp(
    vector::InsertStridedSliceOp insertStridedSlice, LayoutsMutRef operands,
    LayoutsMutRef results) {
  // The layout of the result must be present.
  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (!consumerLayoutAttr)
    return;

  auto srcVecType = insertStridedSlice.getSourceVectorType();
  auto resVecType = insertStridedSlice.getDestVectorType();
  const auto *uArch = xegpu::uArch::getUArch(
      xegpu::getChipStr(insertStridedSlice).value_or(""));
  if (!uArch)
    return;

  auto requiredResLayoutAttr = xegpu::setupInsertStridedSliceResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);
  results[0] = requiredResLayoutAttr;

  auto srcLayoutAttr = xegpu::inferInsertStridedSliceSourceLayout(
      requiredResLayoutAttr, resVecType.getShape(), srcVecType.getShape());
  operands[0] = srcLayoutAttr;
  operands[1] = requiredResLayoutAttr;
}

/// Propagate the layout of the result to the tensor descriptor, mask and
/// offset operands in LoadGatherOp.
void LayoutPropagation::visitLoadGatherOp(xegpu::LoadGatherOp load,
                                          LayoutsMutRef operands,
                                          LayoutsMutRef results) {
  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = load.getLayoutAttr();
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(load).value_or(""));
  if (!uArch)
    return;
  VectorType resVecTy = load.getValueType();
  // `contiguity` says how many neighbouring elements one lane may take in a
  // single access. Absent, only one element per lane is safe.
  int contigChunkSize = load.getContiguity().value_or(1);

  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
    if (layoutKind == xegpu::LayoutKind::InstData && consumerLayoutAttr &&
        !consumerLayoutAttr.getEffectiveLaneLayoutAsInt().empty()) {
      const auto uArchInstruction =
          dyn_cast<xegpu::uArch::LoadGatherInstruction>(
              uArch->getInstruction(xegpu::uArch::InstructionKind::LoadGather));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeScatterLoadLaneLayoutFromInstData(
          anchorLayoutAttr, consumerLayoutAttr, resVecTy.getElementType(),
          uArchInstruction, uArch->getSubgroupSize());
      if (!completed) {
        load.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      requiredAnchorLayoutAttr = *completed;
      load.setLayoutAttr(requiredAnchorLayoutAttr);
    }
  } else if (!resVecTy) {
    load.emitWarning("Not propagating, non-vector payload supplied.");
    return;
  } else {
    auto numSgOrErr =
        getNumSgOrFail(load, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (failed(numSgOrErr))
      return;
    requiredAnchorLayoutAttr = xegpu::setupLoadGatherAnchorLayout(
        layoutKind, resVecTy, contigChunkSize, consumerLayoutAttr,
        numSgOrErr.value_or(0), uArch);
    load.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  // Propagate the new layout to the tensor descriptor operand.
  if (isa<xegpu::TensorDescType>(load.getSourceType()))
    operands[0] = requiredAnchorLayoutAttr;
  // The offset and mask operands share the value's anchor layout.
  operands[1] = requiredAnchorLayoutAttr;
  operands[2] = requiredAnchorLayoutAttr;
}

/// Set the layout for the value, tensor descriptor, offset and mask operands
/// in the StoreScatterOp.
void LayoutPropagation::visitStoreScatterOp(xegpu::StoreScatterOp storeScatter,
                                            LayoutsMutRef operands,
                                            LayoutsMutRef results) {

  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = storeScatter.getLayoutAttr();
  const auto *uArch =
      xegpu::uArch::getUArch(getChipStr(storeScatter).value_or(""));
  if (!uArch)
    return;
  VectorType srcVecTy = storeScatter.getValueType();
  // `contiguity` says how many neighbouring elements one lane may write in a
  // single access. Absent, only one element per lane is safe.
  int contigChunkSize = storeScatter.getContiguity().value_or(1);

  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
    if (layoutKind == xegpu::LayoutKind::InstData) {
      const auto uArchInstruction =
          dyn_cast<xegpu::uArch::StoreScatterInstruction>(uArch->getInstruction(
              xegpu::uArch::InstructionKind::StoreScatter));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeScatterStoreLaneLayoutFromInstData(
          anchorLayoutAttr, srcVecTy.getElementType(), uArchInstruction,
          uArch->getSubgroupSize());
      if (!completed) {
        markFailure(
            storeScatter,
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      requiredAnchorLayoutAttr = *completed;
      storeScatter.setLayoutAttr(requiredAnchorLayoutAttr);
    }
  } else {
    if (!srcVecTy) {
      storeScatter.emitWarning("Not propagating, non-vector payload supplied.");
      return;
    }
    auto numSgOrErr =
        getNumSgOrFail(storeScatter, uArch->getSubgroupSize(), nullptr);
    if (failed(numSgOrErr))
      return;
    requiredAnchorLayoutAttr = xegpu::setupStoreScatterAnchorLayout(
        layoutKind, srcVecTy, contigChunkSize, numSgOrErr.value_or(0), uArch);
    if (!requiredAnchorLayoutAttr) {
      markFailure(storeScatter,
                  "Failed to determine required layout for store scatter.");
      return;
    }
    storeScatter.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  // Propagate the payload operand layout
  operands[0] = requiredAnchorLayoutAttr;
  // Propagate the destination (if tdesc) operand layout
  if (isa<xegpu::TensorDescType>(storeScatter.getDestType()))
    operands[1] = requiredAnchorLayoutAttr;
  // The offset and mask operands share the value's anchor layout.
  operands[2] = requiredAnchorLayoutAttr;
  operands[3] = requiredAnchorLayoutAttr;
}

void LayoutPropagation::visitLoadMatrixOp(xegpu::LoadMatrixOp loadMatrixOp,
                                          LayoutsMutRef operands,
                                          LayoutsMutRef results) {

  xegpu::DistributeLayoutAttr consumerLayoutAttr = results[0];
  xegpu::DistributeLayoutAttr anchorLayout = loadMatrixOp.getLayoutAttr();

  // only need to set anchor layout, no need to porpagate to memdesc and
  // offset
  if (!hasParamsOfLayoutKind(anchorLayout)) {
    VectorType resVecTy =
        llvm::cast<VectorType>(loadMatrixOp.getRes().getType());
    const auto *uArch =
        xegpu::uArch::getUArch(getChipStr(loadMatrixOp).value_or(""));
    if (!uArch)
      return;
    int chunkSize =
        1; // placeHolder for future use when LoadMatrix supports coalescing
    auto numSgOrErr = getNumSgOrFail(loadMatrixOp, uArch->getSubgroupSize(),
                                     consumerLayoutAttr);
    if (failed(numSgOrErr))
      return;
    auto requiredAnchorLayoutAttr = xegpu::setupLoadMatrixAnchorLayout(
        layoutKind, resVecTy, chunkSize, consumerLayoutAttr,
        numSgOrErr.value_or(0), uArch);
    loadMatrixOp.setLayoutAttr(requiredAnchorLayoutAttr);
  }
}

void LayoutPropagation::visitStoreMatrixOp(xegpu::StoreMatrixOp storeMatrix,
                                           LayoutsMutRef operands,
                                           LayoutsMutRef results) {
  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = storeMatrix.getLayoutAttr();
  VectorType srcVecTy = llvm::cast<VectorType>(storeMatrix.getData().getType());
  const auto *uArch =
      xegpu::uArch::getUArch(getChipStr(storeMatrix).value_or(""));
  if (!uArch)
    return;
  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
    if (layoutKind == xegpu::LayoutKind::InstData) {
      const auto uArchInstruction =
          dyn_cast<xegpu::uArch::StoreScatterInstruction>(uArch->getInstruction(
              xegpu::uArch::InstructionKind::StoreScatter));
      if (!uArchInstruction)
        return;
      auto completed = xegpu::completeScatterStoreLaneLayoutFromInstData(
          anchorLayoutAttr, srcVecTy.getElementType(), uArchInstruction,
          uArch->getSubgroupSize());
      if (!completed) {
        markFailure(
            storeMatrix,
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      requiredAnchorLayoutAttr = *completed;
      storeMatrix.setLayoutAttr(requiredAnchorLayoutAttr);
    }
  } else {
    int chunkSize =
        1; // placeHolder for future use when StoreMatrix supports coalescing
    auto numSgOrErr =
        getNumSgOrFail(storeMatrix, uArch->getSubgroupSize(), nullptr);
    if (failed(numSgOrErr))
      return;
    requiredAnchorLayoutAttr = xegpu::setupStoreMatrixAnchorLayout(
        layoutKind, srcVecTy, chunkSize, numSgOrErr.value_or(0), uArch);
    if (!requiredAnchorLayoutAttr) {
      markFailure(storeMatrix,
                  "Failed to determine required layout for store matrix.");
      return;
    }
    storeMatrix.setLayoutAttr(requiredAnchorLayoutAttr);
  }
  operands[0] = requiredAnchorLayoutAttr;
}

//===----------------------------------------------------------------------===//
// LayoutPropagation: backward scan
//===----------------------------------------------------------------------===//

/// Set `layout` on the type of a tensor_desc value.
static void setTensorDescLayout(Value value,
                                xegpu::DistributeLayoutAttr layout) {
  auto tdescTy = cast<xegpu::TensorDescType>(value.getType());
  value.setType(xegpu::TensorDescType::get(
      tdescTy.getContext(), tdescTy.getShape(), tdescTy.getElementType(),
      tdescTy.getEncoding(), layout));
}

/// Update the function signature with the layouts of its tensor_desc
/// arguments. Every vector and tensor_desc argument must have a layout.
static LogicalResult
updateFunctionSignature(FunctionOpInterface funcOp,
                        xegpu::GetLayoutFnTy getLayoutOfValue) {
  // Only process functions whose type is a standard MLIR FunctionType.
  // Functions using a different type representation (e.g. llvm.func with
  // LLVMFunctionType) are not targets for XeGPU layout propagation, and
  // calling setType(FunctionType{}) on them would corrupt their type.
  if (!isa<FunctionType>(funcOp.getFunctionType()))
    return success();
  for (BlockArgument arg : funcOp.getArguments()) {
    if (!isa<VectorType, xegpu::TensorDescType>(arg.getType()))
      continue;
    xegpu::DistributeLayoutAttr layout = getLayoutOfValue(arg);
    if (!layout) {
      LLVM_DEBUG(DBGS() << "Expecting layout for function argument: " << arg
                        << " but got none.\n");
      return failure();
    }
    if (isa<xegpu::TensorDescType>(arg.getType()))
      setTensorDescLayout(arg, layout);
  }
  // NOTE: We assume that function results are not expected to have layouts.
  funcOp.setType(FunctionType::get(funcOp.getContext(),
                                   funcOp.getFunctionBody().getArgumentTypes(),
                                   funcOp.getResultTypes()));
  return success();
}

LogicalResult LayoutPropagation::run() {
  xegpu::walkBackward(scopeRoot, *this);
  // An anchor op with no determinable layout cannot be lowered; stop before
  // the forward fill fabricates degenerate layouts from unlabeled values.
  return failure(propagationFailed);
}

void LayoutPropagation::visitFunction(FunctionOpInterface funcOp) {
  // The arguments are decided once the whole body has been visited.
  if (failed(updateFunctionSignature(
          funcOp, [&](Value val) { return getLayout(val); })))
    markFailure(funcOp, "Failed to update operation with the layout.");
}

void LayoutPropagation::visitRegularOp(Operation *op) {
  SmallVector<xegpu::DistributeLayoutAttr> resultLayouts = llvm::map_to_vector(
      op->getResults(), [&](Value result) { return getLayout(result); });
  SmallVector<xegpu::DistributeLayoutAttr> operandLayouts(op->getNumOperands());
  visitOperation(op, operandLayouts, resultLayouts);

  // Recorded in reverse so that, if the op demands different layouts of the
  // same value on several operands, the first operand wins.
  for (unsigned i : llvm::reverse(llvm::seq(op->getNumOperands())))
    recordDemand(op->getOperand(i), operandLayouts[i]);

  // Record the result layouts in the IR; an anchor op carries its own layout.
  // A result takes its consumers' layout unless the transfer function replaced
  // it with the layout the op itself requires (e.g. a reduction's slice
  // layout). Vector results are recorded, and so is the scalar result of a
  // reduction, whose layout its transfer function sets (a consumer may demand
  // the same one, as the accumulator of another reduction):
  // ResolveLayoutConflicts derives the reduction's operand layout from it and
  // anchors the scalar with a convert_layout.
  if (isa<xegpu::AnchorLayoutInterface>(op))
    return;
  for (auto [result, layout] :
       llvm::zip_equal(op->getResults(), resultLayouts)) {
    if (!layout)
      continue;
    if (isa<xegpu::TensorDescType>(result.getType()))
      setTensorDescLayout(result, layout);
    else if (isa<VectorType>(result.getType()) ||
             isa<vector::MultiDimReductionOp, vector::ReductionOp>(op))
      xegpu::setTemporaryLayout(result, layout);
  }
}

void LayoutPropagation::visitTerminator(
    RegionBranchTerminatorOpInterface terminator) {
  auto regionOp = cast<RegionBranchOpInterface>(terminator->getParentOp());
  RegionBranchSuccessorMapping mapping;
  regionOp.getSuccessorOperandInputMapping(mapping,
                                           RegionBranchPoint(terminator));
  for (const auto &[successorOperand, successorInputs] : mapping) {
    // Propagate backward into the region: the terminator operand is demanded
    // the layout of the value it feeds, so that its producer inside the region
    // is laid out for it. (The opposite direction, giving the region result the
    // layout of the operand feeding it, is done by visitRegionOp.) An operand
    // may feed both a region argument (the next iteration or the next region)
    // and a region result (on exit); the region argument wins when it has a
    // layout, e.g. scf.while's "after" region yield feeding the "before"
    // region, which is walked first. Otherwise the region result's layout,
    // decided by its consumers after the region op, is used, e.g. for scf.for's
    // yield, whose iter_arg is reached only through the back edge, or for
    // scf.condition, whose "after" region has not been walked yet.
    xegpu::DistributeLayoutAttr layout;
    for (Value input : successorInputs)
      if (isa<BlockArgument>(input) && (layout = getLayout(input)))
        break;
    for (Value input : successorInputs)
      if (isa<OpResult>(input) && !layout)
        layout = getLayout(input);
    recordDemand(successorOperand->get(), layout);

    // Restrict the input IR: a region argument tied to no init operand
    // (scf.while's "after" arguments) must be fed by a pass-through of a region
    // argument, as nothing else identifies the layout the region carries: a
    // region argument reads its layout from its tied init operand.
    auto loop = dyn_cast<LoopLikeOpInterface>(regionOp.getOperation());
    for (auto arg : llvm::make_isa_range<BlockArgument>(successorInputs)) {
      if (isa<VectorType>(arg.getType()) &&
          !(loop && loop.getTiedLoopInit(arg)) &&
          !isa<BlockArgument>(successorOperand->get())) {
        markFailure(terminator,
                    "unsupported region structure: the successor argument it "
                    "feeds is not tied to an init operand, so its value must "
                    "be passed through from predecessor region argument.");
      }
    }
  }
}

/// Which values at a region boundary are filled from a terminator's operands.
enum class FillTarget { RegionResults, RegionResultsAndLoopInits };

/// Gives each region result fed by `terminator`, and with
/// FillTarget::RegionResultsAndLoopInits each tied loop init, the layout of the
/// terminator operand feeding it, unless it already has one. Vector operands
/// for which `getLayoutOfValue` returns null are skipped; returns false if
/// there is any.
static bool
fillRegionBoundaryFromTerminator(RegionBranchTerminatorOpInterface terminator,
                                 xegpu::GetLayoutFnTy getLayoutOfValue,
                                 FillTarget target) {
  auto regionOp = dyn_cast<RegionBranchOpInterface>(terminator->getParentOp());
  if (!regionOp)
    return true;
  auto loop = dyn_cast<LoopLikeOpInterface>(regionOp.getOperation());
  RegionBranchSuccessorMapping mapping;
  regionOp.getSuccessorOperandInputMapping(mapping,
                                           RegionBranchPoint(terminator));
  bool allLaidOut = true;
  for (const auto &[successorOperand, successorInputs] : mapping) {
    Value operand = successorOperand->get();
    if (!isa<VectorType>(operand.getType()))
      continue;
    xegpu::DistributeLayoutAttr layout = getLayoutOfValue(operand);
    if (!layout) {
      allLaidOut = false;
      continue;
    }
    for (Value input : successorInputs) {
      // A region result: %r gets the layout of %v in
      //   %r = scf.if %c -> (vector<...>) { scf.yield %v } else { ... }
      // An operand feeds at most one region result; its other successor
      // inputs are region arguments.
      if (auto result = dyn_cast<OpResult>(input)) {
        if (!xegpu::getDistributeLayoutAttr(result))
          xegpu::setTemporaryLayout(result, layout);
        continue;
      }
      if (target == FillTarget::RegionResults)
        continue;
      // A loop init, around the back edge: %init gets the layout of %n in
      //   scf.for ... iter_args(%acc = %init) {
      //     %n = arith.addf %acc, %x
      //     scf.yield %n
      //   }
      // so %acc, which reads its layout from %init, gets it too. A splat
      // constant %init without a layout takes it as well.
      OpOperand *init =
          loop ? loop.getTiedLoopInit(cast<BlockArgument>(input)) : nullptr;
      if (!init || xegpu::getTemporaryLayout(*init))
        continue;
      xegpu::setTemporaryLayout(*init, layout);
      auto initResult = dyn_cast<OpResult>(init->get());
      if (initResult &&
          !isa<xegpu::AnchorLayoutInterface>(initResult.getOwner()) &&
          !xegpu::getDistributeLayoutAttr(initResult))
        xegpu::setTemporaryLayout(initResult, layout);
    }
  }
  return allLaidOut;
}

void LayoutPropagation::visitRegionOp(RegionBranchOpInterface regionOp) {
  // The init operands take the layout of the region argument they feed (and
  // tensor_desc region arguments get it on their type). An init whose region
  // argument has no layout yet (it is constrained only through the loop back
  // edge) is filled by the forward pass from the yield operand instead
  // (FillTarget::RegionResultsAndLoopInits).
  (void)xegpu::propagateRegionArgsToInits(
      regionOp, [&](Value val) { return getLayout(val); });
  for (OpOperand &operand : llvm::reverse(regionOp->getOpOperands()))
    recordDemand(operand.get(), xegpu::getTemporaryLayout(operand));

  // Record the region result layouts (the opposite direction of the demand
  // visitTerminator propagates): a result takes the layout of the terminator
  // operand feeding it. Several regions may each have a terminator feeding the
  // results (scf.if's then/else, scf.index_switch's cases); a multi-block
  // region (e.g. scf.execute_region) may have several exits. scf.for and
  // scf.while regions have a single block. All these terminators' operands
  // have the result's layout: visitTerminator demands it of each of them, and
  // if a producer requires its own layout instead (an anchor op, or e.g. a
  // reduction), ResolveLayoutConflicts converts that operand to the result's
  // layout. So the first terminator whose operand has a layout sets it. Only
  // an operand defined inside the region op has a decided layout yet; a result
  // left without a layout (e.g. that of a cross-iteration accumulation) is
  // filled by the forward pass.
  auto getDecidedLayout = [&](Value val) -> xegpu::DistributeLayoutAttr {
    if (!regionOp->isAncestor(val.getParentBlock()->getParentOp()))
      return nullptr;
    return xegpu::getDistributeLayoutAttr(val);
  };
  for (Region &region : regionOp->getRegions())
    for (Block &block : region)
      if (auto terminator =
              dyn_cast_if_present<RegionBranchTerminatorOpInterface>(
                  block.empty() ? nullptr : &block.back()))
        (void)fillRegionBoundaryFromTerminator(terminator, getDecidedLayout,
                                               FillTarget::RegionResults);
}

void LayoutPropagation::print(llvm::raw_ostream &os) {
  auto printLayout = [&](Value val) {
    if (xegpu::DistributeLayoutAttr layout = getLayout(val))
      os << layout;
    else
      os << "Not assigned.";
    os << "\n";
  };
  auto printFunctionResult = [&](FunctionOpInterface funcOp) {
    os << "function: " << funcOp.getName() << ":\n";
    for (BlockArgument arg : funcOp.getArguments()) {
      os << "argument: " << arg << "\n";
      os << "layout  : ";
      printLayout(arg);
    }
    funcOp.walk([&](Operation *op) {
      if (op->getResults().empty())
        return;
      os << "op    : ";
      // For control-flow ops, print the op name only.
      if (isa<BranchOpInterface, RegionBranchOpInterface>(op))
        os << op->getName();
      else
        op->print(os);
      os << "\n";
      for (auto [i, r] : llvm::enumerate(op->getResults())) {
        os << "layout for result #" << i << ": ";
        printLayout(r);
      }
    });
  };

  auto modOp = dyn_cast<ModuleOp>(scopeRoot);
  if (!modOp)
    return;
  for (auto funcOp : modOp.getOps<FunctionOpInterface>())
    printFunctionResult(funcOp);
  for (auto gpuModOp : modOp.getOps<gpu::GPUModuleOp>())
    for (auto gpuFuncOp : gpuModOp.getOps<FunctionOpInterface>())
      printFunctionResult(gpuFuncOp);
}

namespace {

//===----------------------------------------------------------------------===//
// ResolveLayoutConflicts
//===----------------------------------------------------------------------===//

/// Helper to get the defining CreateNdDescOp of a tensor descriptor value.
/// This function tries to find the defining CreateNdDescOp recursively
/// accross control-flow boundaries.
static xegpu::CreateNdDescOp getDefiningCreateNdDescOp(Value tdescValue) {
  // Try to get the defining CreateNdDescOp of the tensor descriptor.
  auto definingOp = tdescValue.getDefiningOp<xegpu::CreateNdDescOp>();
  if (definingOp)
    return definingOp;
  // If tdescValue is an argument, try to get the tied init value from the
  // parent loop-like op.
  if (auto arg = dyn_cast<BlockArgument>(tdescValue)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      if (tiedInit)
        return getDefiningCreateNdDescOp(tiedInit->get());
    }
  }
  // If not found, return null.
  return nullptr;
}

struct ResolveLayoutConflicts {
  ResolveLayoutConflicts(Operation *parentOp)
      : parentOp(parentOp), builder(parentOp->getContext()) {}
  LogicalResult run();

private:
  Operation *parentOp;
  OpBuilder builder;
  LogicalResult resolveTensorDescConsumer(OpOperand &operand);
  LogicalResult resolveVectorConsumer(OpOperand &operand);
  LogicalResult assignResultLayout(OpResult &result);
};

} // namespace

LogicalResult ResolveLayoutConflicts::run() {
  // Scan all operations in the parent op and resolve layout conflicts at
  // tensor descriptor and vector use points.
  auto r = parentOp->walk([&](Operation *op) -> WalkResult {
    for (OpResult result : op->getResults()) {
      // if the operation inputs vector and output scalar, like multi-reduction
      // we need to check if the result has layout and add a convert_layout to
      // serve as anchor op for the reduction op's layout.
      if (result.getType().isIntOrFloat() &&
          (isa<vector::MultiDimReductionOp>(op) ||
           isa<vector::ReductionOp>(op))) {
        auto res = assignResultLayout(result);
        if (failed(res)) {
          DBGS() << "Failed to assign layout for scalar consumer of reduction "
                 << *op << "\n";
          return WalkResult::interrupt();
        }
      }
      // If the op is a region branch op with a vector result that has no uses,
      // we need to add a convert_layout to serve as an anchor op for the
      // result's layout.
      if (isa<VectorType>(result.getType()) && result.use_empty() &&
          isa<RegionBranchOpInterface>(op)) {
        auto res = assignResultLayout(result);
        if (failed(res)) {
          DBGS() << "Failed to assign layout for vector consumer of region op "
                 << *op << "\n";
          return WalkResult::interrupt();
        }
      }
    }
    for (OpOperand &operand : op->getOpOperands()) {
      // Handle conflicts in tensor descriptor operands.
      Type operandType = operand.get().getType();
      if (isa<xegpu::AnchorLayoutInterface>(op) &&
          isa<xegpu::TensorDescType>(operandType)) {
        auto res = resolveTensorDescConsumer(operand);
        if (failed(res)) {
          DBGS() << "Failed to resolve tensor descriptor consumer: " << *op
                 << "\n";
          return WalkResult::interrupt();
        }
      }
      // Handle conflicts in vector operands.
      if (isa<VectorType>(operandType)) {
        auto res = resolveVectorConsumer(operand);
        if (failed(res)) {
          DBGS() << "Failed to resolve vector consumer: " << *op << "\n";
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });

  if (r.wasInterrupted())
    return failure();

  LLVM_DEBUG({
    DBGS() << "IR after resolving layout conflicts:\n";
    parentOp->dump();
  });

  return success();
}

LogicalResult ResolveLayoutConflicts::assignResultLayout(OpResult &result) {
  Operation *producerOp = result.getDefiningOp();
  auto producerLayout = xegpu::getDistributeLayoutAttr(result);
  // Insert a convert_layout op to assign the layout.
  builder.setInsertionPointAfterValue(result);
  auto convertOp = xegpu::ConvertLayoutOp::create(
      builder, producerOp->getLoc(), result.getType(), result, producerLayout,
      producerLayout);
  result.replaceAllUsesExcept(convertOp.getResult(), convertOp);
  return success();
}

LogicalResult
ResolveLayoutConflicts::resolveVectorConsumer(OpOperand &operand) {
  Value vectorValue = operand.get();
  Operation *consumerOp = operand.getOwner();
  // Get the current layout of the vector value.
  auto producerLayout = xegpu::getDistributeLayoutAttr(vectorValue);
  if (!producerLayout) {
    if (auto vectorTy = dyn_cast<VectorType>(vectorValue.getType());
        vectorTy && vectorTy.getRank() > 1)
      consumerOp->emitWarning("Expected layout for non-1D vectors.");
    return success(); // uniform non-tensor-data vector does not require
                      // layout
  }
  // getConsumerLayoutAt also covers region-carried operands (loop init and
  // yield operands), so a layout conflict there is reconciled below rather than
  // silently trusted to region forwarding.
  auto consumerLayout = xegpu::getConsumerLayoutAt(operand);
  if (!consumerLayout) {
    if (isa<func::ReturnOp>(consumerOp) || isa<gpu::ReturnOp>(consumerOp))
      return success();
    return consumerOp->emitError(
        "No consumer layout found for vector operand.");
  }

  // If layouts are same, no conflict exists, return success.
  if (consumerLayout.isEqualTo(producerLayout))
    return success();

  // Consumer is a convert_layout: retarget its input_layout to the producer
  // instead of chaining a second convert. Always safe (single source
  // operand).
  if (auto consumerConvert = dyn_cast<xegpu::ConvertLayoutOp>(consumerOp)) {
    consumerConvert.setInputLayoutAttr(producerLayout);
    return success();
  }

  // Producer is a convert_layout feeding only this use: retarget its
  // target_layout to the consumer instead of appending another convert.
  if (auto producerConvert =
          vectorValue.getDefiningOp<xegpu::ConvertLayoutOp>();
      producerConvert && vectorValue.hasOneUse()) {
    // Pin the effective input before retargeting target, else an omitted
    // input_layout would follow target and make the conversion a no-op.
    producerConvert.setInputLayoutAttr(
        producerConvert.getEffectiveInputLayout());
    producerConvert.setTargetLayoutAttr(consumerLayout);
    return success();
  }

  // If the producer is trivially rematerializable (e.g. `vector.step`, splat
  // `arith.constant`), clone it and stamp the consumer's expected layout on
  // the clone instead of inserting a `xegpu.convert_layout`. The convert
  // would otherwise lower to a cross-subgroup data movement through SLM at
  // WG-to-SG distribution time, which is more expensive than
  // recomputing a pure value generator.
  if (auto *producerOp = vectorValue.getDefiningOp();
      producerOp && producerOp->getNumResults() == 1 &&
      isa<OpResult>(vectorValue) &&
      xegpu::isTriviallyRematerializable(producerOp)) {
    builder.setInsertionPointAfter(producerOp);
    Operation *clone = builder.clone(*producerOp);
    OpResult cloneResult = clone->getResult(0);
    // Drop the inherited producer layout so the new layout takes effect
    xegpu::removeLayoutAttr(cloneResult);
    xegpu::setDistributeLayoutAttr(cloneResult, consumerLayout);
    operand.set(cloneResult);
    return success();
  }

  // Insert a convert_layout op to resolve the conflict.
  builder.setInsertionPointAfterValue(vectorValue);
  auto convertOp = xegpu::ConvertLayoutOp::create(
      builder, consumerOp->getLoc(), vectorValue.getType(), vectorValue,
      producerLayout, consumerLayout);

  // Update the operand to use the converted value.
  operand.set(convertOp.getResult());
  return success();
}

LogicalResult
ResolveLayoutConflicts::resolveTensorDescConsumer(OpOperand &operand) {
  Operation *consumerOp = operand.getOwner();
  Value tdescValue = operand.get();
  auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(consumerOp);
  auto currTDescType = dyn_cast<xegpu::TensorDescType>(tdescValue.getType());
  assert(anchorOp && currTDescType &&
         "Expected anchor layout op and tensor descriptor consumer.");
  Attribute currLayout = currTDescType.getLayout();
  Attribute expectedLayout = anchorOp.getAnchorLayout();
  // A conflict exists in tensor descriptor operand if tensor descriptor's
  // layout is different from the anchor layout expected by the consumer.
  if (expectedLayout && currLayout && expectedLayout != currLayout) {
    // Try to get the defining CreateNdDescOp of the tensor descriptor.
    auto conflictingCreateNdOp = getDefiningCreateNdDescOp(tdescValue);
    if (!conflictingCreateNdOp) {
      DBGS() << "Unable to find defining CreateNdDescOp for tensor descriptor: "
             << tdescValue << "\n";
      return failure();
    }
    // Duplicate the CreateNdDescOp with the expected layout.
    builder.setInsertionPointAfter(conflictingCreateNdOp);
    auto newTensorDescType = xegpu::TensorDescType::get(
        conflictingCreateNdOp.getContext(), currTDescType.getShape(),
        currTDescType.getElementType(), currTDescType.getEncoding(),
        expectedLayout);
    auto newOp = xegpu::CreateNdDescOp::create(
        builder, consumerOp->getLoc(), TypeRange{newTensorDescType},
        conflictingCreateNdOp->getOperands(),
        conflictingCreateNdOp.getProperties(),
        conflictingCreateNdOp->getDiscardableAttrDictionary().getValue());
    // Replace the tensor descriptor operand in the consumer op with the new
    // tensor descriptor.
    consumerOp->replaceUsesOfWith(tdescValue, newOp.getResult());
  }
  return success();
}

using GetLayoutFnTy = function_ref<xegpu::DistributeLayoutAttr(Value)>;

/// Update an operation with the layout of its results. For a vector result a
/// temporary layout attribute is added to the op; for a tensor descriptor
/// result the layout is written into its type.
///
/// If the global propagation left a result without a layout, forward-fill it
/// locally from the operand layouts.
static LogicalResult updateOpWithForwardFill(mlir::OpBuilder &builder,
                                             mlir::Operation *op,
                                             GetLayoutFnTy getLayoutOfValue) {
  // Iterate over all the results.
  for (OpResult result : op->getResults()) {
    Type resultType = result.getType();
    // Layouts are needed only for vector and tensor descriptor types.
    if (!isa<VectorType, xegpu::TensorDescType>(resultType))
      continue;
    // A result without a layout is inferred from the layouts of the operands.
    xegpu::DistributeLayoutAttr layout = getLayoutOfValue(result);
    if (!layout) {
      // Gather operand layouts, indexed by operand number.
      SmallVector<xegpu::DistributeLayoutAttr> srcLayouts;
      srcLayouts.reserve(op->getNumOperands());
      bool anyAssigned = false;
      for (Value operand : op->getOperands()) {
        auto srclayout = xegpu::getDistributeLayoutAttr(operand);
        srcLayouts.push_back(srclayout);
        anyAssigned |= (srclayout != nullptr);
      }
      if (anyAssigned) {
        layout =
            xegpu::inferResultLayoutFromSourceForNonAnchorOp(op, srcLayouts);
      }
    }
    // If the result is a tensor descriptor type, update the tensor desc type
    // with layout.
    if (auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(resultType)) {
      auto typeWithLayout = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
      result.setType(typeWithLayout);
      continue;
    }
    // If the result is a vector type, add a temporary layout attribute to the
    // op.
    xegpu::setDistributeLayoutAttr(result, layout);
  }
  return success();
}

/// Optimize elementwise operations by sinking costly layout conversion.
///
///   %m  = vector.create_mask ...                        {L1}
///   %d  = arith.mulf ...                                {L1}
///   %cm = xegpu.convert_layout %m : L1 -> L2
///   %cd = xegpu.convert_layout %d : L1 -> L2
///   %s  = arith.select %cm, %cd, %splat                 {L2}
///
/// becomes
///
///   %s' = arith.select %m, %d, %splat                   {L1}
///   %s  = xegpu.convert_layout %s' : L1 -> L2
///
/// We only sink if a coarser layout applies to an elementwise op
///
void xegpu::sinkElementwiseConversions(OpBuilder &builder,
                                       Operation *parentOp) {
  llvm::SmallSetVector<xegpu::ConvertLayoutOp, 8> deadConverts;
  parentOp->walk([&](Operation *op) {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return;
    OpResult result = op->getResult(0);
    if (!isa<VectorType>(result.getType()))
      return;
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getDistributeLayoutAttr(result);
    if (!resultLayout)
      return;

    // Ensure that all feeding conversions have the same layout.
    SmallVector<std::pair<OpOperand *, xegpu::ConvertLayoutOp>> conversions;
    xegpu::DistributeLayoutAttr uniformConvSrcLayout;
    for (OpOperand &operand : op->getOpOperands()) {
      Value operandValue = operand.get();
      if (!isa<VectorType>(operandValue.getType()))
        continue;
      auto operandConversionOp =
          operandValue.getDefiningOp<xegpu::ConvertLayoutOp>();
      // Those that are not a conversion should be easily materializable with
      // another layout.
      if (!operandConversionOp) {
        Operation *definingOp = operandValue.getDefiningOp();
        if (!definingOp || !xegpu::isTriviallyRematerializable(definingOp))
          return;
        continue;
      }
      xegpu::DistributeLayoutAttr input =
          operandConversionOp.getEffectiveInputLayout();
      if (!input)
        return;
      if (!uniformConvSrcLayout)
        uniformConvSrcLayout = input;
      else if (!uniformConvSrcLayout.isEqualTo(input))
        return;
      conversions.emplace_back(&operand, operandConversionOp);
    }

    if (!uniformConvSrcLayout)
      return;

    // Sink only when it leaves the op coarser.
    // Conversion's source layout must be larger than the elemwise result
    // layout.
    SmallVector<int64_t> resultLayoutInstData =
        resultLayout.getEffectiveInstDataAsInt();
    SmallVector<int64_t> sourceInstData =
        uniformConvSrcLayout.getEffectiveInstDataAsInt();
    if (resultLayoutInstData.empty() || sourceInstData.empty() ||
        computeProduct(sourceInstData) <= computeProduct(resultLayoutInstData))
      return;

    // Rewire the conversions, then rematerialize the remaining operands in the
    // source layout.
    llvm::SmallDenseSet<unsigned> rewiredOpIdxs;
    for (auto [operand, convert] : conversions) {
      operand->set(convert.getSource());
      rewiredOpIdxs.insert(operand->getOperandNumber());
    }
    for (OpOperand &operand : op->getOpOperands()) {
      Value operandValue = operand.get();
      if (!isa<VectorType>(operandValue.getType()))
        continue;
      if (rewiredOpIdxs.contains(operand.getOperandNumber()))
        continue;
      Operation *definingOp = operandValue.getDefiningOp();
      assert(definingOp && xegpu::isTriviallyRematerializable(definingOp) &&
             "operand should have been rejected above");
      // Rematerialize with uniform source layout.
      builder.setInsertionPointAfter(definingOp);
      Operation *clone = builder.clone(*definingOp);
      OpResult cloneResult =
          clone->getResult(cast<OpResult>(operandValue).getResultNumber());
      xegpu::removeLayoutAttr(cloneResult);
      xegpu::setDistributeLayoutAttr(cloneResult, uniformConvSrcLayout);
      operand.set(cloneResult);
    }

    // Run the op in the source layout and bridge its result back, so that
    // `getConsumerLayoutAt` now reports the source layout for every operand.
    builder.setInsertionPointAfterValue(result);
    auto newConvOp = xegpu::ConvertLayoutOp::create(
        builder, op->getLoc(), result.getType(), result, uniformConvSrcLayout,
        resultLayout);
    result.replaceAllUsesExcept(newConvOp.getResult(), newConvOp);
    xegpu::removeLayoutAttr(result);
    xegpu::setDistributeLayoutAttr(result, uniformConvSrcLayout);

    for (auto [operand, convert] : conversions)
      if (convert.getResult().use_empty())
        deadConverts.insert(convert);
  });

  // Memory effects block dce
  for (xegpu::ConvertLayoutOp convert : deadConverts)
    convert.erase();
}

namespace {
struct XeGPUPropagateLayoutPass final
    : public xegpu::impl::XeGPUPropagateLayoutBase<XeGPUPropagateLayoutPass> {
  XeGPUPropagateLayoutPass() = default;
  XeGPUPropagateLayoutPass(const XeGPUPropagateLayoutPass &other) = default;
  XeGPUPropagateLayoutPass(xegpu::XeGPUPropagateLayoutOptions options)
      : XeGPUPropagateLayoutBase(std::move(options)) {}
  void runOnOperation() override;
};

} // namespace

/// Forward pass, in program order, filling the layouts the backward scan left
/// unassigned: values with no consumer demanding a layout, chiefly the
/// loop-carried value of a cross-iteration accumulation, which is constrained
/// only through the loop back edge. A result is inferred from the layouts of
/// its operands; a region boundary takes the layout of the terminator operand
/// feeding it.
static LogicalResult
forwardFillUnassignedLayouts(OpBuilder &builder, Operation *target,
                             xegpu::GetLayoutFnTy getLayoutOfValue) {
  WalkResult walkResult = target->walk([&](Operation *op) -> WalkResult {
    LogicalResult r = success();
    if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op))
      r = success(fillRegionBoundaryFromTerminator(
          terminator, getLayoutOfValue, FillTarget::RegionResultsAndLoopInits));
    else if (op != target &&
             !isa<RegionBranchOpInterface, FunctionOpInterface>(op))
      r = updateOpWithForwardFill(builder, op, getLayoutOfValue);
    if (failed(r)) {
      op->emitError("Failed to update operation with the layout.");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // Report the values left without a layout once the pass is complete: a value
  // may get its layout after its definition is visited (e.g. a loop init, at
  // the yield around the back edge).
  target->walk([](Operation *op) {
    for (OpResult result : op->getResults())
      if (isa<VectorType, xegpu::TensorDescType>(result.getType()) &&
          !result.use_empty() && !xegpu::getDistributeLayoutAttr(result))
        op->emitWarning("op has users but no layout assigned for its result");
  });
  return success();
}

LogicalResult xegpu::propagateLayouts(OpBuilder &builder, Operation *target,
                                      LayoutKind layoutKind,
                                      unsigned /*indexBitWidth*/,
                                      bool printOnly) {
  LayoutPropagation propagation(target, layoutKind);
  if (failed(propagation.run()))
    return failure();
  // Print the analysis result and exit. (for debugging purposes)
  if (printOnly) {
    propagation.print(llvm::outs());
    return success();
  }
  // A value's layout is read from the IR, falling back to the one the backward
  // scan decided for it (e.g. a vector function argument).
  return forwardFillUnassignedLayouts(
      builder, target, [&](Value val) -> xegpu::DistributeLayoutAttr {
        if (xegpu::DistributeLayoutAttr layout =
                xegpu::getDistributeLayoutAttr(val))
          return layout;
        return propagation.getLayout(val);
      });
}

LogicalResult xegpu::resolveLayoutConflicts(Operation *target) {
  ResolveLayoutConflicts resolver(target);
  return resolver.run();
}

void XeGPUPropagateLayoutPass::runOnOperation() {

  xegpu::removeTemporaryLayoutAttrs(getOperation());

  xegpu::LayoutKind layoutKind;
  if (this->layoutKind == "lane") {
    layoutKind = xegpu::LayoutKind::Lane;
  } else if (this->layoutKind == "inst") {
    layoutKind = xegpu::LayoutKind::InstData;
  } else if (this->layoutKind == "subgroup") {
    layoutKind = xegpu::LayoutKind::Subgroup;
  } else {
    getOperation()->emitError("Unsupported layout kind option: " +
                              this->layoutKind);
    signalPassFailure();
    return;
  }
  OpBuilder builder(&getContext());
  if (failed(xegpu::propagateLayouts(builder, getOperation(), layoutKind,
                                     this->indexBitWidth, this->printOnly))) {
    signalPassFailure();
    return;
  }
  // Resolve layout conflicts if any.
  if (failed(xegpu::resolveLayoutConflicts(getOperation()))) {
    signalPassFailure();
    return;
  }
  if (layoutKind == xegpu::LayoutKind::InstData)
    xegpu::sinkElementwiseConversions(builder, getOperation());
}
