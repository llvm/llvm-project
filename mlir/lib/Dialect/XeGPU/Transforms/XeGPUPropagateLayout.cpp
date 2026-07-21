//===- XeGPUPropagateLayout.cpp - XeGPU Layout Propagation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUPROPAGATELAYOUT
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-propagate-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::dataflow;

namespace {

//===----------------------------------------------------------------------===//
// LayoutInfo
//===----------------------------------------------------------------------===//

/// Helper class for tracking the analysis state of an mlir value. For layout
/// propagation, the analysis state is simply the distribution layout of
/// each value. The distribution layout information is encapsulated using
/// xegpu::DistributeLayoutAttr class which can hold information about any type
/// of distribution layout that XeGPU dialect supports. Purpose of this analysis
/// to propagate some unique distribution layout for each value in the program
/// starting from a set of anchor operations (like DPAS, StoreNd, etc.). Note
/// that analysis will reach a fixed point when all values are reached some
/// layout and, analysis does not try to modify any already assigned layouts.
///
/// Given this, LayoutInfo  satisifies the following properties:
///  1) A LayoutInfo value can be in one of two states - `assigned` or `not
///  assigned`.
///  2) Two LayoutInfo values are equal if they are both not assigned, or both
///  assigned with the same layout.
///  3) The meet operator works as follows:
///     - If only one side is assigned, return that side.
///     - If both sides are assigned, prefer the layout demanded by the user
///       that is nearer to the producer in program order (smaller
///       `programOrder`); on a tie keep lhs.
///
/// The `programOrder` field records the program-order index of the consumer op
/// that demanded the layout (stamped via
/// `LayoutInfoPropagation::makeLayoutInfo` from
/// `LayoutInfoPropagation::currentProgramOrder`). During this backward analysis
/// a value can be demanded by several users; keeping the nearest one tends to
/// preserve a consumer's layout as far up the def chain as possible, minimizing
/// layout conversions. This is a hint, not an optimum. `programOrder` is never
/// propagated up the chain - each visited op stamps its own index - so it is
/// excluded from `operator==`.

struct LayoutInfo {
private:
  xegpu::DistributeLayoutAttr storage = nullptr;
  // Program-order index of the consumer op that demanded this layout. Smaller
  // means nearer to the producer. Unassigned/unknown demands sort last.
  int64_t programOrder = std::numeric_limits<int64_t>::max();

public:
  LayoutInfo() = default;
  LayoutInfo(const xegpu::DistributeLayoutAttr &layout, int64_t programOrder)
      : storage(layout), programOrder(programOrder) {}

  // Equality by assignment state and, when both assigned, by the layout:
  //  - one assigned, the other not -> not equal;
  //  - both unassigned             -> equal;
  //  - both assigned               -> equal iff the layouts match.
  bool operator==(const LayoutInfo &other) const {
    if (isAssigned() != other.isAssigned())
      return false;
    if (!isAssigned())
      return true;
    return storage.isEqualTo(other.storage);
  }

  static LayoutInfo meet(const LayoutInfo &lhs, const LayoutInfo &rhs);

  static LayoutInfo join(const LayoutInfo &lhs, const LayoutInfo &rhs);

  void print(raw_ostream &os) const;

  bool isAssigned() const { return storage != nullptr; }

  SmallVector<int> getLaneLayout() const;

  SmallVector<int> getLaneData() const;

  SmallVector<int> getInstData() const;

  SmallVector<int> getSgLayout() const;

  SmallVector<int> getSgData() const;

  SmallVector<int> getOrder() const;

  bool isSliceLayout() const {
    if (!isAssigned())
      return false;
    return isa<xegpu::SliceAttr>(storage);
  }

  int64_t getRank() const {
    if (!isAssigned())
      return -1;
    return storage.getRank();
  }

  Attribute get() { return storage; }
  void set(const xegpu::DistributeLayoutAttr &layout) { storage = layout; }
};

void LayoutInfo::print(raw_ostream &os) const {
  if (isAssigned()) {
    os << storage;
  } else {
    os << "Not assigned.";
  }
}

LayoutInfo LayoutInfo::meet(const LayoutInfo &lhs, const LayoutInfo &rhs) {
  if (!lhs.isAssigned())
    return rhs;
  if (!rhs.isAssigned())
    return lhs;
  // Prefer the demand from the user nearer to the producer in program order.
  // Distinct users always have distinct indices, so this decides every
  // real conflict; on a tie (same op, or both unknown) keep lhs.
  if (rhs.programOrder < lhs.programOrder)
    return rhs;
  return lhs;
}

/// Since this is a backward analysis, join method is not used.
LayoutInfo LayoutInfo::join(const LayoutInfo &lhs, const LayoutInfo &rhs) {
  llvm_unreachable("Join should not be triggered by layout propagation.");
}

//===----------------------------------------------------------------------===//
// LayoutInfoLattice
//===----------------------------------------------------------------------===//

/// Lattice holding the LayoutInfo for each value.
struct LayoutInfoLattice : public Lattice<LayoutInfo> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutInfoLattice)
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// LayoutInfoPropagation
//===----------------------------------------------------------------------===//

/// Backward data flow analysis to propagate the lane_layout and lane_data of
/// each value in the program. Currently, the layouts for operands DPAS,
/// StoreNd, and StoreScatter are fixed (known before propagation). Purpose of
/// this analysis is to propagate those known layouts to all their producers and
/// (other) consumers.
class LayoutInfoPropagation
    : public SparseBackwardDataFlowAnalysis<LayoutInfoLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutInfoPropagation)

private:
  xegpu::LayoutKind layoutKind;
  unsigned indexBitWidth;

  // Program-order index of every op, built lazily on first use via a pre-order
  // walk of the top-level module/function (matching printed-IR order). Used to
  // tell which consumer of a value is nearer to its producer.
  DenseMap<Operation *, int64_t> programOrder;
  // Returns the program-order index of `op`, populating `programOrder` from
  // `op`'s top-level ancestor on first call.
  int64_t getProgramOrder(Operation *op);

  int64_t currentProgramOrder = std::numeric_limits<int64_t>::max();
  LayoutInfo makeLayoutInfo(const xegpu::DistributeLayoutAttr &layout) {
    return LayoutInfo(layout, currentProgramOrder);
  }

  void visitDpasOp(xegpu::DpasOp dpas, ArrayRef<LayoutInfoLattice *> operands,
                   ArrayRef<const LayoutInfoLattice *> results);

  void visitDpasMxOp(xegpu::DpasMxOp dpasMx,
                     ArrayRef<LayoutInfoLattice *> operands,
                     ArrayRef<const LayoutInfoLattice *> results);

  void visitStoreNdOp(xegpu::StoreNdOp store,
                      ArrayRef<LayoutInfoLattice *> operands,
                      ArrayRef<const LayoutInfoLattice *> results);

  void visitStoreScatterOp(xegpu::StoreScatterOp storeScatter,
                           ArrayRef<LayoutInfoLattice *> operands,
                           ArrayRef<const LayoutInfoLattice *> results);

  void visitLoadNdOp(xegpu::LoadNdOp load,
                     ArrayRef<LayoutInfoLattice *> operands,
                     ArrayRef<const LayoutInfoLattice *> results);

  void visitLoadGatherOp(xegpu::LoadGatherOp load,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitTransposeOp(vector::TransposeOp transpose,
                        ArrayRef<LayoutInfoLattice *> operands,
                        ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorBitcastOp(vector::BitCastOp bitcast,
                            ArrayRef<LayoutInfoLattice *> operands,
                            ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorInterleaveOp(vector::InterleaveOp interleave,
                               ArrayRef<LayoutInfoLattice *> operands,
                               ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorDeinterleaveOp(vector::DeinterleaveOp deinterleave,
                                 ArrayRef<LayoutInfoLattice *> operands,
                                 ArrayRef<const LayoutInfoLattice *> results);

  void visitPrefetchNdOp(xegpu::PrefetchNdOp prefetch,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorMultiReductionOp(vector::MultiDimReductionOp reduction,
                                   ArrayRef<LayoutInfoLattice *> operands,
                                   ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorReductionOp(vector::ReductionOp reduction,
                              ArrayRef<LayoutInfoLattice *> operands,
                              ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorBroadCastOp(vector::BroadcastOp broadcast,
                              ArrayRef<LayoutInfoLattice *> operands,
                              ArrayRef<const LayoutInfoLattice *> results);
  void visitShapeCastOp(vector::ShapeCastOp shapeCast,
                        ArrayRef<LayoutInfoLattice *> operands,
                        ArrayRef<const LayoutInfoLattice *> results);
  void
  visitInsertStridedSliceOp(vector::InsertStridedSliceOp insertStridedSlice,
                            ArrayRef<LayoutInfoLattice *> operands,
                            ArrayRef<const LayoutInfoLattice *> results);

  void visitLoadMatrixOp(xegpu::LoadMatrixOp load,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitStoreMatrixOp(xegpu::StoreMatrixOp store,
                          ArrayRef<LayoutInfoLattice *> operands,
                          ArrayRef<const LayoutInfoLattice *> results);

  void visitLoadGatherOp(xegpu::LoadMatrixOp load,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitStoreScatterOp(xegpu::StoreMatrixOp store,
                           ArrayRef<LayoutInfoLattice *> operands,
                           ArrayRef<const LayoutInfoLattice *> results);

  void visitConvertLayoutOp(xegpu::ConvertLayoutOp convertLayout,
                            ArrayRef<LayoutInfoLattice *> operands,
                            ArrayRef<const LayoutInfoLattice *> results);

  bool hasParamsOfLayoutKind(xegpu::DistributeLayoutAttr anchorLayout);

public:
  LayoutInfoPropagation(DataFlowSolver &solver,
                        SymbolTableCollection &symbolTable,
                        xegpu::LayoutKind layoutKind, unsigned indexBitWidth)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        layoutKind(layoutKind), indexBitWidth(indexBitWidth) {}
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutInfoLattice *> operands,
                 ArrayRef<const LayoutInfoLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override {};

  void visitCallOperand(OpOperand &operand) override {};

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {};

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results) override {
  };

  void setToExitState(LayoutInfoLattice *lattice) override {
    (void)lattice->meet(LayoutInfo());
  }
};
} // namespace

int64_t LayoutInfoPropagation::getProgramOrder(Operation *op) {
  auto it = programOrder.find(op);
  if (it != programOrder.end())
    return it->second;
  // First time we see this op's tree: number every op under its top-level
  // ancestor in pre-order (i.e. printed-IR order). Nested ops (e.g. inside an
  // scf.for body) get an index between their parent and the parent's next
  // sibling, so a use inside a loop is "nearer" than a use after it.
  Operation *root = op;
  while (root->getParentOp())
    root = root->getParentOp();
  int64_t counter = 0;
  root->walk<WalkOrder::PreOrder>(
      [&](Operation *o) { programOrder[o] = counter++; });
  return programOrder.lookup(op);
}

LogicalResult LayoutInfoPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Stamp demands pushed by this op with its program-order index so `meet` can
  // prefer the nearest consumer.
  currentProgramOrder = getProgramOrder(op);
  TypeSwitch<Operation *>(op)
      .Case(
          [&](xegpu::DpasOp dpasOp) { visitDpasOp(dpasOp, operands, results); })
      .Case([&](xegpu::DpasMxOp dpasMxOp) {
        visitDpasMxOp(dpasMxOp, operands, results);
      })
      .Case([&](xegpu::StoreNdOp storeNdOp) {
        visitStoreNdOp(storeNdOp, operands, results);
      })
      .Case([&](xegpu::StoreScatterOp storeScatterOp) {
        visitStoreScatterOp(storeScatterOp, operands, results);
      })
      .Case([&](xegpu::LoadNdOp loadNdOp) {
        visitLoadNdOp(loadNdOp, operands, results);
      })
      .Case([&](xegpu::LoadGatherOp loadGatherOp) {
        visitLoadGatherOp(loadGatherOp, operands, results);
      })
      .Case([&](xegpu::PrefetchNdOp prefetchNdOp) {
        visitPrefetchNdOp(prefetchNdOp, operands, results);
      })
      .Case([&](vector::TransposeOp transposeOp) {
        visitTransposeOp(transposeOp, operands, results);
      })
      .Case([&](vector::BitCastOp bitcastOp) {
        visitVectorBitcastOp(bitcastOp, operands, results);
      })
      .Case([&](vector::InterleaveOp interleaveOp) {
        visitVectorInterleaveOp(interleaveOp, operands, results);
      })
      .Case([&](vector::DeinterleaveOp deinterleaveOp) {
        visitVectorDeinterleaveOp(deinterleaveOp, operands, results);
      })
      .Case([&](vector::MultiDimReductionOp reductionOp) {
        visitVectorMultiReductionOp(reductionOp, operands, results);
      })
      .Case([&](vector::ReductionOp reductionOp) {
        visitVectorReductionOp(reductionOp, operands, results);
      })
      .Case([&](vector::BroadcastOp broadcastOp) {
        visitVectorBroadCastOp(broadcastOp, operands, results);
      })
      .Case([&](vector::ShapeCastOp shapeCastOp) {
        visitShapeCastOp(shapeCastOp, operands, results);
      })
      .Case([&](vector::InsertStridedSliceOp insertStridedSliceOp) {
        visitInsertStridedSliceOp(insertStridedSliceOp, operands, results);
      })
      .Case([&](xegpu::LoadMatrixOp loadMatrixOp) {
        visitLoadMatrixOp(loadMatrixOp, operands, results);
      })
      .Case([&](xegpu::StoreMatrixOp storeMatrixOp) {
        visitStoreMatrixOp(storeMatrixOp, operands, results);
      })
      .Case([&](xegpu::ConvertLayoutOp convertLayoutOp) {
        visitConvertLayoutOp(convertLayoutOp, operands, results);
      })
      // All other ops.
      .Default([&](Operation *op) {
        for (const LayoutInfoLattice *resultInfo : results) {
          if (!resultInfo->getValue().isAssigned())
            continue;
          for (auto [operandInfo, operand] :
               llvm::zip(operands, op->getOpOperands())) {
            // If the operand type is not a vector or tensor descriptor, skip
            // it.
            if (!isa<xegpu::TensorDescType, VectorType>(
                    operand.get().getType()))
              continue;
            // Propagate the result layout to the operand.
            meet(operandInfo, *resultInfo);
          }
        }
      });

  return success();
}

bool LayoutInfoPropagation::hasParamsOfLayoutKind(
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
getNumSg(Operation *op, const int sgSize,
         xegpu::DistributeLayoutAttr consumerLayout = nullptr) {
  // first look for the number of subgroups required by the consumer layout
  if (consumerLayout) {
    auto sgLayout = consumerLayout.getEffectiveSgLayoutAsInt();
    if (!sgLayout.empty())
      return llvm::product_of(sgLayout);
  }
  // Oblivious to workitem layout, the total count matters.
  auto gpuFunc = op->getParentOfType<gpu::GPUFuncOp>();
  if (!gpuFunc)
    return failure();
  auto knownBlockSize = gpuFunc.getKnownBlockSize();
  if (!knownBlockSize.has_value())
    return failure();
  const int flatBlockSize = llvm::product_of(knownBlockSize.value());
  return flatBlockSize / sgSize;
}

void LayoutInfoPropagation::visitPrefetchNdOp(
    xegpu::PrefetchNdOp prefetch, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  LayoutInfo prefetchLayout;
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(prefetch).value_or(""));
  if (!uArch)
    return;
  xegpu::DistributeLayoutAttr anchorLayout = prefetch.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    prefetchLayout = makeLayoutInfo(anchorLayout);
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
      prefetchLayout = makeLayoutInfo(*completed);
    }
  } else {
    auto tdescTy = prefetch.getTensorDescType();
    auto numSgOrErr = getNumSg(prefetch, uArch->getSubgroupSize());
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      prefetch.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }

    auto layoutAttr = xegpu::setupPrefetchNdAnchorLayout(
        layoutKind, tdescTy, numSgOrErr.value_or(0), uArch);
    if (!layoutAttr) {
      prefetch.emitWarning(
          "Failed to determine required layout for prefetch_nd.");
      return;
    }
    prefetchLayout = makeLayoutInfo(layoutAttr);
    prefetch.setLayoutAttr(layoutAttr);
  }
  // Propagate the layout to the source tensor descriptor.
  propagateIfChanged(operands[0], operands[0]->meet(prefetchLayout));
}

void LayoutInfoPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  Type resultTy = reduction.getDestType();
  // The layout of the result must be present.
  LayoutInfo resLayoutInfo = results[0]->getValue();

  xegpu::DistributeLayoutAttr consumerLayoutAttr;
  if (!resultTy.isIntOrFloat()) {
    if (!resLayoutInfo.isAssigned())
      return;
    consumerLayoutAttr =
        dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());
  }

  VectorType sourceTy = reduction.getSourceVectorType();
  SmallVector<int64_t> reductionDims(reduction.getReductionDims());

  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(reduction).value_or(""));
  if (!uArch)
    return;

  auto numSgOrErr =
      getNumSg(reduction, uArch->getSubgroupSize(), consumerLayoutAttr);
  if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
    reduction.emitWarning(
        "Unable to determine the number of subgroups for the operation.");
    return;
  }

  // The result layout represents the layout requirements of the operation.
  // it is recorded to anchor layout or temporary layout.
  // it must be honored for current op and may conflict with the layout
  // propagated from consumer op, the conflict is resolved in later phase by
  // converting the required result layout to the consumer layout
  auto requiredResLayoutAttr = xegpu::setupMultiReductionResultLayout(
      layoutKind, sourceTy, consumerLayoutAttr, reductionDims,
      numSgOrErr.value_or(0), uArch);

  xegpu::setTemporaryLayout(reduction->getResult(0), requiredResLayoutAttr);

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferMultiReductionSourceLayout(
      requiredResLayoutAttr, reductionDims);

  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
  // Accumulator should have the same layout as the result.
  propagateIfChanged(operands[1],
                     operands[1]->meet(makeLayoutInfo(requiredResLayoutAttr)));
}

void LayoutInfoPropagation::visitVectorReductionOp(
    vector::ReductionOp reduction, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  VectorType sourceTy = reduction.getSourceVectorType();
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(reduction).value_or(""));
  if (!uArch)
    return;

  auto requiredResLayoutAttr =
      xegpu::setupReductionResultLayout(layoutKind, sourceTy, uArch);
  xegpu::setTemporaryLayout(reduction->getResult(0), requiredResLayoutAttr);

  auto srcLayoutAttr = xegpu::inferReductionSourceLayout(requiredResLayoutAttr);
  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
  if (reduction.getAcc())
    propagateIfChanged(
        operands[1], operands[1]->meet(makeLayoutInfo(requiredResLayoutAttr)));
}

void LayoutInfoPropagation::visitVectorBroadCastOp(
    vector::BroadcastOp broadcast, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  // Only consider vector to vector broadcasts for now.
  VectorType resultTy = broadcast.getResultVectorType();
  VectorType sourceTy = dyn_cast<VectorType>(broadcast.getSourceType());
  // skip layout propagation for non-vector source operand.
  if (!sourceTy)
    return;

  auto srcShape = sourceTy.getShape();
  auto resShape = resultTy.getShape();

  auto resultLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  xegpu::DistributeLayoutAttr srcLayoutAttr =
      xegpu::inferBroadcastSourceLayout(resultLayoutAttr, resShape, srcShape);

  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
}

void LayoutInfoPropagation::visitShapeCastOp(
    vector::ShapeCastOp shapeCast, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;
  ArrayRef<int64_t> resShape = shapeCast.getResultVectorType().getShape();
  ArrayRef<int64_t> srcShape = shapeCast.getSourceVectorType().getShape();
  auto resultLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  xegpu::DistributeLayoutAttr srcLayoutAttr =
      xegpu::inferShapeCastSourceLayout(resultLayoutAttr, resShape, srcShape);
  // TODO: turn this into a real pass failure once propagation failures are
  // wired to signalPassFailure().
  if (!srcLayoutAttr) {
    shapeCast.emitWarning("Failed to infer source layout for shape_cast; "
                          "unsupported shape-cast pattern.");
    return;
  }

  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
}

/// Set the layouts for DPAS A, B, and C operands.
void LayoutInfoPropagation::visitDpasOp(
    xegpu::DpasOp dpas, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo dpasALayout;
  LayoutInfo dpasBLayout;
  LayoutInfo dpasCDLayout;

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
    dpasALayout = makeLayoutInfo(anchorLayoutA);
    dpasBLayout = makeLayoutInfo(anchorLayoutB);
    dpasCDLayout = makeLayoutInfo(anchorLayoutCD);
    if (layoutKind == xegpu::LayoutKind::InstData) {
      auto completed = xegpu::completeDpasLaneLayoutFromInstData(
          anchorLayoutA, anchorLayoutB, anchorLayoutCD, aTy, bTy, cdTy, uArch);
      if (!completed) {
        dpas.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      auto [completedA, completedB, completedCD] = *completed;
      dpas.setLayoutAAttr(completedA);
      dpas.setLayoutBAttr(completedB);
      dpas.setLayoutCdAttr(completedCD);
      dpasALayout = makeLayoutInfo(completedA);
      dpasBLayout = makeLayoutInfo(completedB);
      dpasCDLayout = makeLayoutInfo(completedCD);
    }
  } else {

    xegpu::DistributeLayoutAttr consumerLayoutAttr = nullptr;
    xegpu::DistributeLayoutAttr requiredCDLayoutAttr, requiredALayout,
        requiredBLayout;

    LayoutInfo consumerLayout = results[0]->getValue();
    if (!consumerLayout.isAssigned())
      return;
    consumerLayoutAttr =
        dyn_cast<xegpu::DistributeLayoutAttr>(consumerLayout.get());

    auto numSgOrErr =
        getNumSg(dpas, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      dpas.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }

    auto layouts =
        xegpu::setupDpasLayout(layoutKind, aTy, bTy, cdTy, consumerLayoutAttr,
                               numSgOrErr.value_or(0), uArch);
    if (!layouts.has_value()) {
      dpas.emitWarning(
          "Failed to determine required layouts for DPAS operands.");
      return;
    }

    std::tie(requiredALayout, requiredBLayout, requiredCDLayoutAttr) = *layouts;

    dpas.setLayoutAAttr(requiredALayout);
    dpas.setLayoutBAttr(requiredBLayout);
    dpas.setLayoutCdAttr(requiredCDLayoutAttr);
    dpasALayout = makeLayoutInfo(requiredALayout);
    dpasBLayout = makeLayoutInfo(requiredBLayout);
    dpasCDLayout = makeLayoutInfo(requiredCDLayoutAttr);
  }
  propagateIfChanged(operands[0], operands[0]->meet(dpasALayout));
  propagateIfChanged(operands[1], operands[1]->meet(dpasBLayout));
  if (operands.size() > 2)
    propagateIfChanged(operands[2], operands[2]->meet(dpasCDLayout));
}

/// Propagate layout for DpasMxOp operands using the layout attributes.
/// DpasMxOp has operands: a, b, acc (optional), scale_a (optional), scale_b
/// (optional)
void LayoutInfoPropagation::visitDpasMxOp(
    xegpu::DpasMxOp dpasMx, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  // Initialize layout variables
  LayoutInfo dpasMxALayout, dpasMxBLayout, dpasMxCDLayout;
  LayoutInfo dpasMxAScaleLayout, dpasMxBScaleLayout;

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
    dpasMxALayout = makeLayoutInfo(anchorLayoutA);
    dpasMxBLayout = makeLayoutInfo(anchorLayoutB);
    dpasMxCDLayout = makeLayoutInfo(anchorLayoutCD);

    // Get scale layouts if available
    xegpu::DistributeLayoutAttr anchorLayoutAScale =
        dpasMx.getLayoutAScaleAttr();
    xegpu::DistributeLayoutAttr anchorLayoutBScale =
        dpasMx.getLayoutBScaleAttr();
    if (anchorLayoutAScale)
      dpasMxAScaleLayout = makeLayoutInfo(anchorLayoutAScale);
    if (anchorLayoutBScale)
      dpasMxBScaleLayout = makeLayoutInfo(anchorLayoutBScale);

    if (layoutKind == xegpu::LayoutKind::InstData) {
      auto completed = xegpu::completeDpasMxLaneLayoutFromInstData(
          anchorLayoutA, anchorLayoutB, anchorLayoutCD, aTy, bTy, cdTy,
          aScaleTy, bScaleTy, uArch);
      if (!completed) {
        dpasMx.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      auto [completedA, completedB, completedCD, completedAScale,
            completedBScale] = *completed;
      dpasMx.setLayoutAAttr(completedA);
      dpasMx.setLayoutBAttr(completedB);
      dpasMx.setLayoutCdAttr(completedCD);
      dpasMxALayout = makeLayoutInfo(completedA);
      dpasMxBLayout = makeLayoutInfo(completedB);
      dpasMxCDLayout = makeLayoutInfo(completedCD);
      if (completedAScale) {
        dpasMx.setLayoutAScaleAttr(completedAScale);
        dpasMxAScaleLayout = makeLayoutInfo(completedAScale);
      }
      if (completedBScale) {
        dpasMx.setLayoutBScaleAttr(completedBScale);
        dpasMxBScaleLayout = makeLayoutInfo(completedBScale);
      }
    }
  } else {
    xegpu::DistributeLayoutAttr consumerLayoutAttr = nullptr;
    xegpu::DistributeLayoutAttr requiredCDLayoutAttr, requiredALayout,
        requiredBLayout, requiredAScaleLayout, requiredBScaleLayout;

    LayoutInfo consumerLayout = results[0]->getValue();
    if (!consumerLayout.isAssigned())
      return;
    consumerLayoutAttr =
        dyn_cast<xegpu::DistributeLayoutAttr>(consumerLayout.get());

    auto numSgOrErr =
        getNumSg(dpasMx, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      dpasMx.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }

    auto layouts = xegpu::setupDpasMxLayout(
        layoutKind, aTy, bTy, cdTy, aScaleTy, bScaleTy, consumerLayoutAttr,
        numSgOrErr.value_or(0), uArch);
    if (!layouts.has_value()) {
      dpasMx.emitWarning(
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

    dpasMxALayout = makeLayoutInfo(requiredALayout);
    dpasMxBLayout = makeLayoutInfo(requiredBLayout);
    dpasMxCDLayout = makeLayoutInfo(requiredCDLayoutAttr);
    if (requiredAScaleLayout)
      dpasMxAScaleLayout = makeLayoutInfo(requiredAScaleLayout);
    if (requiredBScaleLayout)
      dpasMxBScaleLayout = makeLayoutInfo(requiredBScaleLayout);
  }

  // Propagate layouts to operands. Because acc, scale_a, scale_b are all
  // optional (AttrSizedOperandSegments), the index of each present operand in
  // `operands` depends on which optionals are actually supplied. Use the
  // op's accessors to determine the correct positional index.
  propagateIfChanged(operands[0], operands[0]->meet(dpasMxALayout));
  propagateIfChanged(operands[1], operands[1]->meet(dpasMxBLayout));
  unsigned idx = 2;
  if (dpasMx.getAcc()) {
    propagateIfChanged(operands[idx], operands[idx]->meet(dpasMxCDLayout));
    ++idx;
  }
  if (dpasMx.getScaleA()) {
    if (dpasMxAScaleLayout.isAssigned())
      propagateIfChanged(operands[idx],
                         operands[idx]->meet(dpasMxAScaleLayout));
    ++idx;
  }
  if (dpasMx.getScaleB()) {
    if (dpasMxBScaleLayout.isAssigned())
      propagateIfChanged(operands[idx],
                         operands[idx]->meet(dpasMxBScaleLayout));
    ++idx;
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutInfoPropagation::visitStoreNdOp(
    xegpu::StoreNdOp store, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo storeLayout;
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(store).value_or(""));
  if (!uArch)
    return;
  xegpu::DistributeLayoutAttr anchorLayout = store.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    storeLayout = makeLayoutInfo(anchorLayout);
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
        store.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      store.setLayoutAttr(*completed);
      storeLayout = makeLayoutInfo(*completed);
    }
  } else {
    auto numSgOrErr = getNumSg(store, uArch->getSubgroupSize());
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      store.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }

    auto layoutAttr = xegpu::setupStoreNdAnchorLayout(
        layoutKind, store.getValueType(), numSgOrErr.value_or(0), uArch);
    if (!layoutAttr) {
      store.emitWarning("Failed to determine required layout for store_nd.");
      return;
    }
    storeLayout = makeLayoutInfo(layoutAttr);
    store.setLayoutAttr(layoutAttr);
  }
  // Propagate the layout to the value operand.
  // Both operands should have the same layout
  for (LayoutInfoLattice *operand : operands)
    propagateIfChanged(operand, operand->meet(storeLayout));
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// LoadNdOp.
void LayoutInfoPropagation::visitLoadNdOp(
    xegpu::LoadNdOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo loadLayout;

  const auto *uArch = xegpu::uArch::getUArch(getChipStr(load).value_or(""));
  if (!uArch)
    return;
  LayoutInfo valueLayout = results[0]->getValue();
  if (!valueLayout.isAssigned())
    return;
  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(valueLayout.get());
  xegpu::DistributeLayoutAttr anchorLayout = load.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    loadLayout = makeLayoutInfo(anchorLayout);
    if (layoutKind == xegpu::LayoutKind::InstData &&
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
      loadLayout = makeLayoutInfo(*completed);
    }
  } else {
    auto numSgOrErr =
        getNumSg(load, uArch->getSubgroupSize(), consumerLayoutAttr);
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      load.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }
    auto layoutAttr = xegpu::setupLoadNdAnchorLayout(
        layoutKind, load.getType(), consumerLayoutAttr, numSgOrErr.value_or(0),
        uArch);
    if (!layoutAttr) {
      load.emitWarning("Failed to determine required layout for load_nd.");
      return;
    }
    loadLayout = makeLayoutInfo(layoutAttr);
    load.setLayoutAttr(layoutAttr);
  }
  // Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(loadLayout));
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// ConvertLayoutOp.
void LayoutInfoPropagation::visitConvertLayoutOp(
    xegpu::ConvertLayoutOp convert, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  LayoutInfo resultLayout = results[0]->getValue();

  // TODO: fix if one of the layouts is a slice layout
  auto targetLayoutAttr =
      dyn_cast<xegpu::LayoutAttr>(convert.getTargetLayoutAttr());
  auto inputLayoutAttr =
      dyn_cast<xegpu::LayoutAttr>(convert.getInputLayoutAttr());

  // The result's propagated layout is authoritative for the converted value.
  // Fill the lane_layout / lane_data / order parameters the target_layout is
  // missing from it (sg_layout / sg_data / inst_data are left as-is), so the
  // target stays consistent with what is actually propagated downstream.
  auto resultLayoutAttr = resultLayout.isAssigned()
                              ? dyn_cast<xegpu::LayoutAttr>(resultLayout.get())
                              : nullptr;
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

  xegpu::DistributeLayoutAttr anchorLayout = convert.getInputLayoutAttr();
  LayoutInfo convertLayout = makeLayoutInfo(anchorLayout);
  // Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(convertLayout));
}

/// For vector::TransposeOp, the layout of the result is transposed and
/// propagated to the operand.
void LayoutInfoPropagation::visitTransposeOp(
    vector::TransposeOp transpose, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Need the layout of transpose result to propagate to the operands.
  LayoutInfo resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resultLayout.get());
  auto srcLayoutAttr = xegpu::inferTransposeSourceLayout(
      consumerLayoutAttr, transpose.getPermutation());

  // Propagate the new layout to the vector operand.
  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
}

/// For vector::BitCastOp, the lane_data of the source layout is changed based
/// on the bit width of the source and result types.
void LayoutInfoPropagation::visitVectorBitcastOp(
    vector::BitCastOp bitcast, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Need the layout of bitcast result to propagate to the operands.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  auto srcVecType = bitcast.getSourceVectorType();
  auto resVecType = bitcast.getResultVectorType();

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(bitcast).value_or(""));
  if (!uArch)
    return;
  auto requiredResLayoutAttr = setupBitCastResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  xegpu::setTemporaryLayout(bitcast->getResult(0), requiredResLayoutAttr);

  int inElemTyBitWidth = srcVecType.getElementType().getIntOrFloatBitWidth();
  int outElemTyBitWidth = resVecType.getElementType().getIntOrFloatBitWidth();

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferBitCastSourceLayout(
      requiredResLayoutAttr, outElemTyBitWidth, inElemTyBitWidth);

  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
}

/// For vector::InterleaveOp, the result has double the innermost dimension
/// size compared to each source operand. The layout is propagated from result
/// to sources, adjusting for the 2x size increase.
void LayoutInfoPropagation::visitVectorInterleaveOp(
    vector::InterleaveOp interleave, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Need the layout of interleave result to propagate to the operands.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  auto srcVecType = interleave.getSourceVectorType();
  auto resVecType = interleave.getResultVectorType();

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(interleave).value_or(""));
  if (!uArch)
    return;

  // Setup the result layout to ensure the source layout can be safely derived
  auto requiredResLayoutAttr = setupInterleaveResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  xegpu::setTemporaryLayout(interleave->getResult(0), requiredResLayoutAttr);

  // Derive the source layout from the result layout (halve the innermost dim)
  auto srcLayoutAttr =
      xegpu::inferInterleaveSourceLayout(requiredResLayoutAttr);

  // Both operands (lhs and rhs) get the same source layout
  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(makeLayoutInfo(srcLayoutAttr)));
}

/// For vector::DeinterleaveOp, the source has double the innermost dimension
/// size compared to each result. The layout is propagated from results to
/// source, adjusting for the 2x size decrease in results.
void LayoutInfoPropagation::visitVectorDeinterleaveOp(
    vector::DeinterleaveOp deinterleave, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Need the layout of deinterleave results to propagate to the operand.
  // Use the first result's layout (both results should have the same layout)
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  // Derive the source layout from the result layout (double the innermost
  // dim) No setup function needed - just infer directly
  auto srcLayoutAttr = xegpu::inferDeinterleaveSourceLayout(consumerLayoutAttr);

  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
}

void LayoutInfoPropagation::visitInsertStridedSliceOp(
    vector::InsertStridedSliceOp insertStridedSlice,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  auto srcVecType = insertStridedSlice.getSourceVectorType();
  auto resVecType = insertStridedSlice.getDestVectorType();

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());
  const auto *uArch = xegpu::uArch::getUArch(
      xegpu::getChipStr(insertStridedSlice).value_or(""));
  if (!uArch)
    return;

  auto requiredResLayoutAttr = xegpu::setupInsertStridedSliceResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);
  xegpu::setTemporaryLayout(insertStridedSlice->getResult(0),
                            requiredResLayoutAttr);

  auto srcLayoutAttr = xegpu::inferInsertStridedSliceSourceLayout(
      requiredResLayoutAttr, resVecType.getShape(), srcVecType.getShape());
  propagateIfChanged(operands[0],
                     operands[0]->meet(makeLayoutInfo(srcLayoutAttr)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(makeLayoutInfo(requiredResLayoutAttr)));
}

/// Propagate the layout of the result to the tensor descriptor, mask and
/// offset operands in LoadGatherOp.
void LayoutInfoPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = load.getLayoutAttr();
  const auto *uArch = xegpu::uArch::getUArch(getChipStr(load).value_or(""));
  if (!uArch)
    return;
  VectorType resVecTy = load.getValueType();
  int chunkSize = load.getChunkSize().value_or(1);

  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;
  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
    if (layoutKind == xegpu::LayoutKind::InstData &&
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
  } else {
    if (!resVecTy) {
      load.emitWarning("Not propagating, non-vector payload supplied.");
      return;
    }
    requiredAnchorLayoutAttr = xegpu::setupLoadGatherAnchorLayout(
        layoutKind, resVecTy, chunkSize, consumerLayoutAttr, uArch);
    load.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  assert((chunkSize <= 1) || (layoutKind != xegpu::LayoutKind::Subgroup));
  auto maskLayoutAttr = xegpu::inferMaskOffsetLayoutForScatterIO(
      requiredAnchorLayoutAttr, chunkSize);
  LayoutInfo maskLayoutInfo = makeLayoutInfo(maskLayoutAttr);
  auto loadLayoutInfo = makeLayoutInfo(requiredAnchorLayoutAttr);

  // Propagate the new layout to the tensor descriptor operand.
  if (isa<xegpu::TensorDescType>(load.getSourceType()))
    propagateIfChanged(operands[0], operands[0]->meet(loadLayoutInfo));
  // Propagate the new layout to the offset and mask operands.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayoutInfo));
  propagateIfChanged(operands[2], operands[2]->meet(maskLayoutInfo));
}

/// Set the layout for the value, tensor descriptor, offset and mask operands
/// in the StoreScatterOp.
void LayoutInfoPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = storeScatter.getLayoutAttr();
  const auto *uArch =
      xegpu::uArch::getUArch(getChipStr(storeScatter).value_or(""));
  if (!uArch)
    return;
  VectorType srcVecTy = storeScatter.getValueType();
  int chunkSize = storeScatter.getChunkSize().value_or(1);

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
        storeScatter.emitWarning(
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
    auto numSgOrErr = getNumSg(storeScatter, uArch->getSubgroupSize());
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      storeScatter.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }
    requiredAnchorLayoutAttr = xegpu::setupStoreScatterAnchorLayout(
        layoutKind, srcVecTy, chunkSize, numSgOrErr.value_or(0), uArch);
    if (!requiredAnchorLayoutAttr) {
      storeScatter.emitWarning(
          "Failed to determine required layout for store scatter.");
      return;
    }
    storeScatter.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  LayoutInfo srcLayoutInfo = makeLayoutInfo(requiredAnchorLayoutAttr);
  assert((chunkSize <= 1) || (layoutKind != xegpu::LayoutKind::Subgroup));
  auto maskLayoutAttr = xegpu::inferMaskOffsetLayoutForScatterIO(
      requiredAnchorLayoutAttr, chunkSize);
  LayoutInfo maskLayoutInfo = makeLayoutInfo(maskLayoutAttr);

  // Propagate the payload operand layout
  propagateIfChanged(operands[0], operands[0]->meet(srcLayoutInfo));
  // Propagate the destination (if tdesc) operand layout
  if (isa<xegpu::TensorDescType>(storeScatter.getDestType()))
    propagateIfChanged(operands[1], operands[1]->meet(srcLayoutInfo));
  // Propagate the new layout to the offset and mask operands.
  propagateIfChanged(operands[2], operands[2]->meet(maskLayoutInfo));
  propagateIfChanged(operands[3], operands[3]->meet(maskLayoutInfo));
}

void LayoutInfoPropagation::visitLoadMatrixOp(
    xegpu::LoadMatrixOp loadMatrixOp, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

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
    auto requiredAnchorLayoutAttr = xegpu::setupLoadMatrixAnchorLayout(
        layoutKind, resVecTy, chunkSize, consumerLayoutAttr, uArch);
    loadMatrixOp.setLayoutAttr(requiredAnchorLayoutAttr);
  }
}

void LayoutInfoPropagation::visitStoreMatrixOp(
    xegpu::StoreMatrixOp storeMatrix, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = storeMatrix.getLayoutAttr();
  LayoutInfo layout;
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
        storeMatrix.emitWarning(
            "Failed to identify lane layouts for the specified inst_data.");
        return;
      }
      requiredAnchorLayoutAttr = *completed;
      storeMatrix.setLayoutAttr(requiredAnchorLayoutAttr);
    }
  } else {
    int chunkSize =
        1; // placeHolder for future use when StoreMatrix supports coalescing
    auto numSgOrErr = getNumSg(storeMatrix, uArch->getSubgroupSize());
    if (layoutKind == xegpu::LayoutKind::Subgroup && failed(numSgOrErr)) {
      storeMatrix.emitWarning(
          "Unable to determine the number of subgroups for the operation.");
      return;
    }
    requiredAnchorLayoutAttr = xegpu::setupStoreMatrixAnchorLayout(
        layoutKind, srcVecTy, chunkSize, numSgOrErr.value_or(0), uArch);
    if (!requiredAnchorLayoutAttr) {
      storeMatrix.emitWarning(
          "Failed to determine required layout for store matrix.");
      return;
    }
    storeMatrix.setLayoutAttr(requiredAnchorLayoutAttr);
  }
  layout = makeLayoutInfo(requiredAnchorLayoutAttr);
  propagateIfChanged(operands[0], operands[0]->meet(layout));
}

namespace {
//===----------------------------------------------------------------------===//
// RunLayoutInfoPropagation
//===----------------------------------------------------------------------===//

/// Driver class for running the LayoutInfoPropagation analysis.
class RunLayoutInfoPropagation {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunLayoutInfoPropagation)

  RunLayoutInfoPropagation(Operation *op, xegpu::LayoutKind layoutKind,
                           unsigned indexBitWidth)
      : target(op) {
    SymbolTableCollection symbolTable;
    loadBaselineAnalyses(solver);
    solver.load<LayoutInfoPropagation>(symbolTable, layoutKind, indexBitWidth);
    (void)solver.initializeAndRun(op);
  }

  LayoutInfo getLayoutInfo(Value val);

  void printAnalysisResult(llvm::raw_ostream &os);

private:
  DataFlowSolver solver;
  const Operation *target;
};
} // namespace

LayoutInfo RunLayoutInfoPropagation::getLayoutInfo(Value val) {
  auto *state = solver.lookupState<LayoutInfoLattice>(val);
  if (!state)
    return {};
  return state->getValue();
}

// Print the analysis result for debugging purposes.
void RunLayoutInfoPropagation::printAnalysisResult(llvm::raw_ostream &os) {
  auto printFunctionResult = [&](FunctionOpInterface funcOp) {
    os << "function: " << funcOp.getName() << ":\n";
    // Function arguments
    for (BlockArgument arg : funcOp.getArguments()) {
      LayoutInfo layout = getLayoutInfo(arg);
      os << "argument: " << arg << "\n";
      os << "layout  : ";
      layout.print(os);
      os << "\n";
    }
    // Function ops
    funcOp.walk([&](Operation *op) {
      // Skip ops that do not have results
      if (op->getResults().empty())
        return;
      os << "op    : ";
      // For control-flow ops, print the op name only.
      if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op))
        os << op->getName();
      else
        op->print(os);
      os << "\n";
      // Print the layout for each result.
      for (auto [i, r] : llvm::enumerate(op->getResults())) {
        LayoutInfo layout = getLayoutInfo(r);
        os << "layout for result #" << i << ": ";
        layout.print(os);
        os << "\n";
      }
    });
  };

  SmallVector<FunctionOpInterface> funcOps;
  if (auto modOp = dyn_cast<ModuleOp>(target)) {
    for (auto funcOp : modOp.getOps<FunctionOpInterface>())
      funcOps.push_back(funcOp);

    // Collect all GpuFuncOps in the module.
    for (auto gpuModOp : modOp.getOps<gpu::GPUModuleOp>()) {
      for (auto gpuFuncOp : gpuModOp.getOps<FunctionOpInterface>())
        funcOps.push_back(gpuFuncOp);
    }
  }
  // Print the analysis result for each function.
  for (FunctionOpInterface funcOp : funcOps)
    printFunctionResult(funcOp);
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

  LLVM_DEBUG({
    DBGS() << "IR after resolving layout conflicts:\n";
    parentOp->dump();
  });

  return r.wasInterrupted() ? failure() : success();
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
  // Region branch ops (e.g. scf.for) and their terminators (e.g. scf.yield)
  // forward their operands to successor region inputs / parent op results;
  // their consumer layout is resolved through that forwarding, not at this
  // use point.
  if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(
          consumerOp))
    return success();

  auto consumerLayout = xegpu::getConsumerLayoutAt(operand);
  if (!consumerLayout)
    return consumerOp->emitError(
        "No consumer layout found for vector operand.");

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
    xegpu::CreateNdDescOp newOp = xegpu::CreateNdDescOp::create(
        builder, consumerOp->getLoc(), newTensorDescType,
        conflictingCreateNdOp->getOperands(),
        conflictingCreateNdOp->getAttrs());
    // Replace the tensor descriptor operand in the consumer op with the new
    // tensor descriptor.
    consumerOp->replaceUsesOfWith(tdescValue, newOp.getResult());
  }
  return success();
}

using GetLayoutFnTy = function_ref<xegpu::DistributeLayoutAttr(Value)>;

/// Update an operation with the layout of its results. If the result type is
/// a vector type, a temporary layout attribute is added to the operation. If
/// the result type is a tensor descriptor type, the type is updated with the
/// layout attribute. The users of the result are also updated with the layout
/// attribute.
///
/// If the global propagation left a result without a layout, forward-fill it
/// locally from the operand layouts.
static LogicalResult updateOpWithForwardFill(mlir::OpBuilder &builder,
                                             mlir::Operation *op,
                                             GetLayoutFnTy getLayoutOfValue) {
  // Region ops (like scf.for) are already handled by the
  // updateControlFlowOps.
  if (mlir::isa<mlir::RegionBranchOpInterface>(op))
    return success();

  // Iterate over all the results.
  for (OpResult result : op->getResults()) {
    Type resultType = result.getType();
    // Layouts are needed only for vector and tensor descriptor types.
    if (!isa<VectorType, xegpu::TensorDescType>(resultType))
      continue;
    // If the result has no layout but has users, emit a warning and continue.
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
    if (!layout && result.getNumUses() > 0) {
      op->emitWarning("op has users but no layout assigned for its result");
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

/// Update the function arguments and results with the layouts.
static LogicalResult updateFunctionOpInterface(mlir::OpBuilder &builder,
                                               mlir::FunctionOpInterface funcOp,
                                               GetLayoutFnTy getLayoutOfValue) {
  // Only process functions whose type is a standard MLIR FunctionType.
  // Functions using a different type representation (e.g. llvm.func with
  // LLVMFunctionType) are not targets for XeGPU layout propagation, and
  // calling setType(FunctionType{}) on them would corrupt their type.
  if (!isa<FunctionType>(funcOp.getFunctionType()))
    return success();
  SmallVector<Type> newArgTypes;
  // Update the function arguments.
  for (BlockArgument arg : funcOp.getArguments()) {
    Type argType = arg.getType();
    newArgTypes.push_back(argType);
    if (!isa<VectorType, xegpu::TensorDescType>(argType))
      continue;
    xegpu::DistributeLayoutAttr layout = getLayoutOfValue(arg);
    if (!layout) {
      LLVM_DEBUG(DBGS() << "Expecting layout for function argument: " << arg
                        << " but got none.\n");
      return failure();
    }
    if (auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(argType)) {
      auto newTdescTy = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
      arg.setType(newTdescTy);
      newArgTypes.back() = newTdescTy;
    }
  }
  // Update the function type with the new argument types.
  // NOTE: We assume that function results are not expected to have layouts.
  funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                   funcOp.getResultTypes()));
  return success();
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

LogicalResult xegpu::propagateLayouts(OpBuilder &builder, Operation *target,
                                      LayoutKind layoutKind,
                                      unsigned indexBitWidth, bool printOnly) {
  RunLayoutInfoPropagation analysis(target, layoutKind, indexBitWidth);
  // Print the analysis result and exit. (for debugging purposes)
  if (printOnly) {
    auto &os = llvm::outs();
    analysis.printAnalysisResult(os);
    return success();
  }
  // Helper to convert LayoutInfo to xegpu::LayoutAttr.
  auto getLayoutFromPropagation =
      [&](Value val) -> xegpu::DistributeLayoutAttr {
    LayoutInfo layout = analysis.getLayoutInfo(val);
    if (auto opResult = dyn_cast<OpResult>(val)) {
      Operation *defOp = opResult.getDefiningOp();
      if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(defOp)) {
        auto anchorLayout = anchorOp.getAnchorLayout();
        if (anchorLayout != nullptr)
          return anchorLayout;
      }
      xegpu::DistributeLayoutAttr requiredResLayoutAttr =
          xegpu::getTemporaryLayout(opResult);
      if (requiredResLayoutAttr != nullptr)
        return requiredResLayoutAttr;
    }
    if (!layout.isAssigned())
      return {};
    xegpu::DistributeLayoutAttr layoutAttr =
        cast<xegpu::DistributeLayoutAttr>(layout.get());
    if (layout.isSliceLayout())
      return cast<xegpu::SliceAttr>(layoutAttr);

    return cast<xegpu::LayoutAttr>(layoutAttr);
  };

  Operation *op = target;
  auto walkResult = op->walk([&](mlir::Block *block) -> WalkResult {
    for (mlir::Operation &op : block->getOperations()) {
      LogicalResult r = success();
      TypeSwitch<Operation *>(&op)
          .Case([&](mlir::RegionBranchTerminatorOpInterface branchTermOp) {
            r = xegpu::propagateYieldOperandsToRegionResults(
                branchTermOp, getLayoutFromPropagation);
          })
          .Case([&](mlir::RegionBranchOpInterface branchOp) {
            r = xegpu::propagateRegionArgsToInits(branchOp,
                                                  getLayoutFromPropagation);
          })
          .Case([&](mlir::FunctionOpInterface funcOp) {
            r = updateFunctionOpInterface(builder, funcOp,
                                          getLayoutFromPropagation);
          })
          .Default([&](Operation *op) {
            r = updateOpWithForwardFill(builder, op, getLayoutFromPropagation);
          });
      if (failed(r)) {
        op.emitError("Failed to update operation with the layout.");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  return success();
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
}
