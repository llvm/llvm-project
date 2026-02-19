//===- XeGPUPropagateLayout.cpp - XeGPU Layout Propagation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

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
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
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
///  2) Two LayoutInfo values are equal if they are both assigned or
///  both not assigned. The concrete value of assigned state does not matter.
///  3) The meet operator works as follows:
///     - If current state is assigned, return the current state. (already
///     a unique layout is assigned. don't change it)
///     - Otherwise, return the other state.

struct LayoutInfo {
private:
  xegpu::DistributeLayoutAttr storage = nullptr;

public:
  LayoutInfo() = default;
  LayoutInfo(const xegpu::DistributeLayoutAttr &layout) : storage(layout) {}

  // Two lattice values are equal if they have `some` layout. The actual
  // content of the layout does not matter.
  bool operator==(const LayoutInfo &other) const {
    return this->isAssigned() == other.isAssigned();
  }

  static LayoutInfo meet(const LayoutInfo &lhs, const LayoutInfo &rhs);

  static LayoutInfo join(const LayoutInfo &lhs, const LayoutInfo &rhs);

  void print(raw_ostream &os) const;

  bool isAssigned() const { return storage != nullptr; }

  LayoutInfo transpose(ArrayRef<int64_t> permutation) const;

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

SmallVector<int> LayoutInfo::getLaneLayout() const {
  if (!isAssigned())
    return {};
  return llvm::map_to_vector(storage.getEffectiveLaneLayoutAsInt(),
                             [](int64_t val) { return static_cast<int>(val); });
}

SmallVector<int> LayoutInfo::getLaneData() const {
  if (!isAssigned())
    return {};
  return llvm::map_to_vector(storage.getEffectiveLaneDataAsInt(),
                             [](int64_t val) { return static_cast<int>(val); });
}

SmallVector<int> LayoutInfo::getInstData() const {
  if (!isAssigned())
    return {};
  return llvm::map_to_vector(storage.getEffectiveInstDataAsInt(),
                             [](int64_t val) { return static_cast<int>(val); });
}

SmallVector<int> LayoutInfo::getSgLayout() const {
  if (!isAssigned())
    return {};
  return llvm::map_to_vector(storage.getEffectiveSgLayoutAsInt(),
                             [](int64_t val) { return static_cast<int>(val); });
}

SmallVector<int> LayoutInfo::getSgData() const {
  if (!isAssigned())
    return {};
  return llvm::map_to_vector(storage.getEffectiveSgDataAsInt(),
                             [](int64_t val) { return static_cast<int>(val); });
}

SmallVector<int> LayoutInfo::getOrder() const {
  if (!isAssigned() || !storage.getOrder())
    return {};
  return llvm::map_to_vector(storage.getOrder().asArrayRef(),
                             [](int64_t val) { return static_cast<int>(val); });
}

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
  return lhs;
}

/// Since this is a backward analysis, join method is not used.
LayoutInfo LayoutInfo::join(const LayoutInfo &lhs, const LayoutInfo &rhs) {
  llvm_unreachable("Join should not be triggered by layout propagation.");
}

/// Construct a new layout with the transposed inst_data or lane_layout,
/// lane_data.
LayoutInfo LayoutInfo::transpose(ArrayRef<int64_t> permutation) const {
  if (!isAssigned())
    return {};
  // Check if the permutation is valid.
  llvm::SmallSet<int64_t, 4> seen(permutation.begin(), permutation.end());
  bool hasDuplicates = seen.size() != permutation.size();
  bool withinRange = llvm::all_of(permutation, [&](int64_t idx) {
    return idx >= 0 && idx < static_cast<int64_t>(permutation.size());
  });

  if (!withinRange || hasDuplicates) {
    assert(false && "Invalid permutation for transpose.");
    return {};
  }

  SmallVector<int32_t> laneLayout;
  SmallVector<int32_t> laneData;
  SmallVector<int32_t> instData;
  SmallVector<int32_t> sgLayout;
  SmallVector<int32_t> sgData;
  SmallVector<int32_t> order;

  for (int64_t idx : permutation) {
    if (getLaneLayout().size()) {
      laneLayout.push_back(static_cast<int32_t>(getLaneLayout()[idx]));
      laneData.push_back(static_cast<int32_t>(getLaneData()[idx]));
    }
    if (getInstData().size())
      instData.push_back(static_cast<int32_t>(getInstData()[idx]));
    if (getSgData().size()) {
      sgLayout.push_back(static_cast<int32_t>(getSgLayout()[idx]));
      sgData.push_back(static_cast<int32_t>(getSgData()[idx]));
    }
    if (getOrder().size()) {
      order.push_back(static_cast<int32_t>(getOrder()[idx]));
    }
  }
  auto orderAttr = order.size()
                       ? DenseI32ArrayAttr::get(storage.getContext(), order)
                       : nullptr;
  xegpu::LayoutAttr layoutAttr;
  if (getLaneLayout().size())
    layoutAttr =
        xegpu::LayoutAttr::get(storage.getContext(), laneLayout, laneData);
  if (getInstData().size())
    layoutAttr = xegpu::LayoutAttr::get(storage.getContext(), instData);
  if (getSgData().size())
    layoutAttr = xegpu::LayoutAttr::get(
        storage.getContext(),
        DenseI32ArrayAttr::get(storage.getContext(), sgLayout),
        DenseI32ArrayAttr::get(storage.getContext(), sgData),
        /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
        /*lane_data =*/nullptr, orderAttr);
  return LayoutInfo(layoutAttr);
}

//===----------------------------------------------------------------------===//
// LayoutInfoLattice
//===----------------------------------------------------------------------===//

/// Lattice holding the LayoutInfo for each value.
struct LayoutInfoLattice : public Lattice<LayoutInfo> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutInfoLattice)
  using Lattice::Lattice;
};

/// Helper Functions to get default layouts. A `default layout` is a layout that
/// is assigned to a value when the layout is not fixed by some anchor operation
/// (like DPAS).

/// Helper Function to get the default layout for uniform values like constants.
/// For 1D vector, lane_layout is [subgroupSize] and lane_data is [1].
/// For 2D vector, lane_layout is [1, subgroupSize] and lane_data is [1, 1].
static LayoutInfo getDefaultSIMTLayoutInfo(mlir::MLIRContext *ctx,
                                           unsigned rank,
                                           const xegpu::uArch::uArch *uArch) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1) {
    return LayoutInfo(
        xegpu::LayoutAttr::get(ctx, {uArch->getSubgroupSize()}, {1}));
  }
  return LayoutInfo(
      xegpu::LayoutAttr::get(ctx, {1, uArch->getSubgroupSize()}, {1, 1}));
}

static LayoutInfo getDefaultSIMTLayoutInfo(mlir::MLIRContext *ctx,
                                           unsigned rank, int subgroupSize) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1) {
    return LayoutInfo(xegpu::LayoutAttr::get(ctx, {subgroupSize}, {1}));
  }
  return LayoutInfo(xegpu::LayoutAttr::get(ctx, {1, subgroupSize}, {1, 1}));
}

/// Helper to get the default layout for 2D block operations.
template <typename Ty>
static LayoutInfo getSIMTLayoutInfoBlockIO(Ty ty,
                                           const xegpu::uArch::uArch *uArch,
                                           unsigned packingSize) {
  // Expecting a 1D or 2D vector.
  assert((ty.getRank() == 1 || ty.getRank() == 2) &&
         "Expected 1D or 2D vector.");
  // Expecting int or float element type.
  assert(ty.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  // If the rank is 1, then return default layout for 1D vector.
  if (ty.getRank() == 1)
    return getDefaultSIMTLayoutInfo(ty.getContext(), 1, uArch);
  // Packing factor is determined by the element type bitwidth.
  unsigned bitwidth = ty.getElementType().getIntOrFloatBitWidth();
  int packingFactor = bitwidth < packingSize ? packingSize / bitwidth : 1;
  return LayoutInfo(xegpu::LayoutAttr::get(
      ty.getContext(), {1, uArch->getSubgroupSize()}, {1, packingFactor}));
}

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
private:
  xegpu::LayoutKind layoutKind;
  void visitDpasOp(xegpu::DpasOp dpas, ArrayRef<LayoutInfoLattice *> operands,
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

  void visitCreateDescOp(xegpu::CreateDescOp createDesc,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitUpdateNdOffsetOp(xegpu::UpdateNdOffsetOp updateNdOffset,
                             ArrayRef<LayoutInfoLattice *> operands,
                             ArrayRef<const LayoutInfoLattice *> results);

  void visitPrefetchNdOp(xegpu::PrefetchNdOp prefetch,
                         ArrayRef<LayoutInfoLattice *> operands,
                         ArrayRef<const LayoutInfoLattice *> results);

  void visitVectorMultiReductionOp(vector::MultiDimReductionOp reduction,
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

  bool hasParamsOfLayoutKind(xegpu::DistributeLayoutAttr anchorLayout);

public:
  LayoutInfoPropagation(DataFlowSolver &solver,
                        SymbolTableCollection &symbolTable,
                        xegpu::LayoutKind layoutKind)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        layoutKind(layoutKind) {}
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

LogicalResult LayoutInfoPropagation::visitOperation(
    Operation *op, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  TypeSwitch<Operation *>(op)
      .Case(
          [&](xegpu::DpasOp dpasOp) { visitDpasOp(dpasOp, operands, results); })
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
      .Case([&](xegpu::CreateDescOp createDescOp) {
        visitCreateDescOp(createDescOp, operands, results);
      })
      .Case([&](xegpu::UpdateNdOffsetOp updateNdOffsetOp) {
        visitUpdateNdOffsetOp(updateNdOffsetOp, operands, results);
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
      .Case([&](vector::MultiDimReductionOp reductionOp) {
        visitVectorMultiReductionOp(reductionOp, operands, results);
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

// This function returns all layouts for the given sgCount, whose sgData:
// 1. Evenly divides the wgShape.
// 2. Is a multiple of instData.
// Example:
//   wgShape = [128, 64], instData = [8, 16], sgCount = 32
// Returns layouts:
//   [(8,4), (16,2)], which correspond to sgData [16,16] and [8,32].
SmallVector<std::pair<int, int>> getValidLayouts(ArrayRef<int64_t> wgShape,
                                                 ArrayRef<int> instData,
                                                 int64_t sgCount) {
  SmallVector<std::pair<int, int>> candidates;
  for (int sgLayout0 = 1; sgLayout0 <= sgCount; ++sgLayout0) {
    if (sgCount % sgLayout0)
      continue;
    int sgLayout1 = sgCount / sgLayout0;
    int sgData0 = wgShape[0] / sgLayout0;
    int sgData1 = wgShape[1] / sgLayout1;
    if ((wgShape[0] % sgLayout0 || wgShape[1] % sgLayout1) ||
        (sgData0 % instData[0] || sgData1 % instData[1]))
      continue;
    candidates.emplace_back(sgLayout0, sgLayout1);
  }
  // Sort primarily by how balanced they are
  // (i.e., minimize the absolute difference between the two dimensions), and
  // secondarily by the first dimension in ascending order.
  llvm::sort(candidates, [](const std::pair<int, int> &lhs,
                            const std::pair<int, int> &rhs) {
    int diffLhs = std::abs(lhs.first - lhs.second);
    int diffRhs = std::abs(rhs.first - rhs.second);
    if (diffLhs != diffRhs)
      return diffLhs < diffRhs;
    return lhs.first < rhs.first;
  });
  return candidates;
}

FailureOr<int64_t> getNumSg(Operation *op, const int sgSize) {
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
  xegpu::DistributeLayoutAttr anchorLayout = prefetch.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    prefetchLayout = LayoutInfo(anchorLayout);
  } else {
    // Here we assign the default layout to the tensor descriptor operand of
    // prefetch.
    auto tdescTy = prefetch.getTensorDescType();

    const auto *uArch = getUArch(getChipStr(prefetch).value_or(""));
    const auto *uArchInstruction =
        dyn_cast<xegpu::uArch::Subgroup2DBlockPrefetchInstruction>(
            uArch->getInstruction(
                xegpu::uArch::InstructionKind::Subgroup2DBlockPrefetch));

    auto blockWHC =
        uArchInstruction->getBlockWidthHeightCount(tdescTy.getElementType());
    if (!blockWHC)
      prefetch.emitWarning("No known block params found for the element type.");
    auto [bWidth, bHeight, bCount] = blockWHC.value();
    SmallVector<int> instData;
    int instWidth = xegpu::getLargestDivisor(
        static_cast<int>(tdescTy.getDimSize(tdescTy.getRank() - 1)), bWidth);
    if (instWidth == -1)
      prefetch.emitWarning(
          "No suitable instruction multiple found for the given shape.");
    if (tdescTy.getRank() == 1)
      instData = {instWidth};
    else {
      int instHeight = xegpu::getLargestDivisor(
          static_cast<int>(tdescTy.getDimSize(tdescTy.getRank() - 2)), bHeight);
      if (instHeight == -1)
        prefetch.emitWarning(
            "No suitable instruction multiple found for the given shape.");
      instData = {instHeight, instWidth};
    }

    if (layoutKind == xegpu::LayoutKind::InstData)
      prefetchLayout =
          LayoutInfo(xegpu::LayoutAttr::get(tdescTy.getContext(), instData));
    else
      prefetchLayout = getSIMTLayoutInfoBlockIO(
          tdescTy, uArch, uArchInstruction->getPackedFormatBitSize());

    prefetch.setLayoutAttr(
        dyn_cast<xegpu::DistributeLayoutAttr>(prefetchLayout.get()));
  }
  // Propagate the layout to the source tensor descriptor.
  propagateIfChanged(operands[0], operands[0]->meet(prefetchLayout));
}

void LayoutInfoPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;

  VectorType sourceTy = reduction.getSourceVectorType();
  SmallVector<int64_t> reductionDims(reduction.getReductionDims());

  const auto *uArch = getUArch(xegpu::getChipStr(reduction).value_or(""));
  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  // The result layout represents the layout requirements of the operation.
  // it is recorded to anchor layout or temporary layout.
  // it must be honored for current op and may conflict with the layout
  // propagated from consumer op, the conflict is resolved in later phase by
  // converting the required result layout to the consumer layout
  auto requiredResLayoutAttr = xegpu::setupMultiReductionResultLayout(
      layoutKind, sourceTy, consumerLayoutAttr, reductionDims, uArch);

  xegpu::setTemporaryLayout(reduction->getResult(0), requiredResLayoutAttr);

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferMultiReductionSourceLayout(
      requiredResLayoutAttr, reductionDims);

  propagateIfChanged(operands[0], operands[0]->meet(LayoutInfo(srcLayoutAttr)));
  // Accumulator should have the same layout as the result.
  propagateIfChanged(operands[1],
                     operands[1]->meet(LayoutInfo(requiredResLayoutAttr)));
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

  size_t dimDiff = resultTy.getRank() - sourceTy.getRank();
  for (size_t i = 0; i < srcShape.size(); i++)
    if ((srcShape[i] == 1) && (resShape[i + dimDiff] != 1))
      broadcast.emitWarning("broadcast must either from low-rank or same-rank "
                            "with unit-dim, mixed scenario is not supported!");

  auto resultLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  xegpu::DistributeLayoutAttr srcLayoutAttr =
      xegpu::inferBroadcastSourceLayout(resultLayoutAttr, resShape, srcShape);

  propagateIfChanged(operands[0], operands[0]->meet(LayoutInfo(srcLayoutAttr)));
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

  propagateIfChanged(operands[0], operands[0]->meet(LayoutInfo(srcLayoutAttr)));
}

/// Propagate the layout of the result tensor to the source tensor descriptor
/// in UpdateNdOffsetOp.
void LayoutInfoPropagation::visitUpdateNdOffsetOp(
    xegpu::UpdateNdOffsetOp updateNdOffset,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  // Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(resultLayout));
}

/// Set the layouts for DPAS A, B, and C operands.
void LayoutInfoPropagation::visitDpasOp(
    xegpu::DpasOp dpas, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo dpasALayout;
  LayoutInfo dpasBLayout;
  LayoutInfo dpasCDLayout;

  xegpu::DistributeLayoutAttr anchorLayoutCD = dpas.getLayoutCdAttr();
  if (hasParamsOfLayoutKind(anchorLayoutCD)) {
    xegpu::DistributeLayoutAttr anchorLayoutA = dpas.getLayoutAAttr();
    xegpu::DistributeLayoutAttr anchorLayoutB = dpas.getLayoutBAttr();
    assert(hasParamsOfLayoutKind(anchorLayoutA) &&
           "Expected anchor layout for DPAS A operand.");
    assert(hasParamsOfLayoutKind(anchorLayoutB) &&
           "Expected anchor layout for DPAS B operand.");
    dpasALayout = LayoutInfo(anchorLayoutA);
    dpasBLayout = LayoutInfo(anchorLayoutB);
    dpasCDLayout = LayoutInfo(anchorLayoutCD);
  } else {
    const auto *uArch = getUArch(getChipStr(dpas).value_or(""));
    VectorType aTy = dpas.getLhsType();
    VectorType bTy = dpas.getRhsType();
    VectorType cdTy = dpas.getResultType();

    xegpu::DistributeLayoutAttr consumerLayoutAttr = nullptr;
    xegpu::DistributeLayoutAttr requiredCDLayoutAttr, requiredALayout,
        requiredBLayout;

    int numSg = 0;
    if (layoutKind == xegpu::LayoutKind::Subgroup) {
      LayoutInfo consumerLayout = results[0]->getValue();
      if (!consumerLayout.isAssigned())
        return;
      consumerLayoutAttr =
          dyn_cast<xegpu::DistributeLayoutAttr>(consumerLayout.get());
      auto numSgOrErr = getNumSg(dpas, uArch->getSubgroupSize());
      if (failed(numSgOrErr)) {
        dpas.emitWarning(
            "Unable to determine the number of subgroups for the operation.");
        return;
      }
      numSg = numSgOrErr.value();
    }
    auto layouts = xegpu::setupDpasLayout(layoutKind, aTy, bTy, cdTy,
                                          consumerLayoutAttr, uArch, numSg);
    if (!layouts.has_value()) {
      dpas.emitWarning(
          "Failed to determine required layouts for DPAS operands.");
      return;
    }

    std::tie(requiredALayout, requiredBLayout, requiredCDLayoutAttr) = *layouts;

    dpas.setLayoutAAttr(requiredALayout);
    dpas.setLayoutBAttr(requiredBLayout);
    dpas.setLayoutCdAttr(requiredCDLayoutAttr);
    dpasALayout = LayoutInfo(requiredALayout);
    dpasBLayout = LayoutInfo(requiredBLayout);
    dpasCDLayout = LayoutInfo(requiredCDLayoutAttr);
  }
  propagateIfChanged(operands[0], operands[0]->meet(dpasALayout));
  propagateIfChanged(operands[1], operands[1]->meet(dpasBLayout));
  if (operands.size() > 2)
    propagateIfChanged(operands[2], operands[2]->meet(dpasCDLayout));
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutInfoPropagation::visitStoreNdOp(
    xegpu::StoreNdOp store, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo storeLayout;
  xegpu::DistributeLayoutAttr anchorLayout = store.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    storeLayout = LayoutInfo(anchorLayout);
  } else {
    const auto *uArch = getUArch(getChipStr(store).value_or(""));
    const auto *uArchInstruction =
        dyn_cast<xegpu::uArch::Subgroup2DBlockStoreInstruction>(
            uArch->getInstruction(
                xegpu::uArch::InstructionKind::Subgroup2DBlockStore));
    VectorType dataTy = store.getValueType();
    auto blockWHC = uArchInstruction->getBlockWidthHeightCount(
        store.getValueType().getElementType());
    if (!blockWHC)
      store.emitWarning("No known block params found for the element type.");
    auto [bWidth, bHeight, bCount] = blockWHC.value();
    SmallVector<int> instData;
    int instWidth = xegpu::getLargestDivisor(
        static_cast<int>(dataTy.getDimSize(dataTy.getRank() - 1)), bWidth);
    if (instWidth == -1)
      store.emitWarning(
          "No suitable instruction multiple found for the given shape.");
    if (dataTy.getRank() == 1)
      instData = {instWidth};
    else {
      int instHeight = xegpu::getLargestDivisor(
          static_cast<int>(dataTy.getDimSize(dataTy.getRank() - 2)), bHeight);
      if (instHeight == -1)
        store.emitWarning(
            "No suitable instruction multiple found for the given shape.");
      instData = {instHeight, instWidth};
    }

    if (layoutKind == xegpu::LayoutKind::InstData)
      storeLayout =
          LayoutInfo(xegpu::LayoutAttr::get(dataTy.getContext(), instData));
    else if (layoutKind == xegpu::LayoutKind::Lane)
      storeLayout =
          getSIMTLayoutInfoBlockIO(store.getValueType(), uArch,
                                   uArchInstruction->getPackedFormatBitSize());
    else { // xegpu::LayoutKind::Subgroup
      auto sgSize = uArch->getSubgroupSize();
      auto numSgOrErr = getNumSg(store, sgSize);
      if (failed(numSgOrErr)) {
        store.emitWarning(
            "Unable to determine the number of subgroups for the operation.");
        return;
      }
      auto sgLayouts = getValidLayouts(store.getValueType().getShape(),
                                       instData, numSgOrErr.value());
      if (sgLayouts.empty()) {
        store.emitWarning(
            "Unable to determine suitable subgroup layout for store value.");
        return;
      }
      SmallVector<int> sgLayout = {sgLayouts[0].first, sgLayouts[0].second};
      SmallVector<int> sgData = {
          static_cast<int>(dataTy.getShape()[0]) / sgLayout[0],
          static_cast<int>(dataTy.getShape()[1]) / sgLayout[1]};
      storeLayout = LayoutInfo(xegpu::LayoutAttr::get(
          dataTy.getContext(),
          DenseI32ArrayAttr::get(dataTy.getContext(), sgLayout),
          DenseI32ArrayAttr::get(dataTy.getContext(), sgData),
          /*inst_data =*/nullptr, /*lane_layout =*/nullptr,
          /*lane_data =*/nullptr, /*order =*/nullptr));
    }
    store.setLayoutAttr(
        dyn_cast<xegpu::DistributeLayoutAttr>(storeLayout.get()));
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
  xegpu::DistributeLayoutAttr anchorLayout = load.getLayoutAttr();
  if (hasParamsOfLayoutKind(anchorLayout)) {
    loadLayout = LayoutInfo(anchorLayout);
  } else {

    LayoutInfo valueLayout = results[0]->getValue();
    // Need the layout of the value to propagate to the tensor descriptor.
    if (!valueLayout.isAssigned())
      return;
    loadLayout = valueLayout;
    // LoadNdOp has the transpose effect. However, at the stage of this analysis
    // this effect is not expected and should be abstracted away. Emit a
    // warning.
    if (auto transpose = load.getTranspose()) {
      load.emitWarning("Transpose effect is not expected for LoadNdOp at "
                       "LayoutInfoPropagation stage.");
      loadLayout = valueLayout.transpose(transpose.value());
    }
    load.setLayoutAttr(dyn_cast<xegpu::DistributeLayoutAttr>(loadLayout.get()));
  }
  // Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(loadLayout));
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
  LayoutInfo newLayout = resultLayout.transpose(transpose.getPermutation());
  // Propagate the new layout to the vector operand.
  propagateIfChanged(operands[0], operands[0]->meet(newLayout));
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
  const auto *uArch = getUArch(xegpu::getChipStr(bitcast).value_or(""));
  auto requiredResLayoutAttr = setupBitCastResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  xegpu::setTemporaryLayout(bitcast->getResult(0), requiredResLayoutAttr);

  int inElemTyBitWidth = srcVecType.getElementType().getIntOrFloatBitWidth();
  int outElemTyBitWidth = resVecType.getElementType().getIntOrFloatBitWidth();

  // derive the source layout from the dominant layout and reduction dims
  auto srcLayoutAttr = xegpu::inferBitCastSourceLayout(
      requiredResLayoutAttr, outElemTyBitWidth, inElemTyBitWidth);

  propagateIfChanged(operands[0], operands[0]->meet(LayoutInfo(srcLayoutAttr)));
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
  const auto *uArch =
      getUArch(xegpu::getChipStr(insertStridedSlice).value_or(""));

  auto requiredResLayoutAttr = xegpu::setupInsertStridedSliceResultLayout(
      layoutKind, srcVecType, resVecType, consumerLayoutAttr, uArch);

  xegpu::setTemporaryLayout(insertStridedSlice->getResult(0),
                            requiredResLayoutAttr);

  auto srcLayoutAttr = xegpu::inferInsertStridedSliceSourceLayout(
      requiredResLayoutAttr, resVecType.getShape(), srcVecType.getShape());

  propagateIfChanged(operands[0], operands[0]->meet(LayoutInfo(srcLayoutAttr)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(LayoutInfo(requiredResLayoutAttr)));
}

/// Propagate the layout of the result to the tensor descriptor, mask and offset
/// operands in LoadGatherOp.
void LayoutInfoPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = load.getLayoutAttr();
  const auto *uArch = getUArch(getChipStr(load).value_or(""));
  auto subgroupSize = uArch->getSubgroupSize();
  VectorType resVecTy = load.getValueType();
  int chunkSize = load.getChunkSize().value_or(1);

  LayoutInfo resLayoutInfo = results[0]->getValue();
  if (!resLayoutInfo.isAssigned())
    return;
  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
  } else {
    if (!resVecTy) {
      load.emitWarning("Not propagating, non-vector payload supplied.");
      return;
    }
    requiredAnchorLayoutAttr = xegpu::setupLoadGatherAnchorLayout(
        layoutKind, resVecTy, chunkSize, consumerLayoutAttr, uArch);
    load.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  auto maskLayoutAttr = requiredAnchorLayoutAttr;
  // Special handling mask layout for chunked ops: Enforce the default xegpu 1D
  // layout for mask.
  if (chunkSize > 1) {
    if (layoutKind == xegpu::LayoutKind::InstData)
      maskLayoutAttr =
          xegpu::LayoutAttr::get(load->getContext(), {subgroupSize});
    else if (layoutKind == xegpu::LayoutKind::Lane)
      maskLayoutAttr =
          xegpu::LayoutAttr::get(load->getContext(), {subgroupSize}, {1});
    else
      assert(false &&
             "chunked StoreScatterOp should not be used at workgroup level");
  }

  LayoutInfo maskLayoutInfo = LayoutInfo(maskLayoutAttr);
  auto loadLayoutInfo = LayoutInfo(requiredAnchorLayoutAttr);

  // Propagate the new layout to the tensor descriptor operand.
  if (isa<xegpu::TensorDescType>(load.getSourceType()))
    propagateIfChanged(operands[0], operands[0]->meet(loadLayoutInfo));
  // Propagate the new layout to the mask and optional offset operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayoutInfo));
  if (load.getOffsets())
    propagateIfChanged(operands[2], operands[2]->meet(maskLayoutInfo));
}

/// Propagate the layout of the descriptor to the vector offset operand in
/// CreateDescOp.
void LayoutInfoPropagation::visitCreateDescOp(
    xegpu::CreateDescOp createDesc, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo descLayout = results[0]->getValue();
  // Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  const auto *uArch = getUArch(getChipStr(createDesc).value_or(""));
  // For offset operand propagate 1D default layout.
  LayoutInfo layout = getDefaultSIMTLayoutInfo(createDesc->getContext(), 1,
                                               uArch->getSubgroupSize());
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, offset and mask operands in
/// the StoreScatterOp.
void LayoutInfoPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  xegpu::DistributeLayoutAttr requiredAnchorLayoutAttr;
  xegpu::DistributeLayoutAttr anchorLayoutAttr = storeScatter.getLayoutAttr();
  const auto *uArch = getUArch(getChipStr(storeScatter).value_or(""));
  auto subgroupSize = uArch->getSubgroupSize();
  VectorType srcVecTy = storeScatter.getValueType();
  int chunkSize = storeScatter.getChunkSize().value_or(1);

  if (hasParamsOfLayoutKind(anchorLayoutAttr)) {
    requiredAnchorLayoutAttr = anchorLayoutAttr;
  } else {
    if (!srcVecTy) {
      storeScatter.emitWarning("Not propagating, non-vector payload supplied.");
      return;
    }
    requiredAnchorLayoutAttr = xegpu::setupStoreScatterAnchorLayout(
        layoutKind, srcVecTy, chunkSize, uArch);
    storeScatter.setLayoutAttr(requiredAnchorLayoutAttr);
  }

  LayoutInfo srcLayoutInfo = LayoutInfo(requiredAnchorLayoutAttr);
  auto maskLayoutAttr = requiredAnchorLayoutAttr;
  // Special handling mask layout for chunked ops: Enforce the default xegpu 1D
  // layout for mask.
  if (chunkSize > 1) {
    if (layoutKind == xegpu::LayoutKind::InstData)
      maskLayoutAttr =
          xegpu::LayoutAttr::get(storeScatter->getContext(), {subgroupSize});
    else if (layoutKind == xegpu::LayoutKind::Lane)
      maskLayoutAttr = xegpu::LayoutAttr::get(storeScatter->getContext(),
                                              {subgroupSize}, {1});
    else
      assert(false &&
             "chunked StoreScatterOp should not be used at workgroup level");
  }

  LayoutInfo maskLayoutInfo = LayoutInfo(maskLayoutAttr);

  // Propagate the payload operand layout
  propagateIfChanged(operands[0], operands[0]->meet(srcLayoutInfo));
  // Propagate the destination (if tdesc) operand layout
  if (isa<xegpu::TensorDescType>(storeScatter.getDestType()))
    propagateIfChanged(operands[1], operands[1]->meet(srcLayoutInfo));
  // Propagate the new layout to the mask and optional offset operand.
  propagateIfChanged(operands[2], operands[2]->meet(maskLayoutInfo));
  if (storeScatter.getOffsets())
    propagateIfChanged(operands[3], operands[3]->meet(maskLayoutInfo));
}

void LayoutInfoPropagation::visitLoadMatrixOp(
    xegpu::LoadMatrixOp loadMatrixOp, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {

  LayoutInfo resLayoutInfo = results[0]->getValue();
  auto consumerLayoutAttr =
      dyn_cast<xegpu::DistributeLayoutAttr>(resLayoutInfo.get());

  xegpu::DistributeLayoutAttr anchorLayout = loadMatrixOp.getLayoutAttr();

  // only need to set anchor layout, no need to porpagate to memdesc and
  // offset
  if (!hasParamsOfLayoutKind(anchorLayout)) {
    VectorType resVecTy =
        llvm::cast<VectorType>(loadMatrixOp.getRes().getType());
    assert(resVecTy.getRank() == 2 && "Expecting 2D vector for store matrix.");
    const auto *uArch = getUArch(getChipStr(loadMatrixOp).value_or(""));
    auto requiredAnchorLayoutAttr = xegpu::setupLoadMatrixAnchorLayout(
        layoutKind, resVecTy, consumerLayoutAttr, uArch);
    loadMatrixOp.setLayoutAttr(requiredAnchorLayoutAttr);
  }
}

// Store matrix is a flavor of scattered store for 2D shapes.
void LayoutInfoPropagation::visitStoreMatrixOp(
    xegpu::StoreMatrixOp storeMatrix, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  xegpu::DistributeLayoutAttr anchorLayout = storeMatrix.getLayoutAttr();
  LayoutInfo layout;
  if (hasParamsOfLayoutKind(anchorLayout)) {
    layout = LayoutInfo(anchorLayout);
  } else {
    VectorType srcVecTy =
        llvm::cast<VectorType>(storeMatrix.getData().getType());
    assert(srcVecTy.getRank() == 2 && "Expecting 2D vector for store matrix.");
    const auto *uArch = getUArch(getChipStr(storeMatrix).value_or(""));
    auto requiredAnchorLayoutAttr =
        xegpu::setupStoreMatrixAnchorLayout(layoutKind, srcVecTy, uArch);
    storeMatrix.setLayoutAttr(requiredAnchorLayoutAttr);
    layout = LayoutInfo(requiredAnchorLayoutAttr);
  }

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

  RunLayoutInfoPropagation(Operation *op, xegpu::LayoutKind layoutKind)
      : target(op) {
    SymbolTableCollection symbolTable;
    loadBaselineAnalyses(solver);
    solver.load<LayoutInfoPropagation>(symbolTable, layoutKind);
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

/// Helper to get the defining CreateNdDescOp of a tensor descriptor value. This
/// function tries to find the defining CreateNdDescOp recursively accross
/// control-flow boundaries.
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

static xegpu::DistributeLayoutAttr
getExpectedLayoutAt(OpOperand &operand,
                    xegpu::DistributeLayoutAttr currLayout) {
  Operation *op = operand.getOwner();
  unsigned idx = operand.getOperandNumber();

  // For vector::BroadcastOp, infer the source layout from the result layout.
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
    auto resLayout = xegpu::getDistributeLayoutAttr(broadcast->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
    auto srcTy = dyn_cast<VectorType>(broadcast.getSourceType());
    if (!srcTy)
      return xegpu::DistributeLayoutAttr();
    return xegpu::inferBroadcastSourceLayout(
        resLayout, broadcast.getResultVectorType().getShape(),
        srcTy.getShape());
  }

  // For vector::MultiDimReductionOp, infer source layout from result layout
  // using reduction dims. Acc operand is expected to have the same layout as
  // the result.
  if (auto reduction = dyn_cast<vector::MultiDimReductionOp>(op)) {
    auto resLayout = xegpu::getDistributeLayoutAttr(reduction->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
    if (idx == 0) {
      SmallVector<int64_t> reductionDims(reduction.getReductionDims());
      return xegpu::inferMultiReductionSourceLayout(resLayout, reductionDims);
    }
    if (idx == 1)
      return resLayout;
  }

  // For vector::BitCastOp, infer source layout from result layout using
  // element type bitwidths.
  if (auto bitcast = dyn_cast<vector::BitCastOp>(op)) {
    auto resLayout = xegpu::getDistributeLayoutAttr(bitcast->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
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
    auto resLayout = xegpu::getDistributeLayoutAttr(shapeCast->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
    return xegpu::inferShapeCastSourceLayout(
        resLayout, shapeCast.getResultVectorType().getShape(),
        shapeCast.getSourceVectorType().getShape());
  }

  // For vector::InsertStridedSliceOp, infer source layout from result layout.
  // Dest vector must have the same layout as the result.
  if (auto insertSlice = dyn_cast<vector::InsertStridedSliceOp>(op)) {
    auto resLayout = xegpu::getDistributeLayoutAttr(insertSlice->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
    if (idx == 0)
      return xegpu::inferInsertStridedSliceSourceLayout(
          resLayout, insertSlice.getDestVectorType().getShape(),
          insertSlice.getSourceVectorType().getShape());
    if (idx == 1)
      return resLayout;
  }
  // For elementwise operations, all operands must have the same layout as the
  // result.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    auto resLayout = xegpu::getDistributeLayoutAttr(op->getResult(0));
    if (!resLayout)
      return xegpu::DistributeLayoutAttr();
    return resLayout;
  }
  // TODO: Handle more cases as needed here.
  // Fallback to currently assigned layout for all other cases. This assumes no
  // conflicts.
  return currLayout;
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
};

} // namespace

LogicalResult ResolveLayoutConflicts::run() {
  // Scan all operations in the parent op and resolve layout conflicts at
  // tensor descriptor and vector use points.
  auto r = parentOp->walk([&](Operation *op) -> WalkResult {
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

  return r.wasInterrupted() ? failure() : success();
}

LogicalResult
ResolveLayoutConflicts::resolveVectorConsumer(OpOperand &operand) {
  Value vectorValue = operand.get();
  Operation *consumerOp = operand.getOwner();
  // Get the current layout of the vector value.
  auto currLayout = xegpu::getDistributeLayoutAttr(vectorValue);
  if (!currLayout) {
    consumerOp->emitError("Vector operand has no layout assigned.");
    return failure();
  }

  // Get the expected layout at this operand.
  auto expectedLayout = getExpectedLayoutAt(operand, currLayout);
  if (!expectedLayout) {
    consumerOp->emitError("No expected layout found for vector operand.");
    return failure();
  }

  // If layouts are same, no conflict exists, return success.
  if (expectedLayout.isEqualTo(currLayout))
    return success();

  // Insert a convert_layout op to resolve the conflict.
  builder.setInsertionPointAfterValue(vectorValue);
  auto convertOp = xegpu::ConvertLayoutOp::create(
      builder, consumerOp->getLoc(), vectorValue.getType(), vectorValue,
      currLayout, expectedLayout);

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
  // TODO: Scattered tensor desc is not supported for now.
  if (currTDescType.isScattered()) {
    DBGS() << "Scattered tensor descriptor not supported: " << tdescValue
           << "\n";
    return failure();
  }
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
static LogicalResult updateOp(mlir::OpBuilder &builder, mlir::Operation *op,
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
    if (!layout && result.getNumUses() > 0) {
      op->emitWarning("op has users but no layout assigned for its result");
      continue;
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

/// Region ops like scf.for need special handling because they have blocks
/// inside. If the blocks have tensor descriptor type as block arguments,
/// thier types must be updated. Also region op can have results that may not
/// have any users (e.g. A and B tiles). They are not assigned a layout by
/// layout analysis because they have no users. However inside the region op
/// corresponding block arguments for these results do have layouts.
/// Therefore, in this case we still need to update the result types with the
/// layout attribute. This function function updates the internal block
/// arguments and the result types of the region op with the assigned layouts.
/// clang-format off
/// Example: scf.for ... iter_args(...) -> (out types) {
///   ^bb0(block types):
///     ...
///   scf.yield ... : (yield types)
/// }
/// clang-format on
/// In this example, at scf.yield, control-flow can transfer to two successor
/// regions. One is the ^bb0 (for loop body) and the other is the scf.for op
/// itself (yield the results). So we update both the block arguments of the
/// successor region (i.e. block types) and the result types of the scf.for op
/// (i.e. out types). Note that yield types are updated by respective
/// producers inside bb0.
static LogicalResult
updateControlFlowOps(mlir::OpBuilder &builder,
                     mlir::RegionBranchTerminatorOpInterface terminator,
                     GetLayoutFnTy getLayoutOfValue) {
  // Only process if the terminator is inside a region branch op.
  auto branchOp = dyn_cast<RegionBranchOpInterface>(terminator->getParentOp());
  if (!branchOp)
    return success();

  RegionBranchSuccessorMapping mapping;
  branchOp.getSuccessorOperandInputMapping(mapping,
                                           RegionBranchPoint(terminator));
  for (const auto &[successorOperand, successorInputs] : mapping) {
    for (Value successorInput : successorInputs) {
      Type inputType = successorInput.getType();
      // We only need to operate on tensor descriptor or vector types.
      if (!isa<xegpu::TensorDescType, VectorType>(inputType))
        continue;
      xegpu::DistributeLayoutAttr successorInputLayout =
          getLayoutOfValue(successorInput);
      xegpu::DistributeLayoutAttr successorOperandLayout =
          getLayoutOfValue(successorOperand->get());

      // If either of the layouts is not assigned, we cannot proceed.
      if (!successorOperandLayout) {
        LLVM_DEBUG(DBGS() << "No layout assigned for forwarded operand in "
                             "branch terminator: "
                          << successorOperand->get() << "\n");
        return failure();
      }
      // We expect the layouts to match.
      if (successorInputLayout &&
          successorInputLayout != successorOperandLayout) {
        LLVM_DEBUG(DBGS() << "Conflicting layouts for region argument and "
                             "operand forwarded as the argument: "
                          << successorInputLayout << " vs "
                          << successorOperandLayout << "\n");
        return failure();
      }
      // Get tensor descriptor type with the layout.
      if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(inputType)) {
        auto newTdescTy = xegpu::TensorDescType::get(
            tdescTy.getContext(), tdescTy.getShape(), tdescTy.getElementType(),
            tdescTy.getEncoding(), successorOperandLayout);
        successorInput.setType(newTdescTy);
        continue;
      }
      // If the type is a vector type and this region argument is an OpResult,
      // set the layout attribute on the OpResult.
      if (auto result = dyn_cast<OpResult>(successorInput))
        xegpu::setDistributeLayoutAttr(result, successorOperandLayout);
    }
  }
  return success();
}

/// Update the function arguments and results with the layouts.
static LogicalResult updateFunctionOpInterface(mlir::OpBuilder &builder,
                                               mlir::FunctionOpInterface funcOp,
                                               GetLayoutFnTy getLayoutOfValue) {
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
                                      LayoutKind layoutKind, bool printOnly) {
  RunLayoutInfoPropagation analysis(target, layoutKind);
  // Print the analysis result and exit. (for debugging purposes)
  if (printOnly) {
    auto &os = llvm::outs();
    analysis.printAnalysisResult(os);
    return success();
  }
  // Helper to convert LayoutInfo to xegpu::LayoutAttr.
  auto getXeGPULayoutForValue = [&](Value val) -> xegpu::DistributeLayoutAttr {
    LayoutInfo layout = analysis.getLayoutInfo(val);
    if (!layout.isAssigned())
      return {};
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
    xegpu::DistributeLayoutAttr layoutAttr =
        cast<xegpu::DistributeLayoutAttr>(layout.get());
    if (layout.isSliceLayout())
      return cast<xegpu::SliceAttr>(layoutAttr);

    return cast<xegpu::LayoutAttr>(layoutAttr);
  };

  Operation *op = target;
  auto walkResult = op->walk([&](mlir::Block *block) -> WalkResult {
    for (mlir::Operation &op : llvm::reverse(block->getOperations())) {
      LogicalResult r = success();
      TypeSwitch<Operation *>(&op)
          .Case([&](mlir::RegionBranchTerminatorOpInterface branchTermOp) {
            r = updateControlFlowOps(builder, branchTermOp,
                                     getXeGPULayoutForValue);
          })
          .Case([&](mlir::FunctionOpInterface funcOp) {
            r = updateFunctionOpInterface(builder, funcOp,
                                          getXeGPULayoutForValue);
          })
          .Default([&](Operation *op) {
            r = updateOp(builder, op, getXeGPULayoutForValue);
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
                                     this->printOnly))) {
    signalPassFailure();
    return;
  }
  // Resolve layout conflicts if any.
  if (failed(xegpu::resolveLayoutConflicts(getOperation()))) {
    signalPassFailure();
    return;
  }
}
