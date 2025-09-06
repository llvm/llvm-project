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
#include "mlir/Dialect/XeGPU/IR/XeGPUTargetInfo.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
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
// Layout
//===----------------------------------------------------------------------===//

/// Helper class to store the ND layout of lanes within a subgroup and data
/// owned by each lane.
struct Layout {
  SmallVector<int64_t, 3> layout;
  Layout() = default;
  Layout(std::initializer_list<int64_t> list) : layout(list) {}
  void print(llvm::raw_ostream &os) const;
  size_t size() const { return layout.size(); }
};

void Layout::print(llvm::raw_ostream &os) const {
  os << llvm::interleaved_array(layout);
}

/// LaneLayout represents the logical layout of lanes within a subgroup when it
/// accesses some value. LaneData represents the logical layout of data owned by
/// each work item.
using LaneLayout = Layout;
using LaneData = Layout;

//===----------------------------------------------------------------------===//
// LayoutInfo
//===----------------------------------------------------------------------===//

/// Helper class for tracking the analysis state of an mlir value. For layout
/// propagation, the analysis state is simply the lane_layout and lane_data of
/// each value. Purpose of this analysis to propagate some unique layout for
/// each value in the program starting from a set of anchor operations (like
/// DPAS, StoreNd, etc.).
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
  LaneLayout laneLayout;
  LaneData laneData;
  xegpu::LayoutAttr layoutAttr;

public:
  LayoutInfo() = default;
  LayoutInfo(const LaneLayout &layout, const LaneData &data)
      : laneLayout(layout), laneData(data) {}

  // Two lattice values are equal if they have `some` layout. The actual
  // content of the layout does not matter.
  bool operator==(const LayoutInfo &other) const {
    return this->isAssigned() == other.isAssigned();
  }

  static LayoutInfo meet(const LayoutInfo &lhs, const LayoutInfo &rhs);

  static LayoutInfo join(const LayoutInfo &lhs, const LayoutInfo &rhs);

  void print(raw_ostream &os) const;

  bool isAssigned() const {
    return laneLayout.size() > 0 && laneData.size() > 0;
  }

  LayoutInfo getTransposedLayout(ArrayRef<int64_t> permutation) const;

  const LaneLayout &getLayout() const { return laneLayout; }
  const LaneData &getData() const { return laneData; }
  ArrayRef<int64_t> getLayoutAsArrayRef() const { return laneLayout.layout; }
  ArrayRef<int64_t> getDataAsArrayRef() const { return laneData.layout; }
};

void LayoutInfo::print(raw_ostream &os) const {
  if (isAssigned()) {
    os << "lane_layout: ";
    laneLayout.print(os);
    os << ", lane_data: ";
    laneData.print(os);
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

/// Get the transposed layout according to the given permutation.
LayoutInfo
LayoutInfo::getTransposedLayout(ArrayRef<int64_t> permutation) const {
  if (!isAssigned())
    return {};
  LaneLayout newLayout;
  LaneData newData;
  for (int64_t idx : permutation) {
    newLayout.layout.push_back(laneLayout.layout[idx]);
    newData.layout.push_back(laneData.layout[idx]);
  }
  return LayoutInfo(newLayout, newData);
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
static LayoutInfo getDefaultSIMTLayoutInfo(unsigned rank) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1)
    return LayoutInfo(LaneLayout({xegpu::targetinfo::subgroupSize}),
                      LaneData({1}));
  return LayoutInfo(LaneLayout({1, xegpu::targetinfo::subgroupSize}),
                    LaneData({1, 1}));
}

/// Helper to get the default layout for a vector type.
static LayoutInfo getDefaultSIMTLayoutInfo(VectorType vectorTy,
                                           bool isScattered = false) {
  // Expecting a 1D or 2D vector.
  assert((vectorTy.getRank() == 1 || vectorTy.getRank() == 2) &&
         "Expected 1D or 2D vector.");
  // Expecting int or float element type.
  assert(vectorTy.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  // If the rank is 1, then return default layout for 1D vector.
  if (vectorTy.getRank() == 1)
    return getDefaultSIMTLayoutInfo(1);
  // Packing factor is determined by the element type bitwidth.
  int packingFactor = 1;
  unsigned bitwidth = vectorTy.getElementType().getIntOrFloatBitWidth();
  if (isScattered) {
    packingFactor =
        bitwidth < xegpu::targetinfo::packedSizeInBitsForGatherScatter
            ? xegpu::targetinfo::packedSizeInBitsForGatherScatter / bitwidth
            : 1;
    return LayoutInfo(LaneLayout({xegpu::targetinfo::subgroupSize, 1}),
                      LaneData({1, packingFactor}));
  }
  if (bitwidth < xegpu::targetinfo::packedSizeInBitsForDefault)
    packingFactor = xegpu::targetinfo::packedSizeInBitsForDefault / bitwidth;
  return LayoutInfo(LaneLayout({1, xegpu::targetinfo::subgroupSize}),
                    LaneData({1, packingFactor}));
}

/// Helper to get the default layout for a vector type.
static LayoutInfo getDefaultSIMTLayoutInfo(xegpu::TensorDescType tdescTy,
                                           bool isScattered = false) {
  // Expecting a 1D or 2D vector.
  assert((tdescTy.getRank() == 1 || tdescTy.getRank() == 2) &&
         "Expected 1D or 2D TensorDesc.");
  // Expecting int or float element type.
  assert(tdescTy.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  // If the rank is 1, then return default layout for 1D vector.
  if (tdescTy.getRank() == 1)
    return getDefaultSIMTLayoutInfo(1);
  // Packing factor is determined by the element type bitwidth.
  unsigned bitwidth = tdescTy.getElementType().getIntOrFloatBitWidth();

  if (isScattered) {
    int packingFactor =
        bitwidth < xegpu::targetinfo::packedSizeInBitsForGatherScatter
            ? xegpu::targetinfo::packedSizeInBitsForGatherScatter / bitwidth
            : 1;
    return LayoutInfo(LaneLayout({xegpu::targetinfo::subgroupSize, 1}),
                      LaneData({1, packingFactor}));
  }

  int packingFactor =
      (bitwidth < xegpu::targetinfo::packedSizeInBitsForDefault)
          ? xegpu::targetinfo::packedSizeInBitsForDefault / bitwidth
          : 1;
  return LayoutInfo(LaneLayout({1, xegpu::targetinfo::subgroupSize}),
                    LaneData({1, packingFactor}));
}

/// Helper Function to get the expected layouts for DPAS operands. `lane_data`
/// is set according to the following criteria:
/// * For A operand, the data must be packed in minimum
/// `packedSizeInBitsForDefault`
/// * For B operand, the data must be packed in minimum
/// `packedSizeInBitsForDpasB`
static LayoutInfo getSIMTLayoutInfoForDPASOperand(VectorType vectorTy,
                                                  unsigned operandNum) {
  Type elementTy = vectorTy.getElementType();
  assert(elementTy.isIntOrFloat() &&
         "Expected int or float type in DPAS operands");
  LaneLayout layout({1, xegpu::targetinfo::subgroupSize});
  // For B operand, data must be packed in minimum `packedDpasBSizeInBits` and
  // must have the VNNI format.
  if (operandNum == 1 && elementTy.getIntOrFloatBitWidth() <
                             xegpu::targetinfo::packedSizeInBitsForDpasB) {
    LaneData data({xegpu::targetinfo::packedSizeInBitsForDpasB /
                       elementTy.getIntOrFloatBitWidth(),
                   1});
    return LayoutInfo(layout, data);
  }
  // Otherwise, return the default layout for the vector type.
  return getDefaultSIMTLayoutInfo(vectorTy);
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

public:
  LayoutInfoPropagation(DataFlowSolver &solver,
                        SymbolTableCollection &symbolTable)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable) {}
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutInfoLattice *> operands,
                 ArrayRef<const LayoutInfoLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override {};

  void visitCallOperand(OpOperand &operand) override {};

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
      .Case<xegpu::DpasOp>(
          [&](auto dpasOp) { visitDpasOp(dpasOp, operands, results); })
      .Case<xegpu::StoreNdOp>(
          [&](auto storeNdOp) { visitStoreNdOp(storeNdOp, operands, results); })
      .Case<xegpu::StoreScatterOp>([&](auto storeScatterOp) {
        visitStoreScatterOp(storeScatterOp, operands, results);
      })
      .Case<xegpu::LoadNdOp>(
          [&](auto loadNdOp) { visitLoadNdOp(loadNdOp, operands, results); })
      .Case<xegpu::LoadGatherOp>([&](auto loadGatherOp) {
        visitLoadGatherOp(loadGatherOp, operands, results);
      })
      .Case<xegpu::CreateDescOp>([&](auto createDescOp) {
        visitCreateDescOp(createDescOp, operands, results);
      })
      .Case<xegpu::UpdateNdOffsetOp>([&](auto updateNdOffsetOp) {
        visitUpdateNdOffsetOp(updateNdOffsetOp, operands, results);
      })
      .Case<xegpu::PrefetchNdOp>([&](auto prefetchNdOp) {
        visitPrefetchNdOp(prefetchNdOp, operands, results);
      })
      .Case<vector::TransposeOp>([&](auto transposeOp) {
        visitTransposeOp(transposeOp, operands, results);
      })
      .Case<vector::BitCastOp>([&](auto bitcastOp) {
        visitVectorBitcastOp(bitcastOp, operands, results);
      })
      .Case<vector::MultiDimReductionOp>([&](auto reductionOp) {
        visitVectorMultiReductionOp(reductionOp, operands, results);
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

void LayoutInfoPropagation::visitPrefetchNdOp(
    xegpu::PrefetchNdOp prefetch, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Here we assign the default layout to the tensor descriptor operand of
  // prefetch.
  auto tdescTy = prefetch.getTensorDescType();
  auto prefetchLayout = getDefaultSIMTLayoutInfo(tdescTy);
  // Propagate the layout to the source tensor descriptor.
  propagateIfChanged(operands[0], operands[0]->meet(prefetchLayout));
}

void LayoutInfoPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout of the result must be present.
  LayoutInfo resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  // We only consider 2D -> 1D reductions at this point.
  VectorType resultTy = llvm::dyn_cast<VectorType>(reduction.getDestType());
  if (!resultTy || resultTy.getRank() != 1) {
    reduction.emitWarning("Expecting output type to be 1D vector.");
    return;
  }
  // Given that the result is 1D, the layout of the operand should be 2D with
  // default layout.
  LayoutInfo operandLayout = getDefaultSIMTLayoutInfo(2);
  propagateIfChanged(operands[0], operands[0]->meet(operandLayout));
  // Accumulator should have the same layout as the result.
  propagateIfChanged(operands[1], operands[1]->meet(resultLayout));
}

/// Propagate the layout of the result tensor to the source tensor descriptor in
/// UpdateNdOffsetOp.
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
  VectorType aTy = dpas.getLhsType();
  VectorType bTy = dpas.getRhsType();
  propagateIfChanged(
      operands[0], operands[0]->meet(getSIMTLayoutInfoForDPASOperand(aTy, 0)));
  propagateIfChanged(
      operands[1], operands[1]->meet(getSIMTLayoutInfoForDPASOperand(bTy, 1)));
  if (operands.size() > 2) {
    VectorType cTy = dpas.getAccType();
    propagateIfChanged(
        operands[2],
        operands[2]->meet(getSIMTLayoutInfoForDPASOperand(cTy, 2)));
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutInfoPropagation::visitStoreNdOp(
    xegpu::StoreNdOp store, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo storeLayout = getDefaultSIMTLayoutInfo(store.getValueType());
  // Both operands should have the same layout
  for (LayoutInfoLattice *operand : operands)
    propagateIfChanged(operand, operand->meet(storeLayout));
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// LoadNdOp.
void LayoutInfoPropagation::visitLoadNdOp(
    xegpu::LoadNdOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo valueLayout = results[0]->getValue();
  // Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;
  LayoutInfo tensorDescLayout = valueLayout;
  // LoadNdOp has the transpose effect. However, at the stage of this analysis
  // this effect is not expected and should be abstracted away. Emit a
  // warning.
  if (auto transpose = load.getTranspose()) {
    load.emitWarning("Transpose effect is not expected for LoadNdOp at "
                     "LayoutInfoPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout(transpose.value());
  }
  // Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
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
  LayoutInfo newLayout =
      resultLayout.getTransposedLayout(transpose.getPermutation());
  // Propagate the new layout to the vector operand.
  propagateIfChanged(operands[0], operands[0]->meet(newLayout));
}

/// For vector::BitCastOp, the lane_data of the source layout is changed based
/// on the bit width of the source and result types.
void LayoutInfoPropagation::visitVectorBitcastOp(
    vector::BitCastOp bitcast, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Need the layout of bitcast result to propagate to the operands.
  LayoutInfo resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  int inElemTyBitWidth =
      bitcast.getSourceVectorType().getElementType().getIntOrFloatBitWidth();
  int outElemTyBitWidth =
      bitcast.getResultVectorType().getElementType().getIntOrFloatBitWidth();

  // NOTE: We do not expect widening or narrowing bitcasts at this stage. Emit
  // a warning and return.
  if (inElemTyBitWidth != outElemTyBitWidth) {
    bitcast.emitWarning("Widening or narrowing bitcasts are not expected at "
                        "layout propagation stage.");
    return;
  }

  propagateIfChanged(operands[0], operands[0]->meet(resultLayout));
}

/// Propagate the layout of the result to the tensor descriptor, mask and offset
/// operands in LoadGatherOp.
void LayoutInfoPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // The layout is strictly determined by the payload type.
  auto payloadTy = dyn_cast<VectorType>(load.getValueType());
  if (!payloadTy) {
    load.emitWarning("Not propagating, non-vector payload supplied.");
    return;
  }
  LayoutInfo layout = getDefaultSIMTLayoutInfo(payloadTy, /*scattered*/ true);

  // Mask operand should have 1D default layout.
  LayoutInfo maskLayout = getDefaultSIMTLayoutInfo(1);

  // Propagate the new layout to the tensor descriptor operand.
  if (isa<xegpu::TensorDescType>(load.getSourceType()))
    propagateIfChanged(operands[0], operands[0]->meet(layout));
  // Propagate the new layout to the mask and optional offset operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
  if (load.getOffsets())
    propagateIfChanged(operands[2], operands[2]->meet(maskLayout));
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
  // For offset operand propagate 1D default layout.
  LayoutInfo layout = getDefaultSIMTLayoutInfo(1);
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, offset and mask operands in
/// the StoreScatterOp.
void LayoutInfoPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Currently, for 2D StoreScatterOp we expect that the height dimension of
  // the tensor descriptor is equal to the subgroup size. This is ensured by
  // the op verifier.
  auto payloadTy = dyn_cast<VectorType>(storeScatter.getValueType());
  if (!payloadTy) {
    storeScatter.emitWarning("Not propagating, non-vector payload supplied.");
    return;
  }
  auto payloadShape = payloadTy.getShape();
  if (payloadShape.size() > 1)
    assert(
        payloadShape[0] == xegpu::targetinfo::subgroupSize &&
        "Expected the first dimension of 2D tensor descriptor to be equal to "
        "subgroup size.");

  LayoutInfo payloadLayout =
      getDefaultSIMTLayoutInfo(payloadTy, /*scattered=*/true);

  LayoutInfo maskLayout = getDefaultSIMTLayoutInfo(1);
  // Propagate the payload operand layout
  propagateIfChanged(operands[0], operands[0]->meet(payloadLayout));
  // Propagate the destination (if tdesc) operand layout
  if (isa<xegpu::TensorDescType>(storeScatter.getDestType()))
    propagateIfChanged(operands[1], operands[1]->meet(payloadLayout));
  // Propagate the new layout to the mask and optional offset operand.
  propagateIfChanged(operands[2], operands[2]->meet(maskLayout));
  if (storeScatter.getOffsets())
    propagateIfChanged(operands[3], operands[3]->meet(maskLayout));
}

namespace {
//===----------------------------------------------------------------------===//
// RunLayoutInfoPropagation
//===----------------------------------------------------------------------===//

/// Driver class for running the LayoutInfoPropagation analysis.
class RunLayoutInfoPropagation {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunLayoutInfoPropagation)

  RunLayoutInfoPropagation(Operation *op) : target(op) {
    SymbolTableCollection symbolTable;
    loadBaselineAnalyses(solver);
    solver.load<LayoutInfoPropagation>(symbolTable);
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

using GetLayoutFnTy = function_ref<xegpu::LayoutAttr(Value)>;
/// Update an operation with the layout of its results. If the result type is a
/// vector type, a temporary layout attribute is added to the operation. If the
/// result type is a tensor descriptor type, the type is updated with the layout
/// attribute. The users of the result are also updated with the layout
/// attribute.
static LogicalResult updateOp(mlir::OpBuilder &builder, mlir::Operation *op,
                              GetLayoutFnTy getLayoutOfValue) {
  // Region ops (like scf.for) are already handled by the updateControlFlowOps.
  if (mlir::isa<mlir::RegionBranchOpInterface>(op))
    return success();

  // Iterate over all the results.
  for (OpResult result : op->getResults()) {
    Type resultType = result.getType();
    // Layouts are needed only for vector and tensor descriptor types.
    if (!isa<VectorType, xegpu::TensorDescType>(resultType))
      continue;
    // If the result has no layout but has users, emit a warning and continue.
    xegpu::LayoutAttr layout = getLayoutOfValue(result);
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
/// inside. If the blocks have tensor descriptor type as block arguments, thier
/// types must be updated. Also region op can have results that may not have any
/// users (e.g. A and B tiles). They are not assigned a layout by layout
/// analysis because they have no users. However inside the region op
/// corresponding block arguments for these results do have layouts. Therefore,
/// in this case we still need to update the result types with the layout
/// attribute. This function function updates the internal block arguments and
/// the result types of the region op with the assigned layouts.
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
/// (i.e. out types). Note that yield types are updated by respective producers
/// inside bb0.
static LogicalResult
updateControlFlowOps(mlir::OpBuilder &builder,
                     mlir::RegionBranchTerminatorOpInterface terminator,
                     GetLayoutFnTy getLayoutOfValue) {
  // Only process if the terminator is inside a region branch op.
  if (!mlir::isa<mlir::RegionBranchOpInterface>(terminator->getParentOp()))
    return success();

  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(terminator->getNumOperands(),
                                              nullptr);
  terminator.getSuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    mlir::OperandRange successorOperands =
        terminator.getSuccessorOperands(successor);
    mlir::ValueRange successorInputs = successor.getSuccessorInputs();
    for (auto [successorOperand, successorInput] :
         llvm::zip(successorOperands, successorInputs)) {
      Type inputType = successorInput.getType();
      // We only need to operate on tensor descriptor or vector types.
      if (!isa<xegpu::TensorDescType, VectorType>(inputType))
        continue;
      xegpu::LayoutAttr successorInputLayout = getLayoutOfValue(successorInput);
      xegpu::LayoutAttr successorOperandLayout =
          getLayoutOfValue(successorOperand);

      // If either of the layouts is not assigned, we cannot proceed.
      if (!successorOperandLayout) {
        LLVM_DEBUG(
            DBGS()
            << "No layout assigned for forwarded operand in branch terminator: "
            << successorOperand << "\n");
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
    xegpu::LayoutAttr layout = getLayoutOfValue(arg);
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
      : XeGPUPropagateLayoutBase(options) {}
  void runOnOperation() override;
};

} // namespace

void XeGPUPropagateLayoutPass::runOnOperation() {
  auto &analysis = getAnalysis<RunLayoutInfoPropagation>();
  // Print the analysis result and exit. (for debugging purposes)
  if (printOnly) {
    auto &os = llvm::outs();
    analysis.printAnalysisResult(os);
    return;
  }
  // Helper to convert LayoutInfo to xegpu::LayoutAttr.
  auto getXeGPULayoutForValue = [&](Value val) -> xegpu::LayoutAttr {
    LayoutInfo layout = analysis.getLayoutInfo(val);
    if (!layout.isAssigned())
      return {};
    return xegpu::LayoutAttr::get(
        val.getContext(), llvm::to_vector_of<int>(layout.getLayoutAsArrayRef()),
        llvm::to_vector_of<int>(layout.getDataAsArrayRef()));
  };

  mlir::OpBuilder builder(&getContext());
  Operation *op = getOperation();
  auto walkResult = op->walk([&](mlir::Block *block) -> WalkResult {
    for (mlir::Operation &op : llvm::reverse(block->getOperations())) {
      LogicalResult r = success();
      TypeSwitch<Operation *>(&op)
          .Case<mlir::RegionBranchTerminatorOpInterface>(
              [&](mlir::RegionBranchTerminatorOpInterface branchTermOp) {
                r = updateControlFlowOps(builder, branchTermOp,
                                         getXeGPULayoutForValue);
              })
          .Case<mlir::FunctionOpInterface>(
              [&](mlir::FunctionOpInterface funcOp) {
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
  if (walkResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}
