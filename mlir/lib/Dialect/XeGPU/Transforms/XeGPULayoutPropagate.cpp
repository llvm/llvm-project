//===- XeGPULayoutPropagate.cpp - XeGPU Layout Propagation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPULAYOUTPROPAGATE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-layout-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::dataflow;

/// HW dependent constants.
/// TODO: These constants should be queried from the target information.
constexpr unsigned subgroupSize = 16; // How many lanes in a subgroup.
/// If DPAS A or B operands have low precision element types they must be packed
/// according to the following sizes.
constexpr unsigned packedSizeInBitsForDefault =
    16; // Minimum packing size per register for DPAS A.
constexpr unsigned packedSizeInBitsForDpasB =
    32; // Minimum packing size per register for DPAS B.

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
  int64_t operator[](size_t idx) const;
};

void Layout::print(llvm::raw_ostream &os) const {
  os << llvm::interleaved_array(layout);
}

int64_t Layout::operator[](size_t idx) const {
  assert(idx < layout.size() && "Index out of bounds.");
  return layout[idx];
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
static LayoutInfo getDefaultLayoutInfo(unsigned rank) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1)
    return LayoutInfo(LaneLayout({subgroupSize}), LaneData({1}));
  return LayoutInfo(LaneLayout({1, subgroupSize}), LaneData({1, 1}));
}

/// Helper to get the default layout for a vector type.
static LayoutInfo getDefaultLayoutInfo(VectorType vectorTy) {
  // Expecting a 1D or 2D vector.
  assert((vectorTy.getRank() == 1 || vectorTy.getRank() == 2) &&
         "Expected 1D or 2D vector.");
  // Expecting int or float element type.
  assert(vectorTy.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  // If the rank is 1, then return default layout for 1D vector.
  if (vectorTy.getRank() == 1)
    return getDefaultLayoutInfo(1);
  // Packing factor is determined by the element type bitwidth.
  int packingFactor = 1;
  unsigned bitwidth = vectorTy.getElementType().getIntOrFloatBitWidth();
  if (bitwidth < packedSizeInBitsForDefault)
    packingFactor = packedSizeInBitsForDefault / bitwidth;
  return LayoutInfo(LaneLayout({1, subgroupSize}),
                    LaneData({1, packingFactor}));
}

/// Helper Function to get the expected layouts for DPAS operands. `lane_data`
/// is set according to the following criteria:
/// * For A operand, the data must be packed in minimum
/// `packedSizeInBitsForDefault`
/// * For B operand, the data must be packed in minimum
/// `packedSizeInBitsForDpasB`
static LayoutInfo getLayoutInfoForDPASOperand(VectorType vectorTy,
                                              unsigned operandNum) {
  Type elementTy = vectorTy.getElementType();
  assert(elementTy.isIntOrFloat() &&
         "Expected int or float type in DPAS operands");
  LaneLayout layout({1, subgroupSize});
  // For B operand, data must be packed in minimum `packedDpasBSizeInBits` and
  // must have the VNNI format.
  if (operandNum == 1 &&
      elementTy.getIntOrFloatBitWidth() < packedSizeInBitsForDpasB) {
    LaneData data(
        {packedSizeInBitsForDpasB / elementTy.getIntOrFloatBitWidth(), 1});
    return LayoutInfo(layout, data);
  }
  // Otherwise, return the default layout for the vector type.
  return getDefaultLayoutInfo(vectorTy);
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
      // No need to propagate the layout to operands in CreateNdDescOp because
      // they are scalars (offsets, sizes, etc.).
      .Case<xegpu::CreateNdDescOp>([&](auto createNdDescOp) {})
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
        for (const LayoutInfoLattice *r : results) {
          for (LayoutInfoLattice *operand : operands) {
            // Propagate the layout of the result to the operand.
            if (r->getValue().isAssigned())
              meet(operand, *r);
          }
        }
      });
  // Add a dependency from each result to program point after the operation.
  for (const LayoutInfoLattice *r : results) {
    addDependency(const_cast<LayoutInfoLattice *>(r), getProgramPointAfter(op));
  }
  return success();
}

void LayoutInfoPropagation::visitPrefetchNdOp(
    xegpu::PrefetchNdOp prefetch, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Here we assign the default layout to the tensor descriptor operand of
  // prefetch.
  auto tdescTy = prefetch.getTensorDescType();
  auto prefetchLayout = getDefaultLayoutInfo(
      VectorType::get(tdescTy.getShape(), tdescTy.getElementType()));
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
  assert(resultLayout.getLayout().size() == 1 &&
         "Expected 1D layout for reduction result.");
  // Given that the result is 1D, the layout of the operand should be 2D with
  // default layout.
  LayoutInfo operandLayout = getDefaultLayoutInfo(2);
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
  propagateIfChanged(operands[0],
                     operands[0]->meet(getLayoutInfoForDPASOperand(aTy, 0)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(getLayoutInfoForDPASOperand(bTy, 1)));
  if (operands.size() > 2) {
    VectorType cTy = dpas.getAccType();
    propagateIfChanged(operands[2],
                       operands[2]->meet(getLayoutInfoForDPASOperand(cTy, 2)));
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutInfoPropagation::visitStoreNdOp(
    xegpu::StoreNdOp store, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo storeLayout = getDefaultLayoutInfo(store.getValueType());
  // Both operands should have the same layout
  for (LayoutInfoLattice *operand : operands) {
    propagateIfChanged(operand, operand->meet(storeLayout));
  }
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
  // this effect is not expected and should be abstracted away. Emit a warning.
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

  // LaneLayout does not change.
  const LaneLayout &newLaneLayout = resultLayout.getLayout();
  const LaneData &currData = resultLayout.getData();
  LaneData newLaneData;
  // It's a widening bitcast
  if (inElemTyBitWidth < outElemTyBitWidth) {
    int ratio = outElemTyBitWidth / inElemTyBitWidth;
    newLaneData = resultLayout.getData()[0] == 1
                      ? LaneData({1, currData[1] * ratio})
                      : LaneData({currData[0] * ratio, 1});
  } else {
    // It's a narrowing bitcast
    int ratio = inElemTyBitWidth / outElemTyBitWidth;
    newLaneData = resultLayout.getData()[0] == 1
                      ? LaneData({1, currData[1] / ratio})
                      : LaneData({currData[0] / ratio, 1});
  }

  propagateIfChanged(operands[0],
                     operands[0]->meet(LayoutInfo(newLaneLayout, newLaneData)));
}

/// Propagate the layout of the result to the tensor descriptor and mask
/// operands in LoadGatherOp.
void LayoutInfoPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  LayoutInfo valueLayout = results[0]->getValue();
  // Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;

  LayoutInfo tensorDescLayout = valueLayout;
  if (load.getTranspose()) {
    // LoadGatherOp has the transpose effect. However, at the stage of this
    // analyis this effect is not expected and should be abstracted away. Emit
    // a warning.
    load.emitWarning("Transpose effect is not expected for LoadGatherOp at "
                     "LayoutInfoPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout({1, 0});
  }
  // Mask operand should have 1D default layout.
  LayoutInfo maskLayout = getDefaultLayoutInfo(1);
  // Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
  // Propagate the new layout to the mask operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
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
  LayoutInfo layout = getDefaultLayoutInfo(1);
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, and mask operands in the
/// StoreScatterOp.
void LayoutInfoPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  // Currently, for 2D StoreScatterOp we expect that the height dimension of
  // the tensor descriptor is equal to the subgroup size. This is ensured by
  // the op verifier.
  ArrayRef<int64_t> tdescShape = storeScatter.getTensorDescType().getShape();
  if (tdescShape.size() > 1)
    assert(
        tdescShape[0] == subgroupSize &&
        "Expected the first dimension of 2D tensor descriptor to be equal to "
        "subgroup size.");

  LayoutInfo valueLayout = getDefaultLayoutInfo(storeScatter.getValueType());
  LayoutInfo storeScatterLayout = valueLayout;
  if (storeScatter.getTranspose()) {
    // StoreScatteOp allows transpose effect. However, at the stage of this
    // analyis this effect is not expected and should be abstracted away. Emit
    // a warning.
    storeScatter.emitWarning("Transpose effect is not expected for "
                             "StoreScatterOp at LayoutInfoPropagation stage.");
    storeScatterLayout = valueLayout.getTransposedLayout({1, 0});
  }
  // Propagate the value layout.
  propagateIfChanged(operands[0], operands[0]->meet(valueLayout));
  // Propagate the tensor descriptor layout.
  propagateIfChanged(operands[1], operands[1]->meet(storeScatterLayout));
  // Use default 1D layout for mask operand.
  LayoutInfo maskLayout = getDefaultLayoutInfo(1);
  propagateIfChanged(operands[2], operands[2]->meet(maskLayout));
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
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
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
[[maybe_unused]] void
RunLayoutInfoPropagation::printAnalysisResult(llvm::raw_ostream &os) {
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
    for (auto funcOp : modOp.getOps<FunctionOpInterface>()) {
      funcOps.push_back(funcOp);
    }
    // Collect all GpuFuncOps in the module.
    for (auto gpuModOp : modOp.getOps<gpu::GPUModuleOp>()) {
      for (auto gpuFuncOp : gpuModOp.getOps<FunctionOpInterface>()) {
        funcOps.push_back(gpuFuncOp);
      }
    }
  }
  // Print the analysis result for each function.
  for (FunctionOpInterface funcOp : funcOps) {
    printFunctionResult(funcOp);
  }
}

using GetLayoutCallbackFnTy = function_ref<xegpu::LayoutAttr(Value)>;
static void updateOp(mlir::OpBuilder &builder, mlir::Operation *op,
                     GetLayoutCallbackFnTy getLayoutOfValue) {

  // Iterate over all the results.
  for (OpResult result : op->getResults()) {
    Type resultType = result.getType();
    // Layouts are needed only for vector and tensor descriptor types.
    if (!isa<VectorType, xegpu::TensorDescType>(resultType))
      continue;
    // If the result has any users, we expect it to have a layout.
    xegpu::LayoutAttr layout = getLayoutOfValue(result);
    if (!layout && result.getNumUses() > 0) {
      LLVM_DEBUG(DBGS() << "Expecting layout for result: " << result
                        << " but got none.\n");
      continue;
    }
    if (auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(resultType)) {
      // TODO: Handle error.
      auto typeWithLayout = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
      result.setType(typeWithLayout);
      continue;
    }
    // If the result is a vector type, add a temporary layout attribute to the
    // op.
    std::string resultLayoutName = xegpu::getLayoutName(result);
    op->setAttr(resultLayoutName, layout);
    // Update all users of the result with the layout.
    for (OpOperand &user : result.getUses()) {
      Operation *owner = user.getOwner();
      // Add temorary layout attribute at the user op.
      std::string attrName = xegpu::getLayoutName(user);
      owner->setAttr(attrName, layout);
    }
  }
}
static void updateBranchTerminatorOpInterface(
    mlir::OpBuilder &builder,
    mlir::RegionBranchTerminatorOpInterface terminator,
    GetLayoutCallbackFnTy getLayoutOfValue) {
  if (!mlir::isa<mlir::RegionBranchOpInterface>(terminator->getParentOp()))
    return;

  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(terminator->getNumOperands(),
                                              nullptr);
  terminator.getSuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    if (!successor.isParent())
      continue;

    mlir::OperandRange operands = terminator.getSuccessorOperands(successor);
    mlir::ValueRange inputs = successor.getSuccessorInputs();
    for (auto [operand, input] : llvm::zip(operands, inputs)) {
      // print arg and inp
      // llvm::errs() << "arg: " << operand << ", inp: " << input << "\n";
      Type inputType = input.getType();
      if (!isa<xegpu::TensorDescType>(inputType))
        continue;
      xegpu::LayoutAttr inputLayout = getLayoutOfValue(input);
      xegpu::LayoutAttr operandLayout = getLayoutOfValue(operand);

      if (!operandLayout) {
        LLVM_DEBUG(DBGS() << "Expecting layout for region successor operand : "
                          << operand << " but got none.\n");
        continue;
      }

      if (inputLayout && inputLayout != operandLayout) {
        LLVM_DEBUG(
            DBGS()
            << "Conflicting layouts for region successor operand and input: "
            << inputLayout << " vs " << operandLayout << "\n");
        continue;
      }
      // Get tensor descriptor type with the layout.
      auto tdescTy = dyn_cast<xegpu::TensorDescType>(inputType);
      auto newTdescTy = xegpu::TensorDescType::get(
          tdescTy.getContext(), tdescTy.getShape(), tdescTy.getElementType(),
          tdescTy.getEncoding(), operandLayout);
      input.setType(newTdescTy);
    }
  }
}
static void updateBranchOpInterface(mlir::OpBuilder &builder,
                                    mlir::RegionBranchOpInterface branch,
                                    GetLayoutCallbackFnTy getLayoutOfValue) {
  mlir::Operation *op = branch.getOperation();
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(op->getNumOperands(), nullptr);
  branch.getEntrySuccessorRegions(operands, successors);
  DenseMap<Value, xegpu::LayoutAttr> resultToLayouts;
  mlir::ValueRange results = op->getResults();

  for (mlir::RegionSuccessor &successor : successors) {
    if (successor.isParent())
      continue;

    mlir::OperandRange operands = branch.getEntrySuccessorOperands(successor);
    mlir::ValueRange inputs = successor.getSuccessorInputs();

    for (auto [operand, input, result] : llvm::zip(operands, inputs, results)) {
      Type inputType = input.getType();
      if (!isa<xegpu::TensorDescType>(inputType))
        continue;
      xegpu::LayoutAttr blockArgLayout = getLayoutOfValue(input);
      xegpu::LayoutAttr initArgLayout = getLayoutOfValue(operand);

      if (!blockArgLayout || !initArgLayout) {
        LLVM_DEBUG(DBGS() << "No layout assigned for block arg: " << input
                          << " or init arg: " << operand << "\n");
        continue;
      }

      // TOOD: We expect these two to match. Data flow analysis will ensure
      // this.
      assert(blockArgLayout == initArgLayout &&
             "Expexing block arg and init arg to have the same layout.");
      // Get tensor descriptor type with the layout.
      auto tdescTy = dyn_cast<xegpu::TensorDescType>(inputType);
      auto newTdescTy = xegpu::TensorDescType::get(
          tdescTy.getContext(), tdescTy.getShape(), tdescTy.getElementType(),
          tdescTy.getEncoding(), blockArgLayout);
      input.setType(newTdescTy);
      // Store the layout for the result.
      if (resultToLayouts.count(result) != 0 &&
          resultToLayouts[result] != blockArgLayout) {
        LLVM_DEBUG(DBGS() << "Conflicting layouts for result: " << result
                          << " - " << resultToLayouts[result] << " vs "
                          << blockArgLayout << "\n");
      } else {
        resultToLayouts[result] = blockArgLayout;
      }
    }
  }
  for (auto [i, r] : llvm::enumerate(op->getResults())) {
    Type resultType = r.getType();
    if (!isa<xegpu::TensorDescType, VectorType>(resultType))
      continue;
    xegpu::LayoutAttr layout = getLayoutOfValue(r);
    if (!layout)
      layout = resultToLayouts[r];
    if (!layout) {
      LLVM_DEBUG(DBGS() << "No layout assigned for vector/tensor desc result:"
                        << r << "\n");
      continue;
    }
    if (auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(resultType)) {
      auto newTdescTy = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
      r.setType(newTdescTy);
      continue;
    }
    // If the result is a vector type, add a temporary layout attribute to
    // the op.
    std::string resultLayoutName = xegpu::getLayoutName(r);
    op->setAttr(resultLayoutName, layout);
    // Update all users of the result with the layout.
    for (OpOperand &user : r.getUses()) {
      Operation *owner = user.getOwner();
      // Add temporary layout attribute at the user op.
      std::string attrName = xegpu::getLayoutName(user);
      owner->setAttr(attrName, layout);
    }
  }
}

static void updateFunctionOpInterface(mlir::OpBuilder &builder,
                                      mlir::FunctionOpInterface funcOp,
                                      GetLayoutCallbackFnTy getLayoutOfValue) {
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
      continue;
    }
    if (auto tensorDescTy = dyn_cast<xegpu::TensorDescType>(argType)) {
      auto newTdescTy = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(), layout);
      arg.setType(newTdescTy);
      newArgTypes.back() = newTdescTy;
      continue;
    }
    // If the argument is a vector type, update all the users of the argument
    // with the layout.
    for (OpOperand &user : arg.getUses()) {
      Operation *owner = user.getOwner();
      std::string attrName = xegpu::getLayoutName(user);
      owner->setAttr(attrName, layout);
    }
  }
  // Update the function type with the new argument types.
  // NOTE: We assume that function results are not expected to have layouts.
  funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                   funcOp.getResultTypes()));
}

namespace {

struct XeGPULayoutPropagatePass final
    : public xegpu::impl::XeGPULayoutPropagateBase<XeGPULayoutPropagatePass> {
  void runOnOperation() override;
};

} // namespace

void XeGPULayoutPropagatePass::runOnOperation() {
  auto &analyis = getAnalysis<RunLayoutInfoPropagation>();

  auto getXeGPULayoutForValue = [&](Value val) -> xegpu::LayoutAttr {
    LayoutInfo layout = analyis.getLayoutInfo(val);
    if (!layout.isAssigned()) {
      return {};
    }
    SmallVector<int, 2> laneLayout, laneData;
    for (auto [layout, data] : llvm::zip_equal(layout.getLayoutAsArrayRef(),
                                               layout.getDataAsArrayRef())) {
      laneLayout.push_back(static_cast<int>(layout));
      laneData.push_back(static_cast<int>(data));
    }
    return xegpu::LayoutAttr::get(val.getContext(), laneLayout, laneData);
  };

  mlir::OpBuilder builder(&getContext());
  Operation *op = getOperation();
  op->walk([&](mlir::Block *block) {
    for (mlir::Operation &op : llvm::reverse(block->getOperations())) {
      if (auto branchTermOp =
              mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(op)) {
        updateBranchTerminatorOpInterface(builder, branchTermOp,
                                          getXeGPULayoutForValue);
        continue;
      }

      if (auto regionBrOp = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op)) {
        updateBranchOpInterface(builder, regionBrOp, getXeGPULayoutForValue);
        continue;
      }

      if (auto funcOp = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
        updateFunctionOpInterface(builder, funcOp, getXeGPULayoutForValue);
        continue;
      }
      updateOp(builder, &op, getXeGPULayoutForValue);
    }
  });
}
