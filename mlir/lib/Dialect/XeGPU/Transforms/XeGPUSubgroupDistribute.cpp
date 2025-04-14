//===- XeGPUSubgroupDistribute.cpp - XeGPU Subgroup Distribute Pass -------===//
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
#include "mlir/Dialect/GPU/Utils/DistributionUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSUBGROUPDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-subgroup-distribute"
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
static const char *const operandLayoutNamePrefix = "layout_operand_";
static const char *const resultLayoutNamePrefix = "layout_result_";

namespace {

///===----------------------------------------------------------------------===///
/// Layout
///===----------------------------------------------------------------------===///

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
  os << "[";
  llvm::interleaveComma(layout, os);
  os << "]";
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

///===----------------------------------------------------------------------===///
/// LayoutInfo
///===----------------------------------------------------------------------===///

/// Helper class for tracking the analysis state of a value. For SGPropagation,
/// the analysis state is simply the lane_layout and lane_data of each value.
/// Purpose of this analysis to propagate some unique layout for each value in
/// the program starting from some known values (like DPAS, StoreNd, etc.).
///
/// Given this, LayoutInfo satisifies the following properties:
///  1) LayoutInfo is a lattice with two states - assigned and not assigned.
///  2) Two LayoutInfo values are equal if they are both assigned or both not
///  assigned. The concrete value of assigned state does not matter.
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

  /// Two lattice values are equal if they have `some` layout. The actual
  /// content of the layout does not matter.
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
  } else
    os << "Not assigned.";
}

LayoutInfo LayoutInfo::meet(const LayoutInfo &lhs, const LayoutInfo &rhs) {
  if (!lhs.isAssigned())
    return rhs;
  return lhs;
}

/// Since this is a backward analysis, join method is not used.
LayoutInfo LayoutInfo::join(const LayoutInfo &lhs, const LayoutInfo &rhs) {
  llvm_unreachable("Join should not be triggered by SGMapPropagation.");
}

/// Get the transposed layout according to the given permutation.
LayoutInfo
LayoutInfo::getTransposedLayout(ArrayRef<int64_t> permutation) const {
  if (!isAssigned())
    return {};
  LaneLayout newLayout;
  LaneData newData;
  for (auto idx : permutation) {
    newLayout.layout.push_back(laneLayout.layout[idx]);
    newData.layout.push_back(laneData.layout[idx]);
  }
  return LayoutInfo(newLayout, newData);
}

///===----------------------------------------------------------------------===///
/// LayoutInfoLattice
///===----------------------------------------------------------------------===///

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
  /// Expecting a 1D or 2D vector.
  assert((vectorTy.getRank() == 1 || vectorTy.getRank() == 2) &&
         "Expected 1D or 2D vector.");
  /// Expecting int or float element type.
  assert(vectorTy.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  /// If the rank is 1, then return default layout for 1D vector.
  if (vectorTy.getRank() == 1)
    return getDefaultLayoutInfo(1);
  /// Packing factor is determined by the element type bitwidth.
  int packingFactor = 1;
  auto bitwidth = vectorTy.getElementType().getIntOrFloatBitWidth();
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
  auto elementTy = vectorTy.getElementType();
  assert(elementTy.isIntOrFloat() &&
         "Expected int or float type in DPAS operands");
  LaneLayout layout({1, subgroupSize});
  /// For B operand, data must be packed in minimum `packedDpasBSizeInBits` and
  /// must have the VNNI format.
  if (operandNum == 1 &&
      elementTy.getIntOrFloatBitWidth() < packedSizeInBitsForDpasB) {
    LaneData data(
        {packedSizeInBitsForDpasB / elementTy.getIntOrFloatBitWidth(), 1});
    return LayoutInfo(layout, data);
  }
  /// Otherwise, return the default layout for the vector type.
  return getDefaultLayoutInfo(vectorTy);
}

///===----------------------------------------------------------------------===///
/// LayoutInfoPropagation
///===----------------------------------------------------------------------===///

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
      /// No need to propagate the layout to operands in CreateNdDescOp because
      /// they are scalars (offsets, sizes, etc.).
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
      /// All other ops.
      .Default([&](Operation *op) {
        for (const LayoutInfoLattice *r : results) {
          for (LayoutInfoLattice *operand : operands) {
            /// Propagate the layout of the result to the operand.
            if (r->getValue().isAssigned())
              meet(operand, *r);
          }
        }
      });
  /// Add a dependency from each result to program point after the operation.
  for (const LayoutInfoLattice *r : results) {
    addDependency(const_cast<LayoutInfoLattice *>(r), getProgramPointAfter(op));
  }
  return success();
}

void LayoutInfoPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  /// The layout of the result must be present.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  /// We only consider 2D -> 1D reductions at this point.
  assert(resultLayout.getLayout().size() == 1 &&
         "Expected 1D layout for reduction result.");
  /// Given that the result is 1D, the layout of the operand should be 2D with
  /// default layout.
  auto operandLayout = getDefaultLayoutInfo(2);
  propagateIfChanged(operands[0], operands[0]->meet(operandLayout));
  /// Accumulator should have the same layout as the result.
  propagateIfChanged(operands[1], operands[1]->meet(resultLayout));
}

/// Propagate the layout of the result tensor to the source tensor descriptor in
/// UpdateNdOffsetOp.
void LayoutInfoPropagation::visitUpdateNdOffsetOp(
    xegpu::UpdateNdOffsetOp updateNdOffset,
    ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  /// The layout of the result must be present.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(resultLayout));
}

/// Set the layouts for DPAS A, B, and C operands.
void LayoutInfoPropagation::visitDpasOp(
    xegpu::DpasOp dpas, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  auto aTy = dpas.getLhsType();
  auto bTy = dpas.getRhsType();
  propagateIfChanged(operands[0],
                     operands[0]->meet(getLayoutInfoForDPASOperand(aTy, 0)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(getLayoutInfoForDPASOperand(bTy, 1)));
  if (operands.size() > 2) {
    auto cTy = dpas.getAccType();
    propagateIfChanged(operands[2],
                       operands[2]->meet(getLayoutInfoForDPASOperand(cTy, 2)));
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void LayoutInfoPropagation::visitStoreNdOp(
    xegpu::StoreNdOp store, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  auto storeLayout = getDefaultLayoutInfo(store.getValueType());
  /// Both operands should have the same layout
  for (LayoutInfoLattice *operand : operands) {
    propagateIfChanged(operand, operand->meet(storeLayout));
  }
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// LoadNdOp.
void LayoutInfoPropagation::visitLoadNdOp(
    xegpu::LoadNdOp load, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;
  LayoutInfo tensorDescLayout = valueLayout;
  /// LoadNdOp has the transpose effect. However, at the stage of this analysis
  /// this effect is not expected and should be abstracted away. Emit a warning.
  if (auto transpose = load.getTranspose()) {
    load.emitWarning("Transpose effect is not expected for LoadNdOp at "
                     "LayoutInfoPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout(transpose.value());
  }
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
}

/// For vector::TransposeOp, the layout of the result is transposed and
/// propagated to the operand.
void LayoutInfoPropagation::visitTransposeOp(
    vector::TransposeOp transpose, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  /// Need the layout of transpose result to propagate to the operands.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  auto newLayout = resultLayout.getTransposedLayout(transpose.getPermutation());
  /// Propagate the new layout to the vector operand.
  propagateIfChanged(operands[0], operands[0]->meet(newLayout));
}

/// For vector::BitCastOp, the lane_data of the source layout is changed based
/// on the bit width of the source and result types.
void LayoutInfoPropagation::visitVectorBitcastOp(
    vector::BitCastOp bitcast, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  /// Need the layout of bitcast result to propagate to the operands.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  auto inElemTyBitWidth =
      bitcast.getSourceVectorType().getElementType().getIntOrFloatBitWidth();
  auto outElemTyBitWidth =
      bitcast.getResultVectorType().getElementType().getIntOrFloatBitWidth();

  /// LaneLayout does not change.
  const LaneLayout &newLaneLayout = resultLayout.getLayout();
  const LaneData &currData = resultLayout.getData();
  LaneData newLaneData;
  /// It's a widening bitcast
  if (inElemTyBitWidth < outElemTyBitWidth) {
    auto ratio = outElemTyBitWidth / inElemTyBitWidth;
    newLaneData = resultLayout.getData()[0] == 1
                      ? LaneData({1, currData[1] * ratio})
                      : LaneData({currData[0] * ratio, 1});
  } else {
    /// It's a narrowing bitcast
    auto ratio = inElemTyBitWidth / outElemTyBitWidth;
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
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;

  LayoutInfo tensorDescLayout = valueLayout;
  if (load.getTranspose()) {
    /// LoadGatherOp has the transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    load.emitWarning("Transpose effect is not expected for LoadGatherOp at "
                     "LayoutInfoPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout({1, 0});
  }
  /// Mask operand should have 1D default layout.
  auto maskLayout = getDefaultLayoutInfo(1);
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
  /// Propagate the new layout to the mask operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
}

/// Propagate the layout of the descriptor to the vector offset operand in
/// CreateDescOp.
void LayoutInfoPropagation::visitCreateDescOp(
    xegpu::CreateDescOp createDesc, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// For offset operand propagate 1D default layout.
  LayoutInfo layout = getDefaultLayoutInfo(1);
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, and mask operands in the
/// StoreScatterOp.
void LayoutInfoPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<LayoutInfoLattice *> operands,
    ArrayRef<const LayoutInfoLattice *> results) {
  /// Currently, for 2D StoreScatterOp we expect that the height dimension of
  /// the tensor descriptor is evenly divisible by the subgroup size.
  /// TODO: Add support for other 2D shapes.
  auto tdescShape = storeScatter.getTensorDescType().getShape();
  if (tdescShape.size() > 1 && tdescShape[0] % subgroupSize != 0) {
    storeScatter.emitError("Height dimension of the tensor descriptor should "
                           "be evenly divisible by the subgroup size.");
    return;
  }
  auto valueLayout = getDefaultLayoutInfo(storeScatter.getValueType());
  LayoutInfo storeScatterLayout = valueLayout;
  if (storeScatter.getTranspose()) {
    /// StoreScatteOp allows transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    storeScatter.emitWarning("Transpose effect is not expected for "
                             "StoreScatterOp at LayoutInfoPropagation stage.");
    storeScatterLayout = valueLayout.getTransposedLayout({1, 0});
  }
  /// Propagate the value layout.
  propagateIfChanged(operands[0], operands[0]->meet(valueLayout));
  /// Propagate the tensor descriptor layout.
  propagateIfChanged(operands[1], operands[1]->meet(storeScatterLayout));
  /// Use default 1D layout for mask operand.
  auto maskLayout = getDefaultLayoutInfo(1);
  propagateIfChanged(operands[2], operands[2]->meet(maskLayout));
}

namespace {

///===----------------------------------------------------------------------===///
/// RunLayoutInfoPropagation
///===----------------------------------------------------------------------===///

/// Driver class for running the LayoutInfoPropagation analysis.
class RunLayoutInfoPropagation {
public:
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

void RunLayoutInfoPropagation::printAnalysisResult(llvm::raw_ostream &os) {
  auto printFunctionResult = [&](FunctionOpInterface funcOp) {
    os << "function: " << funcOp.getName() << ":\n";
    // Function arguments
    for (auto arg : funcOp.getArguments()) {
      auto layout = getLayoutInfo(arg);
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
      /// For control-flow ops, print the op name only.
      if (isa<BranchOpInterface>(op) || isa<RegionBranchOpInterface>(op))
        os << op->getName();
      else
        op->print(os);
      os << "\n";
      /// Print the layout for each result.
      for (auto [i, r] : llvm::enumerate(op->getResults())) {
        auto layout = getLayoutInfo(r);
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
    /// Collect all GpuFuncOps in the module.
    for (auto gpuModOp : modOp.getOps<gpu::GPUModuleOp>()) {
      for (auto gpuFuncOp : gpuModOp.getOps<FunctionOpInterface>()) {
        funcOps.push_back(gpuFuncOp);
      }
    }
  }
  /// Print the analysis result for each function.
  for (auto funcOp : funcOps) {
    printFunctionResult(funcOp);
  }
}

void attachLayoutAttributeToUsers(Value v, xegpu::LayoutAttr layout) {
  for (OpOperand &user : v.getUses()) {
    Operation *owner = user.getOwner();
    unsigned operandNumber = user.getOperandNumber();
    /// Use a generic name for ease of querying the layout attribute later.
    std::string attrName =
        operandLayoutNamePrefix + std::to_string(operandNumber);
    owner->setAttr(attrName, layout);
  }
}

static LogicalResult attachLayoutAttributes(
    Operation *top, llvm::function_ref<LayoutInfo(Value)> getPropagatedLayout) {
  /// Helper to convert the layout info to the xegpu::LayoutAttr.
  auto getLayoutInfoForResult = [&](Value r) -> xegpu::LayoutAttr {
    auto layout = getPropagatedLayout(r);
    if (!layout.isAssigned())
      return {};
    SmallVector<int, 2> laneLayout, laneData;
    for (auto [layout, data] : llvm::zip_equal(layout.getLayoutAsArrayRef(),
                                               layout.getDataAsArrayRef())) {
      laneLayout.push_back(static_cast<int>(layout));
      laneData.push_back(static_cast<int>(data));
    }
    return xegpu::LayoutAttr::get(r.getContext(), laneLayout, laneData);
  };
  /// Attach the layout attributes to the results of the operations.
  auto walkResult = top->walk([&](Operation *op) {
    /// For function ops, propagate the argument layout to the users.
    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      for (auto arg : func.getArguments()) {
        auto layoutInfo = getLayoutInfoForResult(arg);
        if (layoutInfo) {
          attachLayoutAttributeToUsers(arg, layoutInfo);
        }
      }
      return WalkResult::advance();
    }
    /// If no results, move on.
    if (op->getNumResults() == 0)
      return WalkResult::advance();
    if (auto tensorDescTy =
            dyn_cast<xegpu::TensorDescType>(op->getResult(0).getType())) {
      auto layoutInfo = getLayoutInfoForResult(op->getResult(0));
      if (!layoutInfo) {
        LLVM_DEBUG(DBGS() << "No layout for result of " << *op << "\n");
        return WalkResult::interrupt();
      }

      /// Clone the op, attach the sg_map to the result tensor descriptor, and
      /// remove the original op.
      OpBuilder builder(op);
      auto *newOp = builder.clone(*op);
      auto newTensorDescTy = xegpu::TensorDescType::get(
          tensorDescTy.getContext(), tensorDescTy.getShape(),
          tensorDescTy.getElementType(), tensorDescTy.getEncoding(),
          layoutInfo);
      newOp->getResult(0).setType(newTensorDescTy);
      op->replaceAllUsesWith(newOp->getResults());
      op->erase();
      return WalkResult::advance();
    }
    /// Otherwise simply attach the sg_map to the op itself.
    for (auto [i, r] : llvm::enumerate(op->getResults())) {
      auto layoutInfo = getLayoutInfoForResult(r);
      if (layoutInfo) {
        auto attrName = resultLayoutNamePrefix + std::to_string(i);
        op->setAttr(attrName, layoutInfo);
        /// Attach the layout attribute to the users of the result.
        attachLayoutAttributeToUsers(r, layoutInfo);
      }
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

static LogicalResult resolveLayoutConflicts(Operation *top) {
  /// TODO: Implement the layout conflict resolution.
  return success();
}

namespace {

///===----------------------------------------------------------------------===///
/// SIMT Distribution Patterns
///===----------------------------------------------------------------------===///

/// Returns the distributed vector type for a source vector type according to
/// the lane_layout. We simply divide each dimension of tensor descriptor shape
/// by corresponding lane_layout dimension. If array_length > 1, that is
/// appended to the front of the disributed shape.
///
/// Examples:
/// | original vector shape | lane_layout | distributed vector shape |
/// |-----------------------|-------------|--------------------------|
/// | 32x16                 | [1, 16]     | 32x1                     |
/// | 32x16                 | [2, 8]      | 16x2                     |
/// | 2x32x16               | [1, 16]     | 2x32x1                   |
FailureOr<VectorType> getDistVecTypeBasedOnLaneLayout(xegpu::LayoutAttr layout,
                                                      VectorType originalType) {
  if (!layout)
    return failure();

  auto laneLayout = layout.getLaneLayout().asArrayRef();
  assert(originalType.getShape().size() >= laneLayout.size() &&
         "Rank of the original vector type should be greater or equal to the "
         "size of the lane layout to distribute the vector type.");
  SmallVector<int64_t> distributedShape(originalType.getShape());
  /// Only distribute the last `laneLayout.size()` dimensions. The remaining
  /// dimensions are not distributed.
  unsigned distributionStart = originalType.getRank() - laneLayout.size();
  for (auto [i, dim] : llvm::enumerate(originalType.getShape())) {
    if (i < distributionStart) {
      continue;
    }
    /// Check if the dimension can be distributed evenly.
    if (dim % laneLayout[i - distributionStart] != 0)
      return failure();
    distributedShape[i] = dim / laneLayout[i - distributionStart];
  }
  return VectorType::get(distributedShape, originalType.getElementType());
}

static VectorType getDistributedVectorType(xegpu::LayoutAttr layout,
                                           VectorType originalType) {
  auto shape = originalType.getShape();
  auto distVecTyOrFailure =
      xegpu::TensorDescType::get(shape, originalType.getElementType(),
                                 /*array_length=*/1, /*boundary_check=*/true,
                                 /*memory_space=*/xegpu::MemorySpace::Global,
                                 layout)
          .getDistributedVectorType();
  assert(llvm::succeeded(distVecTyOrFailure) &&
         "Failed to compute distributed vector type for the given vector type");
  return distVecTyOrFailure.value();
}

static xegpu::TensorDescType dropLayouts(xegpu::TensorDescType tensorDesc) {
  return xegpu::TensorDescType::get(
      tensorDesc.getContext(), tensorDesc.getShape(),
      tensorDesc.getElementType(), tensorDesc.getEncoding(),
      xegpu::LayoutAttr());
}

template <typename T>
static Value resolveDistributedTy(Value orig, T expected,
                                  PatternRewriter &rewriter) {
  /// If orig and expected types are the same, return orig.
  if (orig.getType() == expected)
    return orig;
  /// If orig is a vector type, create a shape cast op to reconcile the types.
  if (auto origVecType = isa<VectorType>(orig.getType())) {
    auto castOp =
        rewriter.create<vector::ShapeCastOp>(orig.getLoc(), expected, orig);
    return castOp.getResult();
  }
  /// If orig is a tensor descriptor type, create an unrealized conversion cast
  /// op to reconcile the types.
  if (auto origTensorDescTy = isa<xegpu::TensorDescType>(orig.getType())) {
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(orig.getLoc(),
                                                              expected, orig);
    return castOp.getResult(0);
  }
  llvm_unreachable("Unsupported type for reconciliation");
  return orig;
}

// static Value reconcileDistributedTensorDescTy(Value orig,
//                                               xegpu::TensorDescType expected,
//                                               PatternRewriter &rewriter) {
//   assert(isa<xegpu::TensorDescType>(orig.getType()) &&
//          "expecting tensor descriptor type");
//   auto origTensorDescTy = cast<xegpu::TensorDescType>(orig.getType());
//   /// No need to reconcile if the types are the same.
//   if (origTensorDescTy == expected)
//     return orig;
//   auto castOp = rewriter.create<UnrealizedConversionCastOp>(orig.getLoc(),
//                                                             expected, orig);
//   return castOp.getResult(0);
// }

// // unify above 2 functions with a template
// template <typename T>
// static Value reconcileDistributedType(Value orig, T expected,
//                                        PatternRewriter &rewriter) {
//   if constexpr (std::is_same_v<T, VectorType>) {
//     return reconcileDistributedVecType(orig, expected, rewriter);
//   } else if constexpr (std::is_same_v<T, xegpu::TensorDescType>) {
//     return reconcileDistributedTensorDescTy(orig, expected, rewriter);
//   } else {
//     static_assert(llvm::is_one_of<T, VectorType,
//     xegpu::TensorDescType>::value,
//                   "Unsupported type for reconciliation");
//   }
//   return orig;
// }

static SmallVector<NamedAttribute>
filterTemporaryLayoutAttributes(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> newAttrs;
  for (auto attr : attrs) {
    if (attr.getName().strref().contains(operandLayoutNamePrefix) ||
        attr.getName().strref().contains(resultLayoutNamePrefix)) {
      continue;
    }
    newAttrs.push_back(attr);
  }
  return newAttrs;
}

/// Given a GPUFuncOp, this pattern creates a new GPUFuncOp and moves the body
/// of the original GPUFuncOp to the new GPUFuncOp such that entire body is
/// contained within a WarpExecuteOnLane0Op.
/// Example:
///
/// ```
///   gpu.func @foo(%arg0: memref<*xf16>) -> vector<8x16xf32> {
///     ...
///     ...
///     gpu.return %result: vector<8x16xf32>
///   }
/// ```
/// To
/// ```
///   gpu.func @foo(%arg0: memref<*xf16>) -> vector<8x16xf32> {
///     %laneid = gpu.lane_id : index
///     %0 = gpu.warp_execute_on_lane_0(%laneid) -> vector<8x16xf32> {
///       ...
///       ...
///       gpu.yield %result: vector<8x16xf32>
///     }
///     return %0
///   }
struct MoveFuncBodyToWarpExecuteOnLane0
    : public OpRewritePattern<gpu::GPUFuncOp> {
  using OpRewritePattern<gpu::GPUFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::GPUFuncOp gpuFuncOp,
                                PatternRewriter &rewriter) const override {
    /// If the function only contains a single void return, skip.
    if (llvm::all_of(gpuFuncOp.getBody().getOps(), [](Operation &op) {
          return isa<gpu::ReturnOp>(op) && !op.getNumOperands();
        }))
      return failure();
    /// If the function already moved inside a warp_execute_on_lane0, skip.
    if (llvm::any_of(gpuFuncOp.getBody().getOps(), [](Operation &op) {
          return isa<gpu::WarpExecuteOnLane0Op>(op);
        }))
      return failure();
    /// Create a new function with the same signature.
    auto newGpuFunc = rewriter.create<gpu::GPUFuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), gpuFuncOp.getFunctionType());
    /// Create a WarpExecuteOnLane0Op with same arguments and results as the
    /// original gpuFuncOp.
    rewriter.setInsertionPointToEnd(&newGpuFunc.getFunctionBody().front());
    auto laneId = rewriter.create<gpu::LaneIdOp>(
        newGpuFunc.getLoc(), rewriter.getIndexType(),
        /** upperBound = **/ mlir::IntegerAttr());
    auto gpuFuncResultType = gpuFuncOp.getFunctionType().getResults();
    auto warpOp = rewriter.create<gpu::WarpExecuteOnLane0Op>(
        laneId.getLoc(), gpuFuncResultType, laneId, subgroupSize,
        newGpuFunc.getArguments(), newGpuFunc.getArgumentTypes());
    auto &warpBodyBlock = warpOp.getBodyRegion().front();
    /// Replace the ReturnOp of the original gpu function with a YieldOp.
    auto origRetunOp =
        cast<gpu::ReturnOp>(gpuFuncOp.getBlocks().back().getTerminator());
    rewriter.setInsertionPointAfter(origRetunOp);
    rewriter.create<gpu::YieldOp>(origRetunOp.getLoc(),
                                  origRetunOp.getOperands());
    rewriter.eraseOp(origRetunOp);
    /// Move the original function body to the WarpExecuteOnLane0Op body.
    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), warpOp.getBodyRegion(),
                                warpOp.getBodyRegion().begin());
    rewriter.eraseBlock(&warpBodyBlock);
    /// Insert a new ReturnOp after the WarpExecuteOnLane0Op.
    rewriter.setInsertionPointAfter(warpOp);
    rewriter.create<gpu::ReturnOp>(newGpuFunc.getLoc(), warpOp.getResults());
    rewriter.replaceOp(gpuFuncOp, newGpuFunc);
    return success();
  }
};

/// Clone a create_nd_tdesc feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op. The warp op
/// will still contain the original op that will not be used by the yield op
/// (and should be cleaned up later with dce). The yield op will bypass the
/// create_nd_tdesc's arguments. Tensor descriptor is not distributed because
/// it is a uniform value accorss all work items within the subgroup.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (!xegpu.tensor_desc<4x8xf32>) {
///     ...
///     %td = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %td
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.create_nd_tdesc %arg0[0, 0]
///               : memref<4x8xf32> -> !xegpu.tensor_desc<4x8xf32>
///     vector.yield %arg0, %dead
///   }
///   %td = xegpu.create_nd_tdesc %r#0[0, 0]: memref<4x8xf32>
///                                 -> !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct CreateNdDescDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(subgroupOp, llvm::IsaPred<xegpu::CreateNdDescOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          subgroupOp, "warp result is not a xegpu::CreateNdDesc op");
    auto descOp = operand->get().getDefiningOp<xegpu::CreateNdDescOp>();
    unsigned operandIdx = operand->getOperandNumber();

    auto srcTypedVal = dyn_cast<TypedValue<MemRefType>>(descOp.getSource());
    if (!srcTypedVal)
      return rewriter.notifyMatchFailure(
          descOp, "expecting a memref typed value as the source");

    auto descOffsets = descOp.getMixedOffsets();

    xegpu::LayoutAttr layout = descOp.getType().getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          descOp, "the tensor descriptor lacks sg_map attribute");

    SmallVector<size_t> newRetIndices;
    SmallVector<Value> newYieldValues;
    SmallVector<Type> newYieldTypes;

    for (auto arg : descOp->getOperands()) {
      newYieldValues.push_back(arg);
      newYieldTypes.push_back(arg.getType());
    }
    rewriter.setInsertionPoint(subgroupOp);
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, subgroupOp, /* new yieled values = */ newYieldValues,
        /* new yielded types = */ newYieldTypes, newRetIndices);

    SmallVector<Value> newDescOperands;
    for (auto i : newRetIndices) {
      newDescOperands.push_back(newWarpOp.getResult(i));
    }
    rewriter.setInsertionPointAfter(newWarpOp);
    auto distributedTensorDescTy =
        dropLayouts(descOp.getType()); /// Distributed tensor descriptor type
                                       /// does not contain layout info.
    auto newDescOp = rewriter.create<xegpu::CreateNdDescOp>(
        newWarpOp.getLoc(), distributedTensorDescTy, newDescOperands,
        descOp->getAttrs());

    Value distributedVal = newWarpOp.getResult(operandIdx);
    rewriter.replaceAllUsesWith(distributedVal, newDescOp);
    return success();
  }
};

/// Sink a store_nd op at the end of enclosing `gpu.warp_execute_on_lane_0`.
/// In case arguments for the store are passed through the warp op interface
/// they would be propagated as returned values. Only the source vector for
/// the store is distributed according to sg_map attribute.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     xegpu.store_nd %arg0, %arg1: vector<4x8xf32>,
///                                 !xegpu.tensor_desc<4x8xf32>
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     gpu.yield %arg0, %arg1: vector<4x8xf32>, !xegpu.tensor_desc<4x8xf32>
///   }
///   xegpu.store_nd %r#0, %r#1: vector<4x1xf32>,
///     !xegpu.tensor_desc<4x8xf32>
///
/// ```
struct StoreNdDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<gpu::YieldOp>(
        subgroupOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Operation *lastNode = yield->getPrevNode();
    auto storeOp = dyn_cast_or_null<xegpu::StoreNdOp>(lastNode);
    if (!storeOp)
      return failure();

    auto tensorDescTy = storeOp.getTensorDescType();
    xegpu::LayoutAttr layout = tensorDescTy.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          storeOp, "the source tensor descriptor lacks sg_map attribute");

    auto distributedTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, storeOp.getValueType());
    if (failed(distributedTypeByWarpOpOrFailure))
      return rewriter.notifyMatchFailure(storeOp,
                                         "Failed to distribute the type");
    VectorType distributedTypeByWarpOp =
        distributedTypeByWarpOpOrFailure.value();

    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, subgroupOp,
        /* new yielded values = */
        ValueRange{storeOp.getValue(), storeOp.getTensorDesc()},
        /* new yielded types = */
        TypeRange{distributedTypeByWarpOp, storeOp.getTensorDescType()},
        newRetIndices);
    /// Create a new store op outside the warp op with the distributed vector
    /// type. Tensor descriptor is not distributed.
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newStoreOperands;

    /// For the value operand, there can be a mismatch between the vector type
    /// distributed by the warp op and (xegpu-specific) distributed type
    /// supported by the store op. Type mismatch must be resolved using
    /// appropriate cast op.
    auto storeNdDistributedValueTyOrFailure =
        storeOp.getTensorDescType().getDistributedVectorType();
    if (failed(storeNdDistributedValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          storeOp, "Failed to get distributed vector type for the store op");
    newStoreOperands.push_back(resolveDistributedTy(
        newWarpOp.getResult(newRetIndices[0]),
        storeNdDistributedValueTyOrFailure.value(), rewriter));
    /// For the tensor descriptor operand, the layout attibute is dropped after
    /// distribution. Types needs to be resolved in this case also.
    auto distributedTensorDescTy = dropLayouts(storeOp.getTensorDescType());
    newStoreOperands.push_back(
        resolveDistributedTy(newWarpOp.getResult(newRetIndices[1]),
                             distributedTensorDescTy, rewriter));

    rewriter.create<xegpu::StoreNdOp>(
        newWarpOp.getLoc(), TypeRange{}, newStoreOperands,
        filterTemporaryLayoutAttributes(storeOp->getAttrs()));
    rewriter.eraseOp(storeOp);
    return success();
  }
};

/// Clone a load_nd feeding into vector.yield op for the enclosing
/// `gpu.warp_execute_on_lane_0` and put it after the warp op.
/// The warp op will still contain the original op that will not be used by
/// the yield op (and should be cleaned up later with dce). The yield op will
/// bypass the load's arguments. Only the loaded vector is distributed
/// according to sg_map attribute and, tensor descriptor types is not
/// distributed.
///
/// Example:
///
/// ```
///   #sg_map_8 = #xegpu.sg_map<wi_layout = [1, 8], wi_data = [1, 1]>
///   %r = gpu.warp_execute_on_lane_0(%laneid) ->
///                   (vector<4x1xf32>) {
///     ...
///     %ld = xegpu.load_nd %arg0, %arg1: !xegpu.tensor_desc<4x8xf32> ->
///       vector<4x8xf32>
///     gpu.yield %ld
///   }
/// ```
/// To
/// ```
///   %r:2 = gpu.warp_execute_on_lane_0(%laneid) -> () {
///     ...
///     %dead = xegpu.load_nd %arg0: !xegpu.tensor_desc<4x8xf32> ->
///     vector<4x8xf32> gpu.yield %arg0, %arg1
///   }
///   %ld = xegpu.load_nd %r#0: !xegpu.tensor_desc<4x8xf32> -> vector<4x1xf32>
///
/// ```
struct LoadNdDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(subgroupOp, llvm::IsaPred<xegpu::LoadNdOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(
          subgroupOp, "warp result is not a xegpu::LoadNd op");

    auto loadOp = operand->get().getDefiningOp<xegpu::LoadNdOp>();
    xegpu::TensorDescType tensorDescTy = loadOp.getTensorDescType();
    xegpu::LayoutAttr layout = tensorDescTy.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(
          loadOp, "the source tensor descriptor lacks sg_map attribute");

    unsigned operandIdx = operand->getOperandNumber();
    VectorType distributedTypeByWarpOp =
        cast<VectorType>(subgroupOp.getResult(operandIdx).getType());

    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, subgroupOp,
        /* new yielded values = */ loadOp.getTensorDesc(),
        /* new yielded types = */ tensorDescTy, newRetIndices);

    /// Create a new load op outside the warp op with the distributed vector
    /// type.
    rewriter.setInsertionPointAfter(newWarpOp);
    auto loadNdDistValueTyOrFailure =
        loadOp.getTensorDescType().getDistributedVectorType();
    if (failed(loadNdDistValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          loadOp, "Failed to get distributed vector type for the load op");
    Value newLoadOp = rewriter.create<xegpu::LoadNdOp>(
        newWarpOp.getLoc(), loadNdDistValueTyOrFailure.value(),
        newWarpOp->getResult(newRetIndices[0]),
        filterTemporaryLayoutAttributes(loadOp->getAttrs()));
    Value distributedVal = newWarpOp.getResult(operandIdx);
    /// There can be a conflict between the vector type distributed by the
    /// warp op and (xegpu-specific) distributed type supported by the load
    /// op. We reconcile these mismatches by inserting a cast.
    newLoadOp =
        resolveDistributedTy(newLoadOp, distributedTypeByWarpOp, rewriter);
    rewriter.replaceAllUsesWith(distributedVal, newLoadOp);
    return success();
  }
};

struct DpasDistribution final : public gpu::WarpDistributionPattern {
  using gpu::WarpDistributionPattern::WarpDistributionPattern;
  LogicalResult matchAndRewrite(gpu::WarpExecuteOnLane0Op subgroupOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand =
        getWarpResult(subgroupOp, llvm::IsaPred<xegpu::DpasOp>);
    if (!operand)
      return rewriter.notifyMatchFailure(subgroupOp,
                                         "warp result is not a xegpu::Dpas op");

    auto dpasOp = operand->get().getDefiningOp<xegpu::DpasOp>();
    unsigned operandIdx = operand->getOperandNumber();
    auto layoutAName =
        llvm::formatv("{0}{1}", operandLayoutNamePrefix, 0).str();
    auto layoutBName =
        llvm::formatv("{0}{1}", operandLayoutNamePrefix, 1).str();
    auto layoutCName = llvm::formatv("{0}{1}", resultLayoutNamePrefix, 0).str();
    xegpu::LayoutAttr layoutA =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutAName);
    xegpu::LayoutAttr layoutB =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutBName);
    xegpu::LayoutAttr layoutOut =
        dpasOp->getAttrOfType<xegpu::LayoutAttr>(layoutCName);
    if (!layoutA || !layoutB || !layoutOut)
      return rewriter.notifyMatchFailure(
          dpasOp,
          "the xegpu::Dpas op lacks layout attribute for A, B or output");

    auto distLhsTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutA, dpasOp.getLhsType());
    auto distRhsTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutB, dpasOp.getRhsType());
    auto distResultTypeByWarpOpOrFailure =
        getDistVecTypeBasedOnLaneLayout(layoutOut, dpasOp.getResultType());
    if (failed(distLhsTypeByWarpOpOrFailure) ||
        failed(distRhsTypeByWarpOpOrFailure) ||
        failed(distResultTypeByWarpOpOrFailure))
      return rewriter.notifyMatchFailure(
          dpasOp,
          "Failed to distribute the A, B or output types in xegpu::Dpas op");

    llvm::SmallVector<Value, 3> newYieldValues{dpasOp.getLhs(),
                                               dpasOp.getRhs()};
    llvm::SmallVector<Type, 3> newYieldTypes{
        distLhsTypeByWarpOpOrFailure.value(),
        distRhsTypeByWarpOpOrFailure.value()};
    /// Dpas acc operand is optional.
    if (dpasOp.getAcc()) {
      newYieldValues.push_back(dpasOp.getAcc());
      newYieldTypes.push_back(distResultTypeByWarpOpOrFailure.value());
    }
    /// Create a new warp op without the dpas.
    SmallVector<size_t> newRetIndices;
    gpu::WarpExecuteOnLane0Op newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, subgroupOp, newYieldValues, newYieldTypes, newRetIndices);

    // Create a new dpas op outside the warp op.
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newDpasOperands;
    SmallVector<VectorType> newDpasOperandExpectedTypes;
    /// Reconcile the distributed types with the original types.
    newDpasOperandExpectedTypes.push_back(
        getDistributedVectorType(layoutA, dpasOp.getLhsType()));
    newDpasOperandExpectedTypes.push_back(
        getDistributedVectorType(layoutB, dpasOp.getRhsType()));
    if (dpasOp.getAcc()) {
      newDpasOperandExpectedTypes.push_back(
          getDistributedVectorType(layoutOut, dpasOp.getResultType()));
    }

    for (auto i : newRetIndices) {
      newDpasOperands.push_back(resolveDistributedTy(
          newWarpOp.getResult(i),
          newDpasOperandExpectedTypes[newDpasOperands.size()], rewriter));
    }
    auto newDpasOp = rewriter.create<xegpu::DpasOp>(
        newWarpOp->getLoc(), distResultTypeByWarpOpOrFailure.value(),
        newDpasOperands, dpasOp->getAttrs());
    Value disributedVal = newWarpOp.getResult(operandIdx);
    /// Reconile the output type.
    disributedVal = resolveDistributedTy(
        disributedVal,
        getDistributedVectorType(layoutOut, dpasOp.getResultType()), rewriter);
    rewriter.replaceAllUsesWith(disributedVal, newDpasOp);
    return success();
  }
};

} // namespace

namespace {
struct XeGPUSubgroupDistributePass final
    : public xegpu::impl::XeGPUSubgroupDistributeBase<
          XeGPUSubgroupDistributePass> {
  XeGPUSubgroupDistributePass() = default;
  XeGPUSubgroupDistributePass(const XeGPUSubgroupDistributePass &other) =
      default;
  XeGPUSubgroupDistributePass(xegpu::XeGPUSubgroupDistributeOptions options)
      : XeGPUSubgroupDistributeBase(options) {}
  void runOnOperation() override;
};
} // namespace

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {
  patterns.add<CreateNdDescDistribution, StoreNdDistribution,
               LoadNdDistribution, DpasDistribution>(patterns.getContext());
}

void XeGPUSubgroupDistributePass::runOnOperation() {
  auto &analyis = getAnalysis<RunLayoutInfoPropagation>();
  // Print the analysis result and exit. (for testing purposes)
  if (printOnly) {
    auto &os = llvm::outs();
    analyis.printAnalysisResult(os);
    return;
  }
  auto getPropagatedLayout = [&](Value val) {
    return analyis.getLayoutInfo(val);
  };
  if (failed(attachLayoutAttributes(getOperation(), getPropagatedLayout)))
    signalPassFailure();
  if (failed(resolveLayoutConflicts(getOperation())))
    signalPassFailure();
  /// Move all operations inside a GPU functions inside
  /// gpu.warp_execute_on_lane0.
  /// We want to avoid ops from hoisted out of the gpu.warp_execute_on_lane0
  /// region.
  // GreedyRewriteConfig config;
  // config.cseConstants = false;
  // config.fold = false;
  // config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveFuncBodyToWarpExecuteOnLane0>(&getContext());

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
  /// Finally, do the SIMD to SIMT distribution.
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUSubgroupDistributePatterns(patterns);
  /// TODO: These are not used at this point.
  auto distributionFn = [](Value val) { return AffineMap(); };
  auto shuffleFn = [](Location loc, OpBuilder &builder, Value val, Value srcIdx,
                      int64_t warpSz) { return Value(); };
  vector::populatePropagateWarpVectorDistributionPatterns(
      patterns, distributionFn, shuffleFn);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
