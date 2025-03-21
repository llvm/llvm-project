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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSUBGROUPDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

/// HW dependent constants.
/// TODO: These constants should be queried from the target information.
constexpr unsigned subgroupSize = 16; // How many work items in a subgroup.
/// If DPAS A or B operands have low precision element types they must be packed
/// according to the following sizes.
constexpr unsigned packedSizeInBitsForDefault =
    16; // Minimum packing size per register for DPAS A.
constexpr unsigned packedSizeInBitsForDpasB =
    32; // Minimum packing size per register for DPAS B.

namespace {

///===----------------------------------------------------------------------===///
/// Layout
///===----------------------------------------------------------------------===///

/// Helper class to store the ND layout of work items within a subgroup and data
/// owned by each work item.
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

/// WiLayout represents the layout of work items within a subgroup when it
/// accesses some value. WiData represents the layout of data owned by each work
/// item.
using WiLayout = Layout;
using WiData = Layout;

///===----------------------------------------------------------------------===///
/// SGMap
///===----------------------------------------------------------------------===///

/// Helper class for tracking the analysis state of a value. For SGPropagation,
/// the analysis state is simply the wi_layout and wi_data of each value.
/// Purpose of this analysis to propagate some unique layout for each value in
/// the program starting from some known values (like DPAS, StoreNd, etc.).
///
/// Given this, SGMap satisifies the following properties:
///  1) SGMap is a lattice with two states - assigned and not assigned.
///  2) Two SGMap values are equal if they are both assigned or both not
///  assigned. The concrete value of assigned state does not matter.
///  3) The meet operator works as follows:
///     - If current state is assigned, return the current state. (already
///     a unique layout is assigned. don't change it)
///     - Otherwise, return the other state.

struct SGMap {
private:
  WiLayout wiLayout;
  WiData wiData;

public:
  SGMap() = default;
  SGMap(const WiLayout &layout, const WiData &data)
      : wiLayout(layout), wiData(data) {}

  /// Two lattice values are equal if they have `some` layout. The actual
  /// content of the layout does not matter.
  bool operator==(const SGMap &other) const {
    return this->isAssigned() == other.isAssigned();
  }

  static SGMap meet(const SGMap &lhs, const SGMap &rhs);

  static SGMap join(const SGMap &lhs, const SGMap &rhs);

  void print(raw_ostream &os) const;

  bool isAssigned() const { return wiLayout.size() > 0 && wiData.size() > 0; }

  SGMap getTransposedLayout(ArrayRef<int64_t> permutation) const;

  const WiLayout &getLayout() const { return wiLayout; }
  const WiData &getData() const { return wiData; }
};

void SGMap::print(raw_ostream &os) const {
  if (isAssigned()) {
    os << "wi_layout: ";
    wiLayout.print(os);
    os << ", wi_data: ";
    wiData.print(os);
  } else
    os << "Not assigned.";
}

SGMap SGMap::meet(const SGMap &lhs, const SGMap &rhs) {
  if (!lhs.isAssigned())
    return rhs;
  return lhs;
}

/// Since this is a backward analysis, join method is not used.
SGMap SGMap::join(const SGMap &lhs, const SGMap &rhs) {
  llvm_unreachable("Join should not be triggered by SGMapPropagation.");
}

/// Get the transposed layout according to the given permutation.
SGMap SGMap::getTransposedLayout(ArrayRef<int64_t> permutation) const {
  if (!isAssigned())
    return {};
  WiLayout newLayout;
  WiData newData;
  for (auto idx : permutation) {
    newLayout.layout.push_back(wiLayout.layout[idx]);
    newData.layout.push_back(wiData.layout[idx]);
  }
  return SGMap(newLayout, newData);
}

///===----------------------------------------------------------------------===///
/// SGMapLattice
///===----------------------------------------------------------------------===///

/// Lattice holding the SGMap for each value.
struct SGMapLattice : public Lattice<SGMap> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SGMapLattice)
  using Lattice::Lattice;
};

/// Helper Functions to get default layouts. A `default layout` is a layout that
/// is assigned to a value when the layout is not fixed by some anchor operation
/// (like DPAS). This is the natural layout work items are arranged in a
/// subgroup.

/// Helper Function to get the default layout for uniform values like constants.
/// For 1D vector, wi_layout is [subgroupSize] and wi_data is [1].
/// For 2D vector, wi_layout is [1, subgroupSize] and wi_data is [1, 1].
static SGMap getDefaultSgMap(unsigned rank) {
  assert((rank == 1 || rank == 2) && "Expected 1D or 2D vector.");
  if (rank == 1)
    return SGMap(WiLayout({subgroupSize}), WiData({1}));
  return SGMap(WiLayout({1, subgroupSize}), WiData({1, 1}));
}

/// Helper to get the default layout for a vector type.
static SGMap getDefaultSgMap(VectorType vectorTy) {
  /// Expecting a 1D or 2D vector.
  assert((vectorTy.getRank() == 1 || vectorTy.getRank() == 2) &&
         "Expected 1D or 2D vector.");
  /// Expecting int or float element type.
  assert(vectorTy.getElementType().isIntOrFloat() &&
         "Expected int or float element type.");
  /// If the rank is 1, then return default layout for 1D vector.
  if (vectorTy.getRank() == 1)
    return getDefaultSgMap(1);
  /// Packing factor is determined by the element type bitwidth.
  int packingFactor = 1;
  auto bitwidth = vectorTy.getElementType().getIntOrFloatBitWidth();
  if (bitwidth < packedSizeInBitsForDefault)
    packingFactor = packedSizeInBitsForDefault / bitwidth;
  return SGMap(WiLayout({1, subgroupSize}), WiData({1, packingFactor}));
}

/// Helper Function to get the expected layouts for DPAS operands. `wi_data` is
/// set according to the following criteria:
/// * For A operand, the data must be packed in minimum
/// `packedSizeInBitsForDefault`
/// * For B operand, the data must be packed in minimum
/// `packedSizeInBitsForDpasB`
static SGMap getSGMapForDPASOperand(VectorType vectorTy, unsigned operandNum) {
  auto elementTy = vectorTy.getElementType();
  assert(elementTy.isIntOrFloat() &&
         "Expected int or float type in DPAS operands");
  WiLayout layout({1, subgroupSize});
  /// For B operand, data must be packed in minimum `packedDpasBSizeInBits` and
  /// must have the VNNI format.
  if (operandNum == 1 &&
      elementTy.getIntOrFloatBitWidth() < packedSizeInBitsForDpasB) {
    WiData data(
        {packedSizeInBitsForDpasB / elementTy.getIntOrFloatBitWidth(), 1});
    return SGMap(layout, data);
  }
  /// Otherwise, return the default layout for the vector type.
  return getDefaultSgMap(vectorTy);
}

///===----------------------------------------------------------------------===///
/// SGMapPropagation
///===----------------------------------------------------------------------===///

/// Backward data flow analysis to propagate the wi_layout and wi_data of each
/// value in the program. Currently, the layouts for operands DPAS, StoreNd, and
/// StoreScatter are fixed (known before propagation). Purpose of this analysis
/// is to propagate those known layouts to all their producers and (other)
/// consumers.
class SGMapPropagation : public SparseBackwardDataFlowAnalysis<SGMapLattice> {
private:
  void visitDpasOp(xegpu::DpasOp dpas, ArrayRef<SGMapLattice *> operands,
                   ArrayRef<const SGMapLattice *> results);

  void visitStoreNdOp(xegpu::StoreNdOp store, ArrayRef<SGMapLattice *> operands,
                      ArrayRef<const SGMapLattice *> results);

  void visitStoreScatterOp(xegpu::StoreScatterOp storeScatter,
                           ArrayRef<SGMapLattice *> operands,
                           ArrayRef<const SGMapLattice *> results);

  void visitLoadNdOp(xegpu::LoadNdOp load, ArrayRef<SGMapLattice *> operands,
                     ArrayRef<const SGMapLattice *> results);

  void visitLoadGatherOp(xegpu::LoadGatherOp load,
                         ArrayRef<SGMapLattice *> operands,
                         ArrayRef<const SGMapLattice *> results);

  void visitTransposeOp(vector::TransposeOp transpose,
                        ArrayRef<SGMapLattice *> operands,
                        ArrayRef<const SGMapLattice *> results);

  void visitVectorBitcastOp(vector::BitCastOp bitcast,
                            ArrayRef<SGMapLattice *> operands,
                            ArrayRef<const SGMapLattice *> results);

  void visitCreateDescOp(xegpu::CreateDescOp createDesc,
                         ArrayRef<SGMapLattice *> operands,
                         ArrayRef<const SGMapLattice *> results);

  void visitUpdateNdOffsetOp(xegpu::UpdateNdOffsetOp updateNdOffset,
                             ArrayRef<SGMapLattice *> operands,
                             ArrayRef<const SGMapLattice *> results);

  void visitVectorMultiReductionOp(vector::MultiDimReductionOp reduction,
                                   ArrayRef<SGMapLattice *> operands,
                                   ArrayRef<const SGMapLattice *> results);

public:
  SGMapPropagation(DataFlowSolver &solver, SymbolTableCollection &symbolTable)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable) {}
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<SGMapLattice *> operands,
                               ArrayRef<const SGMapLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override {};

  void visitCallOperand(OpOperand &operand) override {};

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<SGMapLattice *> operands,
                         ArrayRef<const SGMapLattice *> results) override {};

  void setToExitState(SGMapLattice *lattice) override {
    (void)lattice->meet(SGMap());
  }
};
} // namespace

LogicalResult
SGMapPropagation::visitOperation(Operation *op,
                                 ArrayRef<SGMapLattice *> operands,
                                 ArrayRef<const SGMapLattice *> results) {
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
        for (const SGMapLattice *r : results) {
          for (SGMapLattice *operand : operands) {
            /// Propagate the layout of the result to the operand.
            if (r->getValue().isAssigned())
              meet(operand, *r);
          }
        }
      });
  /// Add a dependency from each result to program point after the operation.
  for (const SGMapLattice *r : results) {
    addDependency(const_cast<SGMapLattice *>(r), getProgramPointAfter(op));
  }
  return success();
}

void SGMapPropagation::visitVectorMultiReductionOp(
    vector::MultiDimReductionOp reduction, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// The layout of the result must be present.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  /// We only consider 2D -> 1D reductions at this point.
  assert(resultLayout.getLayout().size() == 1 &&
         "Expected 1D layout for reduction result.");
  /// Given that the result is 1D, the layout of the operand should be 2D with
  /// default layout.
  auto operandLayout = getDefaultSgMap(2);
  propagateIfChanged(operands[0], operands[0]->meet(operandLayout));
  /// Accumulator should have the same layout as the result.
  propagateIfChanged(operands[1], operands[1]->meet(resultLayout));
}

/// Propagate the layout of the result tensor to the source tensor descriptor in
/// UpdateNdOffsetOp.
void SGMapPropagation::visitUpdateNdOffsetOp(
    xegpu::UpdateNdOffsetOp updateNdOffset, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// The layout of the result must be present.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(resultLayout));
}

/// Set the layouts for DPAS A, B, and C operands.
void SGMapPropagation::visitDpasOp(xegpu::DpasOp dpas,
                                   ArrayRef<SGMapLattice *> operands,
                                   ArrayRef<const SGMapLattice *> results) {
  auto aTy = dpas.getLhsType();
  auto bTy = dpas.getRhsType();
  propagateIfChanged(operands[0],
                     operands[0]->meet(getSGMapForDPASOperand(aTy, 0)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(getSGMapForDPASOperand(bTy, 1)));
  if (operands.size() > 2) {
    auto cTy = dpas.getAccType();
    propagateIfChanged(operands[2],
                       operands[2]->meet(getSGMapForDPASOperand(cTy, 2)));
  }
}

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void SGMapPropagation::visitStoreNdOp(xegpu::StoreNdOp store,
                                      ArrayRef<SGMapLattice *> operands,
                                      ArrayRef<const SGMapLattice *> results) {
  auto storeLayout = getDefaultSgMap(store.getValueType());
  /// Both operands should have the same layout
  for (SGMapLattice *operand : operands) {
    propagateIfChanged(operand, operand->meet(storeLayout));
  }
}

/// Propagate the layout of the value to the tensor descriptor operand in
/// LoadNdOp.
void SGMapPropagation::visitLoadNdOp(xegpu::LoadNdOp load,
                                     ArrayRef<SGMapLattice *> operands,
                                     ArrayRef<const SGMapLattice *> results) {
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;
  SGMap tensorDescLayout = valueLayout;
  /// LoadNdOp has the transpose effect. However, at the stage of this analysis
  /// this effect is not expected and should be abstracted away. Emit a warning.
  if (auto transpose = load.getTranspose()) {
    load.emitWarning("Transpose effect is not expected for LoadNdOp at "
                     "SGMapPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout(transpose.value());
  }
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
}

/// For vector::TransposeOp, the layout of the result is transposed and
/// propagated to the operand.
void SGMapPropagation::visitTransposeOp(
    vector::TransposeOp transpose, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// Need the layout of transpose result to propagate to the operands.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  auto newLayout = resultLayout.getTransposedLayout(transpose.getPermutation());
  /// Propagate the new layout to the vector operand.
  propagateIfChanged(operands[0], operands[0]->meet(newLayout));
}

/// For vector::BitCastOp, the wi_data of the source layout is changed based on
/// the bit width of the source and result types.
void SGMapPropagation::visitVectorBitcastOp(
    vector::BitCastOp bitcast, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// Need the layout of bitcast result to propagate to the operands.
  auto resultLayout = results[0]->getValue();
  if (!resultLayout.isAssigned())
    return;
  auto inElemTyBitWidth =
      bitcast.getSourceVectorType().getElementType().getIntOrFloatBitWidth();
  auto outElemTyBitWidth =
      bitcast.getResultVectorType().getElementType().getIntOrFloatBitWidth();

  /// WiLayout does not change.
  const WiLayout &newWiLayout = resultLayout.getLayout();
  const WiData &currData = resultLayout.getData();
  WiData newWiData;
  /// It's a widening bitcast
  if (inElemTyBitWidth < outElemTyBitWidth) {
    auto ratio = outElemTyBitWidth / inElemTyBitWidth;
    newWiData = resultLayout.getData()[0] == 1
                    ? WiData({1, currData[1] * ratio})
                    : WiData({currData[0] * ratio, 1});
  } else {
    /// It's a narrowing bitcast
    auto ratio = inElemTyBitWidth / outElemTyBitWidth;
    newWiData = resultLayout.getData()[0] == 1
                    ? WiData({1, currData[1] / ratio})
                    : WiData({currData[0] / ratio, 1});
  }

  propagateIfChanged(operands[0],
                     operands[0]->meet(SGMap(newWiLayout, newWiData)));
}

/// Propagate the layout of the result to the tensor descriptor and mask
/// operands in LoadGatherOp.
void SGMapPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;

  SGMap tensorDescLayout = valueLayout;
  if (load.getTranspose()) {
    /// LoadGatherOp has the transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    load.emitWarning("Transpose effect is not expected for LoadGatherOp at "
                     "SGMapPropagation stage.");
    tensorDescLayout = valueLayout.getTransposedLayout({1, 0});
  }
  /// Mask operand should have 1D default layout.
  auto maskLayout = getDefaultSgMap(1);
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
  /// Propagate the new layout to the mask operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
}

/// Propagate the layout of the descriptor to the vector offset operand in
/// CreateDescOp.
void SGMapPropagation::visitCreateDescOp(
    xegpu::CreateDescOp createDesc, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// For offset operand propagate 1D default layout.
  SGMap layout = getDefaultSgMap(1);
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, and mask operands in the
/// StoreScatterOp.
void SGMapPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// Currently, for 2D StoreScatterOp we expect that the height dimension of
  /// the tensor descriptor is evenly divisible by the subgroup size.
  /// TODO: Add support for other 2D shapes.
  auto tdescShape = storeScatter.getTensorDescType().getShape();
  if (tdescShape.size() > 1 && tdescShape[0] % subgroupSize != 0) {
    storeScatter.emitError("Height dimension of the tensor descriptor should "
                           "be evenly divisible by the subgroup size.");
    return;
  }
  auto valueLayout = getDefaultSgMap(storeScatter.getValueType());
  SGMap storeScatterLayout = valueLayout;
  if (storeScatter.getTranspose()) {
    /// StoreScatteOp allows transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    storeScatter.emitWarning("Transpose effect is not expected for "
                             "StoreScatterOp at SGMapPropagation stage.");
    storeScatterLayout = valueLayout.getTransposedLayout({1, 0});
  }
  /// Propagate the value layout.
  propagateIfChanged(operands[0], operands[0]->meet(valueLayout));
  /// Propagate the tensor descriptor layout.
  propagateIfChanged(operands[1], operands[1]->meet(storeScatterLayout));
  /// Use default 1D layout for mask operand.
  auto maskLayout = getDefaultSgMap(1);
  propagateIfChanged(operands[2], operands[2]->meet(maskLayout));
}

namespace {

///===----------------------------------------------------------------------===///
/// RunSGMapPropagation
///===----------------------------------------------------------------------===///

/// Driver class for running the SGMapPropagation analysis.
class RunSGMapPropagation {
public:
  RunSGMapPropagation(Operation *op) : target(op) {
    SymbolTableCollection symbolTable;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<SGMapPropagation>(symbolTable);
    (void)solver.initializeAndRun(op);
  }

  SGMap getSGMap(Value val);

  void printAnalysisResult(llvm::raw_ostream &os);

private:
  DataFlowSolver solver;
  const Operation *target;
};
} // namespace

SGMap RunSGMapPropagation::getSGMap(Value val) {
  auto *state = solver.lookupState<SGMapLattice>(val);
  if (!state)
    return {};
  return state->getValue();
}

void RunSGMapPropagation::printAnalysisResult(llvm::raw_ostream &os) {
  auto printFunctionResult = [&](FunctionOpInterface funcOp) {
    os << "function: " << funcOp.getName() << ":\n";
    // Function arguments
    for (auto arg : funcOp.getArguments()) {
      auto layout = getSGMap(arg);
      os << "argument: " << arg << "\n";
      os << "sg_map  : ";
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
      /// Print the sg_map for each result.
      for (auto [i, r] : llvm::enumerate(op->getResults())) {
        auto layout = getSGMap(r);
        os << "sg_map for result #" << i << ": ";
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

void XeGPUSubgroupDistributePass::runOnOperation() {
  Operation *op = getOperation();
  RunSGMapPropagation solver(op);

  // Print the analysis result and exit.
  if (printOnly) {
    auto &os = llvm::outs();
    solver.printAnalysisResult(os);
    return;
  }
}
