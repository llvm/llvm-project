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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
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

constexpr unsigned subgroupSize = 16;
constexpr unsigned packedASizeInBits = 16;
constexpr unsigned packedBSizeInBits = 32;

namespace {

///===----------------------------------------------------------------------===///
/// Layout
///===----------------------------------------------------------------------===///

/// Helper class to store the ND layout of work items within a subgroup and data
/// owned by each work item.
struct Layout {
  SmallVector<int64_t, 3> layout;
  Layout() = default;
  Layout(const Layout &other) = default;
  Layout(std::initializer_list<int64_t> list) : layout(list) {}
  void print(llvm::raw_ostream &os) const;
  size_t size() const { return layout.size(); }
  int64_t operator[](size_t idx) const { return layout[idx]; }
};

void Layout::print(llvm::raw_ostream &os) const {
  os << "[";
  llvm::interleaveComma(layout, os);
  os << "]";
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
  WiLayout layout;
  WiData data;

public:
  SGMap() = default;
  SGMap(const SGMap &other) = default;
  SGMap(const WiLayout &layout, const WiData &data)
      : layout(layout), data(data) {}

  /// Two lattice values are equal if they have `some` layout. The actual
  /// content of the layout does not matter.
  bool operator==(const SGMap &other) const {
    return this->isAssigned() == other.isAssigned();
  }

  static SGMap meet(const SGMap &lhs, const SGMap &rhs);

  static SGMap join(const SGMap &lhs, const SGMap &rhs);

  void print(raw_ostream &os) const;

  bool isAssigned() const { return layout.size() > 0 && data.size() > 0; }

  SGMap getTransposedLayout(ArrayRef<int64_t> permutation) const;

  const WiLayout &getLayout() const { return layout; }
  const WiData &getData() const { return data; }
};

void SGMap::print(raw_ostream &os) const {
  if (isAssigned()) {
    os << "wi_layout: ";
    layout.print(os);
    os << ", wi_data: ";
    data.print(os);
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
    newLayout.layout.push_back(layout.layout[idx]);
    newData.layout.push_back(data.layout[idx]);
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

/// Helper Function to get the expected layouts for DPAS operands.
static SGMap getSGMapForDPASOperand(Type operandTy, unsigned operandNum) {
  int packingFactorForB = packedBSizeInBits / operandTy.getIntOrFloatBitWidth();
  int packingFactorForA =
      operandTy.getIntOrFloatBitWidth() < packedBSizeInBits
          ? packedASizeInBits / operandTy.getIntOrFloatBitWidth()
          : 1;
  return SGMap(WiLayout({1, subgroupSize}),
               WiData({operandNum == 1 ? packingFactorForB : 1,
                       operandNum == 0 ? packingFactorForA : 1}));
}

/// Helper Function to get the default layout for a given type. Usually this is,
/// wi_layout = [1, subgroupSize] and wi_data = [1, 1].
/// However, the minimum granularity of data access per work item is 16-bits.
/// So, if the bitwidth of the type is less than 16, we need to pack the data to
/// 16-bits.
static SGMap getDefaultSgMap(Type ty) {
  int packingFactor = 1;
  if (ty.getIntOrFloatBitWidth() < packedASizeInBits)
    packingFactor = packedBSizeInBits / ty.getIntOrFloatBitWidth();
  return SGMap(WiLayout({1, subgroupSize}), WiData({1, packingFactor}));
}

/// Helper Function to get the default layout representing constants.
static SGMap getDefaultSgMap() {
  return SGMap(WiLayout({1, subgroupSize}), WiData({1, 1}));
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

  void visitCreateNdDescOp(xegpu::CreateNdDescOp createNdDesc,
                           ArrayRef<SGMapLattice *> operands,
                           ArrayRef<const SGMapLattice *> results);

  void visitCreateDescOp(xegpu::CreateDescOp createDesc,
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
  if (auto dpas = dyn_cast<xegpu::DpasOp>(op))
    visitDpasOp(dpas, operands, results);
  else if (auto store = dyn_cast<xegpu::StoreNdOp>(op))
    visitStoreNdOp(store, operands, results);
  else if (auto load = dyn_cast<xegpu::LoadNdOp>(op))
    visitLoadNdOp(load, operands, results);
  else if (auto transpose = dyn_cast<vector::TransposeOp>(op))
    visitTransposeOp(transpose, operands, results);
  else if (auto bitcast = dyn_cast<vector::BitCastOp>(op))
    visitVectorBitcastOp(bitcast, operands, results);
  else if (auto loadGather = dyn_cast<xegpu::LoadGatherOp>(op))
    visitLoadGatherOp(loadGather, operands, results);
  else if (auto createNdDesc = dyn_cast<xegpu::CreateNdDescOp>(op))
    visitCreateNdDescOp(createNdDesc, operands, results);
  else if (auto createDesc = dyn_cast<xegpu::CreateDescOp>(op))
    visitCreateDescOp(createDesc, operands, results);
  else if (auto storeScatter = dyn_cast<xegpu::StoreScatterOp>(op))
    visitStoreScatterOp(storeScatter, operands, results);
  /// All other ops
  else {
    for (const SGMapLattice *r : results) {
      for (SGMapLattice *operand : operands) {
        /// Propagate the layout of the result to the operand.
        if (r->getValue().isAssigned())
          meet(operand, *r);
      }
    }
  }
  /// Add a dependency from each reult to program point after the operation.
  /// NOTE: not sure if this is required, but all other similar analysis do
  /// this.
  for (const SGMapLattice *r : results) {
    addDependency(const_cast<SGMapLattice *>(r), getProgramPointAfter(op));
  }
  return success();
}

/// Set the layouts for DPAS A, B, and C operands.
void SGMapPropagation::visitDpasOp(xegpu::DpasOp dpas,
                                   ArrayRef<SGMapLattice *> operands,
                                   ArrayRef<const SGMapLattice *> results) {
  auto aTy = dpas.getLhsType().getElementType();
  auto bTy = dpas.getRhsType().getElementType();
  propagateIfChanged(operands[0],
                     operands[0]->meet(getSGMapForDPASOperand(aTy, 0)));
  propagateIfChanged(operands[1],
                     operands[1]->meet(getSGMapForDPASOperand(bTy, 1)));
  if (operands.size() > 2) {
    auto cTy = dpas.getAccType().getElementType();
    propagateIfChanged(operands[2],
                       operands[2]->meet(getSGMapForDPASOperand(cTy, 2)));
  }
};

/// Set the layout for the value and tensor descriptor operands in StoreNdOp.
void SGMapPropagation::visitStoreNdOp(xegpu::StoreNdOp store,
                                      ArrayRef<SGMapLattice *> operands,
                                      ArrayRef<const SGMapLattice *> results) {
  auto storeLayout =
      getDefaultSgMap(store.getTensorDescType().getElementType());
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
  /// LoadNdOp has the transpose effect. However, at the stage of this analyis
  /// this effect is not expected and should be abstracted away. Emit a warning.
  /// TODO: Handle this case properly when `order` is introduced in the sg_map.
  if (auto transpose = load.getTranspose()) {
    load.emitWarning("Transpose effect is not expected for LoadNdOp");
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
  auto operandLayout = results[0]->getValue();
  if (!operandLayout.isAssigned())
    return;
  auto newLayout =
      operandLayout.getTransposedLayout(transpose.getPermutation());
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

  SGMap tensorDescLayout;
  if (load.getTranspose()) {
    /// LoadGatherOp has the transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    /// TODO: Handle this case properly when `order` is introduced in the
    /// sg_map.
    load.emitWarning("Transpose effect is not expected for LoadGatherOp");
    tensorDescLayout = valueLayout.getTransposedLayout({1, 0});
  } else
    tensorDescLayout = valueLayout;
  /// Mask operand should have the same layout as the value but with wi_data =
  /// [1, 1]
  SGMap maskLayout = SGMap(valueLayout.getLayout(), WiData({1, 1}));
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
  /// Propagate the new layout to the mask operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
}

/// Propagate the layout of the descriptor to the operands in CreateNdDescOp.
void SGMapPropagation::visitCreateNdDescOp(
    xegpu::CreateNdDescOp createNdDesc, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(descLayout));
  /// For all other operands propagate the descriptor layout.
  SGMap layout = getDefaultSgMap();
  for (size_t i = 1; i < operands.size(); ++i) {
    propagateIfChanged(operands[i], operands[i]->meet(layout));
  }
}

/// Propagate the layout of the descriptor to the source and offset operands in
/// CreateDescOp.
void SGMapPropagation::visitCreateDescOp(
    xegpu::CreateDescOp createDesc, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(descLayout));
  /// For offset operand propagate the default layout.
  SGMap layout = getDefaultSgMap();
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

/// Set the layout for the value, tensor descriptor, and mask operands in the
/// StoreScatterOp.
void SGMapPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto valueLayout =
      getDefaultSgMap(storeScatter.getTensorDescType().getElementType());
  SGMap storeScatterLayout;
  if (storeScatter.getTranspose()) {
    /// StoreScatteOp allows transpose effect. However, at the stage of this
    /// analyis this effect is not expected and should be abstracted away. Emit
    /// a warning.
    /// TODO: Handle this case properly when `order` is introduced in the
    /// sg_map.
    storeScatter.emitWarning(
        "Transpose effect is not expected for StoreScatterOp");
    storeScatterLayout = valueLayout.getTransposedLayout({1, 0});
  } else
    storeScatterLayout = valueLayout;
  /// Propagate the value layout.
  propagateIfChanged(operands[0], operands[0]->meet(valueLayout));
  /// Propagate the tensor descriptor layout.
  propagateIfChanged(operands[1], operands[1]->meet(storeScatterLayout));
  /// Use default layout for mask operand.
  auto maskLayout = getDefaultSgMap();
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
  if (auto modOp = dyn_cast<ModuleOp>(target)) {
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
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
    }
  }
}

namespace {
struct XeGPUSubgroupDistributePass final
    : public xegpu::impl::XeGPUSubgroupDistributeBase<
          XeGPUSubgroupDistributePass> {
  XeGPUSubgroupDistributePass() = default;
  XeGPUSubgroupDistributePass(const XeGPUSubgroupDistributePass &other)
      : xegpu::impl::XeGPUSubgroupDistributeBase<XeGPUSubgroupDistributePass>(
            other) {
    this->printOnly = other.printOnly;
  }
  void runOnOperation() override;
  /// Print sg map propagation analysis result and exit for testing purposes.
  Option<bool> printOnly{*this, "print-analysis-only", llvm::cl::init(false),
                         llvm::cl::desc("Print the result of the subgroup map "
                                        "propagation analysis and exit.")};
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

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {}
