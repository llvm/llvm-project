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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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

using WiLayout = Layout;
using WiData = Layout;

struct SGMap {
private:
  WiLayout layout;
  WiData data;

public:
  SGMap() = default;
  SGMap(const SGMap &other) = default;
  SGMap(const WiLayout &layout, const WiData &data)
      : layout(layout), data(data) {}

  // Two lattice values are equal if they have `some` layout. The actual
  // content of the layout does not matter.
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
    os << "Layout: ";
    layout.print(os);
    os << ", Data: ";
    data.print(os);
  } else
    os << "Not initialized";
}

SGMap SGMap::meet(const SGMap &lhs, const SGMap &rhs) {
  if (!lhs.isAssigned())
    return rhs;
  return lhs;
}

SGMap SGMap::join(const SGMap &lhs, const SGMap &rhs) {
  // Should not be triggered by this analysis, but required by `Lattice<T>`
  llvm_unreachable("Join should not be triggered by this test");
}

SGMap SGMap::getTransposedLayout(ArrayRef<int64_t> permutation) const {
  if (!isAssigned())
    return {};
  WiLayout newLayout;
  WiData newData;
  for (auto idx : permutation) {
    newLayout.layout.push_back(layout.layout[idx]);
    newData.layout.push_back(data.layout[idx]);
  }
  return SGMap(newLayout, data);
}

struct SGMapLattice : public Lattice<SGMap> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SGMapLattice)
  using Lattice::Lattice;
};

/// Helper Functions
///

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

static SGMap getDefaultSgMap(Type ty) {
  int packingFactor = 1;
  if (ty.getIntOrFloatBitWidth() < packedASizeInBits)
    packingFactor = packedBSizeInBits / ty.getIntOrFloatBitWidth();
  return SGMap(WiLayout({1, subgroupSize}), WiData({1, packingFactor}));
}

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

  void visitBranchOperand(OpOperand &operand) override{};

  void visitCallOperand(OpOperand &operand) override{};

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<SGMapLattice *> operands,
                         ArrayRef<const SGMapLattice *> results) override{};

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

void SGMapPropagation::visitLoadNdOp(xegpu::LoadNdOp load,
                                     ArrayRef<SGMapLattice *> operands,
                                     ArrayRef<const SGMapLattice *> results) {
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;
  SGMap tensorDescLayout = valueLayout;
  if (auto transpose = load.getTranspose())
    tensorDescLayout = valueLayout.getTransposedLayout(transpose.value());
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
}

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

void SGMapPropagation::visitLoadGatherOp(
    xegpu::LoadGatherOp load, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto valueLayout = results[0]->getValue();
  /// Need the layout of the value to propagate to the tensor descriptor.
  if (!valueLayout.isAssigned())
    return;
  /// LoadGatherOp has the transpose effect, so propagate the transposed layout
  /// to the tensor descriptor.
  SGMap tensorDescLayout = valueLayout.getTransposedLayout({1, 0});
  /// Mask operand should have the same layout as the value but with wi_data =
  /// [1, 1]
  SGMap maskLayout = SGMap(valueLayout.getLayout(), WiData({1, 1}));
  /// Propagate the new layout to the tensor descriptor operand.
  propagateIfChanged(operands[0], operands[0]->meet(tensorDescLayout));
  /// Propagate the new layout to the mask operand.
  propagateIfChanged(operands[1], operands[1]->meet(maskLayout));
}

void SGMapPropagation::visitCreateNdDescOp(
    xegpu::CreateNdDescOp createNdDesc, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(descLayout));
  /// For all other operands propagate the same layout with wi_data = [1, 1]
  SGMap layout = SGMap(descLayout.getLayout(), WiData({1, 1}));
  for (size_t i = 1; i < operands.size(); ++i) {
    propagateIfChanged(operands[i], operands[i]->meet(layout));
  }
}

void SGMapPropagation::visitCreateDescOp(
    xegpu::CreateDescOp createDesc, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  auto descLayout = results[0]->getValue();
  /// Need the layout of the descriptor to propagate to the operands.
  if (!descLayout.isAssigned())
    return;
  /// Propagate the layout to the source operand.
  propagateIfChanged(operands[0], operands[0]->meet(descLayout));
  /// For offset operand propagate the same layout with wi_data = [1, 1]
  SGMap layout = SGMap(descLayout.getLayout(), WiData({1, 1}));
  propagateIfChanged(operands[1], operands[1]->meet(layout));
}

void SGMapPropagation::visitStoreScatterOp(
    xegpu::StoreScatterOp storeScatter, ArrayRef<SGMapLattice *> operands,
    ArrayRef<const SGMapLattice *> results) {
  /// StoreScatterOp has the transpose effect. Value has the regular layout,
  /// while the tensor descriptor has the transposed layout.
  auto valueLayout =
      getDefaultSgMap(storeScatter.getTensorDescType().getElementType());
  auto storeScatterLayout = valueLayout.getTransposedLayout({1, 0});
  /// Propagate the value layout.
  propagateIfChanged(operands[0], operands[0]->meet(valueLayout));
  /// Propagate the tensor descriptor layout.
  propagateIfChanged(operands[1], operands[1]->meet(storeScatterLayout));
  /// Propagate the mask layout.
  propagateIfChanged(operands[2], operands[2]->meet(valueLayout));
}

namespace {

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
        auto layouts = getSGMap(op->getResult(0));
        os << "op    : ";
        op->print(os);
        os << "\n";
        os << "sg_map: ";
        layouts.print(os);
        os << "\n";
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
  Option<bool> printOnly{
      *this, "print-only", llvm::cl::init(false),
      llvm::cl::desc(
          "Print the result of the subgroup map propagation analysis")};
};
} // namespace

void XeGPUSubgroupDistributePass::runOnOperation() {
  Operation *op = getOperation();

  RunSGMapPropagation solver(op);

  // Print analysis results
  if (printOnly) {
    auto &os = llvm::outs();
    solver.printAnalysisResult(os);
  }
}

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {}
