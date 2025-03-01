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
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
struct Layout2D {
  SmallVector<int64_t, 2> layout;
  Layout2D() = default;
  Layout2D(int64_t x, int64_t y) { layout.insert(layout.end(), {x, y}); }
  bool operator==(const Layout2D &rhs) const {
    return this->layout == rhs.layout;
  }
  bool operator<(const Layout2D &rhs) const {
    return this->layout < rhs.layout;
  }
  void print(llvm::raw_ostream &os) const {
    os << "{";
    llvm::interleave(
        layout, os, [&](int64_t a) { os << a; }, ", ");
    os << "}";
  }
};

using WiLayout = Layout2D;
using WiData = Layout2D;

struct SGMapInfo {
  Layout2D wiLayout;
  Layout2D wiData;
  SGMapInfo() = default;
  SGMapInfo(const Layout2D &layout, const Layout2D &data, unsigned bitWidth)
      : wiLayout(layout), wiData(data) {}
  bool operator==(const SGMapInfo &rhs) const {
    return this->wiLayout == rhs.wiLayout && this->wiData == rhs.wiData;
  }
  bool operator<(const SGMapInfo &rhs) const {
    return this->wiLayout < rhs.wiLayout || this->wiData < rhs.wiData;
  }
  void print(llvm::raw_ostream &os) const {
    os << "{";
    os << "layout: ";
    wiLayout.print(os);
    os << ", ";
    os << "data: ";
    wiData.print(os);
    os << "}";
  }
};

struct SGMapLatticeValue {
private:
  std::set<SGMapInfo> layouts;

public:
  SGMapLatticeValue() = default;
  SGMapLatticeValue(const SGMapLatticeValue &other) = default;
  SGMapLatticeValue(const WiLayout &layout, const WiData &data) {
    layouts.insert(SGMapInfo(layout, data, 16));
  }

  bool operator==(const SGMapLatticeValue &other) const {
    return this->layouts == other.layouts;
  }

  /// This function depends on a partial ordering of the lattice values.
  static SGMapLatticeValue meet(const SGMapLatticeValue &lhs,
                                const SGMapLatticeValue &rhs) {
    SGMapLatticeValue res = lhs;
    (void)res.addLayouts(rhs.layouts);
    return res;
  }

  static SGMapLatticeValue join(const SGMapLatticeValue &lhs,
                                const SGMapLatticeValue &rhs) {
    // Should not be triggered by this analysis, but required by `Lattice<T>`
    llvm_unreachable("Join should not be triggered by this test");
  }

  ChangeResult addLayouts(const std::set<SGMapInfo> &layouts) {
    int sizeBefore = this->layouts.size();
    this->layouts.insert(layouts.begin(), layouts.end());
    int sizeAfter = this->layouts.size();
    return sizeBefore == sizeAfter ? ChangeResult::NoChange
                                   : ChangeResult::Change;
  }

  void print(raw_ostream &os) const {
    os << "[";
    llvm::interleave(
        layouts, os, [&](const SGMapInfo &a) { a.print(os); }, ", ");
    os << "]";
  }

  void clear() { layouts.clear(); }

  std::set<SGMapInfo> getLayouts() const { return layouts; }
};

struct SGMap : public Lattice<SGMapLatticeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SGMap)
  using Lattice::Lattice;
};

static SGMapLatticeValue getSGMapForDPASOperand(Type operandTy,
                                                unsigned operandNum) {
  int packingFactorForB = packedBSizeInBits / operandTy.getIntOrFloatBitWidth();
  int packingFactorForA =
      operandTy.getIntOrFloatBitWidth() < packedBSizeInBits
          ? packedASizeInBits / operandTy.getIntOrFloatBitWidth()
          : 1;
  return SGMapLatticeValue(WiLayout(1, subgroupSize),
                           WiData(operandNum == 1 ? packingFactorForB : 1,
                                  operandNum == 0 ? packingFactorForA : 1));
}

class SGMapPropagation : public SparseBackwardDataFlowAnalysis<SGMap> {
public:
  SGMapPropagation(DataFlowSolver &solver, SymbolTableCollection &symbolTable)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable) {}
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<SGMap *> operands,
                               ArrayRef<const SGMap *> results) override;

  void visitBranchOperand(OpOperand &operand) override{};

  void visitCallOperand(OpOperand &operand) override{};

  void visitExternalCall(CallOpInterface call, ArrayRef<SGMap *> operands,
                         ArrayRef<const SGMap *> results) override{};

  void setToExitState(SGMap *lattice) override {
    (void)lattice->meet(SGMapLatticeValue());
  }
};
} // namespace

LogicalResult
SGMapPropagation::visitOperation(Operation *op, ArrayRef<SGMap *> operands,
                                 ArrayRef<const SGMap *> results) {
  /// Handle dpas
  if (auto dpas = dyn_cast<xegpu::DpasOp>(op)) {
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
    return success();
  }
  for (const SGMap *r : results) {
    // For each operand assume a default layout.
    for (SGMap *operand : operands) {
      meet(operand, *r);
    }
    addDependency(const_cast<SGMap *>(r), getProgramPointAfter(op));
  }
  return success();
}

void xegpu::populateXeGPUSubgroupDistributePatterns(
    RewritePatternSet &patterns) {}

namespace {

class RunSGMapPropagation {
public:
  RunSGMapPropagation(Operation *op) {
    SymbolTableCollection symbolTable;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<SGMapPropagation>(symbolTable);
    (void)solver.initializeAndRun(op);
  }

  SGMapLatticeValue getSGMap(Value val) {
    auto *state = solver.lookupState<SGMap>(val);
    if (!state)
      return {};
    return state->getValue();
  }

private:
  DataFlowSolver solver;
};

struct XeGPUSubgroupDistributePass final
    : public xegpu::impl::XeGPUSubgroupDistributeBase<
          XeGPUSubgroupDistributePass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUSubgroupDistributePass::runOnOperation() {
  Operation *op = getOperation();

  RunSGMapPropagation solver(op);

  // Print analysis results
  auto &os = llvm::outs();
  // check if op is a function
  // llvm::errs() << op->getName() << "\n";
  if (auto modOp = dyn_cast<ModuleOp>(op)) {
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      os << "SGMap for " << funcOp.getName() << ":\n";
      // Function args
      for (auto arg : funcOp.getArguments()) {
        auto layouts = solver.getSGMap(arg);
        os << "SGMap for " << arg << ": ";
        layouts.print(os);
        os << "\n";
      }
      // Function ops
      funcOp.walk([&](Operation *op) {
        if (op->getResults().empty())
          return;
        auto layouts = solver.getSGMap(op->getResult(0));
        os << "SGMap for " << op->getName() << ": ";
        layouts.print(os);
        os << "\n";
      });
    }
  }
}
