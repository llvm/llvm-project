//===- TestDeadCodeAnalysis.cpp - Test dead code analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

/// Print the liveness of every block, control-flow edge, and the predecessors
/// of all regions, callables, and calls.
static void printAnalysisResults(DataFlowSolver &solver, Operation *op,
                                 raw_ostream &os) {
  op->walk([&](Operation *op) {
    auto tag = op->getAttrOfType<StringAttr>("tag");
    if (!tag)
      return;
    os << tag.getValue() << ":\n";
    for (Region &region : op->getRegions()) {
      os << " region #" << region.getRegionNumber() << "\n";
      for (Block &block : region) {
        os << "  ";
        block.printAsOperand(os);
        os << " = ";
        auto *live = solver.lookupState<Executable>(&block);
        if (live)
          os << *live;
        else
          os << "dead";
        os << "\n";
        for (Block *pred : block.getPredecessors()) {
          os << "   from ";
          pred->printAsOperand(os);
          os << " = ";
          auto *live = solver.lookupState<Executable>(
              solver.getProgramPoint<CFGEdge>(pred, &block));
          if (live)
            os << *live;
          else
            os << "dead";
          os << "\n";
        }
      }
      if (!region.empty()) {
        auto *preds = solver.lookupState<PredecessorState>(&region.front());
        if (preds)
          os << "region_preds: " << *preds << "\n";
      }
    }
    auto *preds = solver.lookupState<PredecessorState>(op);
    if (preds)
      os << "op_preds: " << *preds << "\n";
  });
}

namespace {
/// This is a simple analysis that implements a transfer function for constant
/// operations.
struct ConstantAnalysis : public DataFlowAnalysis {
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        if (failed(visit(op)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint point) override {
    Operation *op = point.get<Operation *>();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<Lattice<ConstantValue>>(op->getResult(0));
      propagateIfChanged(
          constant, constant->join(ConstantValue(value, op->getDialect())));
    }
    return success();
  }
};

/// This is a simple pass that runs dead code analysis with no constant value
/// provider. It marks everything as live.
struct TestDeadCodeAnalysisPass
    : public PassWrapper<TestDeadCodeAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDeadCodeAnalysisPass)

  StringRef getArgument() const override { return "test-dead-code-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<ConstantAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();
    printAnalysisResults(solver, op, llvm::errs());
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestDeadCodeAnalysisPass() {
  PassRegistration<TestDeadCodeAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
