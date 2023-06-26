//===- TestLivenessAnalysis.cpp - Test liveness analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
struct TestLivenessAnalysisPass
    : public PassWrapper<TestLivenessAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessAnalysisPass)

  StringRef getArgument() const override { return "test-liveness-analysis"; }

  void runOnOperation() override {
    Operation *op = getOperation();

    SymbolTableCollection symbolTable;

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LivenessAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::outs();
    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const Liveness *liveness = solver.lookupState<Liveness>(operand);
        assert(liveness && "expected a sparse lattice");
        os << " operand #" << index << ": ";
        liveness->print(os);
        os << "\n";
      }
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        const Liveness *liveness = solver.lookupState<Liveness>(operand);
        assert(liveness && "expected a sparse lattice");
        os << " result #" << index << ": ";
        liveness->print(os);
        os << "\n";
      }
    });
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestLivenessAnalysisPass() {
  PassRegistration<TestLivenessAnalysisPass>();
}
} // end namespace test
} // end namespace mlir
