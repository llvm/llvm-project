//===-- TestMetadataAnalsys.cpp - dataflow tutorial -------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the implementation of TestMetadataAnalysisPass.
//
//===----------------------------------------------------------------------===//

#include "MetadataAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dataflow;

namespace mlir {
namespace {
class TestMetadataAnalysisPass
    : public PassWrapper<TestMetadataAnalysisPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMetadataAnalysisPass)
  StringRef getArgument() const final { return "test-metadata-analysis"; }
  StringRef getDescription() const final { return "Tests metadata analysis"; }
  TestMetadataAnalysisPass() = default;
  TestMetadataAnalysisPass(const TestMetadataAnalysisPass &) {}
  void runOnOperation() override {
    Operation *op = getOperation();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<MetadataAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // If an op has more than one result, then the lattice is the same for each
    // result, and we just print one of the results.
    op->walk([&](Operation *op) {
      if (op->getNumResults()) {
        Value result = op->getResult(0);
        auto lattice = solver.lookupState<MetadataLatticeValueLattice>(result);
        lattice->print(llvm::outs());
      }
    });
  }
};
} // namespace

namespace test {
void registerTestMetadataAnalysisPass() {
  PassRegistration<TestMetadataAnalysisPass>();
}
} // namespace test
} // namespace mlir
