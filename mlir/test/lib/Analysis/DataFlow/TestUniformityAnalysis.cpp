//===- TestUniformityAnalysis.cpp - Test uniformity analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
/// Reports the uniformity of the results of every operation carrying a `tag`
/// string attribute, and the uniformity of its execution, as a remark.
struct TestUniformityAnalysisPass
    : public PassWrapper<TestUniformityAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUniformityAnalysisPass)

  StringRef getArgument() const override { return "test-uniformity-analysis"; }
  StringRef getDescription() const override {
    return "Test uniformity analysis by reporting the uniformity of the "
           "results and of the execution of every operation with a `tag` "
           "attribute";
  }

  void runOnOperation() override {
    Operation *rootOp = getOperation();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<UniformityAnalysis>();
    if (failed(solver.initializeAndRun(rootOp)))
      return signalPassFailure();

    rootOp->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "uniformity of \"" << tag.getValue() << "\": results = [";
      llvm::interleaveComma(op->getResults(), os, [&](Value result) {
        os << getUniformity(solver, result);
      });
      os << "], execution = " << getExecutionUniformity(solver, op);
      op->emitRemark() << message;
    });
  }
};
} // end anonymous namespace

namespace mlir::test {
void registerTestUniformityAnalysisPass() {
  PassRegistration<TestUniformityAnalysisPass>();
}
} // end namespace mlir::test
