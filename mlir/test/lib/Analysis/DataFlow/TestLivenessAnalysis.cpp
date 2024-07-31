//===- TestLivenessAnalysis.cpp - Test liveness analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/DataFlow/LivenessAnalysis.h>

#include <cassert>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

using namespace mlir;
using namespace mlir::dataflow;

namespace {

struct TestLivenessAnalysisPass
    : public PassWrapper<TestLivenessAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessAnalysisPass)

  StringRef getArgument() const override { return "test-liveness-analysis"; }

  void runOnOperation() override {
    auto &livenessAnalysis = getAnalysis<RunLivenessAnalysis>();

    Operation *op = getOperation();

    raw_ostream &os = llvm::outs();

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const Liveness *liveness = livenessAnalysis.getLiveness(operand);
        assert(liveness && "expected a sparse lattice");
        os << " operand #" << index << ": ";
        liveness->print(os);
        os << "\n";
      }
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        const Liveness *liveness = livenessAnalysis.getLiveness(operand);
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
