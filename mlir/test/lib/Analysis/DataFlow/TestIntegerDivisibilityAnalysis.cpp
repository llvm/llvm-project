//===- TestIntegerDivisibilityAnalysis.cpp - Test int divisibility --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
struct TestIntegerDivisibilityAnalysisPass
    : public PassWrapper<TestIntegerDivisibilityAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestIntegerDivisibilityAnalysisPass)

  StringRef getArgument() const override {
    return "test-int-divisibility-analysis";
  }
  StringRef getDescription() const override {
    return "Test integer divisibility analysis by annotating "
           "'test.int_divisibility' ops with the divisibility of their "
           "operand.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect>();
  }

  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = &getContext();

    // The pass is rooted on `test.int_divisibility` ops, which are expected
    // to have a single operand for which to annotate divisibility information.
    SmallVector<std::pair<Operation *, Value>> queryOps;
    rootOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "test.int_divisibility" &&
          op->getNumOperands() == 1)
        queryOps.emplace_back(op, op->getOperand(0));
    });

    DataFlowSolver solver;
    // DeadCodeAnalysis is the base analysis that allows the solver to traverse
    // control flow. It is required by IntegerDivisibilityAnalysis.
    solver.load<DeadCodeAnalysis>();
    // SparseConstantPropagation allows the solver to call
    // visitNonControlFlowArguments and analyze arguments like loop induction
    // variables.
    solver.load<SparseConstantPropagation>();
    solver.load<IntegerDivisibilityAnalysis>();
    if (failed(solver.initializeAndRun(rootOp)))
      return signalPassFailure();

    for (auto &[op, value] : queryOps) {
      const auto *lattice =
          solver.lookupState<IntegerDivisibilityLattice>(value);
      if (!lattice || lattice->getValue().isUninitialized()) {
        op->setAttr("divisibility", StringAttr::get(context, "uninitialized"));
        continue;
      }

      // Format for the divisibility information is "udiv = X, sdiv = Y".
      const auto &div = lattice->getValue().getValue();
      std::string result;
      llvm::raw_string_ostream os(result);
      os << "udiv = " << div.udiv() << ", sdiv = " << div.sdiv();
      op->setAttr("divisibility", StringAttr::get(context, result));
    }
  }
};
} // end anonymous namespace

namespace mlir::test {
void registerTestIntegerDivisibilityAnalysisPass() {
  PassRegistration<TestIntegerDivisibilityAnalysisPass>();
}
} // end namespace mlir::test
