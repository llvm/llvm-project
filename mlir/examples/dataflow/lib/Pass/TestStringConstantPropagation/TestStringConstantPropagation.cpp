//===-- TestStringConstantPropagation.cpp - dataflow tutorial ---*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the implementation of TestStringConstantPropagation.
//
//===----------------------------------------------------------------------===//

#include "StringConstantPropagation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dataflow;

namespace mlir {
namespace {
class TestStringConstantPropagation
    : public PassWrapper<TestStringConstantPropagation, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStringConstantPropagation)
  StringRef getArgument() const final {
    return "test-string-constant-propagation";
  }
  StringRef getDescription() const final {
    return "Tests string constant propagation";
  }
  TestStringConstantPropagation() = default;
  TestStringConstantPropagation(const TestStringConstantPropagation &) {}
  void runOnOperation() override {
    Operation *op = getOperation();
    DataFlowSolver solver;
    // Load the analysis.
    solver.load<StringConstantPropagation>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Query the results and do something...
    op->walk([&](Operation *op) {
      if (op->getNumResults()) {
        Value result = op->getResult(0);
        auto stringConstant = solver.lookupState<StringConstant>(result);
        llvm::outs() << OpWithFlags(op, OpPrintingFlags().skipRegions())
                     << " : ";
        stringConstant->print(llvm::outs());
      }
    });
  }
};
} // namespace

namespace test {
void registerTestStringConstantPropagation() {
  PassRegistration<TestStringConstantPropagation>();
}
} // namespace test
} // namespace mlir
