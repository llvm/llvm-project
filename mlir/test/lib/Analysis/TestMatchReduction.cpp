//===- TestMatchReduction.cpp - Test the match reduction utility ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a test pass for the match reduction utility.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

void printReductionResult(Operation *redRegionOp, unsigned numOutput,
                          Value reducedValue,
                          ArrayRef<Operation *> combinerOps) {
  if (reducedValue) {
    redRegionOp->emitRemark("Reduction found in output #") << numOutput << "!";
    redRegionOp->emitRemark("Reduced Value: ") << reducedValue;
    for (Operation *combOp : combinerOps)
      redRegionOp->emitRemark("Combiner Op: ") << *combOp;

    return;
  }

  redRegionOp->emitRemark("Reduction NOT found in output #")
      << numOutput << "!";
}

struct TestMatchReductionPass
    : public PassWrapper<TestMatchReductionPass,
                         InterfacePass<FunctionOpInterface>> {
  StringRef getArgument() const final { return "test-match-reduction"; }
  StringRef getDescription() const final {
    return "Test the match reduction utility.";
  }

  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    func->emitRemark("Testing function");

    func.walk<WalkOrder::PreOrder>([](Operation *op) {
      if (isa<FunctionOpInterface>(op))
        return;

      // Limit testing to ops with only one region.
      if (op->getNumRegions() != 1)
        return;

      Region &region = op->getRegion(0);
      if (!region.hasOneBlock())
        return;

      // We expect all the tested region ops to have 1 input by default. The
      // remaining arguments are assumed to be outputs/reductions and there must
      // be at least one.
      // TODO: Extend it to support more generic cases.
      Block &regionEntry = region.front();
      auto args = regionEntry.getArguments();
      if (args.size() < 2)
        return;

      auto outputs = args.drop_front();
      for (int i = 0, size = outputs.size(); i < size; ++i) {
        SmallVector<Operation *, 4> combinerOps;
        Value reducedValue = matchReduction(outputs, i, combinerOps);
        printReductionResult(op, i, reducedValue, combinerOps);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestMatchReductionPass() {
  PassRegistration<TestMatchReductionPass>();
}
} // namespace test
} // namespace mlir
