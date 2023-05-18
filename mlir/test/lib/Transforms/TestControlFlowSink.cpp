//===- TestControlFlowSink.cpp - Test control-flow sink pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tests the control-flow sink utilities by implementing an example
// control-flow sink pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"

using namespace mlir;

namespace {
/// An example control-flow sink pass to test the control-flow sink utilites.
/// This pass will sink ops named `test.sink_me` and tag them with an attribute
/// `was_sunk` into the first region of `test.sink_target` ops.
struct TestControlFlowSinkPass
    : public PassWrapper<TestControlFlowSinkPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestControlFlowSinkPass)

  /// Get the command-line argument of the test pass.
  StringRef getArgument() const final { return "test-control-flow-sink"; }
  /// Get the description of the test pass.
  StringRef getDescription() const final {
    return "Test control-flow sink pass";
  }

  /// Runs the pass on the function.
  void runOnOperation() override {
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto shouldMoveIntoRegion = [](Operation *op, Region *region) {
      return region->getRegionNumber() == 0 &&
             op->getName().getStringRef() == "test.sink_me";
    };
    auto moveIntoRegion = [](Operation *op, Region *region) {
      Block &entry = region->front();
      op->moveBefore(&entry, entry.begin());
      op->setAttr("was_sunk",
                  Builder(op).getI32IntegerAttr(region->getRegionNumber()));
    };

    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "test.sink_target")
        return;
      SmallVector<Region *> regions =
          llvm::to_vector(RegionRange(op->getRegions()));
      controlFlowSink(regions, domInfo, shouldMoveIntoRegion, moveIntoRegion);
    });
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestControlFlowSink() {
  PassRegistration<TestControlFlowSinkPass>();
}
} // end namespace test
} // end namespace mlir
