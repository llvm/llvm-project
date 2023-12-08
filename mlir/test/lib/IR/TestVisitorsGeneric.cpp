//===- TestIRVisitorsGeneric.cpp - Pass to test the Generic IR visitors ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static std::string getStageDescription(const WalkStage &stage) {
  if (stage.isBeforeAllRegions())
    return "before all regions";
  if (stage.isAfterAllRegions())
    return "after all regions";
  return "before region #" + std::to_string(stage.getNextRegion());
}

namespace {
/// This pass exercises generic visitor with void callbacks and prints the order
/// and stage in which operations are visited.
struct TestGenericIRVisitorPass
    : public PassWrapper<TestGenericIRVisitorPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGenericIRVisitorPass)

  StringRef getArgument() const final { return "test-generic-ir-visitors"; }
  StringRef getDescription() const final { return "Test generic IR visitors."; }
  void runOnOperation() override {
    Operation *outerOp = getOperation();
    int stepNo = 0;
    outerOp->walk([&](Operation *op, const WalkStage &stage) {
      llvm::outs() << "step " << stepNo++ << " op '" << op->getName() << "' "
                   << getStageDescription(stage) << "\n";
    });

    // Exercise static inference of operation type.
    outerOp->walk([&](test::TwoRegionOp op, const WalkStage &stage) {
      llvm::outs() << "step " << stepNo++ << " op '" << op->getName() << "' "
                   << getStageDescription(stage) << "\n";
    });
  }
};

/// This pass exercises the generic visitor with non-void callbacks and prints
/// the order and stage in which operations are visited. It will interrupt the
/// walk based on attributes peesent in the IR.
struct TestGenericIRVisitorInterruptPass
    : public PassWrapper<TestGenericIRVisitorInterruptPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestGenericIRVisitorInterruptPass)

  StringRef getArgument() const final {
    return "test-generic-ir-visitors-interrupt";
  }
  StringRef getDescription() const final {
    return "Test generic IR visitors with interrupts.";
  }
  void runOnOperation() override {
    Operation *outerOp = getOperation();
    int stepNo = 0;

    auto walker = [&](Operation *op, const WalkStage &stage) {
      if (auto interruptBeforeAall =
              op->getAttrOfType<BoolAttr>("interrupt_before_all"))
        if (interruptBeforeAall.getValue() && stage.isBeforeAllRegions())
          return WalkResult::interrupt();

      if (auto interruptAfterAll =
              op->getAttrOfType<BoolAttr>("interrupt_after_all"))
        if (interruptAfterAll.getValue() && stage.isAfterAllRegions())
          return WalkResult::interrupt();

      if (auto interruptAfterRegion =
              op->getAttrOfType<IntegerAttr>("interrupt_after_region"))
        if (stage.isAfterRegion(
                static_cast<int>(interruptAfterRegion.getInt())))
          return WalkResult::interrupt();

      if (auto skipBeforeAall = op->getAttrOfType<BoolAttr>("skip_before_all"))
        if (skipBeforeAall.getValue() && stage.isBeforeAllRegions())
          return WalkResult::skip();

      if (auto skipAfterAll = op->getAttrOfType<BoolAttr>("skip_after_all"))
        if (skipAfterAll.getValue() && stage.isAfterAllRegions())
          return WalkResult::skip();

      if (auto skipAfterRegion =
              op->getAttrOfType<IntegerAttr>("skip_after_region"))
        if (stage.isAfterRegion(static_cast<int>(skipAfterRegion.getInt())))
          return WalkResult::skip();

      llvm::outs() << "step " << stepNo++ << " op '" << op->getName() << "' "
                   << getStageDescription(stage) << "\n";
      return WalkResult::advance();
    };

    // Interrupt the walk based on attributes on the operation.
    auto result = outerOp->walk(walker);

    if (result.wasInterrupted())
      llvm::outs() << "step " << stepNo++ << " walk was interrupted\n";

    // Exercise static inference of operation type.
    result = outerOp->walk([&](test::TwoRegionOp op, const WalkStage &stage) {
      return walker(op, stage);
    });

    if (result.wasInterrupted())
      llvm::outs() << "step " << stepNo++ << " walk was interrupted\n";
  }
};

struct TestGenericIRBlockVisitorInterruptPass
    : public PassWrapper<TestGenericIRBlockVisitorInterruptPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestGenericIRBlockVisitorInterruptPass)

  StringRef getArgument() const final {
    return "test-generic-ir-block-visitors-interrupt";
  }
  StringRef getDescription() const final {
    return "Test generic IR visitors with interrupts, starting with Blocks.";
  }

  void runOnOperation() override {
    int stepNo = 0;

    auto walker = [&](Block *block) {
      for (Operation &op : *block)
        if (op.getAttrOfType<BoolAttr>("interrupt"))
          return WalkResult::interrupt();

      llvm::outs() << "step " << stepNo++ << "\n";
      return WalkResult::advance();
    };

    auto result = getOperation()->walk(walker);
    if (result.wasInterrupted())
      llvm::outs() << "step " << stepNo++ << " walk was interrupted\n";
  }
};

struct TestGenericIRRegionVisitorInterruptPass
    : public PassWrapper<TestGenericIRRegionVisitorInterruptPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestGenericIRRegionVisitorInterruptPass)

  StringRef getArgument() const final {
    return "test-generic-ir-region-visitors-interrupt";
  }
  StringRef getDescription() const final {
    return "Test generic IR visitors with interrupts, starting with Regions.";
  }

  void runOnOperation() override {
    int stepNo = 0;

    auto walker = [&](Region *region) {
      for (Operation &op : region->getOps())
        if (op.getAttrOfType<BoolAttr>("interrupt"))
          return WalkResult::interrupt();

      llvm::outs() << "step " << stepNo++ << "\n";
      return WalkResult::advance();
    };

    auto result = getOperation()->walk(walker);
    if (result.wasInterrupted())
      llvm::outs() << "step " << stepNo++ << " walk was interrupted\n";
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestGenericIRVisitorsPass() {
  PassRegistration<TestGenericIRVisitorPass>();
  PassRegistration<TestGenericIRVisitorInterruptPass>();
  PassRegistration<TestGenericIRBlockVisitorInterruptPass>();
  PassRegistration<TestGenericIRRegionVisitorInterruptPass>();
}

} // namespace test
} // namespace mlir
