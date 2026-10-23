//===------ TestCompositePass.cpp --- composite test pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the composite pass utility.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
/// A pass that increments an attribute on the operation on every run,
/// guaranteeing that it never reaches a fixed point. Used to test
/// `CompositeFixedPointPass`'s convergence-failure handling.
struct TestIncrementAttrPass
    : public PassWrapper<TestIncrementAttrPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIncrementAttrPass)

  StringRef getArgument() const final { return "test-increment-attr"; }
  StringRef getDescription() const final {
    return "Test pass that increments an attribute on the operation on "
           "every run, so it never reaches a fixed point";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    int64_t counter = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("test.counter"))
      counter = attr.getInt();
    op->setAttr("test.counter", Builder(op).getI64IntegerAttr(counter + 1));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestCompositePass() {
  registerPassPipeline(
      "test-composite-fixed-point-pass", "Test composite pass",
      [](OpPassManager &pm, StringRef optionsStr,
         function_ref<LogicalResult(const Twine &)> errorHandler) {
        if (!optionsStr.empty())
          return failure();

        pm.addPass(createCompositeFixedPointPass(
            "TestCompositePass", [](OpPassManager &p) {
              p.addPass(createCanonicalizerPass());
              p.addPass(createCSEPass());
            }));
        return success();
      },
      [](function_ref<void(const detail::PassOptions &)>) {});

  PassRegistration<TestIncrementAttrPass>();
}
} // namespace test
} // namespace mlir
