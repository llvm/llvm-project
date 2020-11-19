//===- TestSidEffects.cpp - Pass to test side effects ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct SideEffectsPass
    : public PassWrapper<SideEffectsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();

    // Walk operations detecting side effects.
    SmallVector<MemoryEffects::EffectInstance, 8> effects;
    module.walk([&](MemoryEffectOpInterface op) {
      effects.clear();
      op.getEffects(effects);

      // Check to see if this operation has any memory effects.
      if (effects.empty()) {
        op.emitRemark() << "operation has no memory effects";
        return;
      }

      for (MemoryEffects::EffectInstance instance : effects) {
        auto diag = op.emitRemark() << "found an instance of ";

        if (isa<MemoryEffects::Allocate>(instance.getEffect()))
          diag << "'allocate'";
        else if (isa<MemoryEffects::Free>(instance.getEffect()))
          diag << "'free'";
        else if (isa<MemoryEffects::Read>(instance.getEffect()))
          diag << "'read'";
        else if (isa<MemoryEffects::Write>(instance.getEffect()))
          diag << "'write'";

        if (instance.getValue())
          diag << " on a value,";

        diag << " on resource '" << instance.getResource()->getName() << "'";
      }
    });

    SmallVector<TestEffects::EffectInstance, 1> testEffects;
    module.walk([&](TestEffectOpInterface op) {
      testEffects.clear();
      op.getEffects(testEffects);

      if (testEffects.empty())
        return;

      for (const TestEffects::EffectInstance &instance : testEffects) {
        op.emitRemark() << "found a parametric effect with "
                        << instance.getParameters();
      }
    });
  }
};
} // end anonymous namespace

namespace mlir {
void registerSideEffectTestPasses() {
  PassRegistration<SideEffectsPass>("test-side-effects",
                                    "Test side effects interfaces");
}
} // namespace mlir
