//===- TestOperationEquals.cpp - Passes to test OperationEquivalence ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This pass illustrates the IR def-use chains through printing.
struct TestOperationEqualPass
    : public PassWrapper<TestOperationEqualPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-operations-equality"; }
  StringRef getDescription() const final { return "Test operations equality."; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Expects two operations at the top-level:
    int opCount = module.getBody()->getOperations().size();
    if (opCount != 2) {
      module.emitError() << "expected 2 top-level ops in the module, got "
                         << opCount;
      return signalPassFailure();
    }
    DenseMap<Value, Value> valuesMap;
    auto mapValue = [&](Value lhs, Value rhs) {
      auto insertion = valuesMap.insert({lhs, rhs});
      return success(insertion.first->second == rhs);
    };

    Operation *first = &module.getBody()->front();
    llvm::outs() << first->getName().getStringRef() << " with attr "
                 << first->getAttrDictionary();
    OperationEquivalence::Flags flags{};
    if (!first->hasAttr("strict_loc_check"))
      flags |= OperationEquivalence::IgnoreLocations;
    if (OperationEquivalence::isEquivalentTo(first, &module.getBody()->back(),
                                             mapValue, mapValue, flags))
      llvm::outs() << " compares equals.\n";
    else
      llvm::outs() << " compares NOT equals!\n";
  }
};
} // end anonymous namespace

namespace mlir {
void registerTestOperationEqualPass() {
  PassRegistration<TestOperationEqualPass>();
}
} // namespace mlir
