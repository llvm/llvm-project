//===- TestModuleCombiner.cpp - Pass to test SPIR-V module combiner lib ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class TestModuleCombinerPass
    : public PassWrapper<TestModuleCombinerPass,
                         OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { return "test-spirv-module-combiner"; }
  StringRef getDescription() const final {
    return "Tests SPIR-V module combiner library";
  }
  TestModuleCombinerPass() = default;
  TestModuleCombinerPass(const TestModuleCombinerPass &) {}
  void runOnOperation() override;

private:
  OwningOpRef<spirv::ModuleOp> combinedModule;
};
} // namespace

void TestModuleCombinerPass::runOnOperation() {
  auto modules = llvm::to_vector<4>(getOperation().getOps<spirv::ModuleOp>());

  OpBuilder combinedModuleBuilder(modules[0]);

  auto listener = [](spirv::ModuleOp originalModule, StringRef oldSymbol,
                     StringRef newSymbol) {
    llvm::outs() << "[" << originalModule.getName() << "] " << oldSymbol
                 << " -> " << newSymbol << "\n";
  };

  combinedModule = spirv::combine(modules, combinedModuleBuilder, listener);

  for (spirv::ModuleOp module : modules)
    module.erase();
}

namespace mlir {
void registerTestSpirvModuleCombinerPass() {
  PassRegistration<TestModuleCombinerPass>();
}
} // namespace mlir
