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
  combinedModule = spirv::combine(modules, combinedModuleBuilder, nullptr);

  for (spirv::ModuleOp module : modules)
    module.erase();
}

namespace mlir {
void registerTestSpirvModuleCombinerPass() {
  PassRegistration<TestModuleCombinerPass> registration(
      "test-spirv-module-combiner", "Tests SPIR-V module combiner library");
}
} // namespace mlir
