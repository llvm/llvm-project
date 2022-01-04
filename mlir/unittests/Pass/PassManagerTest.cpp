//===- PassManagerTest.cpp - PassManager unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"

#include <memory>

using namespace mlir;
using namespace mlir::detail;

namespace {
/// Analysis that operates on any operation.
struct GenericAnalysis {
  GenericAnalysis(Operation *op) : isFunc(isa<FuncOp>(op)) {}
  const bool isFunc;
};

/// Analysis that operates on a specific operation.
struct OpSpecificAnalysis {
  OpSpecificAnalysis(FuncOp op) : isSecret(op.getName() == "secret") {}
  const bool isSecret;
};

/// Simple pass to annotate a FuncOp with the results of analysis.
struct AnnotateFunctionPass
    : public PassWrapper<AnnotateFunctionPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp op = getOperation();
    Builder builder(op->getParentOfType<ModuleOp>());

    auto &ga = getAnalysis<GenericAnalysis>();
    auto &sa = getAnalysis<OpSpecificAnalysis>();

    op->setAttr("isFunc", builder.getBoolAttr(ga.isFunc));
    op->setAttr("isSecret", builder.getBoolAttr(sa.isSecret));
  }
};

TEST(PassManagerTest, OpSpecificAnalysis) {
  MLIRContext context;
  Builder builder(&context);

  // Create a module with 2 functions.
  OwningModuleRef module(ModuleOp::create(UnknownLoc::get(&context)));
  for (StringRef name : {"secret", "not_secret"}) {
    FuncOp func =
        FuncOp::create(builder.getUnknownLoc(), name,
                       builder.getFunctionType(llvm::None, llvm::None));
    func.setPrivate();
    module->push_back(func);
  }

  // Instantiate and run our pass.
  PassManager pm(&context);
  pm.addNestedPass<FuncOp>(std::make_unique<AnnotateFunctionPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Verify that each function got annotated with expected attributes.
  for (FuncOp func : module->getOps<FuncOp>()) {
    ASSERT_TRUE(func->getAttr("isFunc").isa<BoolAttr>());
    EXPECT_TRUE(func->getAttr("isFunc").cast<BoolAttr>().getValue());

    bool isSecret = func.getName() == "secret";
    ASSERT_TRUE(func->getAttr("isSecret").isa<BoolAttr>());
    EXPECT_EQ(func->getAttr("isSecret").cast<BoolAttr>().getValue(), isSecret);
  }
}

namespace {
struct InvalidPass : Pass {
  InvalidPass() : Pass(TypeID::get<InvalidPass>(), StringRef("invalid_op")) {}
  StringRef getName() const override { return "Invalid Pass"; }
  void runOnOperation() override {}

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<InvalidPass>(
        *static_cast<const InvalidPass *>(this));
  }
};
} // namespace

TEST(PassManagerTest, InvalidPass) {
  MLIRContext context;
  context.allowUnregisteredDialects();

  // Create a module
  OwningModuleRef module(ModuleOp::create(UnknownLoc::get(&context)));

  // Add a single "invalid_op" operation
  OpBuilder builder(&module->getBodyRegion());
  OperationState state(UnknownLoc::get(&context), "invalid_op");
  builder.insert(Operation::create(state));

  // Register a diagnostic handler to capture the diagnostic so that we can
  // check it later.
  std::unique_ptr<Diagnostic> diagnostic;
  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    diagnostic = std::make_unique<Diagnostic>(std::move(diag));
  });

  // Instantiate and run our pass.
  PassManager pm(&context);
  pm.nest("invalid_op").addPass(std::make_unique<InvalidPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(failed(result));
  ASSERT_TRUE(diagnostic.get() != nullptr);
  EXPECT_EQ(
      diagnostic->str(),
      "'invalid_op' op trying to schedule a pass on an unregistered operation");

  // Check that clearing the pass manager effectively removed the pass.
  pm.clear();
  result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Check that adding the pass at the top-level triggers a fatal error.
  ASSERT_DEATH(pm.addPass(std::make_unique<InvalidPass>()), "");
}

} // namespace
