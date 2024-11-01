//===- PassManagerTest.cpp - PassManager unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"

#include <memory>

using namespace mlir;
using namespace mlir::detail;

namespace {
/// Analysis that operates on any operation.
struct GenericAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericAnalysis)

  GenericAnalysis(Operation *op) : isFunc(isa<func::FuncOp>(op)) {}
  const bool isFunc;
};

/// Analysis that operates on a specific operation.
struct OpSpecificAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSpecificAnalysis)

  OpSpecificAnalysis(func::FuncOp op) : isSecret(op.getName() == "secret") {}
  const bool isSecret;
};

/// Simple pass to annotate a func::FuncOp with the results of analysis.
struct AnnotateFunctionPass
    : public PassWrapper<AnnotateFunctionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateFunctionPass)

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    Builder builder(op->getParentOfType<ModuleOp>());

    auto &ga = getAnalysis<GenericAnalysis>();
    auto &sa = getAnalysis<OpSpecificAnalysis>();

    op->setAttr("isFunc", builder.getBoolAttr(ga.isFunc));
    op->setAttr("isSecret", builder.getBoolAttr(sa.isSecret));
  }
};

TEST(PassManagerTest, OpSpecificAnalysis) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create a module with 2 functions.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (StringRef name : {"secret", "not_secret"}) {
    auto func = func::FuncOp::create(
        builder.getUnknownLoc(), name,
        builder.getFunctionType(std::nullopt, std::nullopt));
    func.setPrivate();
    module->push_back(func);
  }

  // Instantiate and run our pass.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addNestedPass<func::FuncOp>(std::make_unique<AnnotateFunctionPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Verify that each function got annotated with expected attributes.
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    ASSERT_TRUE(isa<BoolAttr>(func->getAttr("isFunc")));
    EXPECT_TRUE(cast<BoolAttr>(func->getAttr("isFunc")).getValue());

    bool isSecret = func.getName() == "secret";
    ASSERT_TRUE(isa<BoolAttr>(func->getAttr("isSecret")));
    EXPECT_EQ(cast<BoolAttr>(func->getAttr("isSecret")).getValue(), isSecret);
  }
}

namespace {
struct InvalidPass : Pass {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InvalidPass)

  InvalidPass() : Pass(TypeID::get<InvalidPass>(), StringRef("invalid_op")) {}
  StringRef getName() const override { return "Invalid Pass"; }
  void runOnOperation() override {}
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

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
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

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
  auto pm = PassManager::on<ModuleOp>(&context);
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
  ASSERT_DEATH(pm.addPass(std::make_unique<InvalidPass>()),
               "Can't add pass 'Invalid Pass' restricted to 'invalid_op' on a "
               "PassManager intended to run on 'builtin.module', did you "
               "intend to nest?");
}

/// Simple pass to annotate a func::FuncOp with the results of analysis.
struct InitializeCheckingPass
    : public PassWrapper<InitializeCheckingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitializeCheckingPass)
  LogicalResult initialize(MLIRContext *ctx) final {
    initialized = true;
    return success();
  }
  bool initialized = false;

  void runOnOperation() override {
    if (!initialized) {
      getOperation()->emitError() << "Pass isn't initialized!";
      signalPassFailure();
    }
  }
};

TEST(PassManagerTest, PassInitialization) {
  MLIRContext context;
  context.allowUnregisteredDialects();

  // Create a module
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

  // Instantiate and run our pass.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addPass(std::make_unique<InitializeCheckingPass>());
  EXPECT_TRUE(succeeded(pm.run(module.get())));

  // Adding a second copy of the pass, we should also initialize it!
  pm.addPass(std::make_unique<InitializeCheckingPass>());
  EXPECT_TRUE(succeeded(pm.run(module.get())));
}

} // namespace
