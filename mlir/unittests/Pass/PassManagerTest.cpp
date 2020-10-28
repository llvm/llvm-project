//===- PassManagerTest.cpp - PassManager unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"

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
/// Note: not using FunctionPass as it skip external functions.
struct AnnotateFunctionPass
    : public PassWrapper<AnnotateFunctionPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp op = getOperation();
    Builder builder(op.getParentOfType<ModuleOp>());

    auto &ga = getAnalysis<GenericAnalysis>();
    auto &sa = getAnalysis<OpSpecificAnalysis>();

    op.setAttr("isFunc", builder.getBoolAttr(ga.isFunc));
    op.setAttr("isSecret", builder.getBoolAttr(sa.isSecret));
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
    module->push_back(func);
  }

  // Instantiate and run our pass.
  PassManager pm(&context);
  pm.addNestedPass<FuncOp>(std::make_unique<AnnotateFunctionPass>());
  LogicalResult result = pm.run(module.get());
  EXPECT_TRUE(succeeded(result));

  // Verify that each function got annotated with expected attributes.
  for (FuncOp func : module->getOps<FuncOp>()) {
    ASSERT_TRUE(func.getAttr("isFunc").isa<BoolAttr>());
    EXPECT_TRUE(func.getAttr("isFunc").cast<BoolAttr>().getValue());

    bool isSecret = func.getName() == "secret";
    ASSERT_TRUE(func.getAttr("isSecret").isa<BoolAttr>());
    EXPECT_EQ(func.getAttr("isSecret").cast<BoolAttr>().getValue(), isSecret);
  }
}

} // end namespace
