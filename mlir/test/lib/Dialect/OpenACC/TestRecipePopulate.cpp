//===- TestRecipePopulate.cpp - Test Recipe Population -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for testing the createAndPopulate methods
// of the recipe operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::acc;

namespace {

struct TestRecipePopulatePass
    : public PassWrapper<TestRecipePopulatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRecipePopulatePass)

  TestRecipePopulatePass() = default;
  TestRecipePopulatePass(const TestRecipePopulatePass &pass)
      : PassWrapper(pass) {
    recipeType = pass.recipeType;
  }

  Pass::Option<std::string> recipeType{
      *this, "recipe-type",
      llvm::cl::desc("Recipe type: private or firstprivate"),
      llvm::cl::init("private")};

  StringRef getArgument() const override { return "test-acc-recipe-populate"; }

  StringRef getDescription() const override {
    return "Test OpenACC recipe population";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<acc::OpenACCDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }
};

void TestRecipePopulatePass::runOnOperation() {
  auto module = getOperation();
  OpBuilder builder(&getContext());

  // Collect all test variables
  SmallVector<std::tuple<Operation *, Value, std::string>> testVars;

  module.walk([&](Operation *op) {
    if (auto varName = op->getAttrOfType<StringAttr>("test.var")) {
      for (auto result : op->getResults()) {
        testVars.push_back({op, result, varName.str()});
      }
    }
  });

  // Generate recipes at module level
  builder.setInsertionPoint(&module.getBodyRegion().front(),
                            module.getBodyRegion().front().begin());

  for (auto [op, var, varName] : testVars) {
    Location loc = op->getLoc();

    std::string recipeName = recipeType.getValue() + "_" + varName;
    ValueRange bounds; // No bounds for memref tests

    if (recipeType == "private") {
      auto recipe = PrivateRecipeOp::createAndPopulate(
          builder, loc, recipeName, var.getType(), varName, bounds);

      if (!recipe) {
        op->emitError("Failed to create private recipe for ") << varName;
      }
    } else if (recipeType == "firstprivate") {
      auto recipe = FirstprivateRecipeOp::createAndPopulate(
          builder, loc, recipeName, var.getType(), varName, bounds);

      if (!recipe) {
        op->emitError("Failed to create firstprivate recipe for ") << varName;
      }
    }
  }
}

} // namespace

namespace mlir {
namespace test {

void registerTestRecipePopulatePass() {
  PassRegistration<TestRecipePopulatePass>();
}

} // namespace test
} // namespace mlir
