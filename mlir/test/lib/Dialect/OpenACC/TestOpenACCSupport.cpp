//===- TestOpenACCSupport.cpp - Test OpenACCSupport Analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for testing the OpenACCSupport analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::acc;

namespace {

struct TestOpenACCSupportPass
    : public PassWrapper<TestOpenACCSupportPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOpenACCSupportPass)

  StringRef getArgument() const override { return "test-acc-support"; }

  StringRef getDescription() const override {
    return "Test OpenACCSupport analysis";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<acc::OpenACCDialect>();
    registry.insert<memref::MemRefDialect>();
  }
};

void TestOpenACCSupportPass::runOnOperation() {
  auto func = getOperation();

  // Get the OpenACCSupport analysis
  OpenACCSupport &support = getAnalysis<OpenACCSupport>();

  // Walk through operations looking for test attributes
  func.walk([&](Operation *op) {
    // Check for test.var_name attribute. This is the marker used to identify
    // the operations that need to be tested for getVariableName.
    if (op->hasAttr("test.var_name")) {
      // For each result of this operation, try to get the variable name
      for (auto result : op->getResults()) {
        std::string foundName = support.getVariableName(result);
        llvm::outs() << "op=" << *op << "\n\tgetVariableName=\"" << foundName
                     << "\"\n";
      }
    }

    // Check for test.recipe_name attribute. This is the marker used to identify
    // the operations that need to be tested for getRecipeName.
    if (auto recipeAttr =
            op->getAttrOfType<RecipeKindAttr>("test.recipe_name")) {
      RecipeKind kind = recipeAttr.getValue();
      // Get the type from the first result if available
      if (op->getNumResults() > 0) {
        Type type = op->getResult(0).getType();
        std::string recipeName =
            support.getRecipeName(kind, type, op->getResult(0));
        llvm::outs() << "op=" << *op
                     << "\n\tgetRecipeName(kind=" << stringifyRecipeKind(kind)
                     << ", type=" << type << ")=\"" << recipeName << "\"\n";
      }
    }

    // Check for test.emit_nyi attribute. This is the marker used to
    // test whether the not yet implemented case is reported correctly.
    if (auto messageAttr = op->getAttrOfType<StringAttr>("test.emit_nyi")) {
      support.emitNYI(op->getLoc(), messageAttr.getValue());
    }
  });
}

} // namespace

namespace mlir {
namespace test {

void registerTestOpenACCSupportPass() {
  PassRegistration<TestOpenACCSupportPass>();
}

} // namespace test
} // namespace mlir
