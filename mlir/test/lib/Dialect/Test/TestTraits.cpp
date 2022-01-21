//===- TestTraits.cpp - Test trait folding --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// Trait Folder.
//===----------------------------------------------------------------------===//

OpFoldResult TestInvolutionTraitFailingOperationFolderOp::fold(
    ArrayRef<Attribute> operands) {
  // This failure should cause the trait fold to run instead.
  return {};
}

OpFoldResult TestInvolutionTraitSuccesfulOperationFolderOp::fold(
    ArrayRef<Attribute> operands) {
  auto argumentOp = getOperand();
  // The success case should cause the trait fold to be supressed.
  return argumentOp.getDefiningOp() ? argumentOp : OpFoldResult{};
}

namespace {
struct TestTraitFolder
    : public PassWrapper<TestTraitFolder, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-trait-folder"; }
  StringRef getDescription() const final { return "Run trait folding"; }
  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(),
                                       RewritePatternSet(&getContext()));
  }
};
} // namespace

namespace mlir {
void registerTestTraitsPass() { PassRegistration<TestTraitFolder>(); }
} // namespace mlir
