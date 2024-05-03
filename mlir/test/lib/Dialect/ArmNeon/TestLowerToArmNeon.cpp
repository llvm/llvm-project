//===- TestLowerToArmNeon.cpp - Test lowering to ArmNeon as a sink pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to ArmNeon as a
// generally usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmNeon/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "test-lower-to-arm-neon"

using namespace mlir;
using namespace mlir::arm_neon;

namespace {
struct TestLowerToArmNeon
    : public PassWrapper<TestLowerToArmNeon, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLowerToArmNeon)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final { return "Tests lower to arm Neon."; }
  TestLowerToArmNeon() = default;
  TestLowerToArmNeon(const TestLowerToArmNeon &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arm_neon::ArmNeonDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void TestLowerToArmNeon::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateLowerContractionToSMMLAPatternPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

namespace mlir {
namespace test {

void registerTestLowerToArmNeon() { PassRegistration<TestLowerToArmNeon>(); }

} // namespace test
} // namespace mlir
