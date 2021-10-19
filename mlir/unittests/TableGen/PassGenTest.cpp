//===- PassGenTest.cpp - TableGen PassGen Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

#include "gmock/gmock.h"

std::unique_ptr<mlir::Pass> createTestPass(int v = 0);

#define GEN_PASS_REGISTRATION
#include "PassGenTest.h.inc"

#define GEN_PASS_CLASSES
#include "PassGenTest.h.inc"

struct TestPass : public TestPassBase<TestPass> {
  explicit TestPass(int v) : extraVal(v) {}

  void runOnOperation() override {}

  std::unique_ptr<mlir::Pass> clone() const {
    return TestPassBase<TestPass>::clone();
  }

  int extraVal;
};

std::unique_ptr<mlir::Pass> createTestPass(int v) {
  return std::make_unique<TestPass>(v);
}

TEST(PassGenTest, PassClone) {
  mlir::MLIRContext context;

  const auto unwrap = [](const std::unique_ptr<mlir::Pass> &pass) {
    return static_cast<const TestPass *>(pass.get());
  };

  const auto origPass = createTestPass(10);
  const auto clonePass = unwrap(origPass)->clone();

  EXPECT_EQ(unwrap(origPass)->extraVal, unwrap(clonePass)->extraVal);
}
