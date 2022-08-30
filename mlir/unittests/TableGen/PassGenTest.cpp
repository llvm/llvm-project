//===- PassGenTest.cpp - TableGen PassGen Tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "gmock/gmock.h"

std::unique_ptr<mlir::Pass> createTestPassWithCustomConstructor(int v = 0);

#define GEN_PASS_DECL_TESTPASS
#define GEN_PASS_DECL_TESTPASSWITHOPTIONS
#define GEN_PASS_DECL_TESTPASSWITHCUSTOMCONSTRUCTOR
#define GEN_PASS_REGISTRATION
#include "PassGenTest.h.inc"

#define GEN_PASS_DEF_TESTPASS
#define GEN_PASS_DEF_TESTPASSWITHOPTIONS
#define GEN_PASS_DEF_TESTPASSWITHCUSTOMCONSTRUCTOR
#include "PassGenTest.h.inc"

struct TestPass : public impl::TestPassBase<TestPass> {
  using TestPassBase::TestPassBase;

  void runOnOperation() override {}

  std::unique_ptr<mlir::Pass> clone() const {
    return TestPassBase<TestPass>::clone();
  }
};

TEST(PassGenTest, defaultGeneratedConstructor) {
  std::unique_ptr<mlir::Pass> pass = createTestPass();
  EXPECT_TRUE(pass.get() != nullptr);
}

TEST(PassGenTest, PassClone) {
  mlir::MLIRContext context;

  const auto unwrap = [](const std::unique_ptr<mlir::Pass> &pass) {
    return static_cast<const TestPass *>(pass.get());
  };

  const auto origPass = createTestPass();
  const auto clonePass = unwrap(origPass)->clone();

  EXPECT_TRUE(clonePass.get() != nullptr);
  EXPECT_TRUE(origPass.get() != clonePass.get());
}

struct TestPassWithOptions
    : public impl::TestPassWithOptionsBase<TestPassWithOptions> {
  using TestPassWithOptionsBase::TestPassWithOptionsBase;

  void runOnOperation() override {}

  std::unique_ptr<mlir::Pass> clone() const {
    return TestPassWithOptionsBase<TestPassWithOptions>::clone();
  }

  int getTestOption() const { return testOption; }

  llvm::ArrayRef<int64_t> getTestListOption() const { return testListOption; }
};

TEST(PassGenTest, PassOptions) {
  mlir::MLIRContext context;

  TestPassWithOptionsOptions options;
  options.testOption = 57;

  llvm::SmallVector<int64_t, 2> testListOption = {1, 2};
  options.testListOption = testListOption;

  const auto unwrap = [](const std::unique_ptr<mlir::Pass> &pass) {
    return static_cast<const TestPassWithOptions *>(pass.get());
  };

  const auto pass = createTestPassWithOptions(options);

  EXPECT_EQ(unwrap(pass)->getTestOption(), 57);
  EXPECT_EQ(unwrap(pass)->getTestListOption()[0], 1);
  EXPECT_EQ(unwrap(pass)->getTestListOption()[1], 2);
}

struct TestPassWithCustomConstructor
    : public impl::TestPassWithCustomConstructorBase<
          TestPassWithCustomConstructor> {
  explicit TestPassWithCustomConstructor(int v) : extraVal(v) {}

  void runOnOperation() override {}

  std::unique_ptr<mlir::Pass> clone() const {
    return TestPassWithCustomConstructorBase<
        TestPassWithCustomConstructor>::clone();
  }

  unsigned int extraVal = 23;
};

std::unique_ptr<mlir::Pass> createTestPassWithCustomConstructor(int v) {
  return std::make_unique<TestPassWithCustomConstructor>(v);
}

TEST(PassGenTest, PassCloneWithCustomConstructor) {
  mlir::MLIRContext context;

  const auto unwrap = [](const std::unique_ptr<mlir::Pass> &pass) {
    return static_cast<const TestPassWithCustomConstructor *>(pass.get());
  };

  const auto origPass = createTestPassWithCustomConstructor(10);
  const auto clonePass = unwrap(origPass)->clone();

  EXPECT_EQ(unwrap(origPass)->extraVal, unwrap(clonePass)->extraVal);
}
