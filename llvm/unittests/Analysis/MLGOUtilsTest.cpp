//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Utils/MLGOUtils.h"
#include "llvm/Analysis/EmitCModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
#include <map>
#include <string>

using namespace llvm;

namespace {

class MockAOTModel final {
  int64_t A = 0;
  int64_t B = 0;
  int64_t R = 0;

public:
  MockAOTModel() = default;
  int LookupArgIndex(const std::string &Name) {
    if (Name == "feed_a")
      return 0;
    if (Name == "feed_b")
      return 1;
    return -1;
  }
  int LookupResultIndex(const std::string &) { return 0; }
  void Run() { R = A + B; }
  void *result_data(int RIndex) { return (RIndex == 0) ? &R : nullptr; }
  void *arg_data(int Index) {
    switch (Index) {
    case 0:
      return &A;
    case 1:
      return &B;
    default:
      return nullptr;
    }
  }
};

class MockEmitCModel1 final {
  int64_t A = 0;
  int64_t B = 0;

public:
  std::map<std::string, void *> reflectionMap;

  MockEmitCModel1() : reflectionMap{{"a", &A}, {"b", &B}} {}

  int64_t operator()() { return A - B; }
};

class MockEmitCModel2 final {
  int64_t A = 0;
  int64_t B = 0;

public:
  std::map<std::string, void *> reflectionMap;

  MockEmitCModel2() : reflectionMap{{"a", &A}, {"b", &B}} {}

  int64_t operator()() { return A + B; }
};

enum class TestModelChoice { Default, Model1, Model2 };

TEST(MLGOUtilsTest, IsReleaseModelValid) {
  // With NoopSavedModelImpl (no embedded AOT model):
  // 1. Default model choice without interactive channel -> invalid
  EXPECT_FALSE(
      (isReleaseModelValid<NoopSavedModelImpl>("", TestModelChoice::Default)));
  // 2. Selected model choice -> valid
  EXPECT_TRUE(
      (isReleaseModelValid<NoopSavedModelImpl>("", TestModelChoice::Model1)));
  EXPECT_TRUE(
      (isReleaseModelValid<NoopSavedModelImpl>("", TestModelChoice::Model2)));
  // 3. Interactive channel specified -> valid regardless of model choice
  EXPECT_TRUE((isReleaseModelValid<NoopSavedModelImpl>(
      "channel", TestModelChoice::Default)));

  // With a custom default model value:
  EXPECT_FALSE((isReleaseModelValid<NoopSavedModelImpl>(
      "", TestModelChoice::Model1,
      /*DefaultModelVal=*/TestModelChoice::Model1)));

  // With a valid embedded AOT model (MockAOTModel):
  EXPECT_TRUE(
      (isReleaseModelValid<MockAOTModel>("", TestModelChoice::Default)));

  // Overload with cl::opt<EnumType>:
  cl::opt<TestModelChoice> OptChoice("test-mlgo-utils-choice",
                                     cl::init(TestModelChoice::Default));
  EXPECT_FALSE((isReleaseModelValid<NoopSavedModelImpl>("", OptChoice)));
  OptChoice = TestModelChoice::Model1;
  EXPECT_TRUE((isReleaseModelValid<NoopSavedModelImpl>("", OptChoice)));
  OptChoice = TestModelChoice::Default;
  EXPECT_TRUE((isReleaseModelValid<NoopSavedModelImpl>("channel", OptChoice)));
  EXPECT_TRUE((isReleaseModelValid<MockAOTModel>("", OptChoice)));
}

TEST(MLGOUtilsTest, CreateReleaseModeModelRunnerModelSelection) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  TensorSpec OutputSpec = TensorSpec::createSpec<int64_t>("result", {1});

  auto CreateRunnerForChoice =
      [&](TestModelChoice Choice) -> std::unique_ptr<MLModelRunner> {
    auto Factory = [&](LLVMContext &C, const std::vector<TensorSpec> &Specs)
        -> std::unique_ptr<MLModelRunner> {
      switch (Choice) {
      case TestModelChoice::Default:
        return nullptr;
      case TestModelChoice::Model1:
        return std::make_unique<EmitCModelRunner<MockEmitCModel1>>(C, Specs);
      case TestModelChoice::Model2:
        return std::make_unique<EmitCModelRunner<MockEmitCModel2>>(C, Specs);
      }
      llvm_unreachable("unknown model choice");
    };
    return createReleaseModeModelRunner<NoopSavedModelImpl,
                                        /*HaveMLIRLowering=*/true>(
        Ctx, Inputs, "decision", "", OutputSpec, Factory);
  };

  // Default choice produces no runner
  EXPECT_EQ(CreateRunnerForChoice(TestModelChoice::Default), nullptr);

  // Model1 choice produces Model1 (A - B)
  auto Runner1 = CreateRunnerForChoice(TestModelChoice::Model1);
  ASSERT_NE(Runner1, nullptr);
  EXPECT_TRUE(EmitCModelRunner<MockEmitCModel1>::classof(Runner1.get()));
  *Runner1->getTensor<int64_t>(0) = 10;
  *Runner1->getTensor<int64_t>(1) = 3;
  EXPECT_EQ(Runner1->evaluate<int64_t>(), 7);

  // Model2 choice produces Model2 (A + B)
  auto Runner2 = CreateRunnerForChoice(TestModelChoice::Model2);
  ASSERT_NE(Runner2, nullptr);
  EXPECT_TRUE(EmitCModelRunner<MockEmitCModel2>::classof(Runner2.get()));
  *Runner2->getTensor<int64_t>(0) = 10;
  *Runner2->getTensor<int64_t>(1) = 3;
  EXPECT_EQ(Runner2->evaluate<int64_t>(), 13);
}

TEST(MLGOUtilsTest, CreateReleaseModeModelRunnerAOTFallback) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  TensorSpec OutputSpec = TensorSpec::createSpec<int64_t>("result", {1});

  auto DummyEmitCFactory =
      [](LLVMContext &,
         const std::vector<TensorSpec> &) -> std::unique_ptr<MLModelRunner> {
    llvm_unreachable(
        "EmitC factory should not be called when HaveMLIRLowering=false");
  };

  auto Runner =
      createReleaseModeModelRunner<MockAOTModel, /*HaveMLIRLowering=*/false>(
          Ctx, Inputs, "result", "", OutputSpec, DummyEmitCFactory);
  ASSERT_NE(Runner, nullptr);
  EXPECT_TRUE(ReleaseModeModelRunner<MockAOTModel>::classof(Runner.get()));
  *Runner->getTensor<int64_t>(0) = 10;
  *Runner->getTensor<int64_t>(1) = 3;
  EXPECT_EQ(Runner->evaluate<int64_t>(), 13);
}

} // namespace
