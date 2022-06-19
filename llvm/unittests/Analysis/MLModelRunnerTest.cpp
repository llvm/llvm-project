//===- MLModelRunnerTest.cpp - test for MLModelRunner ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
// This is a mock of the kind of AOT-generated model evaluator. It has 2 tensors
// of shape {1}, and 'evaluation' adds them.
// The interface is the one expected by ReleaseModelRunner.
class MockAOTModel final {
  int64_t A = 0;
  int64_t B = 0;
  int64_t R = 0;

public:
  MockAOTModel() = default;
  int LookupArgIndex(const std::string &Name) {
    if (Name == "prefix_a")
      return 0;
    if (Name == "prefix_b")
      return 1;
    return -1;
  }
  int LookupResultIndex(const std::string &) { return 0; }
  void Run() { R = A + B; }
  void *result_data(int RIndex) {
    if (RIndex == 0)
      return &R;
    return nullptr;
  }
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
} // namespace llvm

TEST(NoInferenceModelRunner, AccessTensors) {
  const std::vector<TensorSpec> Inputs{
      TensorSpec::createSpec<int64_t>("F1", {1}),
      TensorSpec::createSpec<int64_t>("F2", {10}),
      TensorSpec::createSpec<float>("F2", {5}),
  };
  LLVMContext Ctx;
  NoInferenceModelRunner NIMR(Ctx, Inputs);
  NIMR.getTensor<int64_t>(0)[0] = 1;
  std::memcpy(NIMR.getTensor<int64_t>(1),
              std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.data(),
              10 * sizeof(int64_t));
  std::memcpy(NIMR.getTensor<float>(2),
              std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}.data(),
              5 * sizeof(float));
  ASSERT_EQ(NIMR.getTensor<int64_t>(0)[0], 1);
  ASSERT_EQ(NIMR.getTensor<int64_t>(1)[8], 9);
  ASSERT_EQ(NIMR.getTensor<float>(2)[1], 0.2f);
}

TEST(ReleaseModeRunner, NormalUse) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<MockAOTModel>>(
      Ctx, Inputs, "", "prefix_");
  *Evaluator->getTensor<int64_t>(0) = 1;
  *Evaluator->getTensor<int64_t>(1) = 2;
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), 3);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(0), 1);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(1), 2);
}

TEST(ReleaseModeRunner, ExtraFeatures) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1}),
                                 TensorSpec::createSpec<int64_t>("c", {1})};
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<MockAOTModel>>(
      Ctx, Inputs, "", "prefix_");
  *Evaluator->getTensor<int64_t>(0) = 1;
  *Evaluator->getTensor<int64_t>(1) = 2;
  *Evaluator->getTensor<int64_t>(2) = -3;
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), 3);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(0), 1);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(1), 2);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(2), -3);
}

TEST(ReleaseModeRunner, ExtraFeaturesOutOfOrder) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{
      TensorSpec::createSpec<int64_t>("a", {1}),
      TensorSpec::createSpec<int64_t>("c", {1}),
      TensorSpec::createSpec<int64_t>("b", {1}),
  };
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<MockAOTModel>>(
      Ctx, Inputs, "", "prefix_");
  *Evaluator->getTensor<int64_t>(0) = 1;         // a
  *Evaluator->getTensor<int64_t>(1) = 2;         // c
  *Evaluator->getTensor<int64_t>(2) = -3;        // b
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), -2); // a + b
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(0), 1);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(1), 2);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(2), -3);
}