//===- MLModelRunnerTest.cpp - test for MLModelRunner ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "gtest/gtest.h"

using namespace llvm;

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
              std::vector<float>{0.1, 0.2, 0.3, 0.4, 0.5}.data(),
              5 * sizeof(float));
  ASSERT_EQ(NIMR.getTensor<int64_t>(0)[0], 1);
  ASSERT_EQ(NIMR.getTensor<int64_t>(1)[8], 9);
  ASSERT_EQ(NIMR.getTensor<float>(2)[1], 0.2f);
}