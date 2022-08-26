//===- TFUtilsTest.cpp - test for TFUtils ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;

extern const char *TestMainArgv0;

// NOTE! This test model is currently also used by test/Transforms/Inline/ML tests
//- relevant if updating this model.
static std::string getModelPath() {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "ir2native_x86_64_model");
  return std::string(InputsDir);
}

// Test observable behavior when no model is provided.
TEST(TFUtilsTest, NoModel) {
  TFModelEvaluator Evaluator("", {}, {});
  EXPECT_FALSE(Evaluator.isValid());
}

// Test we can correctly load a savedmodel and evaluate it.
TEST(TFUtilsTest, LoadAndExecuteTest) {
  // We use the ir2native model for test. We know it has one feature of
  // dimension (1, 214)
  const static int64_t KnownSize = 214;
  std::vector<TensorSpec> InputSpecs{TensorSpec::createSpec<int32_t>(
      "serving_default_input_1", {1, KnownSize})};
  std::vector<TensorSpec> OutputSpecs{
      TensorSpec::createSpec<float>("StatefulPartitionedCall", {1})};

  TFModelEvaluator Evaluator(getModelPath(), InputSpecs, OutputSpecs);
  EXPECT_TRUE(Evaluator.isValid());

  int32_t *V = Evaluator.getInput<int32_t>(0);
  // Fill it up with 1's, we know the output.
  for (auto I = 0; I < KnownSize; ++I) {
    V[I] = 1;
  }
  {
    auto ER = Evaluator.evaluate();
    EXPECT_TRUE(ER.hasValue());
    float Ret = *ER->getTensorValue<float>(0);
    EXPECT_EQ(static_cast<int64_t>(Ret), 80);
    EXPECT_EQ(ER->getUntypedTensorValue(0),
              reinterpret_cast<const void *>(ER->getTensorValue<float>(0)));
  }
  // The input vector should be unchanged
  for (auto I = 0; I < KnownSize; ++I) {
    EXPECT_EQ(V[I], 1);
  }
  // Zero-out the unused position '0' of the instruction histogram, which is
  // after the first 9 calculated values. Should the the same result.
  V[9] = 0;
  {
    auto ER = Evaluator.evaluate();
    EXPECT_TRUE(ER.hasValue());
    float Ret = *ER->getTensorValue<float>(0);
    EXPECT_EQ(static_cast<int64_t>(Ret), 80);
  }
}

// Test incorrect input setup
TEST(TFUtilsTest, EvalError) {
  // We use the ir2native model for test. We know it has one feature of
  // dimension (1, 214)
  const static int64_t KnownSize = 213;
  std::vector<TensorSpec> InputSpecs{TensorSpec::createSpec<int32_t>(
      "serving_default_input_1", {1, KnownSize})};
  std::vector<TensorSpec> OutputSpecs{
      TensorSpec::createSpec<float>("StatefulPartitionedCall", {1})};

  TFModelEvaluator Evaluator(getModelPath(), InputSpecs, OutputSpecs);
  EXPECT_FALSE(Evaluator.isValid());
}

TEST(TFUtilsTest, UnsupportedFeature) {
  const static int64_t KnownSize = 214;
  std::vector<TensorSpec> InputSpecs{
      TensorSpec::createSpec<int32_t>("serving_default_input_1",
                                      {1, KnownSize}),
      TensorSpec::createSpec<float>("this_feature_does_not_exist", {2, 5})};

  LLVMContext Ctx;
  auto Evaluator = ModelUnderTrainingRunner::createAndEnsureValid(
      Ctx, getModelPath(), "StatefulPartitionedCall", InputSpecs,
      {LoggedFeatureSpec{
          TensorSpec::createSpec<float>("StatefulPartitionedCall", {1}),
          None}});
  int32_t *V = Evaluator->getTensor<int32_t>(0);
  // Fill it up with 1s, we know the output.
  for (auto I = 0; I < KnownSize; ++I)
    V[I] = 1;

  float *F = Evaluator->getTensor<float>(1);
  for (auto I = 0; I < 2 * 5; ++I)
    F[I] = 3.14 + I;
  float Ret = Evaluator->evaluate<float>();
  EXPECT_EQ(static_cast<int64_t>(Ret), 80);
  // The input vector should be unchanged
  for (auto I = 0; I < KnownSize; ++I)
    EXPECT_EQ(V[I], 1);
  for (auto I = 0; I < 2 * 5; ++I)
    EXPECT_FLOAT_EQ(F[I], 3.14 + I);
}
