//===- TFUtilsTest.cpp - test for TFUtils ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Utils/TFUtils.h"
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
  EXPECT_TRUE(Evaluator.isValid());

  int32_t *V = Evaluator.getInput<int32_t>(0);
  // Fill it up with 1's, we know the output.
  for (auto I = 0; I < KnownSize; ++I) {
    V[I] = 1;
  }
  auto ER = Evaluator.evaluate();
  EXPECT_FALSE(ER.hasValue());
  EXPECT_FALSE(Evaluator.isValid());
}

TEST(TFUtilsTest, JSONParsing) {
  auto Value = json::parse(
      R"({"name": "tensor_name", 
        "port": 2, 
        "type": "int32_t", 
        "shape":[1,4]
        })");
  EXPECT_TRUE(!!Value);
  LLVMContext Ctx;
  Optional<TensorSpec> Spec = getTensorSpecFromJSON(Ctx, *Value);
  EXPECT_TRUE(Spec.hasValue());
  EXPECT_EQ(*Spec, TensorSpec::createSpec<int32_t>("tensor_name", {1, 4}, 2));
}

TEST(TFUtilsTest, JSONParsingInvalidTensorType) {
  auto Value = json::parse(
      R"(
        {"name": "tensor_name", 
        "port": 2, 
        "type": "no such type", 
        "shape":[1,4]
        }
      )");
  EXPECT_TRUE(!!Value);
  LLVMContext Ctx;
  auto Spec = getTensorSpecFromJSON(Ctx, *Value);
  EXPECT_FALSE(Spec.hasValue());
}

TEST(TFUtilsTest, TensorSpecSizesAndTypes) {
  auto Spec1D = TensorSpec::createSpec<int16_t>("Hi1", {1});
  auto Spec2D = TensorSpec::createSpec<int16_t>("Hi2", {1, 1});
  auto Spec1DLarge = TensorSpec::createSpec<float>("Hi3", {10});
  auto Spec3DLarge = TensorSpec::createSpec<float>("Hi3", {2, 4, 10});
  EXPECT_TRUE(Spec1D.isElementType<int16_t>());
  EXPECT_FALSE(Spec3DLarge.isElementType<double>());
  EXPECT_EQ(Spec1D.getElementCount(), 1U);
  EXPECT_EQ(Spec2D.getElementCount(), 1U);
  EXPECT_EQ(Spec1DLarge.getElementCount(), 10U);
  EXPECT_EQ(Spec3DLarge.getElementCount(), 80U);
  EXPECT_EQ(Spec3DLarge.getElementByteSize(), sizeof(float));
  EXPECT_EQ(Spec1D.getElementByteSize(), sizeof(int16_t));
}

TEST(TFUtilsTest, Logger) {
  std::vector<Logger::LoggedFeatureSpec> Features;
  Features.push_back(
      {TensorSpec::createSpec<float>("the_float", {2, 3}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {2}),
                      std::string("alternate_name")});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, true);
  float F00[]{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  int64_t F01[]{2, 3};

  L.logTensorValue(0, F00, 6);
  L.logTensorValue(1, F01, 2);
  L.logReward<float>(3.4);
  float F10[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  int64_t F11[]{-2, -3};
  L.logTensorValue(0, F10, 6);
  L.logTensorValue(1, F11, 2);
  L.logReward<float>(-3.0);
  const auto *Expected = R"(feature_lists: {
  feature_list: {
    key: "the_float" value: {
      feature: { float_list: { value: [0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01] } }
      feature: { float_list: { value: [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00] } }
    }
  }
  feature_list: {
    key: "alternate_name" value: {
      feature: { int64_list: { value: [2, 3] } }
      feature: { int64_list: { value: [-2, -3] } }
    }
  }
  feature_list: {
    key: "reward" value: {
      feature: { float_list: { value: [3.400000e+00] } }
      feature: { float_list: { value: [-3.000000e+00] } }
    }
  }
}
)";
  std::string Result;
  raw_string_ostream OS(Result);
  L.print(OS);
  EXPECT_EQ(Result, Expected);
}

TEST(TFUtilsTest, LoggerNoReward) {
  std::vector<Logger::LoggedFeatureSpec> Features;
  Features.push_back(
      {TensorSpec::createSpec<float>("the_float", {2, 3}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {2}),
                      std::string("alternate_name")});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, false);
  float F00[]{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  int64_t F01[]{2, 3};

  L.logTensorValue(0, F00, 6);
  L.logTensorValue(1, F01, 2);
  float F10[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  int64_t F11[]{-2, -3};
  L.logTensorValue(0, F10, 6);
  L.logTensorValue(1, F11, 2);
  const auto *Expected = R"(feature_lists: {
  feature_list: {
    key: "the_float" value: {
      feature: { float_list: { value: [0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01] } }
      feature: { float_list: { value: [0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00] } }
    }
  }
  feature_list: {
    key: "alternate_name" value: {
      feature: { int64_list: { value: [2, 3] } }
      feature: { int64_list: { value: [-2, -3] } }
    }
  }
}
)";
  std::string Result;
  raw_string_ostream OS(Result);
  L.print(OS);
  EXPECT_EQ(Result, Expected);
}

TEST(TFUtilsTest, LoggerFinalReward) {
  std::vector<Logger::LoggedFeatureSpec> Features;
  Features.push_back({TensorSpec::createSpec<float>("the_float", {1}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {1}), None});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, true);
  for (size_t I = 0; I < 3; ++I) {
    float F = static_cast<float>(I);
    L.logTensorValue(0, &F);
    L.logTensorValue(1, &I);
  }
  L.logFinalReward<float>(3.14);
  const auto *Expected = R"(feature_lists: {
  feature_list: {
    key: "the_float" value: {
      feature: { float_list: { value: [0.000000e+00] } }
      feature: { float_list: { value: [1.000000e+00] } }
      feature: { float_list: { value: [2.000000e+00] } }
    }
  }
  feature_list: {
    key: "the_int" value: {
      feature: { int64_list: { value: [0] } }
      feature: { int64_list: { value: [1] } }
      feature: { int64_list: { value: [2] } }
    }
  }
  feature_list: {
    key: "reward" value: {
      feature: { float_list: { value: [0.000000e+00] } }
      feature: { float_list: { value: [0.000000e+00] } }
      feature: { float_list: { value: [3.140000e+00] } }
    }
  }
}
)";
  std::string Result;
  raw_string_ostream OS(Result);
  L.print(OS);
  EXPECT_EQ(Result, Expected);
}
