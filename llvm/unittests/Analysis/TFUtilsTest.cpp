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
  std::vector<std::string> InputNames{"serving_default_input_1"};
  std::vector<std::string> OutputName{"StatefulPartitionedCall"};
  const static int64_t KnownSize = 214;

  TFModelEvaluator Evaluator(getModelPath(), InputNames, OutputName);
  static const std::vector<int64_t> Dim{1, KnownSize};

  EXPECT_TRUE(Evaluator.isValid());
  Evaluator.initInput<int32_t>(0, Dim);

  int32_t *V = Evaluator.getInput<int32_t>(0);
  // Fill it up with 1's, we know the output.
  for (auto I = 0; I < KnownSize; ++I) {
    V[I] = 1;
  }
  {
    auto ER = Evaluator.evaluate();
    EXPECT_TRUE(ER.hasValue());
    float Ret = *ER->getTensorValue<float>(0);
    EXPECT_EQ(static_cast<size_t>(Ret), 80);
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
    EXPECT_EQ(static_cast<size_t>(Ret), 80);
  }
}

// Test incorrect input setup
TEST(TFUtilsTest, EvalError) {
  // We use the ir2native model for test. We know it has one feature of
  // dimension (1, 214)
  std::vector<std::string> InputNames{"serving_default_input_1"};
  std::vector<std::string> OutputName{"StatefulPartitionedCall"};
  const static int64_t KnownSize = 213;

  TFModelEvaluator Evaluator(getModelPath(), InputNames, OutputName);
  static const std::vector<int64_t> Dim{1, KnownSize};

  EXPECT_TRUE(Evaluator.isValid());
  Evaluator.initInput<int32_t>(0, Dim);

  int32_t *V = Evaluator.getInput<int32_t>(0);
  // Fill it up with 1's, we know the output.
  for (auto I = 0; I < KnownSize; ++I) {
    V[I] = 1;
  }
  auto ER = Evaluator.evaluate();
  EXPECT_FALSE(ER.hasValue());
  EXPECT_FALSE(Evaluator.isValid());
}
