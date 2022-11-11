//===- TrainingLoggerTest.cpp - test for TrainingLogger -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "google/protobuf/struct.pb.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
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

// NOTE! This test model is currently also used by test/Transforms/Inline/ML
// tests
//- relevant if updating this model.

#define PROTO_CHECKER(FNAME, TYPE, INDEX, EXP)                                 \
  do {                                                                         \
    const auto &V = Expected.feature_lists()                                   \
                        .feature_list()                                        \
                        .at(FNAME)                                             \
                        .feature(INDEX)                                        \
                        .TYPE()                                                \
                        .value();                                              \
    for (auto I = 0; I < V.size(); ++I)                                        \
      EXPECT_EQ(V.at(I), EXP[I]);                                              \
  } while (false)

TEST(TrainingLoggerTest, Logger) {
  std::vector<LoggedFeatureSpec> Features;
  Features.push_back(
      {TensorSpec::createSpec<float>("the_float", {2, 3}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {2}),
                      std::string("alternate_name")});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, true);
  const float F00[]{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const int64_t F01[]{2, 3};

  L.logFloatValue(0, F00);
  L.logInt64Value(1, F01);
  L.logFloatReward(3.4);
  const float F10[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  const int64_t F11[]{-2, -3};
  L.logFloatValue(0, F10);
  L.logInt64Value(1, F11);
  L.logFloatReward(-3.0);
  std::string Result;
  raw_string_ostream OS(Result);
  L.flush(OS);

  tensorflow::SequenceExample Expected;
  ASSERT_TRUE(Expected.ParseFromString(Result));
  PROTO_CHECKER("the_float", float_list, 0, F00);
  PROTO_CHECKER("the_float", float_list, 1, F10);
  PROTO_CHECKER("alternate_name", int64_list, 0, F01);
  PROTO_CHECKER("alternate_name", int64_list, 1, F11);
  float R0[]{3.4};
  float R1[]{-3.0};
  PROTO_CHECKER("reward", float_list, 0, R0);
  PROTO_CHECKER("reward", float_list, 1, R1);
}

TEST(TrainingLoggerTest, LoggerInt32FeaturesAndReward) {
  std::vector<LoggedFeatureSpec> Features;
  Features.push_back(
      {TensorSpec::createSpec<float>("the_float", {2, 3}), None});
  Features.push_back({TensorSpec::createSpec<int32_t>("the_int", {2}),
                      std::string("alternate_name")});

  auto Rewards = TensorSpec::createSpec<int32_t>("reward", {1});
  Logger L(Features, Rewards, true);
  const float F00[]{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const int32_t F01[]{2, 3};

  L.logFloatValue(0, F00);
  L.logInt32Value(1, F01);
  L.logInt32Reward(3);
  const float F10[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  const int32_t F11[]{-2, -3};
  L.logFloatValue(0, F10);
  L.logInt32Value(1, F11);
  L.logInt32Reward(-3);
  std::string Result;
  raw_string_ostream OS(Result);
  L.flush(OS);

  tensorflow::SequenceExample Expected;
  ASSERT_TRUE(Expected.ParseFromString(Result));
  PROTO_CHECKER("the_float", float_list, 0, F00);
  PROTO_CHECKER("the_float", float_list, 1, F10);
  PROTO_CHECKER("alternate_name", int64_list, 0, F01);
  PROTO_CHECKER("alternate_name", int64_list, 1, F11);
  int32_t R0[]{3};
  int32_t R1[]{-3};
  PROTO_CHECKER("reward", int64_list, 0, R0);
  PROTO_CHECKER("reward", int64_list, 1, R1);
}

TEST(TrainingLoggerTest, LoggerNoReward) {
  std::vector<LoggedFeatureSpec> Features;
  Features.push_back(
      {TensorSpec::createSpec<float>("the_float", {2, 3}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {2}),
                      std::string("alternate_name")});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, false);
  const float F00[]{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const int64_t F01[]{2, 3};

  L.logFloatValue(0, F00);
  L.logInt64Value(1, F01);
  const float F10[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  const int64_t F11[]{-2, -3};
  L.logFloatValue(0, F10);
  L.logInt64Value(1, F11);

  std::string Result;
  raw_string_ostream OS(Result);
  L.flush(OS);
  tensorflow::SequenceExample Expected;
  ASSERT_TRUE(Expected.ParseFromString(Result));
  PROTO_CHECKER("the_float", float_list, 0, F00);
  PROTO_CHECKER("the_float", float_list, 1, F10);
  PROTO_CHECKER("alternate_name", int64_list, 0, F01);
  PROTO_CHECKER("alternate_name", int64_list, 1, F11);
}

TEST(TrainingLoggerTest, LoggerFinalReward) {
  std::vector<LoggedFeatureSpec> Features;
  Features.push_back({TensorSpec::createSpec<float>("the_float", {1}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {1}), None});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  Logger L(Features, Rewards, true);
  for (int64_t I = 0; I < 3; ++I) {
    float F = static_cast<float>(I);
    L.logFloatValue(0, &F);
    L.logInt64Value(1, &I);
  }
  L.logFloatFinalReward(3.14);
  std::string Result;
  raw_string_ostream OS(Result);
  L.flush(OS);
  const float Zero[]{0.0};
  const float R[]{3.14};
  tensorflow::SequenceExample Expected;
  ASSERT_TRUE(Expected.ParseFromString(Result));
  PROTO_CHECKER("reward", float_list, 0, Zero);
  PROTO_CHECKER("reward", float_list, 1, Zero);
  PROTO_CHECKER("reward", float_list, 2, R);
}

TEST(TrainingLoggerTest, LoggerGroup) {
  std::vector<LoggedFeatureSpec> Features;
  Features.push_back({TensorSpec::createSpec<float>("the_float", {1}), None});
  Features.push_back({TensorSpec::createSpec<int64_t>("the_int", {1}), None});

  auto Rewards = TensorSpec::createSpec<float>("reward", {1});
  StringMap<std::unique_ptr<Logger>> Loggers;
  std::vector<std::string> Names{"a", "b"};
  size_t Bump = 0;
  for (auto Name : Names) {
    auto L = std::make_unique<Logger>(Features, Rewards, true);
    for (int64_t I = 0; I < 3; ++I) {
      float F = static_cast<float>(I) + Bump;
      L->logFloatValue(0, &F);
      L->logInt64Value(1, &I);
    }
    L->logFloatFinalReward(3.14 + Bump);
    Loggers.insert(std::make_pair(Name, std::move(L)));
  }
  std::string Result;
  raw_string_ostream OS(Result);
  Logger::flushLogs(OS, Loggers);
  google::protobuf::Struct Expected;
  ASSERT_TRUE(Expected.ParseFromString(Result));
  EXPECT_EQ(Expected.fields_size(), 2);
  EXPECT_TRUE(Expected.fields().contains("a"));
  EXPECT_TRUE(Expected.fields().contains("b"));
}
