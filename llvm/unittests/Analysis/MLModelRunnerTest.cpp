//===- MLModelRunnerTest.cpp - test for MLModelRunner ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/InteractiveModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <atomic>
#include <thread>

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

TEST(InteractiveModelRunner, Evaluation) {
  LLVMContext Ctx;
  // Test the interaction with an external advisor by asking for advice twice.
  // Use simple values, since we use the Logger underneath, that's tested more
  // extensively elsewhere.
  std::vector<TensorSpec> Inputs{
      TensorSpec::createSpec<int64_t>("a", {1}),
      TensorSpec::createSpec<int64_t>("b", {1}),
      TensorSpec::createSpec<int64_t>("c", {1}),
  };
  TensorSpec AdviceSpec = TensorSpec::createSpec<float>("advice", {1});

  // Create the 2 files. Ideally we'd create them as named pipes, but that's not
  // quite supported by the generic API.
  std::error_code EC;
  SmallString<64> FromCompilerName;
  SmallString<64> ToCompilerName;
  int FromCompilerFD = 0;
  int ToCompilerFD = 0;
  ASSERT_EQ(sys::fs::createTemporaryFile("InteractiveModelRunner_Evaluation",
                                         "temp", FromCompilerFD,
                                         FromCompilerName),
            std::error_code());

  ASSERT_EQ(sys::fs::createTemporaryFile("InteractiveModelRunner_Evaluation",
                                         "temp", ToCompilerFD, ToCompilerName),
            std::error_code());

  raw_fd_stream FromCompiler(FromCompilerName, EC);
  EXPECT_FALSE(EC);
  raw_fd_ostream ToCompiler(ToCompilerName, EC);
  EXPECT_FALSE(EC);
  FileRemover Cleanup1(FromCompilerName);
  FileRemover Cleanup2(ToCompilerName);
  InteractiveModelRunner Evaluator(Ctx, Inputs, AdviceSpec, FromCompilerName,
                                   ToCompilerName);

  Evaluator.switchContext("hi");

  // Helper to read headers and other json lines.
  SmallVector<char, 1024> Buffer;
  auto ReadLn = [&]() {
    Buffer.clear();
    while (true) {
      char Chr = 0;
      auto Read = FromCompiler.read(&Chr, 1);
      EXPECT_GE(Read, 0);
      if (!Read)
        continue;
      if (Chr == '\n')
        return StringRef(Buffer.data(), Buffer.size());
      Buffer.push_back(Chr);
    }
  };
  // See include/llvm/Analysis/Utils/TrainingLogger.h
  // First comes the header
  auto Header = json::parse(ReadLn());
  EXPECT_FALSE(Header.takeError());
  EXPECT_NE(Header->getAsObject()->getArray("features"), nullptr);
  // Then comes the context
  EXPECT_FALSE(json::parse(ReadLn()).takeError());

  // Since the evaluator sends the features over and then blocks waiting for
  // an answer, we must spawn a thread playing the role of the advisor / host:
  std::atomic<int> SeenObservations = 0;
  std::thread Advisor([&]() {
    EXPECT_EQ(SeenObservations, 0);
    int64_t Features[3] = {0};
    auto FullyRead = [&]() {
      size_t InsPt = 0;
      const size_t ToRead = 3 * Inputs[0].getTotalTensorBufferSize();
      char *Buff = reinterpret_cast<char *>(Features);
      while (InsPt < ToRead) {
        auto Read = FromCompiler.read(Buff + InsPt, ToRead - InsPt);
        EXPECT_GE(Read, 0);
        InsPt += Read;
      }
    };
    // Observation
    EXPECT_FALSE(json::parse(ReadLn()).takeError());
    // Tensor values
    FullyRead();
    // a "\n"
    char Chr = 0;
    while (FromCompiler.read(&Chr, 1) == 0) {
    }
    EXPECT_EQ(Chr, '\n');
    EXPECT_EQ(Features[0], 42);
    EXPECT_EQ(Features[1], 43);
    EXPECT_EQ(Features[2], 100);
    ++SeenObservations;

    // Send the advice
    float Advice = 42.0012;
    ToCompiler.write(reinterpret_cast<const char *>(&Advice),
                     AdviceSpec.getTotalTensorBufferSize());
    ToCompiler.flush();

    // Second observation, and same idea as above
    EXPECT_FALSE(json::parse(ReadLn()).takeError());
    FullyRead();
    while (FromCompiler.read(&Chr, 1) == 0) {
    }
    EXPECT_EQ(Chr, '\n');
    EXPECT_EQ(Features[0], 10);
    EXPECT_EQ(Features[1], -2);
    EXPECT_EQ(Features[2], 1);
    ++SeenObservations;
    Advice = 50.30;
    ToCompiler.write(reinterpret_cast<const char *>(&Advice),
                     AdviceSpec.getTotalTensorBufferSize());
    ToCompiler.flush();
  });

  EXPECT_EQ(SeenObservations, 0);
  *Evaluator.getTensor<int64_t>(0) = 42;
  *Evaluator.getTensor<int64_t>(1) = 43;
  *Evaluator.getTensor<int64_t>(2) = 100;
  float Ret = Evaluator.evaluate<float>();
  EXPECT_EQ(SeenObservations, 1);
  EXPECT_FLOAT_EQ(Ret, 42.0012);

  *Evaluator.getTensor<int64_t>(0) = 10;
  *Evaluator.getTensor<int64_t>(1) = -2;
  *Evaluator.getTensor<int64_t>(2) = 1;
  Ret = Evaluator.evaluate<float>();
  EXPECT_EQ(SeenObservations, 2);
  EXPECT_FLOAT_EQ(Ret, 50.30);
  Advisor.join();
}