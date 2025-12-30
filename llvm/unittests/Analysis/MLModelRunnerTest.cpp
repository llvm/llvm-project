//===- MLModelRunnerTest.cpp - test for MLModelRunner ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/InteractiveModelRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>

using namespace llvm;

namespace llvm {
// This is a mock of the kind of AOT-generated model evaluator. It has 2 tensors
// of shape {1}, and 'evaluation' adds them.
// The interface is the one expected by ReleaseModelRunner.
class MockAOTModelBase {
protected:
  int64_t A = 0;
  int64_t B = 0;
  int64_t R = 0;

public:
  MockAOTModelBase() = default;
  virtual ~MockAOTModelBase() = default;

  virtual int LookupArgIndex(const std::string &Name) {
    if (Name == "prefix_a")
      return 0;
    if (Name == "prefix_b")
      return 1;
    return -1;
  }
  int LookupResultIndex(const std::string &) { return 0; }
  virtual void Run() = 0;
  virtual void *result_data(int RIndex) {
    if (RIndex == 0)
      return &R;
    return nullptr;
  }
  virtual void *arg_data(int Index) {
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

class AdditionAOTModel final : public MockAOTModelBase {
public:
  AdditionAOTModel() = default;
  void Run() override { R = A + B; }
};

class DiffAOTModel final : public MockAOTModelBase {
public:
  DiffAOTModel() = default;
  void Run() override { R = A - B; }
};

static const char *M1Selector = "the model that subtracts";
static const char *M2Selector = "the model that adds";

static MD5::MD5Result Hash1 = MD5::hash(arrayRefFromStringRef(M1Selector));
static MD5::MD5Result Hash2 = MD5::hash(arrayRefFromStringRef(M2Selector));
class ComposedAOTModel final {
  DiffAOTModel M1;
  AdditionAOTModel M2;
  uint64_t Selector[2] = {0};

  bool isHashSameAsSelector(const std::pair<uint64_t, uint64_t> &Words) const {
    return Selector[0] == Words.first && Selector[1] == Words.second;
  }
  MockAOTModelBase *getModel() {
    if (isHashSameAsSelector(Hash1.words()))
      return &M1;
    if (isHashSameAsSelector(Hash2.words()))
      return &M2;
    llvm_unreachable("Should be one of the two");
  }

public:
  ComposedAOTModel() = default;
  int LookupArgIndex(const std::string &Name) {
    if (Name == "prefix_model_selector")
      return 2;
    return getModel()->LookupArgIndex(Name);
  }
  int LookupResultIndex(const std::string &Name) {
    return getModel()->LookupResultIndex(Name);
  }
  void *arg_data(int Index) {
    if (Index == 2)
      return Selector;
    return getModel()->arg_data(Index);
  }
  void *result_data(int RIndex) { return getModel()->result_data(RIndex); }
  void Run() { getModel()->Run(); }
};

static EmbeddedModelRunnerOptions makeOptions() {
  EmbeddedModelRunnerOptions Opts;
  Opts.setFeedPrefix("prefix_");
  return Opts;
}
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
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<AdditionAOTModel>>(
      Ctx, Inputs, "", makeOptions());
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
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<AdditionAOTModel>>(
      Ctx, Inputs, "", makeOptions());
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
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<AdditionAOTModel>>(
      Ctx, Inputs, "", makeOptions());
  *Evaluator->getTensor<int64_t>(0) = 1;         // a
  *Evaluator->getTensor<int64_t>(1) = 2;         // c
  *Evaluator->getTensor<int64_t>(2) = -3;        // b
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), -2); // a + b
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(0), 1);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(1), 2);
  EXPECT_EQ(*Evaluator->getTensor<int64_t>(2), -3);
}

// We expect an error to be reported early if the user tried to specify a model
// selector, but the model in fact doesn't support that.
TEST(ReleaseModelRunner, ModelSelectorNoInputFeaturePresent) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  EXPECT_DEATH((void)std::make_unique<ReleaseModeModelRunner<AdditionAOTModel>>(
                   Ctx, Inputs, "", makeOptions().setModelSelector(M2Selector)),
               "A model selector was specified but the underlying model does "
               "not expose a model_selector input");
}

TEST(ReleaseModelRunner, ModelSelectorNoSelectorGiven) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  EXPECT_DEATH(
      (void)std::make_unique<ReleaseModeModelRunner<ComposedAOTModel>>(
          Ctx, Inputs, "", makeOptions()),
      "A model selector was not specified but the underlying model requires "
      "selecting one because it exposes a model_selector input");
}

// Test that we correctly set up the model_selector tensor value. We are only
// responsbile for what happens if the user doesn't specify a value (but the
// model supports the feature), or if the user specifies one, and we correctly
// populate the tensor, and do so upfront (in case the model implementation
// needs that for subsequent tensor buffer lookups).
TEST(ReleaseModelRunner, ModelSelector) {
  LLVMContext Ctx;
  std::vector<TensorSpec> Inputs{TensorSpec::createSpec<int64_t>("a", {1}),
                                 TensorSpec::createSpec<int64_t>("b", {1})};
  // This explicitly asks for M1
  auto Evaluator = std::make_unique<ReleaseModeModelRunner<ComposedAOTModel>>(
      Ctx, Inputs, "", makeOptions().setModelSelector(M1Selector));
  *Evaluator->getTensor<int64_t>(0) = 1;
  *Evaluator->getTensor<int64_t>(1) = 2;
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), -1);

  // Ask for M2
  Evaluator = std::make_unique<ReleaseModeModelRunner<ComposedAOTModel>>(
      Ctx, Inputs, "", makeOptions().setModelSelector(M2Selector));
  *Evaluator->getTensor<int64_t>(0) = 1;
  *Evaluator->getTensor<int64_t>(1) = 2;
  EXPECT_EQ(Evaluator->evaluate<int64_t>(), 3);

  // Asking for a model that's not supported isn't handled by our infra and we
  // expect the model implementation to fail at a point.
}

#if defined(LLVM_ON_UNIX)
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
  llvm::unittest::TempDir Tmp("tmpdir", /*Unique=*/true);
  SmallString<128> FromCompilerName(Tmp.path().begin(), Tmp.path().end());
  SmallString<128> ToCompilerName(Tmp.path().begin(), Tmp.path().end());
  sys::path::append(FromCompilerName, "InteractiveModelRunner_Evaluation.out");
  sys::path::append(ToCompilerName, "InteractiveModelRunner_Evaluation.in");
  EXPECT_EQ(::mkfifo(FromCompilerName.c_str(), 0666), 0);
  EXPECT_EQ(::mkfifo(ToCompilerName.c_str(), 0666), 0);

  FileRemover Cleanup1(FromCompilerName);
  FileRemover Cleanup2(ToCompilerName);

  // Since the evaluator sends the features over and then blocks waiting for
  // an answer, we must spawn a thread playing the role of the advisor / host:
  std::atomic<int> SeenObservations = 0;
  // Start the host first to make sure the pipes are being prepared. Otherwise
  // the evaluator will hang.
  std::thread Advisor([&]() {
    // Open the writer first. This is because the evaluator will try opening
    // the "input" pipe first. An alternative that avoids ordering is for the
    // host to open the pipes RW.
    raw_fd_ostream ToCompiler(ToCompilerName, EC);
    EXPECT_FALSE(EC);
    int FromCompilerHandle = 0;
    EXPECT_FALSE(
        sys::fs::openFileForRead(FromCompilerName, FromCompilerHandle));
    sys::fs::file_t FromCompiler =
        sys::fs::convertFDToNativeFile(FromCompilerHandle);
    EXPECT_EQ(SeenObservations, 0);
    // Helper to read headers and other json lines.
    SmallVector<char, 1024> Buffer;
    auto ReadLn = [&]() {
      Buffer.clear();
      while (true) {
        char Chr = 0;
        auto ReadOrErr = sys::fs::readNativeFile(FromCompiler, {&Chr, 1});
        EXPECT_FALSE(ReadOrErr.takeError());
        if (!*ReadOrErr)
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
    EXPECT_NE(Header->getAsObject()->getObject("advice"), nullptr);
    // Then comes the context
    EXPECT_FALSE(json::parse(ReadLn()).takeError());

    int64_t Features[3] = {0};
    auto FullyRead = [&]() {
      size_t InsPt = 0;
      const size_t ToRead = 3 * Inputs[0].getTotalTensorBufferSize();
      char *Buff = reinterpret_cast<char *>(Features);
      while (InsPt < ToRead) {
        auto ReadOrErr = sys::fs::readNativeFile(
            FromCompiler, {Buff + InsPt, ToRead - InsPt});
        EXPECT_FALSE(ReadOrErr.takeError());
        InsPt += *ReadOrErr;
      }
    };
    // Observation
    EXPECT_FALSE(json::parse(ReadLn()).takeError());
    // Tensor values
    FullyRead();
    // a "\n"
    char Chr = 0;
    auto ReadNL = [&]() {
      do {
        auto ReadOrErr = sys::fs::readNativeFile(FromCompiler, {&Chr, 1});
        EXPECT_FALSE(ReadOrErr.takeError());
        if (*ReadOrErr == 1)
          break;
      } while (true);
    };
    ReadNL();
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
    ReadNL();
    EXPECT_EQ(Chr, '\n');
    EXPECT_EQ(Features[0], 10);
    EXPECT_EQ(Features[1], -2);
    EXPECT_EQ(Features[2], 1);
    ++SeenObservations;
    Advice = 50.30;
    ToCompiler.write(reinterpret_cast<const char *>(&Advice),
                     AdviceSpec.getTotalTensorBufferSize());
    ToCompiler.flush();
    sys::fs::closeFile(FromCompiler);
  });

  InteractiveModelRunner Evaluator(Ctx, Inputs, AdviceSpec, FromCompilerName,
                                   ToCompilerName);

  Evaluator.switchContext("hi");

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
#endif
