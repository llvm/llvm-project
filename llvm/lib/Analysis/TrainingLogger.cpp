//===- TrainingLogger.cpp - mlgo feature/reward logging -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logging infrastructure for extracting features and
// rewards for mlgo policy training.
//
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_API)

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "google/protobuf/struct.pb.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/core/example/example.pb.h"
#include <cassert>
#include <numeric>

using namespace llvm;

using google::protobuf::Message;
using google::protobuf::TextFormat;

static cl::opt<bool>
    ProtobufTextMode("tfutils-text-log", cl::init(false), cl::Hidden,
                     cl::desc("Output textual (human-readable) protobuf."));

namespace {

void serialize(const Message &SE, std::string *OutStr) {
  if (ProtobufTextMode) {
    TextFormat::PrintToString(SE, OutStr);
  } else {
    *OutStr = SE.SerializeAsString();
  }
}
} // namespace

namespace llvm {

class LoggerDataImpl {
protected:
  const std::vector<TensorSpec> LoggedFeatureSpecs;
  const TensorSpec RewardSpec;
  const bool IncludeReward;
  LoggerDataImpl(const std::vector<TensorSpec> &LoggedSpecs,
                 const TensorSpec &RewardSpec, bool IncludeReward)
      : LoggedFeatureSpecs(LoggedSpecs), RewardSpec(RewardSpec),
        IncludeReward(IncludeReward) {}
  virtual void logRewardImpl(const char *Value, size_t Size) = 0;

public:
  // flush the logged info to a stream and clear the log contents.
  virtual void flush(std::string *Str) = 0;
  virtual char *addNewTensor(size_t FeatureID) = 0;
  virtual size_t getNrRecords() const = 0;
  virtual ~LoggerDataImpl() = default;

  template <typename T> void logReward(T Value) {
    logRewardImpl(reinterpret_cast<const char *>(&Value), sizeof(T));
  }
};

class TFSequenceExampleLoggerDataImpl : public LoggerDataImpl {
  std::vector<tensorflow::FeatureList> FeatureLists;
  tensorflow::FeatureList Reward;

  bool isSelfConsistent(const tensorflow::SequenceExample &SE,
                        size_t NrRecords) const {
    bool Ret = true;
    for (const auto &TSpecs : LoggedFeatureSpecs) {
      const auto &Name = TSpecs.name();
      const auto &FL = SE.feature_lists().feature_list().at(Name).feature();
      if (NrRecords != static_cast<size_t>(FL.size())) {
        dbgs() << "[TF-UTILS]: " << Name << " has missing records. Expected "
               << NrRecords << " got " << FL.size() << "\n";
        Ret = false;
      }
    }
    if (IncludeReward && static_cast<size_t>(SE.feature_lists()
                                                 .feature_list()
                                                 .at(RewardSpec.name())
                                                 .feature()
                                                 .size()) != NrRecords) {
      dbgs() << "[TF-UTILS]: reward is missing records.\n";
      Ret = false;
    }
    return Ret;
  }

  void transferLog(tensorflow::SequenceExample &SE) {
    auto *FL = SE.mutable_feature_lists()->mutable_feature_list();
    if (IncludeReward)
      (*FL)[RewardSpec.name()] = std::move(Reward);
    assert(FeatureLists.size() == LoggedFeatureSpecs.size());
    for (size_t I = 0; I < FeatureLists.size(); ++I) {
      const auto &LFS = LoggedFeatureSpecs[I];
      (*FL)[LFS.name()] = std::move(FeatureLists[I]);
    }
  }

public:
  TFSequenceExampleLoggerDataImpl(const std::vector<TensorSpec> &LoggedSpecs,
                                  const TensorSpec &RewardSpec,
                                  bool IncludeReward)
      : LoggerDataImpl(LoggedSpecs, RewardSpec, IncludeReward),
        FeatureLists(LoggedFeatureSpecs.size()) {}

  // flush the logged info to a stream and clear the log contents.
  void flush(std::string *Str) override {
    size_t NrRecords = getNrRecords();
    (void)NrRecords;
    tensorflow::SequenceExample SE;
    transferLog(SE);
    assert(isSelfConsistent(SE, NrRecords));
    serialize(SE, Str);
  }

  char *addNewTensor(size_t FeatureID) override {
    const auto &Spec = LoggedFeatureSpecs[FeatureID];
    if (Spec.isElementType<float>()) {
      auto *RF = FeatureLists[FeatureID]
                     .add_feature()
                     ->mutable_float_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0.0);
      return reinterpret_cast<char *>(RF->mutable_data());
    } else if (Spec.isElementType<int32_t>() || Spec.isElementType<int64_t>()) {
      auto *RF = FeatureLists[FeatureID]
                     .add_feature()
                     ->mutable_int64_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0);
      return reinterpret_cast<char *>(RF->mutable_data());
    }
    llvm_unreachable("Unsupported tensor type.");
  }

  void logRewardImpl(const char *Value, size_t Size) override {
    assert(IncludeReward);
    if (RewardSpec.isElementType<float>())
      Reward.add_feature()->mutable_float_list()->add_value(
          *reinterpret_cast<const float *>(Value));
    else if (RewardSpec.isElementType<int32_t>())
      Reward.add_feature()->mutable_int64_list()->add_value(
          *reinterpret_cast<const int32_t *>(Value));
    else if (RewardSpec.isElementType<int64_t>())
      Reward.add_feature()->mutable_int64_list()->add_value(
          *reinterpret_cast<const int64_t *>(Value));
    else
      llvm_unreachable("Unsupported tensor type.");
  }

  size_t getNrRecords() const override {
    return FeatureLists.empty() ? 0 : FeatureLists[0].feature().size();
  }
};
} // namespace llvm

Logger::Logger(const std::vector<TensorSpec> &FeatureSpecs,
               const TensorSpec &RewardSpec, bool IncludeReward)
    : FeatureSpecs(FeatureSpecs), RewardSpec(RewardSpec),
      IncludeReward(IncludeReward),
      LoggerData(std::make_unique<TFSequenceExampleLoggerDataImpl>(
          FeatureSpecs, RewardSpec, IncludeReward)) {}

Logger::~Logger() {}

#define LOG_REWARD(NAME, TYPE)                                                 \
  void Logger::log##NAME##Reward(TYPE Value) {                                 \
    assert(IncludeReward);                                                     \
    LoggerData->logReward(Value);                                              \
  }

LOG_REWARD(Float, float)
LOG_REWARD(Int32, int32_t)
LOG_REWARD(Int64, int64_t)
#undef LOG_REWARD

#define LOG_FINAL_REWARD(NAME, TYPE)                                           \
  void Logger::log##NAME##FinalReward(TYPE Value) {                            \
    assert(RewardSpec.isElementType<TYPE>());                                  \
    for (size_t I = 1; I < LoggerData->getNrRecords(); ++I)                    \
      log##NAME##Reward(0);                                                    \
    log##NAME##Reward(Value);                                                  \
  }

LOG_FINAL_REWARD(Float, float)
LOG_FINAL_REWARD(Int32, int32_t)
LOG_FINAL_REWARD(Int64, int64_t)
#undef LOG_FINAL_REWARD

void Logger::logFloatValue(size_t FeatureID, const float *Value) {
  assert(FeatureSpecs[FeatureID].isElementType<float>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logInt64Value(size_t FeatureID, const int64_t *Value) {
  assert(FeatureSpecs[FeatureID].isElementType<int64_t>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logInt32Value(size_t FeatureID, const int32_t *Value) {
  assert(FeatureSpecs[FeatureID].isElementType<int32_t>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logSpecifiedTensorValue(size_t FeatureID, const char *RawData) {
  const auto &Spec = FeatureSpecs[FeatureID];
  char *Buff = addEntryAndGetFloatOrInt64Buffer(FeatureID);
  if (Spec.isElementType<int32_t>())
    for (size_t I = 0; I < Spec.getElementCount(); ++I)
      (reinterpret_cast<int64_t *>(Buff))[I] =
          static_cast<int64_t>((reinterpret_cast<const int32_t *>(RawData))[I]);
  else if (Spec.isElementType<int64_t>() || Spec.isElementType<float>())
    std::memcpy(Buff, RawData,
                Spec.getElementCount() * Spec.getElementByteSize());
  else
    llvm_unreachable("Unsupported tensor type");
}

char *Logger::addEntryAndGetFloatOrInt64Buffer(size_t FeatureID) {
  return reinterpret_cast<char *>(LoggerData->addNewTensor(FeatureID));
}

void Logger::flush(std::string *Str) { LoggerData->flush(Str); }

void Logger::flush(raw_ostream &OS) {
  std::string Buff;
  LoggerData->flush(&Buff);
  OS << Buff;
}

void Logger::flushLogs(raw_ostream &OS,
                       const StringMap<std::unique_ptr<Logger>> &Loggers) {
  google::protobuf::Struct Msg;
  for (const auto &NamedLogger : Loggers) {
    tensorflow::SequenceExample SE;
    const auto &Logger = NamedLogger.second;
    std::string Unencoded;
    if (Logger->LoggerData->getNrRecords() > 0)
      Logger->flush(&Unencoded);

    (*Msg.mutable_fields())[NamedLogger.first().str()]
        .mutable_string_value()
        ->append(ProtobufTextMode ? Unencoded : encodeBase64(Unencoded));
  }

  std::string OutStr;
  serialize(Msg, &OutStr);
  OS << OutStr;
}
#endif // defined(LLVM_HAVE_TF_API)
