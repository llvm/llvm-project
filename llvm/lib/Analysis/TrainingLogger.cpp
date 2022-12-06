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
#include "llvm/Analysis/TensorSpec.h"
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

static cl::opt<bool>
    UseSimpleLogger("tfutils-use-simplelogger", cl::init(false), cl::Hidden,
                    cl::desc("Output simple (non-protobuf) log."));

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

// The design goals of the simple logger are:
// - no dependencies that llvm doesn't already have.
// - support streaming, so that we don't need to buffer data during compilation
// - 0-decoding tensor values. Tensor values are potentially very large buffers
// of scalars. Because of their potentially large size, avoiding
// serialization/deserialization overhead is preferred.
//
// The simple logger produces an output of the form (each line item on its line)
// - header: a json object describing the data that will follow.
// - context: e.g. function name, for regalloc, or "default" for module-wide
// optimizations like the inliner. This is the context to which the subsequent
// data corresponds.
// - observation number.
// - tensor values - raw bytes of the tensors, in the order given in the header.
// The values are in succession, i.e. no separator is found between successive
// tensor values. At the end, there is a new line character.
// - [score] - this is optional, and is present if it was present in the header.
// Currently, for final rewards, we output "0" scores after each observation,
// except for the last one.
// <repeat>
// The file should be read as binary, but the reason we use newlines is mostly
// ease of debugging: the log can be opened in a text editor and, while tensor
// values are inscrutable, at least the sequence of data can be easily observed.
// Of course, the buffer of tensor values could contain '\n' bytes. A reader
// should use the header information to know how much data to read for the
// tensor values, and not use line information for that.
//
// An example reader, used for test, is available at
// Analysis/models/log_reader.py
//
// Example:
// {"features":[list of TensorSpecs], "score":<a tensor spec>}
// {"context": "aFunction"}
// {"observation": 0}
// <bytes>
// {"outcome": 0}
// <bytes for the tensor corresponding to the "score" spec in the header>
// {"observation": 1}
// ...
// {"context": "anotherFunction"}
// {"observation": 0}
// ...
//
class SimpleLoggerDataImpl : public LoggerDataImpl {
  std::vector<std::unique_ptr<char[]>> FeatureStorage;
  std::vector<std::unique_ptr<char[]>> RewardStorage;

  raw_ostream &dumpHeader(raw_ostream &OS) const {
    json::OStream JOS(OS);
    JOS.object([&]() {
      JOS.attributeArray("features", [&]() {
        for (const auto &TS : LoggedFeatureSpecs)
          TS.toJSON(JOS);
      });
      if (IncludeReward) {
        JOS.attributeBegin("score");
        RewardSpec.toJSON(JOS);
        JOS.attributeEnd();
      }
    });
    OS << "\n";
    return OS;
  }

  raw_ostream &startContext(raw_ostream &OS, StringRef Name) const {
    json::OStream JOS(OS);
    JOS.object([&]() { JOS.attribute("context", Name); });
    OS << "\n";
    return OS;
  }

  raw_ostream &startObservation(raw_ostream &OS, size_t Nr) const {
    json::OStream JOS(OS);
    JOS.object([&]() { JOS.attribute("observation", Nr); });
    OS << "\n";
    return OS;
  }

  raw_ostream &writeOutcome(raw_ostream &OS,
                            size_t CurrentObservationID) const {
    if (IncludeReward) {
      OS << "\n";
      json::OStream JOS(OS);
      JOS.object([&]() { JOS.attribute("outcome", CurrentObservationID); });
      OS << "\n";
      OS.write(RewardStorage[CurrentObservationID].get(),
               RewardSpec.getTotalTensorBufferSize());
    }
    OS << "\n";
    return OS;
  }
  void flush(std::string *Str) override {
    llvm_unreachable("Use the ostream implementation");
  }

  char *addNewTensor(size_t FeatureID) override {
    return FeatureStorage
        .emplace_back(
            new char[LoggedFeatureSpecs[FeatureID].getTotalTensorBufferSize()])
        .get();
  }

  size_t getNrRecords() const override {
    assert(FeatureStorage.size() % LoggedFeatureSpecs.size() == 0);
    return FeatureStorage.size() / LoggedFeatureSpecs.size();
  }

  void logRewardImpl(const char *Value, size_t Size) override {
    std::memcpy(RewardStorage.emplace_back(new char[Size]).get(), Value, Size);
  }

public:
  SimpleLoggerDataImpl(const std::vector<TensorSpec> &LoggedSpecs,
                       const TensorSpec &RewardSpec, bool IncludeReward)
      : LoggerDataImpl(LoggedSpecs, RewardSpec, IncludeReward) {}

  raw_ostream &flush(raw_ostream &OS, bool WithHeader = true,
                     StringRef Context = "default") const {
    if (WithHeader)
      dumpHeader(OS);
    startContext(OS, Context);
    size_t CurrentObservationID = 0;
    for (size_t I = 0; I < FeatureStorage.size(); ++I) {
      size_t TensorID = I % LoggedFeatureSpecs.size();
      if (TensorID == 0) {
        CurrentObservationID = I / LoggedFeatureSpecs.size();
        startObservation(OS, CurrentObservationID);
      }
      OS.write(FeatureStorage[I].get(),
               LoggedFeatureSpecs[TensorID].getTotalTensorBufferSize());
      if (TensorID == LoggedFeatureSpecs.size() - 1) {
        writeOutcome(OS, CurrentObservationID);
      }
    }
    return OS;
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
      IncludeReward(IncludeReward) {
  if (UseSimpleLogger)
    LoggerData = std::make_unique<SimpleLoggerDataImpl>(
        FeatureSpecs, RewardSpec, IncludeReward);
  else
    LoggerData = std::make_unique<TFSequenceExampleLoggerDataImpl>(
        FeatureSpecs, RewardSpec, IncludeReward);
}

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
  if (UseSimpleLogger) {
    reinterpret_cast<SimpleLoggerDataImpl *>(LoggerData.get())->flush(OS);
  } else {
    std::string Buff;
    LoggerData->flush(&Buff);
    OS << Buff;
  }
}

void Logger::flushLogs(raw_ostream &OS,
                       const StringMap<std::unique_ptr<Logger>> &Loggers) {
  if (UseSimpleLogger) {
    bool IsFirst = true;
    for (const auto &NamedLogger : Loggers) {
      auto *Impl = NamedLogger.second->LoggerData.get();
      reinterpret_cast<const SimpleLoggerDataImpl *>(Impl)->flush(
          OS, IsFirst, NamedLogger.first());
      IsFirst = false;
    }
  } else {
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
}
#endif // defined(LLVM_HAVE_TF_API)
