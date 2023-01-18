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

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <numeric>

using namespace llvm;

// FIXME(mtrofin): remove the flag altogether
static cl::opt<bool>
    UseSimpleLogger("tfutils-use-simplelogger", cl::init(true), cl::Hidden,
                    cl::desc("Output simple (non-protobuf) log."));

raw_ostream &Logger::dumpHeader(raw_ostream &OS) const {
  json::OStream JOS(OS);
  JOS.object([&]() {
    JOS.attributeArray("features", [&]() {
      for (const auto &TS : FeatureSpecs)
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

raw_ostream &Logger::startContext(raw_ostream &OS, StringRef Name) const {
  json::OStream JOS(OS);
  JOS.object([&]() { JOS.attribute("context", Name); });
  OS << "\n";
  return OS;
}

raw_ostream &Logger::startObservation(raw_ostream &OS, size_t Nr) const {
  json::OStream JOS(OS);
  JOS.object([&]() { JOS.attribute("observation", static_cast<int64_t>(Nr)); });
  OS << "\n";
  return OS;
}

raw_ostream &Logger::writeOutcome(raw_ostream &OS,
                                  size_t CurrentObservationID) const {
  if (IncludeReward) {
    OS << "\n";
    json::OStream JOS(OS);
    JOS.object([&]() {
      JOS.attribute("outcome", static_cast<int64_t>(CurrentObservationID));
    });
    OS << "\n";
    OS.write(RewardStorage[CurrentObservationID].get(),
             RewardSpec.getTotalTensorBufferSize());
  }
  OS << "\n";
  return OS;
}

char *Logger::addNewTensor(size_t FeatureID) {
  return FeatureStorage
      .emplace_back(
          new char[FeatureSpecs[FeatureID].getTotalTensorBufferSize()])
      .get();
}

size_t Logger::getNrRecords() const {
  assert(FeatureStorage.size() % FeatureSpecs.size() == 0);
  return FeatureStorage.size() / FeatureSpecs.size();
}

void Logger::logRewardImpl(const char *Value, size_t Size) {
  std::memcpy(RewardStorage.emplace_back(new char[Size]).get(), Value, Size);
}

raw_ostream &Logger::flush(raw_ostream &OS, bool WithHeader,
                           StringRef Context) const {
  if (WithHeader)
    dumpHeader(OS);
  startContext(OS, Context);
  size_t CurrentObservationID = 0;
  for (size_t I = 0; I < FeatureStorage.size(); ++I) {
    size_t TensorID = I % FeatureSpecs.size();
    if (TensorID == 0) {
      CurrentObservationID = I / FeatureSpecs.size();
      startObservation(OS, CurrentObservationID);
    }
    OS.write(FeatureStorage[I].get(),
             FeatureSpecs[TensorID].getTotalTensorBufferSize());
    if (TensorID == FeatureSpecs.size() - 1) {
      writeOutcome(OS, CurrentObservationID);
    }
  }
  return OS;
}

#define LOG_REWARD(NAME, TYPE)                                                 \
  void Logger::log##NAME##Reward(TYPE Value) {                                 \
    assert(IncludeReward);                                                     \
    (void)IncludeReward;                                                       \
    logReward(Value);                                                          \
  }

LOG_REWARD(Float, float)
LOG_REWARD(Int32, int32_t)
LOG_REWARD(Int64, int64_t)
#undef LOG_REWARD

#define LOG_FINAL_REWARD(NAME, TYPE)                                           \
  void Logger::log##NAME##FinalReward(TYPE Value) {                            \
    assert(RewardSpec.isElementType<TYPE>());                                  \
    for (size_t I = 1; I < getNrRecords(); ++I)                                \
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
  return reinterpret_cast<char *>(addNewTensor(FeatureID));
}

void Logger::flushLogs(raw_ostream &OS,
                       const StringMap<std::unique_ptr<Logger>> &Loggers) {
  bool IsFirst = true;
  for (const auto &NamedLogger : Loggers) {
    NamedLogger.second->flush(OS, IsFirst, NamedLogger.first());
    IsFirst = false;
  }
}
