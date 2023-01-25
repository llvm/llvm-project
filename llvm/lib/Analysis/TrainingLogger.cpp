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

void Logger::writeHeader() {
  json::OStream JOS(*OS);
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
  *OS << "\n";
}

void Logger::switchContext(StringRef Name) {
  CurrentContext = Name.str();
  json::OStream JOS(*OS);
  JOS.object([&]() { JOS.attribute("context", Name); });
  *OS << "\n";
}

void Logger::startObservation() {
  auto I = ObservationIDs.insert({CurrentContext, 0});
  size_t NewObservationID = I.second ? 0 : ++I.first->second;
  json::OStream JOS(*OS);
  JOS.object([&]() {
    JOS.attribute("observation", static_cast<int64_t>(NewObservationID));
  });
  *OS << "\n";
}

void Logger::endObservation() { *OS << "\n"; }

void Logger::logRewardImpl(const char *RawData) {
  assert(IncludeReward);
  json::OStream JOS(*OS);
  JOS.object([&]() {
    JOS.attribute("outcome", static_cast<int64_t>(
                                 ObservationIDs.find(CurrentContext)->second));
  });
  *OS << "\n";
  writeTensor(RewardSpec, RawData);
  *OS << "\n";
}

Logger::Logger(std::unique_ptr<raw_ostream> OS,
               const std::vector<TensorSpec> &FeatureSpecs,
               const TensorSpec &RewardSpec, bool IncludeReward)
    : OS(std::move(OS)), FeatureSpecs(FeatureSpecs), RewardSpec(RewardSpec),
      IncludeReward(IncludeReward) {
  writeHeader();
}
