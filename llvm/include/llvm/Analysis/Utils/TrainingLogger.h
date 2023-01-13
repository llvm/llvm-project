//===- TrainingLogger.h - mlgo feature/reward logging  ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_UTILS_TRAININGLOGGER_H
#define LLVM_ANALYSIS_UTILS_TRAININGLOGGER_H

#include "llvm/Config/llvm-config.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/JSON.h"

#include <memory>
#include <vector>

namespace llvm {

/// Logging utility - given an ordered specification of features, and assuming
/// a scalar reward, allow logging feature values and rewards.
/// The assumption is that, for an event to be logged (i.e. a set of feature
/// values and a reward), the user calls the log* API for each feature exactly
/// once, providing the index matching the position in the feature spec list
/// provided at construction. The example assumes the first feature's element
/// type is float, the second is int64, and the reward is float:
///
/// event 0:
///   logFloatValue(0, ...)
///   logInt64Value(1, ...)
///   ...
///   logFloatReward(...)
/// event 1:
///   logFloatValue(0, ...)
///   logInt64Value(1, ...)
///   ...
///   logFloatReward(...)
///
/// At the end, call print to generate the log.
/// Alternatively, don't call logReward at the end of each event, just
/// log{Float|Int32|Int64}FinalReward at the end.
class LoggerDataImpl;
class Logger final {
public:
  /// Construct a Logger. If IncludeReward is false, then logReward or
  /// logFinalReward shouldn't be called, and the reward feature won't be
  /// printed out.
  /// NOTE: the FeatureSpecs are expected to be in the same order (i.e. have
  /// corresponding indices) with any MLModelRunner implementations
  /// corresponding to the model being trained/logged.
  Logger(const std::vector<TensorSpec> &FeatureSpecs,
         const TensorSpec &RewardSpec, bool IncludeReward);

  ~Logger();

  void logFloatReward(float Value);
  void logInt32Reward(int32_t Value);
  void logInt64Reward(int64_t Value);

  void logFloatFinalReward(float Value);
  void logInt32FinalReward(int32_t Value);
  void logInt64FinalReward(int64_t Value);

  void logFloatValue(size_t FeatureID, const float *Value);
  void logInt32Value(size_t FeatureID, const int32_t *Value);
  void logInt64Value(size_t FeatureID, const int64_t *Value);

  void logSpecifiedTensorValue(size_t FeatureID, const char *RawData);

  // Warning! For int32_t, the return is set up for int64_t, so the caller needs
  // to piecemeal cast their int32_t values.
  // FIXME: let's drop int32_t support. While it's supported by evaluator, it's
  // not supported by the tensorflow::SequenceExample proto. For small values,
  // we can consider using bytes.
  char *addEntryAndGetFloatOrInt64Buffer(size_t FeatureID);

  // Flush the content of the log to the stream, clearing the stored data in the
  // process.
  void flush(std::string *Str);
  void flush(raw_ostream &OS);

  // Flush a set of logs that are produced from the same module, e.g.
  // per-function regalloc traces.
  static void flushLogs(raw_ostream &OS,
                        const StringMap<std::unique_ptr<Logger>> &Loggers);

private:
  std::vector<TensorSpec> FeatureSpecs;
  TensorSpec RewardSpec;
  const bool IncludeReward;
  std::unique_ptr<LoggerDataImpl> LoggerData;
};

} // namespace llvm
#endif // LLVM_ANALYSIS_UTILS_TRAININGLOGGER_H
