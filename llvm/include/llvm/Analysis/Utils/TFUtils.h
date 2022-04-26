//===- TFUtils.h - utilities for tensorflow C API ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_UTILS_TFUTILS_H
#define LLVM_ANALYSIS_UTILS_TFUTILS_H

#include "llvm/Config/llvm-config.h"

#ifdef LLVM_HAVE_TF_API
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/JSON.h"

#include <memory>
#include <vector>

namespace llvm {

/// Load a SavedModel, find the given inputs and outputs, and setup storage
/// for input tensors. The user is responsible for correctly dimensioning the
/// input tensors and setting their values before calling evaluate().
/// To initialize:
/// - construct the object
/// - initialize the input tensors using initInput. Indices must correspond to
///   indices in the InputNames used at construction.
/// To use:
/// - set input values by using getInput to get each input tensor, and then
///   setting internal scalars, for all dimensions (tensors are row-major:
///   https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/c/c_api.h#L205)
/// - call evaluate. The input tensors' values are not consumed after this, and
///   may still be read.
/// - use the outputs in the output vector
class TFModelEvaluatorImpl;
class EvaluationResultImpl;

/// Logging utility - given an ordered specification of features, and assuming
/// a scalar reward, allow logging feature values and rewards, and then print
/// as tf.train.SequenceExample text protobuf.
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
/// At the end, call print to generate the protobuf.
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
  Logger(const std::vector<LoggedFeatureSpec> &FeatureSpecs,
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
  // per-function regalloc traces, as a google::protobuf::Struct message.
  static void flushLogs(raw_ostream &OS,
                        const StringMap<std::unique_ptr<Logger>> &Loggers);

private:
  std::vector<LoggedFeatureSpec> FeatureSpecs;
  TensorSpec RewardSpec;
  const bool IncludeReward;
  std::unique_ptr<LoggerDataImpl> LoggerData;
};

class TFModelEvaluator final {
public:
  /// The result of a model evaluation. Handles the lifetime of the output
  /// tensors, which means that their values need to be used before
  /// the EvaluationResult's dtor is called.
  class EvaluationResult {
  public:
    EvaluationResult(const EvaluationResult &) = delete;
    EvaluationResult &operator=(const EvaluationResult &Other) = delete;

    EvaluationResult(EvaluationResult &&Other);
    EvaluationResult &operator=(EvaluationResult &&Other);

    ~EvaluationResult();

    /// Get a (const) pointer to the first element of the tensor at Index.
    template <typename T> T *getTensorValue(size_t Index) {
      return static_cast<T *>(getUntypedTensorValue(Index));
    }

    template <typename T> const T *getTensorValue(size_t Index) const {
      return static_cast<T *>(getUntypedTensorValue(Index));
    }

    /// Get a (const) pointer to the untyped data of the tensor.
    void *getUntypedTensorValue(size_t Index);
    const void *getUntypedTensorValue(size_t Index) const;

  private:
    friend class TFModelEvaluator;
    EvaluationResult(std::unique_ptr<EvaluationResultImpl> Impl);
    std::unique_ptr<EvaluationResultImpl> Impl;
  };

  TFModelEvaluator(StringRef SavedModelPath,
                   const std::vector<TensorSpec> &InputSpecs,
                   const std::vector<TensorSpec> &OutputSpecs,
                   const char *Tags = "serve");
  TFModelEvaluator(StringRef SavedModelPath,
                   const std::vector<TensorSpec> &InputSpecs,
                   function_ref<TensorSpec(size_t)> GetOutputSpecs,
                   size_t OutputSpecsSize, const char *Tags = "serve");

  ~TFModelEvaluator();
  TFModelEvaluator(const TFModelEvaluator &) = delete;
  TFModelEvaluator(TFModelEvaluator &&) = delete;

  /// Evaluate the model, assuming it is valid. Returns None if the evaluation
  /// fails or the model is invalid, or an EvaluationResult otherwise. The
  /// inputs are assumed to have been already provided via getInput(). When
  /// returning None, it also invalidates this object.
  Optional<EvaluationResult> evaluate();

  /// Provides access to the input vector.
  template <typename T> T *getInput(size_t Index) {
    return static_cast<T *>(getUntypedInput(Index));
  }

  /// Returns true if the tensorflow model was loaded successfully, false
  /// otherwise.
  bool isValid() const { return !!Impl; }

  /// Untyped access to input.
  void *getUntypedInput(size_t Index);

private:
  std::unique_ptr<TFModelEvaluatorImpl> Impl;
};

} // namespace llvm

#endif // LLVM_HAVE_TF_API
#endif // LLVM_ANALYSIS_UTILS_TFUTILS_H
