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

/// TensorSpec encapsulates the specification of a tensor: its dimensions, or
/// "shape" (row-major), its type (see TensorSpec::getDataType specializations
/// for supported types), its name and port (see "TensorFlow: Large-Scale
/// Machine Learning on Heterogeneous Distributed Systems", section 4.2, para 2:
/// https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
///
/// TensorSpec is used to set up a TFModelEvaluator by describing the expected
/// inputs and outputs.
class TensorSpec final {
public:
  template <typename T>
  static TensorSpec createSpec(const std::string &Name,
                               const std::vector<int64_t> &Shape,
                               int Port = 0) {
    return TensorSpec(Name, Port, getDataType<T>(), Shape);
  }

  const std::string &name() const { return Name; }
  int port() const { return Port; }
  int typeIndex() const { return TypeIndex; }
  const std::vector<int64_t> &shape() const { return Shape; }

  bool operator==(const TensorSpec &Other) const {
    return Name == Other.Name && Port == Other.Port &&
           TypeIndex == Other.TypeIndex && Shape == Other.Shape;
  }

  bool operator!=(const TensorSpec &Other) const { return !(*this == Other); }

  /// Get the number of elements in a tensor with this shape.
  size_t getElementCount() const { return ElementCount; }
  /// Get the size, in bytes, of one element.
  size_t getElementByteSize() const;

  template <typename T> bool isElementType() const {
    return getDataType<T>() == TypeIndex;
  }

private:
  TensorSpec(const std::string &Name, int Port, int TypeIndex,
             const std::vector<int64_t> &Shape);

  template <typename T> static int getDataType() {
    llvm_unreachable("Undefined tensor type");
  }

  std::string Name;
  int Port = 0;
  int TypeIndex = 0;
  std::vector<int64_t> Shape;
  size_t ElementCount = 0;
};

/// Construct a TensorSpec from a JSON dictionary of the form:
/// { "name": <string>,
///   "port": <int>,
///   "type": <string. Use LLVM's types, e.g. float, double, int64_t>,
///   "shape": <array of ints> }
/// For the "type" field, see the C++ primitive types used in
/// TFUTILS_SUPPORTED_TYPES.
Optional<TensorSpec> getTensorSpecFromJSON(LLVMContext &Ctx,
                                           const json::Value &Value);

/// Logging utility - given an ordered specification of features, and assuming
/// a scalar reward, allow logging feature values and rewards, and then print
/// as tf.train.SequenceExample text protobuf.
/// The assumption is that, for an event to be logged (i.e. a set of feature
/// values and a reward), the user calls the log* API for each feature exactly
/// once, providing the index matching the position in the feature spec list
/// provided at construction:
/// event 0:
///   logTensorValue(0, ...)
///   logTensorValue(1, ...)
///   ...
///   logReward(...)
/// event 1:
///   logTensorValue(0, ...)
///   logTensorValue(1, ...)
///   ...
///   logReward(...)
///
/// At the end, call print to generate the protobuf.
class Logger final {
public:
  struct LoggedFeatureSpec {
    TensorSpec Spec;
    Optional<std::string> LoggingName;
  };

  /// Construct a Logger. If IncludeReward is false, then logReward shouldn't
  /// be called, and the reward feature won't be printed out.
  Logger(const std::vector<LoggedFeatureSpec> &FeatureSpecs,
         const TensorSpec &RewardSpec, bool IncludeReward)
      : FeatureSpecs(FeatureSpecs), RewardSpec(RewardSpec),
        RawLogData(FeatureSpecs.size() + IncludeReward),
        IncludeReward(IncludeReward) {}

  template <typename T> void logReward(T Value) {
    assert(IncludeReward);
    logTensorValue(RawLogData.size() - 1, &Value);
  }

  template <typename T>
  void logTensorValue(size_t FeatureID, const T *Value, size_t Size = 1) {
    const char *Start = reinterpret_cast<const char *>(Value);
    const char *End = Start + sizeof(T) * Size;
    RawLogData[FeatureID].insert(RawLogData[FeatureID].end(), Start, End);
  }

  void print(raw_ostream &OS);

private:
  std::vector<LoggedFeatureSpec> FeatureSpecs;
  TensorSpec RewardSpec;
  /// RawData has one entry per feature, plus one more for the reward.
  /// Each feature's values are then stored in a vector, in succession.
  /// This means the ith event is stored at [*][i]
  std::vector<std::vector<char>> RawLogData;
  const bool IncludeReward;
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

private:
  void *getUntypedInput(size_t Index);
  std::unique_ptr<TFModelEvaluatorImpl> Impl;
};

/// List of supported types, as a pair:
/// - C++ type
/// - enum name (implementation-specific)
#define TFUTILS_SUPPORTED_TYPES(M)                                             \
  M(float, TF_FLOAT)                                                           \
  M(double, TF_DOUBLE)                                                         \
  M(int8_t, TF_INT8)                                                           \
  M(uint8_t, TF_UINT8)                                                         \
  M(int16_t, TF_INT16)                                                         \
  M(uint16_t, TF_UINT16)                                                       \
  M(int32_t, TF_INT32)                                                         \
  M(uint32_t, TF_UINT32)                                                       \
  M(int64_t, TF_INT64)                                                         \
  M(uint64_t, TF_UINT64)

#define TFUTILS_GETDATATYPE_DEF(T, E)                                          \
  template <> int TensorSpec::getDataType<T>();

TFUTILS_SUPPORTED_TYPES(TFUTILS_GETDATATYPE_DEF)

#undef TFUTILS_GETDATATYPE_DEF
} // namespace llvm

#endif // LLVM_HAVE_TF_API
#endif // LLVM_ANALYSIS_UTILS_TFUTILS_H
