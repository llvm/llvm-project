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

class TFModelEvaluator final {
public:
  /// The result of a model evaluation. Handles the lifetime of the output
  /// tensors, which means that their values need to be used before
  /// the EvaluationResult's dtor is called.
  class EvaluationResult {
  public:
    EvaluationResult(const EvaluationResult &) = delete;
    EvaluationResult(EvaluationResult &&Other);
    ~EvaluationResult();

    /// Get a pointer to the first element of the tensor at Index.
    template <typename T> T *getTensorValue(size_t Index) {
      return static_cast<T *>(getUntypedTensorValue(Index));
    }

  private:
    friend class TFModelEvaluator;
    EvaluationResult(std::unique_ptr<EvaluationResultImpl> Impl);
    void *getUntypedTensorValue(size_t Index);
    std::unique_ptr<EvaluationResultImpl> Impl;
  };

  TFModelEvaluator(StringRef SavedModelPath,
                   const std::vector<std::string> &InputNames,
                   const std::vector<std::string> &OutputNames,
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

  /// Initialize the input at Index as a tensor of the given type and
  /// dimensions.
  template <typename T>
  void initInput(size_t Index, const std::vector<int64_t> &Dimensions) {
    return initInput(Index, getModelTypeIndex<T>(), Dimensions);
  }

private:
  void *getUntypedInput(size_t Index);
  template <typename T> int getModelTypeIndex();
  void initInput(size_t Index, int TypeIndex,
                 const std::vector<int64_t> &Dimensions);

  std::unique_ptr<TFModelEvaluatorImpl> Impl;
};

template <> int TFModelEvaluator::getModelTypeIndex<float>();
template <> int TFModelEvaluator::getModelTypeIndex<double>();
template <> int TFModelEvaluator::getModelTypeIndex<int8_t>();
template <> int TFModelEvaluator::getModelTypeIndex<uint8_t>();
template <> int TFModelEvaluator::getModelTypeIndex<int16_t>();
template <> int TFModelEvaluator::getModelTypeIndex<uint16_t>();
template <> int TFModelEvaluator::getModelTypeIndex<int32_t>();
template <> int TFModelEvaluator::getModelTypeIndex<uint32_t>();
template <> int TFModelEvaluator::getModelTypeIndex<int64_t>();
template <> int TFModelEvaluator::getModelTypeIndex<uint64_t>();

} // namespace llvm

#endif // LLVM_HAVE_TF_API
#endif // LLVM_ANALYSIS_UTILS_TFUTILS_H
