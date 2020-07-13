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

#ifdef LLVM_HAVE_TF_API
#include "tensorflow/c/c_api.h"
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
/// - prepare an output vector of TF_Output* type, with the correct number of
/// outputs (i.e. same as OutputNames). Initialize the vector with nullptr
/// values.
/// - call evaluate. The input tensors' values are not consumed after this, and
///   may still be read.
/// - use the outputs in the output vector
/// - deallocate each output tensor in the output vector, using TF_DeleteTensor.
class TFModelEvaluator final {
public:
  /// The result of a model evaluation. Handles the lifetime of the output
  /// TF_Tensor objects, which means that their values need to be used before
  /// the EvaluationResult's dtor is called.
  class EvaluationResult {
  public:
    ~EvaluationResult() {
      for (auto *P : Output)
        if (P)
          TF_DeleteTensor(P);
    }

    EvaluationResult(const EvaluationResult &) = delete;
    EvaluationResult(EvaluationResult &&Other)
        : OutputSize(Other.OutputSize), Output(std::move(Other.Output)) {
      Other.Output.clear();
    };

    /// Get a pointer to the first element of the tensor at Index.
    template <typename T> T *getTensorValue(size_t Index) {
      return static_cast<T *>(TF_TensorData(Output[Index]));
    }

  private:
    friend class TFModelEvaluator;
    EvaluationResult(size_t OutputSize)
        : OutputSize(OutputSize), Output(OutputSize){};

    const size_t OutputSize;
    std::vector<TF_Tensor *> Output;
  };

  using TFGraphPtr = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
  using TFSessionOptionsPtr =
      std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
  using TFStatusPtr = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

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
  /// returning None, it also marks the object invalid. Pass an Output vector
  /// with the same size as OutputNames, but with nullptr values. evaluate()
  /// will populate it with tensors, matching in index the corresponding
  /// OutputNames. The caller is responsible for the deallocation of those
  /// tensors, using TF_DeleteTensor.
  Optional<EvaluationResult> evaluate();

  /// Provides access to the input vector. It is already dimensioned correctly,
  /// but the values need to be allocated by the user.
  std::vector<TF_Tensor *> &getInput() { return Input; }

  /// Returns true if the tensorflow model was loaded successfully, false
  /// otherwise.
  bool isValid() const { return !!Session; }

  /// Initialize the input at Index as a tensor of the given type and dimensions
  void initInput(int Index, TF_DataType Type,
                 const std::vector<int64_t> &Dimensions);

private:
  /// The objects necessary for carrying out an evaluation of the SavedModel.
  /// They are expensive to set up, and we maintain them accross all the
  /// evaluations of the model.
  TF_Session *Session = nullptr;
  TFGraphPtr Graph;
  TFSessionOptionsPtr Options;

  /// The specification of the input nodes.
  std::vector<TF_Output> InputFeed;

  /// The input tensors. They must match by index of the corresponding InputFeed
  /// value. We set up the tensors once and just mutate theirs scalars before
  /// each evaluation. The input tensors keep their value after an evaluation.
  std::vector<TF_Tensor *> Input;

  /// The specification of the output nodes. When evaluating, the tensors in the
  /// output tensor vector must match by index the corresponding element in the
  /// OutputFeed.
  std::vector<TF_Output> OutputFeed;

  /// Reusable utility for deleting the session.
  void deleteSession();

  /// Reusable utility for ensuring we can bind the requested Name to a node in
  /// the SavedModel Graph.
  bool checkReportAndReset(const TF_Output &Output, StringRef Name);
};
} // namespace llvm

#endif // LLVM_HAVE_TF_API
#endif // LLVM_ANALYSIS_UTILS_TFUTILS_H
