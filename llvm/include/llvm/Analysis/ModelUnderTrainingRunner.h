//===- ModelUnderTrainingRunner.h -- 'development' mode runner --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_ANALYSIS_MODELUNDERTRAININGRUNNER_H
#define LLVM_ANALYSIS_MODELUNDERTRAININGRUNNER_H

#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Config/llvm-config.h"

#ifdef LLVM_HAVE_TF_API
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// ModelUnderTrainingRunner - training mode implementation. It uses TF C APIs
/// to dynamically load and evaluate a TF SavedModel
/// (https://www.tensorflow.org/guide/saved_model). Runtime performance is
/// sacrificed for ease of use while training.
class ModelUnderTrainingRunner final : public MLModelRunner {
public:
  // Disallows copy and assign.
  ModelUnderTrainingRunner(const ModelUnderTrainingRunner &) = delete;
  ModelUnderTrainingRunner &
  operator=(const ModelUnderTrainingRunner &) = delete;

  const std::vector<LoggedFeatureSpec> &outputLoggedFeatureSpecs() const {
    return OutputSpecs;
  }

  const Optional<TFModelEvaluator::EvaluationResult> &
  lastEvaluationResult() const {
    return LastEvaluationResult;
  }
  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::Development;
  }

  static std::unique_ptr<ModelUnderTrainingRunner>
  createAndEnsureValid(LLVMContext &Ctx, const std::string &ModelPath,
                       StringRef DecisionName,
                       const std::vector<TensorSpec> &InputSpecs,
                       StringRef OutputSpecsPathOverride = "");
  static std::unique_ptr<ModelUnderTrainingRunner>
  createAndEnsureValid(LLVMContext &Ctx, const std::string &ModelPath,
                       StringRef DecisionName,
                       const std::vector<TensorSpec> &InputSpecs,
                       const std::vector<LoggedFeatureSpec> &OutputSpecs);

private:
  ModelUnderTrainingRunner(LLVMContext &Ctx, const std::string &ModelPath,
                           const std::vector<TensorSpec> &InputSpecs,
                           const std::vector<LoggedFeatureSpec> &OutputSpecs);

  std::unique_ptr<TFModelEvaluator> Evaluator;
  const std::vector<LoggedFeatureSpec> OutputSpecs;
  Optional<TFModelEvaluator::EvaluationResult> LastEvaluationResult;
  void *evaluateUntyped() override;
  bool isValid() const { return !!Evaluator; }
};

} // namespace llvm
#endif // define(LLVM_HAVE_TF_API)
#endif // LLVM_ANALYSIS_MODELUNDERTRAININGRUNNER_H
