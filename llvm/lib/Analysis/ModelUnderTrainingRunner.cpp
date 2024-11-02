//===- ModelUnderTrainingRunner.cpp - 'development' mode runner -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a MLModelRunner for 'development' mode, i.e. evaluation
// happens off a model that's provided from the command line and is interpreted.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_API)

#include "llvm/Analysis/ModelUnderTrainingRunner.h"

using namespace llvm;

ModelUnderTrainingRunner::ModelUnderTrainingRunner(
    LLVMContext &Ctx, const std::string &ModelPath,
    const std::vector<TensorSpec> &InputSpecs,
    const std::vector<LoggedFeatureSpec> &OutputSpecs)
    : MLModelRunner(Ctx, MLModelRunner::Kind::Development, InputSpecs.size()),
      OutputSpecs(OutputSpecs) {
  Evaluator = std::make_unique<TFModelEvaluator>(
      ModelPath, InputSpecs, [&](size_t I) { return OutputSpecs[I].Spec; },
      OutputSpecs.size());
  if (!Evaluator || !Evaluator->isValid()) {
    Ctx.emitError("Failed to create saved model evaluator");
    Evaluator.reset();
    return;
  }

  for (size_t I = 0, E = InputSpecs.size(); I < E; ++I) {
    setUpBufferForTensor(I, InputSpecs[I], Evaluator->getUntypedInput(I));
  }
}

void *ModelUnderTrainingRunner::evaluateUntyped() {
  LastEvaluationResult = Evaluator->evaluate();
  if (!LastEvaluationResult.hasValue()) {
    Ctx.emitError("Error evaluating model.");
    return nullptr;
  }
  return LastEvaluationResult->getUntypedTensorValue(0);
}

std::unique_ptr<ModelUnderTrainingRunner>
ModelUnderTrainingRunner::createAndEnsureValid(
    LLVMContext &Ctx, const std::string &ModelPath, StringRef DecisionName,
    const std::vector<TensorSpec> &InputSpecs,
    StringRef OutputSpecsPathOverride) {
  if (auto MaybeOutputSpecs = loadOutputSpecs(Ctx, DecisionName, ModelPath,
                                              OutputSpecsPathOverride))
    return createAndEnsureValid(Ctx, ModelPath, DecisionName, InputSpecs,
                                *MaybeOutputSpecs);
  Ctx.emitError("Could not load the policy model from the provided path");
  return nullptr;
}

std::unique_ptr<ModelUnderTrainingRunner>
ModelUnderTrainingRunner::createAndEnsureValid(
    LLVMContext &Ctx, const std::string &ModelPath, StringRef DecisionName,
    const std::vector<TensorSpec> &InputSpecs,
    const std::vector<LoggedFeatureSpec> &OutputSpecs) {
  std::unique_ptr<ModelUnderTrainingRunner> MUTR;
  MUTR.reset(
      new ModelUnderTrainingRunner(Ctx, ModelPath, InputSpecs, OutputSpecs));
  if (MUTR && MUTR->isValid())
    return MUTR;

  Ctx.emitError("Could not load or create model evaluator.");
  return nullptr;
}

#endif // defined(LLVM_HAVE_TF_API)
