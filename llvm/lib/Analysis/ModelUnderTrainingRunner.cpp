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
    : MLModelRunner(Ctx), OutputSpecs(OutputSpecs) {
  Evaluator = std::make_unique<TFModelEvaluator>(
      ModelPath, InputSpecs, [&](size_t I) { return OutputSpecs[I].Spec; },
      OutputSpecs.size());
  if (!Evaluator || !Evaluator->isValid()) {
    Ctx.emitError("Failed to create inliner saved model evaluator");
    Evaluator.reset();
    return;
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

void *ModelUnderTrainingRunner::getTensorUntyped(size_t Index) {
  return Evaluator->getUntypedInput(Index);
}

#endif // defined(LLVM_HAVE_TF_API)
