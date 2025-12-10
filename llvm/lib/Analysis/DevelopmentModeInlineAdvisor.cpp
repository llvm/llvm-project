//===- DevelopmentModeInlineAdvisor.cpp - runtime-loadable model runner  --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a model runner using TFLite, allowing the
// loading of a model from a command line option.
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TFLITE)

#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/MLInlineAdvisor.h"
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

#include <optional>
#include <vector>

using namespace llvm;

static cl::opt<std::string> TrainingLog(
    "training-log", cl::Hidden,
    cl::desc("Path where the development - mode inlining log is saved."));

static cl::opt<std::string> TFModelUnderTrainingPath(
    "ml-inliner-model-under-training", cl::Hidden,
    cl::desc(R"(Path to SavedModel from the previous training iteration.
The directory is also expected to contain a JSON specification of the 
outputs expected to be logged, where the first entry must be the 
inlining decision. The file containing the specification should be 
called output_spec.json. The expected JSON value is an array of 
dictionaries. Each dictionary should have 2 keys: 

- "tensor_spec, followed by the TensorSpec description of the
output; and 
- "logging_name", a string indicating the name to use when
logging the output values. 

Example:
[
  {
    "logging_name" : "some_name", 
    "tensor_spec" : { 
      "name" : "model_name", 
      "port" : 0,
      "shape" : [2, 3],
      "type" : "float"
      }
  }
]

The first value must always correspond to the decision.)"));

static cl::opt<std::string> TFOutputSpecOverride(
    "ml-inliner-output-spec-override", cl::Hidden,
    cl::desc("Override the path to the output spec json file. See "
             "-ml-inliner-model-under-training documentation for the "
             "specification of that file."));

static cl::opt<std::string> TFFeedPrefix("ml-inliner-trained-model-feed-prefix",
                                         cl::Hidden, cl::init("action_"),
                                         cl::desc("Prefix for feature names."));

namespace {
/// An InlineEvent, used by TrainingLogger.
struct InlineEvent {
  /// What the default policy's decision would have been.
  int64_t DefaultDecision = 0;

  /// What we advised. When training off the default policy, this is the same as
  /// DefaultDecision.
  int64_t AdvisedDecision = 0;

  /// What actually happened. This would be 'false' in the case of an inline
  /// error, even if AdvisedDecision were true, otherwise it agrees with
  /// AdvisedDecision.
  bool Effect = false;
};

/// Collect data we may use for training a model.
class TrainingLogger final {
public:
  TrainingLogger(StringRef LogFileName, const ModelUnderTrainingRunner *MUTR,
                 const std::vector<TensorSpec> &FeatureMap);

  /// Log one inlining event.
  void logInlineEvent(const InlineEvent &Event,
                      const MLModelRunner &ModelRunner);

private:
  StringRef LogFileName;
  const ModelUnderTrainingRunner *const MUTR;
  const std::vector<TensorSpec> &FeatureMap;

  std::unique_ptr<Logger> L;
  BitVector Effects;
  /// Set these 2 clearly OOB, to make sure we set them later.
  size_t DefaultDecisionPos = std::numeric_limits<size_t>::max();
  size_t DecisionPos = std::numeric_limits<size_t>::max();
};

/// An extension of the MLInlineAdvisor for the 'development' mode, targeting
/// the offline training scenario. Note that training happens outside of the
/// compiler, this facility is concerned with producing training data ("logs").
/// This InlineAdvisor can operate in the following modes:
///
/// 1) collect logs for the default policy. This is useful for bootstrapping
/// training, which will be considerably faster by starting from a reasonable
/// policy.
///
/// 2) collect logs for the ML policy, using a model from a previous
/// training. Potentially, that model uses internally some small random
/// perturbation of its weights, to induce exploration (setting this up is the
/// responsibility of the training algorithm). The logs would then be used to
/// retrain and improve on this model.
///
/// 3) use the provided model, with no logging. This is useful for end to end
/// validation - the model, in this case, is a release candidate and shouldn't
/// have random perturbations. It is a convenience feature: rather than needing
/// to take the release candidate model and compile it in 'release' mode,
/// validate it, then potentially discard it, it's easier to just pass the model
/// to the compiler, albeit compilation would be slower, as a one-off. Once the
/// model behaves satisfactorily, it can be compiled AOT, for efficiency, in
/// release mode. The expectation is that a well-trained model provides a good
/// policy over a sufficiently diverse codebase, over many changes (i.e.
/// training happens seldom).
class DevelopmentModeMLInlineAdvisor : public MLInlineAdvisor {
public:
  DevelopmentModeMLInlineAdvisor(
      Module &M, ModuleAnalysisManager &MAM,
      std::function<
          std::unique_ptr<MLModelRunner>(const std::vector<TensorSpec> &)>
          GetModelRunner,
      std::function<bool(CallBase &)> GetDefaultAdvice);

  std::unique_ptr<MLInlineAdvice>
  getAdviceFromModel(CallBase &CB, OptimizationRemarkEmitter &ORE) override;

private:
  bool isLogging() const { return !!Logger; }
  std::unique_ptr<MLInlineAdvice> getMandatoryAdviceImpl(CallBase &CB) override;

  const bool IsDoingInference;
  std::unique_ptr<TrainingLogger> Logger;
};

/// A variant of MLInlineAdvice that tracks all non-trivial inlining
/// decisions, for training/logging.
class LoggingMLInlineAdvice : public MLInlineAdvice {
public:
  LoggingMLInlineAdvice(DevelopmentModeMLInlineAdvisor *Advisor, CallBase &CB,
                        OptimizationRemarkEmitter &ORE, bool Recommendation,
                        TrainingLogger &Logger, bool DefaultDecision,
                        bool Mandatory = false)
      : MLInlineAdvice(Advisor, CB, ORE, Recommendation), Logger(Logger),
        DefaultDecision(DefaultDecision), Mandatory(Mandatory) {}

  virtual ~LoggingMLInlineAdvice() = default;

private:
  DevelopmentModeMLInlineAdvisor *getAdvisor() const {
    return static_cast<DevelopmentModeMLInlineAdvisor *>(Advisor);
  }
  void recordInliningImpl() override {
    MLInlineAdvice::recordInliningImpl();
    log(/*Success=*/true);
  }

  void recordInliningWithCalleeDeletedImpl() override {
    MLInlineAdvice::recordInliningWithCalleeDeletedImpl();
    log(/*Success=*/true);
  }

  void recordUnsuccessfulInliningImpl(const InlineResult &Result) override {
    MLInlineAdvice::recordUnsuccessfulInliningImpl(Result);
    log(/*Success=*/false);
  }

  void recordUnattemptedInliningImpl() override {
    MLInlineAdvice::recordUnattemptedInliningImpl();
    log(/*Success=*/false);
  }

  void log(bool Success) {
    if (Mandatory)
      return;
    InlineEvent Event;
    Event.AdvisedDecision = isInliningRecommended();
    Event.DefaultDecision = DefaultDecision;
    Event.Effect = Success;
    Logger.logInlineEvent(Event, getAdvisor()->getModelRunner());
  }

  TrainingLogger &Logger;
  const int64_t DefaultDecision;
  const int64_t Mandatory;
};

static const std::vector<TensorSpec> TrainingOnlyFeatures{
    TensorSpec::createSpec<float>(TFFeedPrefix + "discount", {1}),
    TensorSpec::createSpec<float>(TFFeedPrefix + "reward", {1}),
    TensorSpec::createSpec<int32_t>(TFFeedPrefix + "step_type", {1})};

// add TFFeedPrefix to the names and also add the "TrainingOnlyFeatures" which
// the model runner needs to see present. We don't set them ourselves or
// interact with them.
static const std::vector<TensorSpec>
convertInputFeatures(const std::vector<TensorSpec> &OriginalFeatures) {
  std::vector<TensorSpec> InputSpecs;
  for (const auto &Feature : OriginalFeatures)
    InputSpecs.push_back(TensorSpec(TFFeedPrefix + Feature.name(), Feature));
  append_range(InputSpecs, TrainingOnlyFeatures);
  return InputSpecs;
}

} // namespace

TrainingLogger::TrainingLogger(StringRef LogFileName,
                               const ModelUnderTrainingRunner *MUTR,
                               const std::vector<TensorSpec> &FeatureMap)
    : LogFileName(LogFileName), MUTR(MUTR), FeatureMap(FeatureMap) {
  // The first output is the inlining decision.
  std::vector<TensorSpec> FT(FeatureMap.begin(), FeatureMap.end());

  if (MUTR)
    append_range(FT, MUTR->extraOutputsForLoggingSpecs());

  DefaultDecisionPos = FT.size();
  FT.push_back(DefaultDecisionSpec);

  DecisionPos = FT.size();
  FT.push_back(InlineDecisionSpec);
  std::error_code EC;
  auto OS = std::make_unique<raw_fd_ostream>(TrainingLog, EC);
  if (EC)
    dbgs() << (EC.message() + ":" + TrainingLog);

  L = std::make_unique<Logger>(std::move(OS), FT,
                               TensorSpec::createSpec<int64_t>(RewardName, {1}),
                               false);
  L->switchContext("");
}

/// Log one inlining event.
void TrainingLogger::logInlineEvent(const InlineEvent &Event,
                                    const MLModelRunner &ModelRunner) {
  L->startObservation();
  size_t CurrentFeature = 0;
  for (; CurrentFeature < FeatureMap.size(); ++CurrentFeature)
    L->logTensorValue(CurrentFeature,
                      reinterpret_cast<const char *>(
                          ModelRunner.getTensorUntyped(CurrentFeature)));

  if (MUTR)
    for (size_t I = 0; I < MUTR->extraOutputsForLoggingSpecs().size(); ++I) {
      const char *RawData =
          reinterpret_cast<const char *>(MUTR->getUntypedExtraOutputValue(I));
      L->logTensorValue(CurrentFeature, RawData);
      ++CurrentFeature;
    }

  assert(CurrentFeature == DefaultDecisionPos);
  L->logTensorValue(DefaultDecisionPos,
                    reinterpret_cast<const char *>(&Event.DefaultDecision));
  L->logTensorValue(DecisionPos,
                    reinterpret_cast<const char *>(&Event.AdvisedDecision));
  L->endObservation();

  // For debugging / later use
  Effects.push_back(Event.Effect);
}

DevelopmentModeMLInlineAdvisor::DevelopmentModeMLInlineAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::function<
        std::unique_ptr<MLModelRunner>(const std::vector<TensorSpec> &)>
        GetModelRunner,
    std::function<bool(CallBase &)> GetDefaultAdvice)
    : MLInlineAdvisor(M, MAM, GetModelRunner, GetDefaultAdvice),
      IsDoingInference(isa<ModelUnderTrainingRunner>(getModelRunner())) {
  // We cannot have the case of neither inference nor logging.
  if (!TrainingLog.empty())
    Logger = std::make_unique<TrainingLogger>(
        TrainingLog, dyn_cast<ModelUnderTrainingRunner>(ModelRunner.get()),
        getFeatureMap());
  assert(IsDoingInference || isLogging());
}

std::unique_ptr<MLInlineAdvice>
DevelopmentModeMLInlineAdvisor::getMandatoryAdviceImpl(CallBase &CB) {
  return std::make_unique<LoggingMLInlineAdvice>(
      /*Advisor=*/this,
      /*CB=*/CB, /*ORE=*/getCallerORE(CB), /*Recommendation=*/true,
      /*Logger=*/*Logger,
      /*DefaultDecision=*/true, /*Mandatory*/ true);
}

std::unique_ptr<MLInlineAdvice>
DevelopmentModeMLInlineAdvisor::getAdviceFromModel(
    CallBase &CB, OptimizationRemarkEmitter &ORE) {
  if (IsDoingInference && !isLogging())
    return MLInlineAdvisor::getAdviceFromModel(CB, ORE);

  bool DefaultAdvice = GetDefaultAdvice(CB);
  auto Recommendation =
      IsDoingInference ? static_cast<bool>(ModelRunner->evaluate<int64_t>())
                       : DefaultAdvice;
  return std::make_unique<LoggingMLInlineAdvice>(
      /*Advisor=*/this,
      /*CB=*/CB, /*ORE=*/ORE, /*Recommendation=*/Recommendation,
      /*Logger=*/*Logger,
      /*DefaultDecision=*/DefaultAdvice);
}

std::unique_ptr<InlineAdvisor> llvm::getDevelopmentModeAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::function<bool(CallBase &)> GetDefaultAdvice) {
  auto &Ctx = M.getContext();
  auto RunnerFactory = [&](const std::vector<TensorSpec> &InputFeatures)
      -> std::unique_ptr<MLModelRunner> {
    std::unique_ptr<MLModelRunner> Runner;
    const std::vector<TensorSpec> ConvertedFeatures =
        convertInputFeatures(InputFeatures);
    if (TFModelUnderTrainingPath.empty())
      Runner.reset(new NoInferenceModelRunner(Ctx, ConvertedFeatures));
    else
      Runner = ModelUnderTrainingRunner::createAndEnsureValid(
          Ctx, TFModelUnderTrainingPath, DecisionName, ConvertedFeatures,
          TFOutputSpecOverride);
    if (!Runner)
      return nullptr;
    return Runner;
  };
  return std::make_unique<DevelopmentModeMLInlineAdvisor>(M, MAM, RunnerFactory,
                                                          GetDefaultAdvice);
}
#endif // defined(LLVM_HAVE_TFLITE)
