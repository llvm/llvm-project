//===- DevelopmentModeInlineAdvisor.cpp - runtime-loadable model runner  --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a model runner using Tensorflow C APIs, allowing the
// loading of a model from a command line option.
//
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_API)

#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"
#include "llvm/Analysis/MLInlineAdvisor.h"
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

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

  /// What the change in size was: size_after - size_before
  int64_t Reward = 0;
};

/// Collect data we may use for training a model, and write it as a textual
/// Tensorflow SequenceExample
/// (https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample)
/// protobuf (https://developers.google.com/protocol-buffers).
/// Because this is a protobuf, we cannot just stream the events as they come.
/// Internally, TrainingLogger stores data in column-major format, because that
/// lines up with how TF SequenceExample represents it.
class TrainingLogger final {
public:
  TrainingLogger(StringRef LogFileName, const ModelUnderTrainingRunner *MUTR);

  /// Log one inlining event.
  void logInlineEvent(const InlineEvent &Event,
                      const MLModelRunner &ModelRunner);

  /// Print the stored tensors.
  void print();

private:
  StringRef LogFileName;
  const ModelUnderTrainingRunner *const MUTR;
  std::unique_ptr<Logger> L;
  BitVector Effects;
  /// There's at least one output. We'll set this to a different value if MUTR
  /// is avaliable.
  size_t OutputCount = 1;
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
      std::unique_ptr<MLModelRunner> ModelRunner,
      std::function<bool(CallBase &)> GetDefaultAdvice,
      std::unique_ptr<TrainingLogger> Logger);

  size_t getTotalSizeEstimate();

  virtual ~DevelopmentModeMLInlineAdvisor();
  void updateNativeSizeEstimate(int64_t Change) {
    *CurrentNativeSize += Change;
  }
  void resetNativeSize(Function *F) {
    PreservedAnalyses PA = PreservedAnalyses::all();
    PA.abandon<InlineSizeEstimatorAnalysis>();
    FAM.invalidate(*F, PA);
  }

  std::unique_ptr<MLInlineAdvice>
  getAdviceFromModel(CallBase &CB, OptimizationRemarkEmitter &ORE) override;

  Optional<size_t> getNativeSizeEstimate(const Function &F) const;

private:
  bool isLogging() const { return !!Logger; }
  std::unique_ptr<MLInlineAdvice> getMandatoryAdviceImpl(CallBase &CB) override;

  std::function<bool(CallBase &)> GetDefaultAdvice;
  const bool IsDoingInference;
  std::unique_ptr<TrainingLogger> Logger;

  const Optional<int32_t> InitialNativeSize;
  Optional<int32_t> CurrentNativeSize;
};

/// A variant of MLInlineAdvice that tracks all non-trivial inlining
/// decisions, for training/logging.
class LoggingMLInlineAdvice : public MLInlineAdvice {
public:
  LoggingMLInlineAdvice(DevelopmentModeMLInlineAdvisor *Advisor, CallBase &CB,
                        OptimizationRemarkEmitter &ORE, bool Recommendation,
                        TrainingLogger &Logger,
                        Optional<size_t> CallerSizeEstimateBefore,
                        Optional<size_t> CalleeSizeEstimateBefore,
                        bool DefaultDecision, bool Mandatory = false)
      : MLInlineAdvice(Advisor, CB, ORE, Recommendation), Logger(Logger),
        CallerSizeEstimateBefore(CallerSizeEstimateBefore),
        CalleeSizeEstimateBefore(CalleeSizeEstimateBefore),
        DefaultDecision(DefaultDecision), Mandatory(Mandatory) {}

  virtual ~LoggingMLInlineAdvice() = default;

private:
  DevelopmentModeMLInlineAdvisor *getAdvisor() const {
    return static_cast<DevelopmentModeMLInlineAdvisor *>(Advisor);
  }
  void recordInliningImpl() override {
    MLInlineAdvice::recordInliningImpl();
    getAdvisor()->resetNativeSize(Caller);
    int Reward = std::numeric_limits<int>::max();
    if (InlineSizeEstimatorAnalysis::isEvaluatorRequested() &&
        !getAdvisor()->isForcedToStop()) {
      int NativeSizeAfter = *getAdvisor()->getNativeSizeEstimate(*Caller) +
                            *CalleeSizeEstimateBefore;
      Reward = NativeSizeAfter -
               (*CallerSizeEstimateBefore + *CalleeSizeEstimateBefore);
      getAdvisor()->updateNativeSizeEstimate(Reward);
    }
    log(Reward, /*Success=*/true);
  }

  void recordInliningWithCalleeDeletedImpl() override {
    MLInlineAdvice::recordInliningWithCalleeDeletedImpl();
    getAdvisor()->resetNativeSize(Caller);
    if (InlineSizeEstimatorAnalysis::isEvaluatorRequested() &&
        !getAdvisor()->isForcedToStop()) {
      int NativeSizeAfter = *getAdvisor()->getNativeSizeEstimate(*Caller);
      int Reward = NativeSizeAfter -
                   (*CallerSizeEstimateBefore + *CalleeSizeEstimateBefore);
      getAdvisor()->updateNativeSizeEstimate(Reward);
      log(Reward, /*Success=*/true);
    } else {
      log(NoReward, /*Success=*/true);
    }
  }

  void recordUnsuccessfulInliningImpl(const InlineResult &Result) override {
    MLInlineAdvice::recordUnsuccessfulInliningImpl(Result);
    log(NoReward, /*Success=*/false);
  }

  void recordUnattemptedInliningImpl() override {
    MLInlineAdvice::recordUnattemptedInliningImpl();
    log(NoReward, /*Success=*/false);
  }

  void log(int64_t Reward, bool Success) {
    if (Mandatory)
      return;
    InlineEvent Event;
    Event.AdvisedDecision = isInliningRecommended();
    Event.DefaultDecision = DefaultDecision;
    Event.Effect = Success;
    Event.Reward = Reward;
    Logger.logInlineEvent(Event, getAdvisor()->getModelRunner());
  }

  static const int64_t NoReward = 0;
  TrainingLogger &Logger;
  const Optional<size_t> CallerSizeEstimateBefore;
  const Optional<size_t> CalleeSizeEstimateBefore;
  const int64_t DefaultDecision;
  const int64_t Mandatory;
};

static const std::vector<TensorSpec> TrainingOnlyFeatures{
    TensorSpec::createSpec<int64_t>(TFFeedPrefix + "inlining_default", {1}),
    TensorSpec::createSpec<float>(TFFeedPrefix + "discount", {1}),
    TensorSpec::createSpec<float>(TFFeedPrefix + "reward", {1}),
    TensorSpec::createSpec<int32_t>(TFFeedPrefix + "step_type", {1})};

static const std::vector<TensorSpec> getInputFeatures() {
  std::vector<TensorSpec> InputSpecs;
  for (size_t I = 0; I < NumberOfFeatures; ++I)
    InputSpecs.push_back(TensorSpec::createSpec<int64_t>(
        TFFeedPrefix + FeatureMap[I].name(), FeatureMap[I].shape()));
  append_range(InputSpecs, TrainingOnlyFeatures);
  return InputSpecs;
}

} // namespace

TrainingLogger::TrainingLogger(StringRef LogFileName,
                               const ModelUnderTrainingRunner *MUTR)
    : LogFileName(LogFileName), MUTR(MUTR) {
  // The first output is the inlining decision.
  if (MUTR)
    OutputCount = MUTR->outputLoggedFeatureSpecs().size();
  std::vector<LoggedFeatureSpec> FT;

  for (size_t I = 0; I < NumberOfFeatures; ++I)
    FT.push_back({FeatureMap.at(I), None});
  if (MUTR && MUTR->outputLoggedFeatureSpecs().size() > 1)
    append_range(FT, drop_begin(MUTR->outputLoggedFeatureSpecs()));

  DefaultDecisionPos = FT.size();
  FT.push_back(
      {TensorSpec::createSpec<int64_t>(DefaultDecisionName, {1}), None});

  DecisionPos = FT.size();
  FT.push_back({TensorSpec::createSpec<int64_t>(DecisionName, {1}), None});

  L = std::make_unique<Logger>(
      FT, TensorSpec::createSpec<int64_t>(RewardName, {1}),
      InlineSizeEstimatorAnalysis::isEvaluatorRequested());
}

/// Log one inlining event.
void TrainingLogger::logInlineEvent(const InlineEvent &Event,
                                    const MLModelRunner &ModelRunner) {
  size_t CurrentFeature = 0;
  for (; CurrentFeature < NumberOfFeatures; ++CurrentFeature) {
    int64_t F = *ModelRunner.getTensor<int64_t>(CurrentFeature);
    L->logInt64Value(CurrentFeature, &F);
  }

  for (size_t I = 1; I < OutputCount; ++I) {
    const auto &Result = *MUTR->lastEvaluationResult();
    const char *RawData =
        reinterpret_cast<const char *>(Result.getUntypedTensorValue(I));
    L->logSpecifiedTensorValue(CurrentFeature, RawData);
    ++CurrentFeature;
  }

  assert(CurrentFeature == DefaultDecisionPos);
  L->logInt64Value(DefaultDecisionPos, &Event.DefaultDecision);
  L->logInt64Value(DecisionPos, &Event.AdvisedDecision);
  if (InlineSizeEstimatorAnalysis::isEvaluatorRequested())
    L->logInt64Reward(Event.Reward);

  // For debugging / later use
  Effects.push_back(Event.Effect);
}

void TrainingLogger::print() {
  std::error_code EC;
  raw_fd_ostream OutFile(LogFileName, EC);
  L->flush(OutFile);
}

DevelopmentModeMLInlineAdvisor::DevelopmentModeMLInlineAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::unique_ptr<MLModelRunner> ModelRunner,
    std::function<bool(CallBase &)> GetDefaultAdvice,
    std::unique_ptr<TrainingLogger> Logger)
    : MLInlineAdvisor(M, MAM, std::move(ModelRunner)),
      GetDefaultAdvice(GetDefaultAdvice),
      IsDoingInference(isa<ModelUnderTrainingRunner>(getModelRunner())),
      Logger(std::move(Logger)),
      InitialNativeSize(isLogging() ? getTotalSizeEstimate() : 0),
      CurrentNativeSize(InitialNativeSize) {
  // We cannot have the case of neither inference nor logging.
  assert(IsDoingInference || isLogging());
}

DevelopmentModeMLInlineAdvisor::~DevelopmentModeMLInlineAdvisor() {
  if (isLogging())
    Logger->print();
}

Optional<size_t>
DevelopmentModeMLInlineAdvisor::getNativeSizeEstimate(const Function &F) const {
  if (!InlineSizeEstimatorAnalysis::isEvaluatorRequested())
    return None;
  auto &R =
      FAM.getResult<InlineSizeEstimatorAnalysis>(const_cast<Function &>(F));
  if (!R) {
    F.getParent()->getContext().emitError(
        "Native size estimator is not present.");
    return 0;
  }
  return *R;
}

std::unique_ptr<MLInlineAdvice>
DevelopmentModeMLInlineAdvisor::getMandatoryAdviceImpl(CallBase &CB) {
  return std::make_unique<LoggingMLInlineAdvice>(
      /*Advisor=*/this,
      /*CB=*/CB, /*ORE=*/getCallerORE(CB), /*Recommendation=*/true,
      /*Logger=*/*Logger,
      /*CallerSizeEstimateBefore=*/getNativeSizeEstimate(*CB.getCaller()),
      /*CalleeSizeEstimateBefore=*/
      getNativeSizeEstimate(*CB.getCalledFunction()),
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
      /*CallerSizeEstimateBefore=*/getNativeSizeEstimate(*CB.getCaller()),
      /*CalleeSizeEstimateBefore=*/
      getNativeSizeEstimate(*CB.getCalledFunction()),
      /*DefaultDecision=*/DefaultAdvice);
}

size_t DevelopmentModeMLInlineAdvisor::getTotalSizeEstimate() {
  if (!InlineSizeEstimatorAnalysis::isEvaluatorRequested())
    return 0;
  size_t Ret = 0;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    Ret += *getNativeSizeEstimate(F);
  }
  return Ret;
}

std::unique_ptr<InlineAdvisor> llvm::getDevelopmentModeAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::function<bool(CallBase &)> GetDefaultAdvice) {
  auto &Ctx = M.getContext();
  std::unique_ptr<MLModelRunner> Runner;
  if (TFModelUnderTrainingPath.empty())
    Runner.reset(new NoInferenceModelRunner(Ctx, getInputFeatures()));
  else
    Runner = ModelUnderTrainingRunner::createAndEnsureValid(
        Ctx, TFModelUnderTrainingPath, DecisionName, getInputFeatures(),
        TFOutputSpecOverride);
  if (!Runner)
    return nullptr;
  std::unique_ptr<TrainingLogger> Logger;
  if (!TrainingLog.empty())
    Logger = std::make_unique<TrainingLogger>(
        TrainingLog, dyn_cast<ModelUnderTrainingRunner>(Runner.get()));

  return std::make_unique<DevelopmentModeMLInlineAdvisor>(
      M, MAM, std::move(Runner), GetDefaultAdvice, std::move(Logger));
}
#endif // defined(LLVM_HAVE_TF_API)
