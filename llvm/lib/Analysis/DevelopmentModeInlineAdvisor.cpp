//===- DevelopmentModeInlineAdvisor.cpp - runtime-loadable model runner  --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a model runner using Tensorflow C APIs, allowing the
// loading of a model from a command line option.
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"
#include "llvm/Analysis/MLInlineAdvisor.h"
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
    cl::desc("Path to SavedModel from the previous training iteration."));

static cl::opt<std::string> TFFeedPrefix("ml-inliner-trained-model-feed-prefix",
                                         cl::Hidden, cl::init("action_"),
                                         cl::desc("Prefix for feature names."));

static cl::opt<std::string> TFDecisionName(
    "ml-inliner-trained-model-decision-name", cl::Hidden,
    cl::init("StatefulPartitionedCall"),
    cl::desc("Name of the graph operation representing the decision."));

namespace {
/// An InlineEvent, used by TrainingLogger.
struct InlineEvent {
  /// What the default policy's decision would have been.
  bool DefaultDecision = false;

  /// What we advised. When training off the default policy, this is the same as
  /// DefaultDecision.
  bool AdvisedDecision = false;

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
  TrainingLogger() {
    for (size_t I = 0; I < NumberOfFeatures; ++I) {
      Features.push_back(InlineFeatures());
    }
  }

  /// Log one inlining event.
  void logInlineEvent(const InlineEvent &Event,
                      const MLModelRunner &ModelRunner) {
    for (size_t I = 0; I < NumberOfFeatures; ++I) {
      Features[I].push_back(ModelRunner.getFeature(I));
    }
    Decisions.push_back(Event.AdvisedDecision);
    Effects.push_back(Event.Effect);
    Rewards.push_back(Event.Reward);
    DefaultDecisions.push_back(Event.DefaultDecision);
  }

  void printTensor(raw_fd_ostream &OutFile) {
    if (DefaultDecisions.empty())
      return;
    OutFile << "feature_lists: {\n";

    for (size_t I = 0; I < Features.size(); I++) {
      writeTensor(OutFile, FeatureNameMap.at(I), Features[I]);
    }
    writeTensor(OutFile, DefaultDecisionName, DefaultDecisions);
    writeTensor(OutFile, DecisionName, Decisions);
    writeTensor(OutFile, RewardName, Rewards);

    OutFile << "}\n";
  }

private:
  template <typename T>
  void writeTensor(raw_fd_ostream &OutFile, StringRef TensorName,
                   const std::vector<T> &Tensor) {
    OutFile << "  feature_list: {\n";
    OutFile << "    key: "
            << "\"" << TensorName << "\" ";
    OutFile << "value: {\n";
    for (const auto &Feature : Tensor) {
      OutFile << "      feature: { int64_list: { value: [" << Feature
              << "] } }\n";
    }
    OutFile << "    }\n";
    OutFile << "  }\n";
  }

  std::vector<InlineFeatures> Features;
  std::vector<bool> DefaultDecisions;
  std::vector<bool> Decisions;
  std::vector<bool> Effects;
  std::vector<int64_t> Rewards;
  std::vector<bool> Mandatory;
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
      std::function<bool(CallBase &)> GetDefaultAdvice, bool IsDoingInference);

  size_t getTotalSizeEstimate();

  virtual ~DevelopmentModeMLInlineAdvisor();
  void updateNativeSizeEstimate(int64_t Change) { CurrentNativeSize += Change; }
  void resetNativeSize(Function *F) {
    FAM.invalidate<InlineSizeEstimatorAnalysis>(*F);
  }

  std::unique_ptr<MLInlineAdvice>
  getMandatoryAdvice(CallBase &CB, OptimizationRemarkEmitter &ORE) override;
  std::unique_ptr<MLInlineAdvice>
  getAdviceFromModel(CallBase &CB, OptimizationRemarkEmitter &ORE) override;

  size_t getNativeSizeEstimate(const Function &F) const;

private:
  bool isLogging() const { return !TrainingLog.empty(); }

  std::function<bool(CallBase &)> GetDefaultAdvice;
  TrainingLogger Logger;
  const bool IsDoingInference;

  const int32_t InitialNativeSize;
  int32_t CurrentNativeSize = 0;
};

/// A variant of MLInlineAdvice that tracks all non-trivial inlining
/// decisions, for training/logging.
class LoggingMLInlineAdvice : public MLInlineAdvice {
public:
  LoggingMLInlineAdvice(DevelopmentModeMLInlineAdvisor *Advisor, CallBase &CB,
                        OptimizationRemarkEmitter &ORE, bool Recommendation,
                        TrainingLogger &Logger, size_t CallerSizeEstimateBefore,
                        size_t CalleeSizeEstimateBefore, bool DefaultDecision)
      : MLInlineAdvice(Advisor, CB, ORE, Recommendation), Logger(Logger),
        CallerSizeEstimateBefore(CallerSizeEstimateBefore),
        CalleeSizeEstimateBefore(CalleeSizeEstimateBefore),
        DefaultDecision(DefaultDecision) {}

  virtual ~LoggingMLInlineAdvice() = default;

private:
  DevelopmentModeMLInlineAdvisor *getAdvisor() const {
    return static_cast<DevelopmentModeMLInlineAdvisor *>(Advisor);
  }
  void recordInliningImpl() override {
    MLInlineAdvice::recordInliningImpl();
    getAdvisor()->resetNativeSize(Caller);
    int Reward = std::numeric_limits<int>::max();
    if (!getAdvisor()->isForcedToStop()) {
      int NativeSizeAfter = getAdvisor()->getNativeSizeEstimate(*Caller) +
                            CalleeSizeEstimateBefore;
      Reward = NativeSizeAfter -
               (CallerSizeEstimateBefore + CalleeSizeEstimateBefore);
      getAdvisor()->updateNativeSizeEstimate(Reward);
    }
    log(Reward, /*Success=*/true);
  }

  void recordInliningWithCalleeDeletedImpl() override {
    MLInlineAdvice::recordInliningWithCalleeDeletedImpl();
    getAdvisor()->resetNativeSize(Caller);
    if (!getAdvisor()->isForcedToStop()) {
      int NativeSizeAfter = getAdvisor()->getNativeSizeEstimate(*Caller);
      int Reward = NativeSizeAfter -
                   (CallerSizeEstimateBefore + CalleeSizeEstimateBefore);
      getAdvisor()->updateNativeSizeEstimate(Reward);
      log(Reward, /*Success=*/true);
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
    InlineEvent Event;
    Event.AdvisedDecision = isInliningRecommended();
    Event.DefaultDecision = DefaultDecision;
    Event.Effect = Success;
    Event.Reward = Reward;
    Logger.logInlineEvent(Event, getAdvisor()->getModelRunner());
  }

  static const int64_t NoReward = 0;
  TrainingLogger &Logger;
  const size_t CallerSizeEstimateBefore;
  const size_t CalleeSizeEstimateBefore;
  const bool DefaultDecision;
};

/// A pseudo model runner. We use it to store feature values when collecting
/// logs for the default policy, but never ask it to 'run'.
class NoInferenceModelRunner : public MLModelRunner {
public:
  NoInferenceModelRunner(LLVMContext &Ctx)
      : MLModelRunner(Ctx), Features(NumberOfFeatures) {}
  void setFeature(FeatureIndex Index, int64_t Value) override {
    Features[static_cast<int>(Index)] = Value;
  }

  int64_t getFeature(int Index) const override { return Features[Index]; }
  bool run() override {
    llvm_unreachable("We shouldn't call run on this model runner.");
  }

private:
  InlineFeatures Features;
};

/// ModelUnderTrainingRunner - training mode implementation. It uses TF C APIs
/// to dynamically load and evaluate a TF SavedModel
/// (https://www.tensorflow.org/guide/saved_model). Runtime performance is
/// sacrificed for ease of use while training.
class ModelUnderTrainingRunner final : public MLModelRunner {
public:
  ModelUnderTrainingRunner(LLVMContext &Ctx, const std::string &ModelPath);

  bool run() override;

  // Disallows copy and assign.
  ModelUnderTrainingRunner(const ModelUnderTrainingRunner &) = delete;
  ModelUnderTrainingRunner &
  operator=(const ModelUnderTrainingRunner &) = delete;

  void setFeature(FeatureIndex Index, int64_t Value) override;
  int64_t getFeature(int Index) const override;
  bool isValid() const { return !!Evaluator; }

private:
  std::unique_ptr<TFModelEvaluator> Evaluator;

  // The training framework needs some additional features, that just need to
  // be set to 0.
  struct TensorSpec {
    std::string Name;
    std::function<void(TFModelEvaluator *, size_t Index,
                       const std::vector<int64_t> &Dim)>
        Initializer;
  };

  const std::vector<TensorSpec> TrainingOnlyFeatures{
      {"inlining_default",
       [](TFModelEvaluator *Evaluator, size_t Index,
          const std::vector<int64_t> &Dim) {
         Evaluator->initInput<int64_t>(Index, Dim);
       }},
      {"discount",
       [](TFModelEvaluator *Evaluator, size_t Index,
          const std::vector<int64_t> &Dim) {
         Evaluator->initInput<float>(Index, Dim);
       }},
      {"reward",
       [](TFModelEvaluator *Evaluator, size_t Index,
          const std::vector<int64_t> &Dim) {
         Evaluator->initInput<float>(Index, Dim);
       }},
      {"step_type", [](TFModelEvaluator *Evaluator, size_t Index,
                       const std::vector<int64_t> &Dim) {
         Evaluator->initInput<int32_t>(Index, Dim);
       }}};
};
} // namespace

DevelopmentModeMLInlineAdvisor::DevelopmentModeMLInlineAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::unique_ptr<MLModelRunner> ModelRunner,
    std::function<bool(CallBase &)> GetDefaultAdvice, bool IsDoingInference)
    : MLInlineAdvisor(M, MAM, std::move(ModelRunner)),
      GetDefaultAdvice(GetDefaultAdvice), IsDoingInference(IsDoingInference),
      InitialNativeSize(isLogging() ? getTotalSizeEstimate() : 0),
      CurrentNativeSize(InitialNativeSize) {
  // We cannot have the case of neither inference nor logging.
  assert(IsDoingInference || isLogging());
}

DevelopmentModeMLInlineAdvisor::~DevelopmentModeMLInlineAdvisor() {
  if (TrainingLog.empty())
    return;
  std::error_code ErrorCode;
  raw_fd_ostream OutFile(TrainingLog, ErrorCode);
  Logger.printTensor(OutFile);
}

size_t
DevelopmentModeMLInlineAdvisor::getNativeSizeEstimate(const Function &F) const {
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
DevelopmentModeMLInlineAdvisor::getMandatoryAdvice(
    CallBase &CB, OptimizationRemarkEmitter &ORE) {
  if (!isLogging())
    return MLInlineAdvisor::getMandatoryAdvice(CB, ORE);
  return std::make_unique<LoggingMLInlineAdvice>(
      /*Advisor=*/this,
      /*CB=*/CB, /*ORE=*/ORE, /*Recommendation=*/true, /*Logger=*/Logger,
      /*CallerSizeEstimateBefore=*/getNativeSizeEstimate(*CB.getCaller()),
      /*CalleeSizeEstimateBefore=*/
      getNativeSizeEstimate(*CB.getCalledFunction()),
      /*DefaultDecision=*/true);
}

std::unique_ptr<MLInlineAdvice>
DevelopmentModeMLInlineAdvisor::getAdviceFromModel(
    CallBase &CB, OptimizationRemarkEmitter &ORE) {
  if (IsDoingInference && !isLogging())
    return MLInlineAdvisor::getAdviceFromModel(CB, ORE);

  bool DefaultAdvice = GetDefaultAdvice(CB);
  auto Recommendation = IsDoingInference ? ModelRunner->run() : DefaultAdvice;
  return std::make_unique<LoggingMLInlineAdvice>(
      /*Advisor=*/this,
      /*CB=*/CB, /*ORE=*/ORE, /*Recommendation=*/Recommendation,
      /*Logger=*/Logger,
      /*CallerSizeEstimateBefore=*/getNativeSizeEstimate(*CB.getCaller()),
      /*CalleeSizeEstimateBefore=*/
      getNativeSizeEstimate(*CB.getCalledFunction()),
      /*DefaultDecision=*/DefaultAdvice);
}

size_t DevelopmentModeMLInlineAdvisor::getTotalSizeEstimate() {
  size_t Ret = 0;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (isFunctionDeleted(&F))
      continue;
    Ret += getNativeSizeEstimate(F);
  }
  return Ret;
}

ModelUnderTrainingRunner::ModelUnderTrainingRunner(LLVMContext &Ctx,
                                                   const std::string &ModelPath)
    : MLModelRunner(Ctx) {
  std::vector<std::string> InputNames;
  std::vector<std::string> OutputNames;
  for (size_t I = 0; I < NumberOfFeatures; ++I)
    InputNames.push_back(TFFeedPrefix + FeatureNameMap[I]);
  for (size_t I = 0; I < TrainingOnlyFeatures.size(); ++I)
    InputNames.push_back(TFFeedPrefix + TrainingOnlyFeatures[I].Name);
  OutputNames.push_back(TFDecisionName);

  Evaluator =
      std::make_unique<TFModelEvaluator>(ModelPath, InputNames, OutputNames);
  if (!Evaluator || !Evaluator->isValid()) {
    Ctx.emitError("Failed to create inliner saved model evaluator");
    Evaluator.reset();
    return;
  }

  static const std::vector<int64_t> Dim{1};

  size_t InputIndex = 0;
  for (; InputIndex < NumberOfFeatures; ++InputIndex) {
    Evaluator->initInput<int64_t>(InputIndex, Dim);
  }

  for (; InputIndex < InputNames.size(); ++InputIndex) {
    TrainingOnlyFeatures[InputIndex - NumberOfFeatures].Initializer(
        Evaluator.get(), InputIndex, Dim);
  }
}

bool ModelUnderTrainingRunner::run() {
  auto ER = Evaluator->evaluate();
  if (!ER.hasValue()) {
    Ctx.emitError("Error evaluating model.");
    return false;
  }
  int64_t Decision = *ER->getTensorValue<int64_t>(0);
  return static_cast<bool>(Decision);
}

int64_t ModelUnderTrainingRunner::getFeature(int Index) const {
  return *Evaluator->getInput<int64_t>(Index);
}

void ModelUnderTrainingRunner::setFeature(FeatureIndex Index, int64_t Value) {
  size_t NumericIndex = static_cast<size_t>(Index);
  *(Evaluator->getInput<int64_t>(NumericIndex)) = Value;
}

std::unique_ptr<InlineAdvisor> llvm::getDevelopmentModeAdvisor(
    Module &M, ModuleAnalysisManager &MAM,
    std::function<bool(CallBase &)> GetDefaultAdvice) {
  auto &Ctx = M.getContext();
  if (TrainingLog.empty() !=
      !InlineSizeEstimatorAnalysis::isEvaluatorRequested()) {
    Ctx.emitError("For development mode, if training logs are requested, then "
                  "a size estimator must be available; either that, or neither "
                  "are specified.");
    return nullptr;
  }

  std::unique_ptr<MLModelRunner> Runner;

  bool IsDoingInference = false;
  if (TFModelUnderTrainingPath.empty())
    Runner.reset(new NoInferenceModelRunner(Ctx));
  else {
    Runner = std::make_unique<ModelUnderTrainingRunner>(
        Ctx, TFModelUnderTrainingPath);
    if (!Runner) {
      Ctx.emitError("Could not load the policy model from the provided path");
      return nullptr;
    }
    IsDoingInference = true;
  }
  return std::make_unique<DevelopmentModeMLInlineAdvisor>(
      M, MAM, std::move(Runner), GetDefaultAdvice, IsDoingInference);
}