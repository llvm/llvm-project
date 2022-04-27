//===- TFUtils.cpp - tensorflow evaluation utilities ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for interfacing with tensorflow C APIs.
//
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_API)

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "google/protobuf/struct.pb.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/core/example/example.pb.h"
#include <cassert>
#include <numeric>

using namespace llvm;

using google::protobuf::Message;
using google::protobuf::TextFormat;

static cl::opt<bool>
    ProtobufTextMode("tfutils-text-log", cl::init(false), cl::Hidden,
                     cl::desc("Output textual (human-readable) protobuf."));

namespace {

using TFGraphPtr = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
using TFSessionOptionsPtr =
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
using TFStatusPtr = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

struct TFInitializer {
  TFInitializer() {
    assert(!IsInitialized && "TFInitialized should be called only once");
    int Argc = 1;
    const char *Name = "";
    const char **NamePtr = &Name;
    TF_InitMain(Name, &Argc, const_cast<char ***>(&NamePtr));
    IsInitialized = true;
  }
  bool IsInitialized = false;
};

llvm::ManagedStatic<TFInitializer> TFLibInitializer;

bool ensureInitTF() { return TFLibInitializer->IsInitialized; }

TFGraphPtr createTFGraph() {
  return TFGraphPtr(TF_NewGraph(), &TF_DeleteGraph);
}

TFStatusPtr createTFStatus() {
  return TFStatusPtr(TF_NewStatus(), &TF_DeleteStatus);
}

TFSessionOptionsPtr createTFSessionOptions() {
  return TFSessionOptionsPtr(TF_NewSessionOptions(), &TF_DeleteSessionOptions);
}

void serialize(const Message &SE, std::string *OutStr) {
  if (ProtobufTextMode) {
    TextFormat::PrintToString(SE, OutStr);
  } else {
    *OutStr = SE.SerializeAsString();
  }
}

int getTFTypeIndex(TensorType TType) {
  switch (TType) {
  case TensorType::Double:
    return TF_DOUBLE;
  case TensorType::Float:
    return TF_FLOAT;
  case TensorType::Int8:
    return TF_INT8;
  case TensorType::UInt8:
    return TF_UINT8;
  case TensorType::Int16:
    return TF_INT16;
  case TensorType::UInt16:
    return TF_UINT16;
  case TensorType::Int32:
    return TF_INT32;
  case TensorType::UInt32:
    return TF_UINT32;
  case TensorType::Int64:
    return TF_INT64;
  case TensorType::UInt64:
    return TF_UINT64;
  case TensorType::Invalid:
    llvm_unreachable("Unknown tensor type");
  }
}
} // namespace

namespace llvm {
class EvaluationResultImpl {
public:
  EvaluationResultImpl(size_t OutputSize)
      : OutputSize(OutputSize), Output(OutputSize){};

  ~EvaluationResultImpl() {
    for (auto *P : Output)
      if (P)
        TF_DeleteTensor(P);
  }

  EvaluationResultImpl(const EvaluationResultImpl &) = delete;
  EvaluationResultImpl(EvaluationResultImpl &&Other) = delete;
  std::vector<TF_Tensor *> &getOutput() { return Output; }

private:
  const size_t OutputSize;
  std::vector<TF_Tensor *> Output;
};

class TFModelEvaluatorImpl {
public:
  TFModelEvaluatorImpl(StringRef SavedModelPath,
                       const std::vector<TensorSpec> &InputSpecs,
                       function_ref<TensorSpec(size_t)> GetOutputSpecs,
                       size_t OutputSpecsSize, const char *Tags);

  bool isValid() const { return IsValid; }
  size_t OutputSize() const { return OutputFeed.size(); }

  void evaluate(TF_Tensor **Output, TF_Status *Status) {
    TF_SessionRun(Session, nullptr, InputFeed.data(), Input.data(),
                  Input.size(), OutputFeed.data(), Output, OutputFeed.size(),
                  nullptr, 0, nullptr, Status);
  }

  void initInput(size_t Index, TF_DataType Type,
                 const std::vector<int64_t> &Dimensions);
  const std::vector<TF_Tensor *> &getInput() const { return Input; }

  ~TFModelEvaluatorImpl();

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

  void invalidate() { IsValid = false; }

  bool IsValid = true;

  /// Reusable utility for ensuring we can bind the requested Name to a node in
  /// the SavedModel Graph.
  bool checkReportAndInvalidate(const TF_Output &Output,
                                const TensorSpec &OutputSpec);
};

class LoggerDataImpl {
  const std::vector<LoggedFeatureSpec> LoggedFeatureSpecs;
  const TensorSpec RewardSpec;
  const bool IncludeReward;

  std::vector<tensorflow::FeatureList> FeatureLists;
  tensorflow::FeatureList Reward;

  bool isSelfConsistent(const tensorflow::SequenceExample &SE,
                        size_t NrRecords) const {
    bool Ret = true;
    for (const auto &TSpecs : LoggedFeatureSpecs) {
      const auto &Name = TSpecs.getLoggingName();
      const auto &FL = SE.feature_lists().feature_list().at(Name).feature();
      if (NrRecords != static_cast<size_t>(FL.size())) {
        dbgs() << "[TF-UTILS]: " << Name << " has missing records. Expected "
               << NrRecords << " got " << FL.size() << "\n";
        Ret = false;
      }
    }
    if (IncludeReward && static_cast<size_t>(SE.feature_lists()
                                                 .feature_list()
                                                 .at(RewardSpec.name())
                                                 .feature()
                                                 .size()) != NrRecords) {
      dbgs() << "[TF-UTILS]: reward is missing records.\n";
      Ret = false;
    }
    return Ret;
  }

  void transferLog(tensorflow::SequenceExample &SE) {
    auto *FL = SE.mutable_feature_lists()->mutable_feature_list();
    if (IncludeReward)
      (*FL)[RewardSpec.name()] = std::move(Reward);
    assert(FeatureLists.size() == LoggedFeatureSpecs.size());
    for (size_t I = 0; I < FeatureLists.size(); ++I) {
      const auto &LFS = LoggedFeatureSpecs[I];
      (*FL)[LFS.getLoggingName()] = std::move(FeatureLists[I]);
    }
  }

public:
  LoggerDataImpl(const std::vector<LoggedFeatureSpec> &LoggedSpecs,
                 const TensorSpec &RewardSpec, bool IncludeReward)
      : LoggedFeatureSpecs(LoggedSpecs), RewardSpec(RewardSpec),
        IncludeReward(IncludeReward), FeatureLists(LoggedFeatureSpecs.size()) {}

  // flush the logged info to a stream and clear the log contents.
  void flush(std::string *Str) {
    size_t NrRecords = getNrRecords();
    (void)NrRecords;
    tensorflow::SequenceExample SE;
    transferLog(SE);
    assert(isSelfConsistent(SE, NrRecords));
    serialize(SE, Str);
  }

  char *addNewTensor(size_t FeatureID) {
    const auto &Spec = LoggedFeatureSpecs[FeatureID].Spec;
    if (Spec.isElementType<float>()) {
      auto *RF = FeatureLists[FeatureID]
                     .add_feature()
                     ->mutable_float_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0.0);
      return reinterpret_cast<char *>(RF->mutable_data());
    } else if (Spec.isElementType<int32_t>() || Spec.isElementType<int64_t>()) {
      auto *RF = FeatureLists[FeatureID]
                     .add_feature()
                     ->mutable_int64_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0);
      return reinterpret_cast<char *>(RF->mutable_data());
    }
    llvm_unreachable("Unsupported tensor type.");
  }

  template <typename T> void logReward(T Value) {
    assert(IncludeReward);
    if (RewardSpec.isElementType<float>())
      Reward.add_feature()->mutable_float_list()->add_value(Value);
    else if (RewardSpec.isElementType<int32_t>() ||
             RewardSpec.isElementType<int64_t>())
      Reward.add_feature()->mutable_int64_list()->add_value(Value);
    else
      llvm_unreachable("Unsupported tensor type.");
  }

  size_t getNrRecords() const {
    return FeatureLists.empty() ? 0 : FeatureLists[0].feature().size();
  }
};
} // namespace llvm

TFModelEvaluatorImpl::TFModelEvaluatorImpl(
    StringRef SavedModelPath, const std::vector<TensorSpec> &InputSpecs,
    function_ref<TensorSpec(size_t)> GetOutputSpecs, size_t OutputSpecsSize,
    const char *Tags = "serve")
    : Graph(createTFGraph()), Options(createTFSessionOptions()),
      InputFeed(InputSpecs.size()), Input(InputSpecs.size()),
      OutputFeed(OutputSpecsSize) {
  if (!ensureInitTF()) {
    errs() << "Tensorflow should have been initialized";
    return;
  }
  auto Status = createTFStatus();

  Session = TF_LoadSessionFromSavedModel(Options.get(), nullptr,
                                         SavedModelPath.str().c_str(), &Tags, 1,
                                         Graph.get(), nullptr, Status.get());
  if (TF_GetCode(Status.get()) != TF_Code::TF_OK) {
    errs() << TF_Message(Status.get());
    invalidate();
  }
  for (size_t I = 0; I < InputSpecs.size(); ++I) {
    auto &InputSpec = InputSpecs[I];
    InputFeed[I] = {
        TF_GraphOperationByName(Graph.get(), (InputSpec.name()).c_str()),
        InputSpec.port()};
    if (!checkReportAndInvalidate(InputFeed[I], InputSpec))
      return;
    initInput(I, static_cast<TF_DataType>(getTFTypeIndex(InputSpec.type())),
              InputSpec.shape());
  }
  for (size_t I = 0; I < OutputSpecsSize; ++I) {
    auto OutputSpec = GetOutputSpecs(I);
    OutputFeed[I] = {
        TF_GraphOperationByName(Graph.get(), (OutputSpec.name()).c_str()),
        OutputSpec.port()};
    if (!checkReportAndInvalidate(OutputFeed[I], OutputSpec))
      return;
  }
}

TFModelEvaluator::TFModelEvaluator(
    StringRef SavedModelPath, const std::vector<TensorSpec> &InputSpecs,
    function_ref<TensorSpec(size_t)> GetOutputSpecs, size_t OutputSpecsSize,
    const char *Tags)
    : Impl(new TFModelEvaluatorImpl(SavedModelPath, InputSpecs, GetOutputSpecs,
                                    OutputSpecsSize, Tags)) {
  if (!Impl->isValid())
    Impl.reset();
}

TFModelEvaluator::TFModelEvaluator(StringRef SavedModelPath,
                                   const std::vector<TensorSpec> &InputSpecs,
                                   const std::vector<TensorSpec> &OutputSpecs,
                                   const char *Tags)
    : TFModelEvaluator(
          SavedModelPath, InputSpecs, [&](size_t I) { return OutputSpecs[I]; },
          OutputSpecs.size(), Tags) {}

TFModelEvaluatorImpl::~TFModelEvaluatorImpl() {
  for (auto *T : Input) {
    TF_DeleteTensor(T);
  }
  if (Session == nullptr)
    return;
  auto Status = createTFStatus();
  TF_DeleteSession(Session, Status.get());
  Session = nullptr;
  if (TF_GetCode(Status.get()) != TF_Code::TF_OK)
    errs() << "Could not delete TF session";
}

bool TFModelEvaluatorImpl::checkReportAndInvalidate(
    const TF_Output &Output, const TensorSpec &OutputSpec) {
  if (Output.oper)
    return true;
  errs() << "Could not find TF_Output named: " + OutputSpec.name();
  IsValid = false;
  return IsValid;
}

Optional<TFModelEvaluator::EvaluationResult> TFModelEvaluator::evaluate() {
  if (!isValid())
    return None;
  std::unique_ptr<EvaluationResultImpl> Ret =
      std::make_unique<EvaluationResultImpl>(Impl->OutputSize());
  auto Status = createTFStatus();
  Impl->evaluate(Ret->getOutput().data(), Status.get());
  if (TF_GetCode(Status.get()) != TF_Code::TF_OK) {
    errs() << TF_Message(Status.get());
    Impl.reset();
    return None;
  }
  return EvaluationResult(std::move(Ret));
}

void TFModelEvaluatorImpl::initInput(size_t Index, TF_DataType Type,
                                     const std::vector<int64_t> &Dimensions) {
  int64_t TotalSize = TF_DataTypeSize(Type);
  for (auto &D : Dimensions)
    TotalSize *= D;

  Input[Index] =
      TF_AllocateTensor(Type, Dimensions.data(), Dimensions.size(), TotalSize);
  std::memset(TF_TensorData(Input[Index]), 0, TotalSize);
}

void *TFModelEvaluator::getUntypedInput(size_t Index) {
  return TF_TensorData(Impl->getInput()[Index]);
}

TFModelEvaluator::EvaluationResult::EvaluationResult(
    std::unique_ptr<EvaluationResultImpl> Impl)
    : Impl(std::move(Impl)) {}

TFModelEvaluator::EvaluationResult::EvaluationResult(EvaluationResult &&Other)
    : Impl(std::move(Other.Impl)) {}

TFModelEvaluator::EvaluationResult &
TFModelEvaluator::EvaluationResult::operator=(EvaluationResult &&Other) {
  Impl = std::move(Other.Impl);
  return *this;
}

void *TFModelEvaluator::EvaluationResult::getUntypedTensorValue(size_t Index) {
  return TF_TensorData(Impl->getOutput()[Index]);
}

const void *
TFModelEvaluator::EvaluationResult::getUntypedTensorValue(size_t Index) const {
  return TF_TensorData(Impl->getOutput()[Index]);
}

TFModelEvaluator::EvaluationResult::~EvaluationResult() {}
TFModelEvaluator::~TFModelEvaluator() {}

Logger::Logger(const std::vector<LoggedFeatureSpec> &FeatureSpecs,
               const TensorSpec &RewardSpec, bool IncludeReward)
    : FeatureSpecs(FeatureSpecs), RewardSpec(RewardSpec),
      IncludeReward(IncludeReward),
      LoggerData(std::make_unique<LoggerDataImpl>(FeatureSpecs, RewardSpec,
                                                  IncludeReward)) {}

Logger::~Logger() {}

#define LOG_REWARD(NAME, TYPE)                                                 \
  void Logger::log##NAME##Reward(TYPE Value) {                                 \
    assert(IncludeReward);                                                     \
    LoggerData->logReward(Value);                                              \
  }

LOG_REWARD(Float, float)
LOG_REWARD(Int32, int32_t)
LOG_REWARD(Int64, int64_t)
#undef LOG_REWARD

#define LOG_FINAL_REWARD(NAME, TYPE)                                           \
  void Logger::log##NAME##FinalReward(TYPE Value) {                            \
    assert(RewardSpec.isElementType<TYPE>());                                  \
    for (size_t I = 1; I < LoggerData->getNrRecords(); ++I)                    \
      log##NAME##Reward(0);                                                    \
    log##NAME##Reward(Value);                                                  \
  }

LOG_FINAL_REWARD(Float, float)
LOG_FINAL_REWARD(Int32, int32_t)
LOG_FINAL_REWARD(Int64, int64_t)
#undef LOG_FINAL_REWARD

void Logger::logFloatValue(size_t FeatureID, const float *Value) {
  assert(FeatureSpecs[FeatureID].Spec.isElementType<float>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logInt64Value(size_t FeatureID, const int64_t *Value) {
  assert(FeatureSpecs[FeatureID].Spec.isElementType<int64_t>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logInt32Value(size_t FeatureID, const int32_t *Value) {
  assert(FeatureSpecs[FeatureID].Spec.isElementType<int32_t>());
  logSpecifiedTensorValue(FeatureID, reinterpret_cast<const char *>(Value));
}

void Logger::logSpecifiedTensorValue(size_t FeatureID, const char *RawData) {
  const auto &Spec = FeatureSpecs[FeatureID].Spec;
  char *Buff = addEntryAndGetFloatOrInt64Buffer(FeatureID);
  if (Spec.isElementType<int32_t>())
    for (size_t I = 0; I < Spec.getElementCount(); ++I)
      (reinterpret_cast<int64_t *>(Buff))[I] =
          static_cast<int64_t>((reinterpret_cast<const int32_t *>(RawData))[I]);
  else if (Spec.isElementType<int64_t>() || Spec.isElementType<float>())
    std::memcpy(Buff, RawData,
                Spec.getElementCount() * Spec.getElementByteSize());
  else
    llvm_unreachable("Unsupported tensor type");
}

char *Logger::addEntryAndGetFloatOrInt64Buffer(size_t FeatureID) {
  return reinterpret_cast<char *>(LoggerData->addNewTensor(FeatureID));
}

void Logger::flush(std::string *Str) { LoggerData->flush(Str); }

void Logger::flush(raw_ostream &OS) {
  std::string Buff;
  LoggerData->flush(&Buff);
  OS << Buff;
}

void Logger::flushLogs(raw_ostream &OS,
                       const StringMap<std::unique_ptr<Logger>> &Loggers) {
  google::protobuf::Struct Msg;
  for (const auto &NamedLogger : Loggers) {
    tensorflow::SequenceExample SE;
    const auto &Logger = NamedLogger.second;
    std::string Unencoded;
    if (Logger->LoggerData->getNrRecords() > 0)
      Logger->flush(&Unencoded);

    (*Msg.mutable_fields())[NamedLogger.first().str()]
        .mutable_string_value()
        ->append(ProtobufTextMode ? Unencoded : encodeBase64(Unencoded));
  }

  std::string OutStr;
  serialize(Msg, &OutStr);
  OS << OutStr;
}
#endif // defined(LLVM_HAVE_TF_API)
