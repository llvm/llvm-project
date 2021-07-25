//===- TFUtils.cpp - tensorflow evaluation utilities ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

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

size_t TensorSpec::getElementByteSize() const {
  return TF_DataTypeSize(static_cast<TF_DataType>(TypeIndex));
}

TensorSpec::TensorSpec(const std::string &Name, int Port, int TypeIndex,
                       const std::vector<int64_t> &Shape)
    : Name(Name), Port(Port), TypeIndex(TypeIndex), Shape(Shape),
      ElementCount(std::accumulate(Shape.begin(), Shape.end(), 1,
                                   std::multiplies<int64_t>())) {}

Optional<TensorSpec> getTensorSpecFromJSON(LLVMContext &Ctx,
                                           const json::Value &Value) {
  auto EmitError = [&](const llvm::Twine &Message) -> Optional<TensorSpec> {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS << Value;
    Ctx.emitError("Unable to parse JSON Value as spec (" + Message + "): " + S);
    return None;
  };
  // FIXME: accept a Path as a parameter, and use it for error reporting.
  json::Path::Root Root("tensor_spec");
  json::ObjectMapper Mapper(Value, Root);
  if (!Mapper)
    return EmitError("Value is not a dict");

  std::string TensorName;
  int TensorPort = -1;
  std::string TensorType;
  std::vector<int64_t> TensorShape;

  if (!Mapper.map<std::string>("name", TensorName))
    return EmitError("'name' property not present or not a string");
  if (!Mapper.map<std::string>("type", TensorType))
    return EmitError("'type' property not present or not a string");
  if (!Mapper.map<int>("port", TensorPort))
    return EmitError("'port' property not present or not an int");
  if (!Mapper.map<std::vector<int64_t>>("shape", TensorShape))
    return EmitError("'shape' property not present or not an int array");

#define PARSE_TYPE(T, E)                                                       \
  if (TensorType == #T)                                                        \
    return TensorSpec::createSpec<T>(TensorName, TensorShape, TensorPort);
  TFUTILS_SUPPORTED_TYPES(PARSE_TYPE)
#undef PARSE_TYPE
  return None;
}

Optional<std::vector<LoggedFeatureSpec>>
loadOutputSpecs(LLVMContext &Ctx, StringRef ExpectedDecisionName,
                StringRef ModelPath, StringRef SpecFileOverride) {
  SmallVector<char, 128> OutputSpecsPath;
  StringRef FileName = SpecFileOverride;
  if (FileName.empty()) {
    llvm::sys::path::append(OutputSpecsPath, ModelPath, "output_spec.json");
    FileName = {OutputSpecsPath.data(), OutputSpecsPath.size()};
  }

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(FileName);
  if (!BufferOrError) {
    Ctx.emitError("Error opening output specs file: " + FileName + " : " +
                  BufferOrError.getError().message());
    return None;
  }
  auto ParsedJSONValues = json::parse(BufferOrError.get()->getBuffer());
  if (!ParsedJSONValues) {
    Ctx.emitError("Could not parse specs file: " + FileName);
    return None;
  }
  auto ValuesArray = ParsedJSONValues->getAsArray();
  if (!ValuesArray) {
    Ctx.emitError("Expected an array of {tensor_spec:<TensorSpec>, "
                  "logging_name:<name>} dictionaries");
    return None;
  }
  std::vector<LoggedFeatureSpec> Ret;
  for (const auto &Value : *ValuesArray)
    if (const auto *Obj = Value.getAsObject())
      if (const auto *SpecPart = Obj->get("tensor_spec"))
        if (auto TensorSpec = getTensorSpecFromJSON(Ctx, *SpecPart))
          if (auto LoggingName = Obj->getString("logging_name")) {
            if (!TensorSpec->isElementType<int64_t>() &&
                !TensorSpec->isElementType<int32_t>() &&
                !TensorSpec->isElementType<float>()) {
              Ctx.emitError(
                  "Only int64, int32, and float tensors are supported. "
                  "Found unsupported type for tensor named " +
                  TensorSpec->name());
              return None;
            }
            Ret.push_back({*TensorSpec, LoggingName->str()});
          }

  if (ValuesArray->size() != Ret.size()) {
    Ctx.emitError(
        "Unable to parse output spec. It should be a json file containing an "
        "array of dictionaries. Each dictionary must have a 'tensor_spec' key, "
        "with a json object describing a TensorSpec; and a 'logging_name' key, "
        "which is a string to use as name when logging this tensor in the "
        "training log.");
    return None;
  }
  if (Ret.empty() || *Ret[0].LoggingName != ExpectedDecisionName) {
    Ctx.emitError("The first output spec must describe the decision tensor, "
                  "and must have the logging_name " +
                  StringRef(ExpectedDecisionName));
    return None;
  }
  return Ret;
}

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

  tensorflow::SequenceExample SE;
  std::vector<tensorflow::FeatureList *> FeatureLists;
  tensorflow::FeatureList *Reward = nullptr;

public:
  LoggerDataImpl(const std::vector<LoggedFeatureSpec> &LoggedSpecs,
                 const TensorSpec &RewardSpec, bool IncludeReward)
      : LoggedFeatureSpecs(LoggedSpecs), RewardSpec(RewardSpec) {
    auto *FL = SE.mutable_feature_lists()->mutable_feature_list();
    if (IncludeReward)
      Reward = &(*FL)[RewardSpec.name()];
    // Allocate first the map entries, then capture their address. We will not
    // mutate the set of features after this (i.e. the pointers won't dangle).
    for (const auto &LFS : LoggedSpecs) {
      (*FL)[LFS.LoggingName ? *LFS.LoggingName : LFS.Spec.name()] = {};
    }
    for (const auto &LFS : LoggedSpecs)
      FeatureLists.push_back(
          &(*FL)[LFS.LoggingName ? *LFS.LoggingName : LFS.Spec.name()]);
  }

  void print(raw_ostream &OS) {
    std::string OutStr;
    if (ProtobufTextMode)
      google::protobuf::TextFormat::PrintToString(SE, &OutStr);
    else
      OutStr = SE.SerializeAsString();

    OS << OutStr;
  }

  char *addNewTensor(size_t FeatureID) {
    const auto &Spec = LoggedFeatureSpecs[FeatureID].Spec;
    if (Spec.isElementType<float>()) {
      auto *RF = FeatureLists[FeatureID]
                     ->add_feature()
                     ->mutable_float_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0.0);
      return reinterpret_cast<char *>(RF->mutable_data());
    } else if (Spec.isElementType<int32_t>() || Spec.isElementType<int64_t>()) {
      auto *RF = FeatureLists[FeatureID]
                     ->add_feature()
                     ->mutable_int64_list()
                     ->mutable_value();
      RF->Resize(Spec.getElementCount(), 0);
      return reinterpret_cast<char *>(RF->mutable_data());
    }
    llvm_unreachable("Unsupported tensor type.");
  }

  template <typename T> void logReward(T Value) {
    if (RewardSpec.isElementType<float>())
      Reward->add_feature()->mutable_float_list()->add_value(Value);
    else if (RewardSpec.isElementType<int32_t>() ||
             RewardSpec.isElementType<int64_t>())
      Reward->add_feature()->mutable_int64_list()->add_value(Value);
    else
      llvm_unreachable("Unsupported tensor type.");
  }

  size_t getNrRecords() const {
    return FeatureLists.empty() ? 0 : FeatureLists[0]->feature().size();
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
    initInput(I, static_cast<TF_DataType>(InputSpec.typeIndex()),
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

#define TFUTILS_GETDATATYPE_IMPL(T, E)                                         \
  template <> int TensorSpec::getDataType<T>() { return E; }

TFUTILS_SUPPORTED_TYPES(TFUTILS_GETDATATYPE_IMPL)

#undef TFUTILS_GETDATATYPE_IMPL

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

void Logger::print(raw_ostream &OS) { LoggerData->print(OS); }
#endif // defined(LLVM_HAVE_TF_API)
