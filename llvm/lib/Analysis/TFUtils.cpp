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

/// Write a list of tensors as a sequence of TensorFlow FeatureList protobufs.
/// The tensors are assumed to be stored contiguously, in row-major format,
/// in the TensorData buffer. Each tensor has the shape given by Spec. The
/// feature name in the output is either the provided LoggingName, if
/// specified, otherwise it's the name of the tensor (as given by Spec).
void writeRawTensorsAsFeatureLists(tensorflow::FeatureLists *FE,
                                   const LoggedFeatureSpec &LoggedSpec,
                                   const char *TensorData, size_t TensorCount,
                                   bool FinalReward = false) {
  const auto &Spec = LoggedSpec.Spec;
  // The 'Feature' protobuf only has 3 possible fields: float_list,
  // int64_list, or bytes_list, so we capture int32 values as int64. We don't
  // support any other types.
  tensorflow::FeatureList &FL = (*FE->mutable_feature_list())[(
      LoggedSpec.LoggingName ? *LoggedSpec.LoggingName : Spec.name())];

  const char *CurrentTensor = TensorData;
  const size_t TensorByteSize =
      Spec.getElementCount() * Spec.getElementByteSize();
  const size_t ElemCount = Spec.getElementCount();
  for (size_t E = 0; E < TensorCount; ++E) {
    const bool ShouldWrite = E + 1 == TensorCount || !FinalReward;

    if (Spec.isElementType<int64_t>()) {
      auto *MF = FL.add_feature()->mutable_int64_list()->mutable_value();
      MF->Resize(ElemCount, 0);
      if (ShouldWrite)
        memcpy(MF->mutable_data(), CurrentTensor, TensorByteSize);
    } else if (Spec.isElementType<int32_t>()) {
      auto *MF = FL.add_feature()->mutable_int64_list()->mutable_value();
      MF->Resize(ElemCount, 0);
      if (ShouldWrite) {
        const int32_t *TD = reinterpret_cast<const int32_t *>(CurrentTensor);
        for (size_t I = 0; I < ElemCount; ++I)
          (*MF)[I] = TD[I];
      }
    } else if (Spec.isElementType<float>()) {
      auto *MF = FL.add_feature()->mutable_float_list()->mutable_value();
      MF->Resize(ElemCount, 0.0);
      if (ShouldWrite)
        memcpy(MF->mutable_data(), CurrentTensor, TensorByteSize);
    } else {
      llvm_unreachable("Unsupported tensor type.");
    }
    if (ShouldWrite)
      CurrentTensor += TensorByteSize;
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

void Logger::print(raw_ostream &OS) {
  tensorflow::SequenceExample SE;

  if (RawLogData.empty())
    return;
  if (RawLogData[0].empty())
    return;
  size_t Tensor0Size = FeatureSpecs[0].Spec.getElementCount() *
                       FeatureSpecs[0].Spec.getElementByteSize();
  size_t NumberOfRecords = RawLogData[0].size() / Tensor0Size;
  if (NumberOfRecords == 0)
    return;
  size_t RewardSize =
      RewardSpec.getElementCount() * RewardSpec.getElementByteSize();
  size_t NumberOfRewards = RawLogData.back().size() / RewardSize;

  tensorflow::FeatureLists *FE = SE.mutable_feature_lists();
  for (size_t I = 0; I < FeatureSpecs.size(); ++I)
    writeRawTensorsAsFeatureLists(FE, FeatureSpecs[I], RawLogData[I].data(),
                                  NumberOfRecords);

  if (IncludeReward)
    writeRawTensorsAsFeatureLists(FE, {RewardSpec, None},
                                  RawLogData.back().data(), NumberOfRecords,
                                  NumberOfRewards == 1);
  std::string OutStr;
  if (ProtobufTextMode) {
    google::protobuf::TextFormat::PrintToString(SE, &OutStr);
  } else {
    OutStr = SE.SerializeAsString();
  }
  OS << OutStr;
}
#endif // defined(LLVM_HAVE_TF_API)
