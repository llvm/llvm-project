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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include <cassert>
#include <numeric>

using namespace llvm;

namespace {

using TFGraphPtr = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
using TFSessionOptionsPtr =
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
using TFStatusPtr = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

struct TFInitializer {
  TFInitializer() {
    int Argc = 1;
    const char *Name = "";
    const char **NamePtr = &Name;
    TF_InitMain(Name, &Argc, const_cast<char ***>(&NamePtr));
  }
};

bool ensureInitTF() {
  static TFInitializer TFLibInitializer;
  return true;
}

TFGraphPtr createTFGraph() {
  return TFGraphPtr(TF_NewGraph(), &TF_DeleteGraph);
}

TFStatusPtr createTFStatus() {
  return TFStatusPtr(TF_NewStatus(), &TF_DeleteStatus);
}

TFSessionOptionsPtr createTFSessionOptions() {
  return TFSessionOptionsPtr(TF_NewSessionOptions(), &TF_DeleteSessionOptions);
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
  size_t NrSupported = 0;
  for (size_t I = 0; I < InputSpecs.size(); ++I) {
    auto &InputSpec = InputSpecs[I];
    InputFeed[I] = {
        TF_GraphOperationByName(Graph.get(), (InputSpec.name()).c_str()),
        InputSpec.port()};
    if (!InputFeed[I].oper) {
      continue;
    }
    if (NrSupported++ != I) {
      errs()
          << "Unsupported features must be placed at the end of the InputSpecs";
      invalidate();
      return;
    }
    if (!checkReportAndInvalidate(InputFeed[I], InputSpec))
      return;
    initInput(I, static_cast<TF_DataType>(getTFTypeIndex(InputSpec.type())),
              InputSpec.shape());
  }
  InputFeed.resize(NrSupported);
  Input.resize(NrSupported);

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
  if (Index < Impl->getInput().size())
    return TF_TensorData(Impl->getInput()[Index]);
  return nullptr;
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

#endif // defined(LLVM_HAVE_TF_API)
