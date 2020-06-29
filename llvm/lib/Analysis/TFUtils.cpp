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

#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include "tensorflow/c/c_api_experimental.h"

#include <cassert>

using namespace llvm;

namespace {

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

TFModelEvaluator::TFGraphPtr createTFGraph() {
  return TFModelEvaluator::TFGraphPtr(TF_NewGraph(), &TF_DeleteGraph);
}

TFModelEvaluator::TFStatusPtr createTFStatus() {
  return TFModelEvaluator::TFStatusPtr(TF_NewStatus(), &TF_DeleteStatus);
}

TFModelEvaluator::TFSessionOptionsPtr createTFSessionOptions() {
  return TFModelEvaluator::TFSessionOptionsPtr(TF_NewSessionOptions(),
                                               &TF_DeleteSessionOptions);
}
} // namespace

TFModelEvaluator::TFModelEvaluator(StringRef SavedModelPath,
                                   const std::vector<std::string> &InputNames,
                                   const std::vector<std::string> &OutputNames,
                                   const char *Tags)
    : Graph(createTFGraph()), Options(createTFSessionOptions()),
      InputFeed(InputNames.size()), Input(InputNames.size()),
      OutputFeed(OutputNames.size()) {
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
    deleteSession();
  }
  for (size_t I = 0; I < InputNames.size(); ++I) {
    InputFeed[I] = {
        TF_GraphOperationByName(Graph.get(), (InputNames[I]).c_str()), 0};
    if (!checkReportAndReset(InputFeed[I], InputNames[I]))
      return;
  }
  for (size_t I = 0; I < OutputNames.size(); ++I) {
    OutputFeed[I] = {
        TF_GraphOperationByName(Graph.get(), (OutputNames[I]).c_str()), 0};
    if (!checkReportAndReset(OutputFeed[I], OutputNames[I]))
      return;
  }
}

TFModelEvaluator::~TFModelEvaluator() {
  for (auto *T : Input) {
    TF_DeleteTensor(T);
  }
  deleteSession();
}

bool TFModelEvaluator::checkReportAndReset(const TF_Output &Output,
                                           StringRef Name) {
  if (Output.oper)
    return true;
  errs() << "Could not find TF_Output named: " + Name;
  deleteSession();
  return false;
}

void TFModelEvaluator::deleteSession() {
  if (Session == nullptr)
    return;
  auto Status = createTFStatus();
  TF_DeleteSession(Session, Status.get());
  Session = nullptr;
  if (TF_GetCode(Status.get()) != TF_Code::TF_OK)
    errs() << "Could not delete TF session";
}

Optional<TFModelEvaluator::EvaluationResult> TFModelEvaluator::evaluate() {
  if (!isValid())
    return None;
  EvaluationResult Ret(OutputFeed.size());
  auto Status = createTFStatus();
  TF_SessionRun(Session, nullptr, InputFeed.data(), Input.data(), Input.size(),
                OutputFeed.data(), Ret.Output.data(), Ret.Output.size(),
                nullptr, 0, nullptr, Status.get());
  if (TF_GetCode(Status.get()) != TF_Code::TF_OK) {
    errs() << TF_Message(Status.get());
    deleteSession();
    return None;
  }
  return Ret;
}

void TFModelEvaluator::initInput(int Index, TF_DataType Type,
                                 const std::vector<int64_t> &Dimensions) {
  int64_t TotalSize = TF_DataTypeSize(Type);
  for (auto &D : Dimensions)
    TotalSize *= D;

  Input[Index] =
      TF_AllocateTensor(Type, Dimensions.data(), Dimensions.size(), TotalSize);
  std::memset(TF_TensorData(Input[Index]), 0, TotalSize);
}