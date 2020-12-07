//===- ReleaseModeModelRunner.cpp - Fast, precompiled model runner  -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a model runner wrapping an AOT compiled ML model.
// Only inference is supported.
//
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_AOT)

#include "llvm/Analysis/InlineModelFeatureMaps.h"
#include "llvm/Analysis/MLInlineAdvisor.h"

// codegen-ed file
#include "InlinerSizeModel.h" // NOLINT

#include <memory>
#include <vector>

using namespace llvm;
namespace {

const char FeedPrefix[] = "feed_";
const char FetchPrefix[] = "fetch_";

/// MLModelRunner - production mode implementation. It uses a AOT-compiled
/// SavedModel for efficient execution.
class ReleaseModeModelRunner final : public MLModelRunner {
public:
  ReleaseModeModelRunner(LLVMContext &Ctx);
  virtual ~ReleaseModeModelRunner() = default;

  bool run() override;

  void setFeature(FeatureIndex Index, int64_t Value) override;
  int64_t getFeature(int Index) const override;

private:
  std::vector<int32_t> FeatureIndices;
  int32_t ResultIndex = -1;
  std::unique_ptr<llvm::InlinerSizeModel> CompiledModel;
};
} // namespace

ReleaseModeModelRunner::ReleaseModeModelRunner(LLVMContext &Ctx)
    : MLModelRunner(Ctx),
      CompiledModel(std::make_unique<llvm::InlinerSizeModel>()) {
  assert(CompiledModel && "The CompiledModel should be valid");

  FeatureIndices.reserve(NumberOfFeatures);

  for (size_t I = 0; I < NumberOfFeatures; ++I) {
    const int Index =
        CompiledModel->LookupArgIndex(FeedPrefix + FeatureNameMap[I]);
    assert(Index >= 0 && "Cannot find Feature in inlining model");
    FeatureIndices[I] = Index;
  }

  ResultIndex =
      CompiledModel->LookupResultIndex(std::string(FetchPrefix) + DecisionName);
  assert(ResultIndex >= 0 && "Cannot find DecisionName in inlining model");
}

int64_t ReleaseModeModelRunner::getFeature(int Index) const {
  return *static_cast<int64_t *>(
      CompiledModel->arg_data(FeatureIndices[Index]));
}

void ReleaseModeModelRunner::setFeature(FeatureIndex Index, int64_t Value) {
  *static_cast<int64_t *>(CompiledModel->arg_data(
      FeatureIndices[static_cast<size_t>(Index)])) = Value;
}

bool ReleaseModeModelRunner::run() {
  CompiledModel->Run();
  return static_cast<bool>(
      *static_cast<int64_t *>(CompiledModel->result_data(ResultIndex)));
}

std::unique_ptr<InlineAdvisor>
llvm::getReleaseModeAdvisor(Module &M, ModuleAnalysisManager &MAM) {
  auto AOTRunner = std::make_unique<ReleaseModeModelRunner>(M.getContext());
  return std::make_unique<MLInlineAdvisor>(M, MAM, std::move(AOTRunner));
}
#endif // defined(LLVM_HAVE_TF_AOT)
