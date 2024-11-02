//===- ReleaseModeModelRunner.h - Fast, precompiled model runner  ---------===//
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

#ifndef LLVM_ANALYSIS_RELEASEMODEMODELRUNNER_H
#define LLVM_ANALYSIS_RELEASEMODEMODELRUNNER_H

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>
#include <vector>

namespace llvm {

/// ReleaseModeModelRunner - production mode implementation of the
/// MLModelRunner. It uses an AOT-compiled SavedModel for efficient execution.
template <class TGen>
class ReleaseModeModelRunner final : public MLModelRunner {
public:
  /// FeatureNames' type should be an indexed collection of std::string, like
  /// std::array or std::vector, that has a size() method.
  template <class FType>
  ReleaseModeModelRunner(LLVMContext &Ctx, const FType &InputSpec,
                         StringRef DecisionName, StringRef FeedPrefix = "feed_",
                         StringRef FetchPrefix = "fetch_")
      : MLModelRunner(Ctx, MLModelRunner::Kind::Release, InputSpec.size()),
        CompiledModel(std::make_unique<TGen>()) {
    assert(CompiledModel && "The CompiledModel should be valid");

    for (size_t I = 0; I < InputSpec.size(); ++I) {
      const int Index =
          CompiledModel->LookupArgIndex(FeedPrefix.str() + InputSpec[I].name());
      void *Buffer = nullptr;
      if (Index >= 0)
        Buffer = CompiledModel->arg_data(Index);
      setUpBufferForTensor(I, InputSpec[I], Buffer);
    }

    ResultIndex = CompiledModel->LookupResultIndex(FetchPrefix.str() +
                                                   DecisionName.str());
    assert(ResultIndex >= 0 && "Cannot find DecisionName in inlining model");
  }

  virtual ~ReleaseModeModelRunner() = default;

  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::Release;
  }

private:
  void *evaluateUntyped() override {
    CompiledModel->Run();
    return CompiledModel->result_data(ResultIndex);
  }

  int32_t ResultIndex = -1;
  std::unique_ptr<TGen> CompiledModel;
};

/// A mock class satisfying the interface expected by ReleaseModeModelRunner for
/// its `TGen` parameter. Useful to avoid conditional compilation complexity, as
/// a compile-time replacement for a real AOT-ed model.
class NoopSavedModelImpl final {
#define NOOP_MODEL_ERRMSG                                                      \
  "The mock AOT-ed saved model is a compile-time stub and should not be "      \
  "called."

public:
  NoopSavedModelImpl() = default;
  int LookupArgIndex(const std::string &) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  int LookupResultIndex(const std::string &) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  void Run() { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  void *result_data(int) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  void *arg_data(int) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
#undef NOOP_MODEL_ERRMSG
};
} // namespace llvm

#endif // LLVM_ANALYSIS_RELEASEMODEMODELRUNNER_H
