//===- EmitCModelRunner.h - Fast, precompiled model runner ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a model runner wrapping an EmitC compiled ML model.
// Only inference is supported.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EMITCMODELRUNNER_H
#define LLVM_ANALYSIS_EMITCMODELRUNNER_H

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"

#include <memory>
#include <type_traits>

namespace llvm {

template <class TGen>
class EmitCModelRunner final : public MLModelRunner {
public:
  template <class FType>
  EmitCModelRunner(LLVMContext &Ctx, const FType &InputSpec,
                   std::unique_ptr<TGen> Model = std::make_unique<TGen>())
      : MLModelRunner(Ctx, MLModelRunner::Kind::Release, InputSpec.size()),
        CompiledModel(std::move(Model)) {
    assert(CompiledModel && "The CompiledModel should be valid");
    for (size_t I = 0; I < InputSpec.size(); ++I)
      populateTensor(I, InputSpec[I]);
  }

  ~EmitCModelRunner() override = default;

  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::Release;
  }

protected:
  void *evaluateUntyped() override { return evaluateImpl(); }

private:
  void populateTensor(size_t Pos, const TensorSpec &Spec) {
    void *Buffer = nullptr;
    auto It = CompiledModel->reflectionMap.find(Spec.name());
    if (It != CompiledModel->reflectionMap.end())
      Buffer = static_cast<void *>(It->second);
    setUpBufferForTensor(Pos, Spec, Buffer);
  }

  using ResultType = decltype(std::declval<TGen>()());

  template <typename R = ResultType>
  std::enable_if_t<!std::is_void_v<R>, void *> evaluateImpl() {
    Result = (*CompiledModel)();
    return &Result;
  }

  template <typename R = ResultType>
  std::enable_if_t<std::is_void_v<R>, void *> evaluateImpl() {
    (*CompiledModel)();
    return nullptr;
  }

  std::conditional_t<std::is_void_v<ResultType>, char, ResultType> Result{};
  std::unique_ptr<TGen> CompiledModel;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EMITCMODELRUNNER_H
