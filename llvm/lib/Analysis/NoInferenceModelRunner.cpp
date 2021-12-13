//===- NoInferenceModelRunner.cpp - noop ML model runner   ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pseudo model runner. We use it to store feature values when collecting
// logs for the default policy, in 'development' mode, but never ask it to
// 'run'.
//===----------------------------------------------------------------------===//
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_API)

#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/Utils/TFUtils.h"

using namespace llvm;

NoInferenceModelRunner::NoInferenceModelRunner(
    LLVMContext &Ctx, const std::vector<TensorSpec> &Inputs)
    : MLModelRunner(Ctx) {
  ValuesBuffer.reserve(Inputs.size());
  for (const auto &TS : Inputs)
    ValuesBuffer.push_back(std::make_unique<char[]>(TS.getElementCount() *
                                                    TS.getElementByteSize()));
}

void *NoInferenceModelRunner::getTensorUntyped(size_t Index) {
  return ValuesBuffer[Index].get();
}
#endif // defined(LLVM_HAVE_TF_API)
