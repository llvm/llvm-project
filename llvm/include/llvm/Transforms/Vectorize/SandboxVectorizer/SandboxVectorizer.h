//===- SandboxVectorizer.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TargetTransformInfo;

class SandboxVectorizerPass : public PassInfoMixin<SandboxVectorizerPass> {
  TargetTransformInfo *TTI = nullptr;

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  bool runImpl(Function &F);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H
