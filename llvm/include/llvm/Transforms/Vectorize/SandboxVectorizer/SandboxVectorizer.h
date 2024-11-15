//===- SandboxVectorizer.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H

#include <memory>

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"

namespace llvm {

class TargetTransformInfo;

class SandboxVectorizerPass : public PassInfoMixin<SandboxVectorizerPass> {
  TargetTransformInfo *TTI = nullptr;

  // The main vectorizer pass.
  sandboxir::BottomUpVec BottomUpVecPass;

  bool runImpl(Function &F);

public:
  // Make sure the constructors/destructors are out-of-line. This works around a
  // problem with -DBUILD_SHARED_LIBS=on where components that depend on the
  // Vectorizer component can't find the vtable for classes like
  // sandboxir::Pass. This way we don't have to make LLVMPasses add a direct
  // dependency on SandboxIR.
  SandboxVectorizerPass();
  SandboxVectorizerPass(SandboxVectorizerPass &&);
  ~SandboxVectorizerPass();

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H
