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

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/PassManager.h"

namespace llvm {

class TargetTransformInfo;

class SandboxVectorizerPass : public PassInfoMixin<SandboxVectorizerPass> {
  TargetTransformInfo *TTI = nullptr;
  AAResults *AA = nullptr;
  ScalarEvolution *SE = nullptr;
  // NOTE: We define the Context as a pass-scope object instead of local object
  // in runOnFunction() because the passes defined in the pass-manager need
  // access to it for registering/deregistering callbacks during construction
  // and destruction.
  std::unique_ptr<sandboxir::Context> Ctx;

  // A pipeline of SandboxIR function passes run by the vectorizer.
  // NOTE: We define this as a pass-scope object to avoid recreating the
  // pass-pipeline every time in runOnFunction(). The downside is that the
  // Context also needs to be defined as a pass-scope object because the passes
  // within FPM may register/unregister callbacks, so they need access to
  // Context.
  sandboxir::FunctionPassManager FPM;

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
