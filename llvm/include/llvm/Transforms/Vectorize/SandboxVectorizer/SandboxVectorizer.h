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
#include "llvm/SandboxIR/PassManager.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"

namespace llvm {

class TargetTransformInfo;

class SandboxVectorizerPass : public PassInfoMixin<SandboxVectorizerPass> {
  TargetTransformInfo *TTI = nullptr;

  // Used to build a RegionPass pipeline to be run on Regions created by the
  // bottom-up vectorization pass.
  sandboxir::PassRegistry PR;

  // The main vectorizer pass.
  std::unique_ptr<sandboxir::BottomUpVec> BottomUpVecPass;

  // The PM containing the pipeline of region passes. It's owned by the pass
  // registry.
  sandboxir::RegionPassManager *RPM;

  bool runImpl(Function &F);
public:
  SandboxVectorizerPass();
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZER_H
