//===-- llvm/Target/TargetVerify/AMDGPUTargetVerifier.h - AMDGPU ---*- C++ -*-===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===----------------------------------------------------------------------===//
////
//// This file defines target verifier interfaces that can be used for some
//// validation of input to the system, and for checking that transformations
//// haven't done something bad. In contrast to the Verifier or Lint, the
//// TargetVerifier looks for constructions invalid to a particular target
//// machine.
////
//// To see what specifically is checked, look at an individual backend's
//// TargetVerifier.
////
////===----------------------------------------------------------------------===//

#ifndef LLVM_AMDGPU_TARGET_VERIFIER_H
#define LLVM_AMDGPU_TARGET_VERIFIER_H

#include "llvm/Target/TargetVerifier.h"

#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"

namespace llvm {

class Function;

class AMDGPUTargetVerifierPass : public TargetVerifierPass {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class AMDGPUTargetVerify : public TargetVerify {
public:
  Module *Mod;

  DominatorTree *DT;
  PostDominatorTree *PDT;
  UniformityInfo *UA;

  AMDGPUTargetVerify(Module *Mod, DominatorTree *DT, PostDominatorTree *PDT, UniformityInfo *UA)
    : TargetVerify(Mod), Mod(Mod), DT(DT), PDT(PDT), UA(UA) {}

  void run(Function &F);
};

} // namespace llvm

#endif // LLVM_AMDGPU_TARGET_VERIFIER_H
