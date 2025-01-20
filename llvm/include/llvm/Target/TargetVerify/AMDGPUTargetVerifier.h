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

namespace llvm {

class Function;

class AMDGPUTargetVerifierPass : public TargetVerifierPass {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_AMDGPU_TARGET_VERIFIER_H
