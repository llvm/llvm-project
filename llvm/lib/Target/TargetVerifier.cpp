//===-- TargetVerifier.cpp - LLVM IR Target Verifier ----------------*- C++ -*-===//
////
///// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
///// See https://llvm.org/LICENSE.txt for license information.
///// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/////
/////===----------------------------------------------------------------------===//
/////
///// This file defines target verifier interfaces that can be used for some
///// validation of input to the system, and for checking that transformations
///// haven't done something bad. In contrast to the Verifier or Lint, the
///// TargetVerifier looks for constructions invalid to a particular target
///// machine.
/////
///// To see what specifically is checked, look at TargetVerifier.cpp or an
///// individual backend's TargetVerifier.
/////
/////===----------------------------------------------------------------------===//

#include "llvm/Target/TargetVerifier.h"
#include "llvm/Target/TargetVerify/AMDGPUTargetVerifier.h"

#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace llvm {

void TargetVerify::run(Function &F, FunctionAnalysisManager &AM) {
  if (TT.isAMDGPU()) {
    auto *UA = &AM.getResult<UniformityInfoAnalysis>(F);
    auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
    auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

    AMDGPUTargetVerify TV(Mod, DT, PDT, UA);
    TV.run(F);

    dbgs() << TV.MessagesStr.str();
    if (!TV.MessagesStr.str().empty()) {
      TV.IsValid = false;
    }
  }
}

} // namespace llvm
