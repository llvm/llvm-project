//===- SIConvertWaveSize.h ----------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_AMDGPU_SICONVERTWAVESIZE_H
#define LLVM_LIB_TARGET_AMDGPU_SICONVERTWAVESIZE_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class SIConvertWaveSizePass : public PassInfoMixin<SIConvertWaveSizePass> {
  /// The target machine.
  const TargetMachine *TM;

public:
  SIConvertWaveSizePass(const TargetMachine &TM)
      : TM(&TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SICONVERTWAVESIZE_H
