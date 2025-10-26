//===- SIConvertWaveSize.h ----------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCONVERTWAVESIZE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCONVERTWAVESIZE_H

#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AMDGPUConvertWaveSizePass : public PassInfoMixin<AMDGPUConvertWaveSizePass> {
  /// The target machine.
  const GCNTargetMachine *TM;

public:
  AMDGPUConvertWaveSizePass(const GCNTargetMachine &TM)
      : TM(&TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCONVERTWAVESIZE_H
