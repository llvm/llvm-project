//===- AMDGPUResourceUsageAnalysis.h ---- analysis of resources -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes how many registers and other resources are used by
/// functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEUSAGEANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEUSAGEANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "SIMachineFunctionInfo.h"

namespace llvm {

class GCNSubtarget;
class MachineFunction;
class TargetMachine;

struct AMDGPUResourceUsageAnalysis : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUResourceUsageAnalysis() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  SIMachineFunctionInfo::SIFunctionResourceInfo
  analyzeResourceUsage(const MachineFunction &MF,
                       uint32_t AssumedStackSizeForDynamicSizeObjects,
                       uint32_t AssumedStackSizeForExternalCall) const;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEUSAGEANALYSIS_H
