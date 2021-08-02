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

#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {

class GCNSubtarget;
class MachineFunction;
class TargetMachine;

struct AMDGPUResourceUsageAnalysis : public CallGraphSCCPass {
  static char ID;

public:
  // Track resource usage for callee functions.
  struct SIFunctionResourceInfo {
    // Track the number of explicitly used VGPRs. Special registers reserved at
    // the end are tracked separately.
    int32_t NumVGPR = 0;
    int32_t NumAGPR = 0;
    int32_t NumExplicitSGPR = 0;
    uint64_t PrivateSegmentSize = 0;
    bool UsesVCC = false;
    bool UsesFlatScratch = false;
    bool HasDynamicallySizedStack = false;
    bool HasRecursion = false;
    bool HasIndirectCall = false;

    int32_t getTotalNumSGPRs(const GCNSubtarget &ST) const;
    int32_t getTotalNumVGPRs(const GCNSubtarget &ST) const;
  };

  AMDGPUResourceUsageAnalysis() : CallGraphSCCPass(ID) {}

  bool runOnSCC(CallGraphSCC &SCC) override;

  bool doInitialization(CallGraph &CG) override {
    CallGraphResourceInfo.clear();
    return CallGraphSCCPass::doInitialization(CG);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.setPreservesAll();
  }

  const SIFunctionResourceInfo &getResourceInfo(const Function *F) const {
    auto Info = CallGraphResourceInfo.find(F);
    assert(Info != CallGraphResourceInfo.end() &&
           "Failed to find resource info for function");
    return Info->getSecond();
  }

private:
  SIFunctionResourceInfo analyzeResourceUsage(const MachineFunction &MF,
                                              const TargetMachine &TM) const;
  void propagateIndirectCallRegisterUsage();

  DenseMap<const Function *, SIFunctionResourceInfo> CallGraphResourceInfo;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEUSAGEANALYSIS_H
