//===-- AMDGPUFormSSAMemoryClauses.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass is a clone of SIFormMemoryClauses intended to run in SSA
/// form, before PHI elimination. It extends the live ranges of registers used
/// as pointers in sequences of adjacent SMEM and VMEM instructions when XNACK
/// is enabled, preventing a load from overwriting a pointer and requiring a
// soft clause break.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUFormSSAMemoryClauses.h"
#include "AMDGPU.h"
#include "AMDGPUFormMemoryClausesImpl.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-form-ssa-memory-clauses"

namespace {

class AMDGPUFormSSAMemoryClausesLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUFormSSAMemoryClausesLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Form SSA Memory Clauses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Unlike SIFormMemoryClauses, we do NOT clear the IsSSA property because
  // this pass is designed to run while the function is still in SSA form.
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUFormSSAMemoryClausesLegacy, DEBUG_TYPE,
                      "AMDGPU Form SSA Memory Clauses", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUFormSSAMemoryClausesLegacy, DEBUG_TYPE,
                    "AMDGPU Form SSA Memory Clauses", false, false)

char AMDGPUFormSSAMemoryClausesLegacy::ID = 0;

char &llvm::AMDGPUFormSSAMemoryClausesID = AMDGPUFormSSAMemoryClausesLegacy::ID;

FunctionPass *llvm::createAMDGPUFormSSAMemoryClausesLegacyPass() {
  return new AMDGPUFormSSAMemoryClausesLegacy();
}

bool AMDGPUFormSSAMemoryClausesLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return AMDGPU::AMDGPUFormMemoryClausesImpl(LIS).run(MF);
}

PreservedAnalyses
AMDGPUFormSSAMemoryClausesPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPU::AMDGPUFormMemoryClausesImpl(&LIS).run(MF);
  return PreservedAnalyses::all();
}
