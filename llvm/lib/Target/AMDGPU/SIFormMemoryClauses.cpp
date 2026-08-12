//===-- SIFormMemoryClauses.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass extends the live ranges of registers used as pointers in
/// sequences of adjacent SMEM and VMEM instructions if XNACK is enabled. A
/// load that would overwrite a pointer would require breaking the soft clause.
/// Artificially extend the live ranges of the pointer operands by adding
/// implicit-def early-clobber operands throughout the soft clause.
///
//===----------------------------------------------------------------------===//

#include "SIFormMemoryClauses.h"
#include "AMDGPU.h"
#include "AMDGPUFormMemoryClausesImpl.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-form-memory-clauses"

namespace {

class SIFormMemoryClausesLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIFormMemoryClausesLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Form memory clauses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFormMemoryClausesLegacy, DEBUG_TYPE,
                      "SI Form memory clauses", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SIFormMemoryClausesLegacy, DEBUG_TYPE,
                    "SI Form memory clauses", false, false)

char SIFormMemoryClausesLegacy::ID = 0;

char &llvm::SIFormMemoryClausesID = SIFormMemoryClausesLegacy::ID;

FunctionPass *llvm::createSIFormMemoryClausesLegacyPass() {
  return new SIFormMemoryClausesLegacy();
}

bool SIFormMemoryClausesLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return AMDGPU::AMDGPUFormMemoryClausesImpl(LIS).run(MF);
}

PreservedAnalyses
SIFormMemoryClausesPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPU::AMDGPUFormMemoryClausesImpl(&LIS).run(MF);
  return PreservedAnalyses::all();
}
