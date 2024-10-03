//===- AMDGPUFixWaveGroupEntry.cpp - modify kernel entry in wavegroup mode ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass modifies kernel entry in wavegroup execution mode
/// idx0 need to be adjusted for per-wave VGPR segment.
/// fp and sp need to be adjusted for per-wave private-memory segment.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-wavegroup-entry"

namespace {

class AMDGPUFixWaveGroupEntry : public ModulePass {
public:
  static char ID;

  AMDGPUFixWaveGroupEntry() : ModulePass(ID) {}

  StringRef getPassName() const override { return "Fix Wave Group Entry"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addRequired<AMDGPUResourceUsageAnalysis>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  bool runOnModule(Module &M) override;

private:
};

} // End anonymous namespace.

char AMDGPUFixWaveGroupEntry::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUFixWaveGroupEntry, DEBUG_TYPE,
                      "SI fix entry for wavegroup execution", false, false)
INITIALIZE_PASS_DEPENDENCY(AMDGPUResourceUsageAnalysis)
INITIALIZE_PASS_END(AMDGPUFixWaveGroupEntry, DEBUG_TYPE,
                    "SI fix entry for wavegroup execution", false, false)

char &llvm::AMDGPUFixWaveGroupEntryID = AMDGPUFixWaveGroupEntry::ID;

ModulePass *llvm::createAMDGPUFixWaveGroupEntryPass() {
  return new AMDGPUFixWaveGroupEntry();
}

bool AMDGPUFixWaveGroupEntry::runOnModule(Module &M) {
  auto ResourceUsage = &getAnalysis<AMDGPUResourceUsageAnalysis>();
  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  bool Changed = false;
  for (Function &F : M.functions()) {
    if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      continue;
    if (!AMDGPU::getWavegroupEnable(F))
      continue;

    auto MF = MMI.getMachineFunction(F);
    auto &Info = ResourceUsage->getResourceInfo();
    auto PerRankVGPRCnt = Info.NumVGPR;

    // Replace the "num vgprs" placeholder that was inserted by frame lowering.
    // We rely on this placeholder only appearing once.
    MachineBasicBlock &Entry = MF->front();
    for (MachineInstr &MI : Entry) {
      bool Found = false;
      for (auto &Op : MI.uses()) {
        if (Op.isTargetIndex() && Op.getIndex() == AMDGPU::TI_NUM_VGPRS) {
          Op.ChangeToImmediate(PerRankVGPRCnt);
          Found = true;
          Changed = true;
          break;
        }
      }
      if (Found)
        break;
    }
  }
  return Changed;
}
