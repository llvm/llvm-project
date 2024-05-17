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

    auto MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;

    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    if (!ST.hasVGPRIndexingRegisters())
      continue;

    auto TII = ST.getInstrInfo();
    auto MFI = MF->getInfo<SIMachineFunctionInfo>();
    auto TFI = ST.getFrameLowering();

    MachineBasicBlock &Entry = MF->front();

    // TODO-GFX13: replace this condition with the real wavegroup attribute
    auto LaneSharedSize = MFI->getLaneSharedSize();
    bool InWaveGroup = (LaneSharedSize > 0);
    if (!InWaveGroup) {
      // assume that hw_reg_gpr_msb_idx0 has been initialized to zero by HW
      continue;
    }

    // For wavegroup execution, we need to do the following:
    // - set IDX0 based upon the VGPR-count-per-wave and the wave-id
    //   in wavegroup.
    // - set fp based upon the stack-size-per-wave and the wave-id in
    //   wavegroup.
    // - adjust sp based upon fp
    // TODO-GFX13: implementation so far only works for the case in which
    // all waves are running the same kernel. It does not handle the cases
    // of rank-specialized wavegroup. For that, We need some meta-data
    // that describes the kernel set, assuming they are all in the same module.

    Register FPReg = MFI->getFrameOffsetReg();
    assert(FPReg != AMDGPU::FP_REG);
    assert(TFI->hasFP(*MF));
    // find the original initialization of FRReg;
    MachineInstr *FPInit = nullptr;
    MachineBasicBlock::iterator I = Entry.begin(), E = Entry.end();
    for (; !FPInit && I != E && !I->isTerminator(); ++I) {
      MachineInstr *MI = &(*I);
      for (auto &Opnd : MI->all_defs()) {
        if (Opnd.getReg() == FPReg) {
          FPInit = MI;
          break;
        }
      }
    }
    assert(FPInit && "cannot find frame-pointer intializer in entry!");
    // find the original initialization of SPReg;
    MachineInstr *SPInit = nullptr;
    Register SPReg = MFI->getStackPtrOffsetReg();
    assert(SPReg != AMDGPU::SP_REG);
    if (TFI->requiresStackPointerReference(*MF)) {
      for (; !SPInit && I != E && !I->isTerminator(); ++I) {
        MachineInstr *MI = &(*I);
        for (auto &Opnd : MI->all_defs()) {
          if (Opnd.getReg() == SPReg) {
            SPInit = MI;
            break;
          }
        }
      }
      assert(SPInit && "cannot find stack-pointer intializer in entry!");
    }
    auto &Info = ResourceUsage->getResourceInfo(&F);
    Changed = true;
    I = Entry.begin();
    DebugLoc DL;
    auto AlignUnit = ST.getStackAlignment();
    auto RankScratchStart = alignTo(LaneSharedSize, AlignUnit);
    auto PerRankScratch = alignTo(Info.PrivateSegmentSize, AlignUnit);
    auto PerRankVGPRCnt = Info.NumVGPR;
    auto WorkGroupSize = ST.getFlatWorkGroupSizes(F).second;
    // the number of wavegroups is always 4
    if (PerRankScratch * WorkGroupSize > ST.getMaxWaveScratchSize() * 4) {
      llvm::DiagnosticInfoStackSize DiagStackSize(
          F, PerRankScratch * WorkGroupSize / 4, ST.getMaxWaveScratchSize(),
          DS_Error);
      F.getContext().diagnose(DiagStackSize);
    }
    assert(PerRankVGPRCnt * WorkGroupSize <=
           ST.getMaxNumVGPRs(ST.getWavefrontSize()) * 4 *
               ST.getWavefrontSize());
    // first get wave-id-in-wavegroup, temporarily reuse FP
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_GETREG_B32), FPReg)
        .addImm(AMDGPU::Hwreg::HwregEncoding::encode(
            AMDGPU::Hwreg::ID_WAVE_GROUP_INFO, 16, 3));
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_MUL_I32), FPReg)
        .addReg(FPReg)
        .addImm(PerRankVGPRCnt);
    // set IDX0
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_SET_GPR_IDX_U32), AMDGPU::IDX0)
        .addReg(FPReg);
    // again get wave-id-in-wavegroup, temporarily reuse FP
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_GETREG_B32), FPReg)
        .addImm(AMDGPU::Hwreg::HwregEncoding::encode(
            AMDGPU::Hwreg::ID_WAVE_GROUP_INFO, 16, 3));
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_MUL_I32), FPReg)
        .addReg(FPReg)
        .addImm(PerRankScratch);
    // set FP
    BuildMI(Entry, I, DL, TII->get(AMDGPU::S_ADD_I32), FPReg)
        .addReg(FPReg)
        .addImm(RankScratchStart);
    // erase the original FP initializer
    FPInit->eraseFromParent();

    if (SPInit) {
      // set SP
      BuildMI(Entry, I, DL, TII->get(AMDGPU::S_ADD_I32), SPReg)
          .addReg(FPReg)
          .addImm(MF->getFrameInfo().getStackSize());
      // erase the original SP initializer
      SPInit->eraseFromParent();
    }
  }
  return Changed;
}
