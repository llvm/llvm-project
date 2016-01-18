//===-- SIPrepareScratchRegs.cpp - Use predicates for control flow --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This pass loads scratch pointer and scratch offset into a register or a
/// frame index which can be used anywhere in the program.  These values will
/// be used for spilling VGPRs.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace {

class SIPrepareScratchRegs : public MachineFunctionPass {

private:
  static char ID;

public:
  SIPrepareScratchRegs() : MachineFunctionPass(ID) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI prepare scratch registers";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace

char SIPrepareScratchRegs::ID = 0;

FunctionPass *llvm::createSIPrepareScratchRegs() {
  return new SIPrepareScratchRegs();
}

bool SIPrepareScratchRegs::runOnMachineFunction(MachineFunction &MF) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo *FrameInfo = MF.getFrameInfo();
  MachineBasicBlock *Entry = &MF.front();
  MachineBasicBlock::iterator I = Entry->begin();
  DebugLoc DL = I->getDebugLoc();

  // FIXME: If we don't have enough VGPRs for SGPR spilling we will need to
  // run this pass.
  if (!MFI->hasSpilledVGPRs())
    return false;

  unsigned ScratchPtrPreloadReg =
      TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_PTR);
  unsigned ScratchOffsetPreloadReg =
      TRI->getPreloadedValue(MF, SIRegisterInfo::SCRATCH_WAVE_OFFSET);

  if (!Entry->isLiveIn(ScratchPtrPreloadReg))
    Entry->addLiveIn(ScratchPtrPreloadReg);

  if (!Entry->isLiveIn(ScratchOffsetPreloadReg))
    Entry->addLiveIn(ScratchOffsetPreloadReg);

  // Load the scratch offset.
  unsigned ScratchOffsetReg =
      TRI->findUnusedRegister(MRI, &AMDGPU::SGPR_32RegClass);
  int ScratchOffsetFI = -1;

  if (ScratchOffsetReg != AMDGPU::NoRegister) {
    // Found an SGPR to use
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::S_MOV_B32), ScratchOffsetReg)
            .addReg(ScratchOffsetPreloadReg);
  } else {
    // No SGPR is available, we must spill.
    ScratchOffsetFI = FrameInfo->CreateSpillStackObject(4,4);
    BuildMI(*Entry, I, DL, TII->get(AMDGPU::SI_SPILL_S32_SAVE))
            .addReg(ScratchOffsetPreloadReg)
            .addFrameIndex(ScratchOffsetFI)
            .addReg(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, RegState::Undef)
            .addReg(AMDGPU::SGPR0, RegState::Undef);
  }


  // Now that we have the scratch pointer and offset values, we need to
  // add them to all the SI_SPILL_V* instructions.

  RegScavenger RS;
  unsigned ScratchRsrcFI = FrameInfo->CreateSpillStackObject(16, 4);
  RS.addScavengingFrameIndex(ScratchRsrcFI);

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    // Add the scratch offset reg as a live-in so that the register scavenger
    // doesn't re-use it.
    if (!MBB.isLiveIn(ScratchOffsetReg) &&
        ScratchOffsetReg != AMDGPU::NoRegister)
      MBB.addLiveIn(ScratchOffsetReg);
    RS.enterBasicBlock(&MBB);

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      MachineInstr &MI = *I;
      RS.forward(I);
      DebugLoc DL = MI.getDebugLoc();
      if (!TII->isVGPRSpill(MI))
        continue;

      // Scratch resource
      unsigned ScratchRsrcReg =
          RS.scavengeRegister(&AMDGPU::SReg_128RegClass, 0);

      uint64_t Rsrc23 = TII->getScratchRsrcWords23();

      unsigned Rsrc0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
      unsigned Rsrc1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);
      unsigned Rsrc2 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2);
      unsigned Rsrc3 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc0)
              .addExternalSymbol("SCRATCH_RSRC_DWORD0")
              .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc1)
              .addExternalSymbol("SCRATCH_RSRC_DWORD1")
              .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc2)
              .addImm(Rsrc23 & 0xffffffff)
              .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc3)
              .addImm(Rsrc23 >> 32)
              .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      // Scratch Offset
      if (ScratchOffsetReg == AMDGPU::NoRegister) {
        ScratchOffsetReg = RS.scavengeRegister(&AMDGPU::SGPR_32RegClass, 0);
        BuildMI(MBB, I, DL, TII->get(AMDGPU::SI_SPILL_S32_RESTORE),
                ScratchOffsetReg)
                .addFrameIndex(ScratchOffsetFI)
                .addReg(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, RegState::Undef)
                .addReg(AMDGPU::SGPR0, RegState::Undef);
      } else if (!MBB.isLiveIn(ScratchOffsetReg)) {
        MBB.addLiveIn(ScratchOffsetReg);
      }

      if (ScratchRsrcReg == AMDGPU::NoRegister ||
          ScratchOffsetReg == AMDGPU::NoRegister) {
        LLVMContext &Ctx = MF.getFunction()->getContext();
        Ctx.emitError("ran out of SGPRs for spilling VGPRs");
        ScratchRsrcReg = AMDGPU::SGPR0;
        ScratchOffsetReg = AMDGPU::SGPR0;
      }
      MI.getOperand(2).setReg(ScratchRsrcReg);
      MI.getOperand(2).setIsKill(true);
      MI.getOperand(2).setIsUndef(false);
      MI.getOperand(3).setReg(ScratchOffsetReg);
      MI.getOperand(3).setIsUndef(false);
      MI.getOperand(3).setIsKill(false);
      MI.addOperand(MachineOperand::CreateReg(Rsrc0, false, true, true));
      MI.addOperand(MachineOperand::CreateReg(Rsrc1, false, true, true));
      MI.addOperand(MachineOperand::CreateReg(Rsrc2, false, true, true));
      MI.addOperand(MachineOperand::CreateReg(Rsrc3, false, true, true));
    }
  }
  return true;
}
