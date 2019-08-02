//===-- DPUFrameLowering.cpp - DPU Frame Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the DPU implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "DPUFrameLowering.h"
#include "DPUInstrInfo.h"
#include "DPUSubtarget.h"
#include "DPUTargetMachine.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define GET_REGINFO_ENUM

#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

#define DEBUG_TYPE "dpu-lower"

bool DPUFrameLowering::hasFP(const MachineFunction &MF) const { return false; }

void DPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const DPUInstrInfo &DPUII =
      *static_cast<const DPUInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const MCRegisterInfo *MRI = MF.getMMI().getContext().getRegisterInfo();
  const DPURegisterInfo &RegInfo =
      *static_cast<const DPURegisterInfo *>(STI.getRegisterInfo());
  const DPUInstrInfo &TII =
      *static_cast<const DPUInstrInfo *>(STI.getInstrInfo());
  DebugLoc DL;
  unsigned CFIIndex;

  // We reserve manually 8 bytes to store d22 (r22r23) at the end of the stack
  // for debug purpose. Not at the beginning because we do not have a frame
  // pointer (pointing at the beginning of the stack) but only a stack pointer
  // (pointing at the end of the stack)
  unsigned int StackSize =
      alignTo(MFI.getStackSize() + STACK_SIZE_FOR_D22, getStackAlignment());
  MFI.setStackSize(StackSize);

  CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createDefCfaOffset(nullptr, StackSize));
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);
  CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
      nullptr, MRI->getDwarfRegNum(DPU::RADD, true), -STACK_SIZE_FOR_D22));
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(MachineInstr::FrameSetup);
  CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
      nullptr, MRI->getDwarfRegNum(DPU::STKP, true), 4 - STACK_SIZE_FOR_D22));
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(MachineInstr::FrameSetup);

  BuildMI(MBB, MBBI, DL, DPUII.get(DPU::SDrir), DPU::STKP)
      .addImm(StackSize - STACK_SIZE_FOR_D22)
      .addReg(DPU::RDFUN);
  BuildMI(MBB, MBBI, DL, DPUII.get(DPU::ADDrri), DPU::STKP)
      .addReg(DPU::STKP)
      .addImm(StackSize);

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (!CSI.empty()) {
    for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
           E = CSI.end(); I != E; ++I) {
      int64_t Offset = MFI.getObjectOffset(I->getFrameIdx());
      unsigned Reg = I->getReg();
      unsigned Reg0 =
          MRI->getDwarfRegNum(RegInfo.getSubReg(Reg, DPU::sub_32bit), true);
      unsigned Reg1 =
          MRI->getDwarfRegNum(RegInfo.getSubReg(Reg, DPU::sub_32bit_hi), true);
      CFIIndex = MF.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, Reg0, Offset - StackSize));
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlag(MachineInstr::FrameSetup);
      CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
          nullptr, Reg1, Offset + 4 - StackSize));
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlag(MachineInstr::FrameSetup);
      ++MBBI;
    }
  }
}

void DPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const DPUInstrInfo &DPUII =
      *static_cast<const DPUInstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc DL = MBBI->getDebugLoc();

  BuildMI(MBB, MBBI, DL, DPUII.get(DPU::LDrri), DPU::RDFUN)
      .addReg(DPU::STKP)
      .addImm(-STACK_SIZE_FOR_D22);
}

MachineBasicBlock::iterator DPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  return MBB.erase(MI);
}

int DPUFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                             unsigned &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Because we use no register but permanently adjust the stack pointer, the
  // frame index reference is directly the frame index.
  FrameReg = DPU::STKP;
  return MFI.getObjectOffset(FI);
}
