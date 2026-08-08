//===-- SuperHRegisterInfo.h - SuperH Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SuperHRegisterInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperHTargetMachine.h"
#include "SuperHFrameLowering.h"
#include "SuperHSubtarget.h"
#include "SuperH.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sh-reginfo"

#define GET_REGINFO_TARGET_DESC
#include "SuperHGenRegisterInfo.inc"



SuperHRegisterInfo::SuperHRegisterInfo(const SuperHSubtarget &ST)
  : SuperHGenRegisterInfo(SH::R0, /*DwarfFlavour*/0, /*EHFlavor*/0,
                         /*PC*/SH::PC), Subtarget(ST) {}

const TargetRegisterClass *SuperHRegisterInfo::getPointerRegClass(unsigned Kind) const {
  return &SH::GPRRegClass;
}

const TargetRegisterClass *SuperHRegisterInfo::intRegClass(unsigned Size) const {
  return &SH::GPRRegClass;
}

const MCPhysReg *SuperHRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SH_SaveList;
}

const uint32_t *SuperHRegisterInfo::getCallPreservedMask(const MachineFunction &MF, CallingConv::ID CC) const {
  return CSR_SH_RegMask; 
}

BitVector SuperHRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // R0 and R1 are always reserved as return slots.
  Reserved.set(SH::R0);
  Reserved.set(SH::R1);

  // Also reserve the stack frame.
  Reserved.set(SH::R14);
  return Reserved;
}

bool SuperHRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj,
                                           unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineInstr &MI = *II;
  DebugLoc dl = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SuperHTargetMachine &TM = (const SuperHTargetMachine &)MF.getTarget();
  const TargetInstrInfo &TII = *TM.getSubtargetImpl(MF.getFunction())->getInstrInfo();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  // NOTE: Stack grows down, so flip this.
  int Offset = -MFI.getObjectOffset(FrameIndex);

  if (MI.getOpcode() == SH::SHFrmIdx) {

    // TODO: Lower frames that can't be expressed in 4 bits.

    Register DstReg = MI.getOperand(0).getReg();
    MachineInstr *New = BuildMI(MBB, MI, dl, TII.get(SH::MOVLD4RmiRn), DstReg)
                        .addReg(SH::R14)
                        .addImm(Offset / 4);

    MI.eraseFromParent();
    return false;
  }
  return false;
}

Register SuperHRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SH::R14;
}