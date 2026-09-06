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
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

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

const MCPhysReg *SuperHRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SH_SaveList;
}

const uint32_t *SuperHRegisterInfo::getCallPreservedMask(const MachineFunction &MF, CallingConv::ID CC) const {
  return CSR_SH_RegMask; 
}

const TargetRegisterClass *
SuperHRegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC,
                                           const MachineFunction &MF) const {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  if (TRI->isTypeLegalForClass(*RC, MVT::i16)) {
    return &SH::GPRRegClass;
  }

  if (TRI->isTypeLegalForClass(*RC, MVT::i8)) {
    return &SH::GPRRegClass;
  }

  if (TRI->isTypeLegalForClass(*RC, MVT::i1)) {
    return &SH::GPRRegClass;
  }

  return TargetRegisterInfo::getLargestLegalSuperClass(RC, MF);
}

BitVector SuperHRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const SuperHFrameLowering *FR = getFrameLowering(MF);

  // R0 is always reserved as some instructions can only write to it.
  Reserved.set(SH::R0);
  
  // R1 is generally used as the temporary storage for addresses.
  Reserved.set(SH::R1);
  
  // Also reserve the stack pointer.
  Reserved.set(SH::R15);

  // Reserve GOT pointer
  if (Subtarget.isPositionIndependent())
    Reserved.set(SH::R12);

  // Reserver frame pointer if it's used.
  if (FR->hasFP(MF))
    Reserved.set(SH::R14);

  return Reserved;
}

static void replaceFI(const MachineFunction &MF, MachineBasicBlock::iterator II,
                      MachineInstr &MI, const DebugLoc &dl,
                      unsigned FIOperandNum, int Offset, Register FramePtr) {

  MI.getOperand(FIOperandNum).ChangeToRegister(FramePtr, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

bool SuperHRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj,
                                           unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineInstr &MI = *II;
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SuperHTargetMachine &TM = (const SuperHTargetMachine &)MF.getTarget();
  const TargetFrameLowering *TFI = TM.getSubtargetImpl(MF.getFunction())->getFrameLowering();
  const TargetInstrInfo &TII = *TM.getSubtargetImpl(MF.getFunction())->getInstrInfo();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  // Get the register offset to fetch.
  Register FrameReg;
  int64_t Offset = TFI->getFrameIndexReference(MF, FrameIndex, FrameReg).getFixed();

  LLVM_DEBUG({
    int64_t Fo = TFI->getFrameIndexReference(MF, FrameIndex, FrameReg).getFixed();
    dbgs()  << "Eliminiate FI " << FrameIndex << " @ SP["
            << -Fo << "]...\n";
  });
  
  replaceFI(MF, II, MI, DL, FIOperandNum, Offset, FrameReg);
  return false;
}

Register SuperHRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SH::R14;
}

Register SuperHRegisterInfo::getFrameRegister() const {
  return SH::R14;
}

Register SuperHRegisterInfo::getStackRegister() const {
  return SH::R15;
}

Register SuperHRegisterInfo::getGOTRegister() const {
  return SH::R12;
}