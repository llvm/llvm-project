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

  // R0 is always reserved as some instructions can only write to it.
  Reserved.set(SH::R0);

  // Reserve GOT pointer
  Reserved.set(SH::R12);
  
  // Also reserve the stack frame and stack pointer.
  Reserved.set(SH::R14);
  Reserved.set(SH::R15);

  // Status Register
  Reserved.set(SH::SR);
  return Reserved;
}

/// getLoadBaseOffset - Gets the base offset to emit a load/store from.
static int64_t getLoadStoreBaseOffset(unsigned Opcode) {
  switch(Opcode) {
  case SH::MOVBL4:
  case SH::MOVBS4:
    return 12;

  case SH::MOVWL4:
  case SH::MOVWS4:
    return 28;

  case SH::MOVLS4:
  case SH::MOVLL4:
    return 60;

  default:
    return 0;
  }
}

/// getLoadBaseOffset - Gets the neccesary bit shift amount for the
/// load/store indexing
static int64_t getLoadStoreOffsetShift(unsigned Opcode) {
  switch(Opcode) {
  case SH::MOVBL4:
  case SH::MOVBS4:
    return 0;

  case SH::MOVWL4:
  case SH::MOVWS4:
    return 1;

  case SH::MOVLS4:
  case SH::MOVLL4:
    return 2;

  default:
    return 0;
  }
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
  int64_t ROff = getLoadStoreBaseOffset(MI.getOpcode());
  int64_t Shift = getLoadStoreOffsetShift(MI.getOpcode());

  // Handle frame loads and stores.
  switch(MI.getOpcode()) {

  // Stack Store
  case SH::MOVBS4:
  case SH::MOVWS4: {
    Register SrcReg = MI.getOperand(0).getReg();
    
    // Expand sequence to
    // mov      <base reg>,r1
    // add      #-ROff,r1
    // mov      <src reg>, r0
    // mov.w/b  r0,@(ROff+Offset,r1)
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::MOV), SH::R1)
      .addReg(FrameReg);
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::ADDI), SH::R1)
      .addReg(SH::R1)
      .addImm(-ROff);
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::MOV), SH::R0)
      .addReg(SrcReg);

    Offset = (ROff-Offset) >> Shift;
    FrameReg = SH::R1;
    break;
  }
  case SH::MOVLS4: {

    // Expand sequence to
    // mov      <base reg>,r1
    // add      #-ROff,r1
    // mov.l    <src reg>,@(ROff+Offset,r1)
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::MOV), SH::R1)
      .addReg(FrameReg);
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::ADDI), SH::R1)
      .addReg(SH::R1)
      .addImm(-ROff);

    Offset = (ROff-Offset) >> Shift;
    FrameReg = SH::R1;
    break;
  }

  // Stack Load
  case SH::MOVBL4:
  case SH::MOVWL4: {
    Register DstReg = MI.getOperand(0).getReg();

    // Expand sequence to
    // mov      <base reg>,r1
    // add      #-ROff,r1
    // mov.w/b  @(ROff+Offset,r1),r0
    // mov      r0,<dst reg>
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::MOV), SH::R1)
      .addReg(FrameReg);
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::ADDI), SH::R1)
      .addReg(SH::R1)
      .addImm(-ROff);

    // This inserts it after our instruction
    BuildMI(*MI.getParent(), MI, DL, TII.get(SH::ADDI), DstReg)
      .addReg(SH::R0)
      .addImm(-ROff);

    Offset = (ROff-Offset) >> Shift;
    FrameReg = SH::R1;
    break;
  }
  case SH::MOVLL4: {

    // Expand sequence to
    // mov    <base reg>,r1
    // add    #-ROff,r1
    // mov.l  @(ROff+Offset,r1),<dst reg>
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::MOV), SH::R1)
      .addReg(FrameReg);
    BuildMI(*MI.getParent(), II, DL, TII.get(SH::ADDI), SH::R1)
      .addReg(SH::R1)
      .addImm(-ROff);

    Offset = (ROff-Offset) >> Shift;
    FrameReg = SH::R1;
    break;
  }
  }

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