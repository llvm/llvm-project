//===- SuperHFrameLowering.cpp - SuperH Frame Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperHTargetFrameLowering class.
//
//===----------------------------------------------------------------------===//


#include "SuperHFrameLowering.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperHInstrInfo.h"
#include "SuperHRegisterInfo.h"
#include "SuperHSubtarget.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "sh-framelowering"

static cl::opt<bool>
AccumOutgoingArgs("sh-accumulate-outgoing-args", cl::Hidden, cl::init(false),
          cl::desc("Reserve space for outgoing arguments in the function prologue."));

using namespace llvm;

// Get amount of times to shift the value in a SP adjustment
// for it to fit.
static unsigned getShiftAmt(uint32_t Val) {
  unsigned R = 0;
  for(unsigned i = 0; i < 4; i++) {
    if (((Val >> (i*8)) & 0xFF))
      R = i;
  }
  return R;
}

// Helper to emit stack pointer adjustment.
static bool emitSPAdj(MachineFunction &MF, MachineBasicBlock &MBB,  MachineBasicBlock::iterator MBBI, int32_t AdjValue) {
  DebugLoc dl;
  const SuperHInstrInfo &TII = *static_cast<const SuperHInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SuperHRegisterInfo &RII = *static_cast<const SuperHRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  MachineInstr::MIFlag MFlag = AdjValue < 0 ? MachineInstr::FrameSetup : MachineInstr::FrameDestroy;
  
  // No stack frame allocation neccesary.
  if (AdjValue == 0)
    return false;

  Register SP = RII.getStackRegister();

  if (AdjValue < 255) {

    // Fast path, emit a single immediate add.
    //    Emit add #-(size),r15
    BuildMI(MBB, MBBI, dl, TII.get(SH::ADDI8Rn), SP)
      .addReg(SP)
      .addImm((int)AdjValue);

    return true;
  }

  // Slow path, shift 8 bits at a time into r0.
  unsigned ToShift = getShiftAmt(AdjValue);

  // Empty R0 in case it had something.
  BuildMI(MBB, MBBI, dl, TII.get(SH::MOVI8Rn), SH::R0)
    .addImm(0)
    .addReg(SH::R0)
    .setMIFlag(MFlag);

  // Shift value in with the following pattern:
  //  or #(byte), r0
  //  shll8 r0
  for(unsigned i = 0; i < ToShift; i++) {
    BuildMI(MBB, MBBI, dl, TII.get(SH::ORI8R0))
      .addImm((AdjValue >> (i*8)) & 0xFF)
      .setMIFlag(MFlag);
    BuildMI(MBB, MBBI, dl, TII.get(SH::SHLL8Rn), SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);
  }

  // Finally negate and add to r15.
  //  neg r0, r0 (if negative displacement)
  //  add r0, r15
  if (AdjValue < 0)
    BuildMI(MBB, MBBI, dl, TII.get(SH::NEGRmRn), SH::R0)
      .addReg(SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);

  BuildMI(MBB, MBBI, dl, TII.get(SH::SUBRmRn), SP)
    .addReg(SH::R0, RegState::Kill)
    .addReg(SP)
    .setMIFlag(MFlag);
  return true;
}

void SuperHFrameLowering::emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  const MCContext &Ctx = MF.getContext();
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  bool HasFP = hasFP(MF);

  Register SP = RII.getStackRegister();
  Register FP = RII.getFrameRegister();
  Register GOT = RII.getGOTRegister();

  // Early exit if we have no frame pointer.
  if (!HasFP) {
    return;
  }

  // Realign stack
  uint32_t StackSize = alignSPAdjust(MFI.getStackSize());
  MFI.setStackSize(StackSize);
  
  if (STI.isPositionIndependent()) {
    // Position Independent Prologue

    // 1. Store GOT to stack and load local GOT.
    if (auto *GOTSym = MF.getPICBaseSymbol()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLRmRndeci), SP)
        .addReg(GOT)
        .setMIFlag(MachineInstr::FrameSetup);
    }

    // 2. Establish Stack Frame
    //    mov.l r14,@-r15
    //    add <stackadj>,r15
    //    mov r15,r14
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLRmRndeci), FP)
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameSetup);
    emitSPAdj(MF, MBB, MBBI, -(int32_t)StackSize);
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOVRmRn), SP)
      .addReg(FP)
      .setMIFlag(MachineInstr::FrameSetup);

    // 4. Save return address to stack.
    if (MFI.hasCalls()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::STSLPRRndeci))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameSetup);
    }

  } else {
    // Position Dependent Prologue

    // 1. Establish Frame Pointer
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLRmRndeci), SP)
      .addReg(FP)
      .setMIFlag(MachineInstr::FrameSetup);

    // 2. Create new stack frame
    emitSPAdj(MF, MBB, MBBI, -(int32_t)StackSize);

    // 3. Save return address to stack.
    if (MFI.hasCalls()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::STSLPRRndeci))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameSetup);
    }
  }
}

void SuperHFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  Register SP = RII.getStackRegister();
  Register FP = RII.getFrameRegister();

  // Early exit if we have no frame pointer.
  if (!hasFP(MF)) {
    return;
  }

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  DebugLoc DL = MBBI->getDebugLoc();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();

  uint32_t StackSize = MFI.getStackSize();

  if (STI.isPositionIndependent()) {

    // 1. Restore return address from stack.
    if (MFI.hasCalls()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::LDSLRminciPR))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }

    // 2. Delete stack frame, restoring stack pointer.
    if (StackSize > 0)
      emitSPAdj(MF, MBB, MBBI, StackSize);

  } else {

    // 1. Restore return address from stack.
    if (MFI.hasCalls()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::LDSLRminciPR))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }

    // 2. Delete stack frame, restoring stack pointer.
    if (StackSize > 0)
      emitSPAdj(MF, MBB, MBBI, StackSize);
  }
}
bool SuperHFrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  
  if (CSI.empty()) {
    return false;
  }

  DebugLoc DL = MBB.findDebugLoc(MI);
  MachineFunction &MF = *MBB.getParent();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  Register SP = RII.getStackRegister();

  for (const CalleeSavedInfo &I : llvm::reverse(CSI)) {
    MCRegister Reg = I.getReg();
    BuildMI(MBB, MI, DL, TII.get(SH::MOVLRmRndeci), SP)
      .addReg(Reg)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  return true;
}

bool SuperHFrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                   MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  
  if (CSI.empty()) {
      return false;
  }

  DebugLoc DL = MBB.findDebugLoc(MI);
  MachineFunction &MF = *MBB.getParent();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  Register SP = RII.getStackRegister();

  for (const CalleeSavedInfo &I : CSI) {
    MCRegister Reg = I.getReg();
    BuildMI(MBB, MI, DL, TII.get(SH::MOVLRminciRn), SP)
      .addReg(Reg)
      .setMIFlag(MachineInstr::FrameDestroy);
  }

  return true;
}

MachineBasicBlock::iterator
SuperHFrameLowering::eliminateCallFramePseudoInstr(MachineFunction &MF, 
                            MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI) const {
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();

  // If call frame is reserved, erase.
  if (hasReservedCallFrame(MF)) {
    return MBB.erase(MI);
  }

  // If frame size is 0, erase.
  int Amount = TII.getFrameSize(*MI);
  if (Amount == 0) {
    return MBB.erase(MI);
  }

  DebugLoc DL = MI->getDebugLoc();
  unsigned int Opcode = MI->getOpcode();
  if (Opcode == TII.getCallFrameSetupOpcode()) {

  } else {
    assert(Opcode == TII.getCallFrameDestroyOpcode());
  }

  return MBB.erase(MI);
}

bool SuperHFrameLowering::canSimplifyCallFramePseudos(
    const MachineFunction &MF) const {
  // Always simplify call frame pseudo instructions, even when
  // hasReservedCallFrame is false.
  return true;
}

bool SuperHFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken();
}

bool SuperHFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return !MFI.hasVarSizedObjects();
}

void SuperHFrameLowering::determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                        RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
}