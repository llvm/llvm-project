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
#include "llvm/Support/TypeSize.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdint>

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
  Register FP = RII.getFrameRegister();

  if (abs(AdjValue) < 255) {

    if (AdjValue < 0) {

      // Fast path, emit a single immediate add.
      //    Emit add #-(size),r15
      BuildMI(MBB, MBBI, dl, TII.get(SH::ADDI), SP)
        .addReg(SP)
        .addImm((int)AdjValue);
      return true;
    }

    if (AdjValue > 0) {

      // Fast path, emit a single immediate add.
      //    Emit add #(size),r14
      BuildMI(MBB, MBBI, dl, TII.get(SH::ADDI), FP)
        .addReg(FP)
        .addImm((int)AdjValue);
      return true;
    }

  }

  // Slow path, shift 8 bits at a time into r0.
  unsigned ToShift = getShiftAmt(AdjValue);

  // Empty R0 in case it had something.
  BuildMI(MBB, MBBI, dl, TII.get(SH::MOVI), SH::R0)
    .addImm(0)
    .addReg(SH::R0)
    .setMIFlag(MFlag);

  // Shift value in with the following pattern:
  //  or #(byte), r0
  //  shll8 r0
  for(unsigned i = 0; i < ToShift; i++) {
    BuildMI(MBB, MBBI, dl, TII.get(SH::ORI))
      .addImm((AdjValue >> (i*8)) & 0xFF)
      .setMIFlag(MFlag);
    BuildMI(MBB, MBBI, dl, TII.get(SH::SHLL8), SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);
  }

  // Finally negate and add to r15.
  //  neg r0, r0 (if negative displacement)
  //  add r0, r15
  if (AdjValue < 0)
    BuildMI(MBB, MBBI, dl, TII.get(SH::NEG), SH::R0)
      .addReg(SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);

  BuildMI(MBB, MBBI, dl, TII.get(SH::SUB), SP)
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

  // Reference register.
  Register Ref = HasFP ? SP : FP;

  LLVM_DEBUG(dbgs() << "Emitting prologue...\n");

  // Realign stack to 32-bit offsets.
  uint32_t StackSize = alignSPAdjust(MFI.getStackSize());
  MFI.setStackSize(StackSize);

  // Store previous frame pointer.
  if (HasFP) {
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLM), FP)
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  // Store Return address on stack (if needed)
  if (MFI.hasCalls()) {
    BuildMI(MBB, MBBI, DL, TII.get(SH::STSMPR))
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII.get(SH::ADDI), SP)
      .addReg(SP)
      .addImm(-4);
  }

  // Create new frame pointer.
  if (HasFP) {
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOV), FP)
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  // // Store GOT
  // if (STI.isPositionIndependent()) {
  //   if (auto *GOTSym = MF.getPICBaseSymbol()) {
  //     BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLM), SP)
  //       .addReg(GOT)
  //       .setMIFlag(MachineInstr::FrameSetup);
  //   }
  // }
}

void SuperHFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  Register SP = RII.getStackRegister();
  Register FP = RII.getFrameRegister();

  LLVM_DEBUG(dbgs() << "Emitting epilogue...\n");

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
      BuildMI(MBB, MBBI, DL, TII.get(SH::LDSMPR))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }

    // 2. Delete stack frame, restoring stack pointer.
    if (StackSize > 0) {
      emitSPAdj(MF, MBB, MBBI, StackSize);
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOV), FP)
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLM), SP)
        .addReg(FP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }

  } else {

    // 1. Restore return address from stack.
    if (MFI.hasCalls()) {
      BuildMI(MBB, MBBI, DL, TII.get(SH::LDSMPR))
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }

    // 2. Delete stack frame, restoring stack pointer.
    //    add <stackadj>,r14
    //    mov r14,r15
    //    mov.l @r15+,r14
    if (StackSize > 0) {
      emitSPAdj(MF, MBB, MBBI, StackSize);
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOV), FP)
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameDestroy);
      BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLM), SP)
        .addReg(FP)
        .setMIFlag(MachineInstr::FrameDestroy);
    }
  }
}

void SuperHFrameLowering::determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                        RegScavenger *RS) const {
  LLVM_DEBUG(dbgs() << "determineCalleeSaves\n");
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
}

bool SuperHFrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  
  LLVM_DEBUG(dbgs() << "Spilling " << CSI.size() << " registers...\n");
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
    BuildMI(MBB, MI, DL, TII.get(SH::MOVLM), SP)
      .addReg(Reg)
      .setMIFlag(MachineInstr::FrameSetup);
  }
  return true;
}

bool SuperHFrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                   MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  
  LLVM_DEBUG(dbgs() << "Restoring " << CSI.size() << " registers...\n");
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
    BuildMI(MBB, MI, DL, TII.get(SH::MOVLM), SP)
      .addReg(Reg)
      .setMIFlag(MachineInstr::FrameDestroy);
  }

  return true;
}

StackOffset
SuperHFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const SuperHSubtarget &Subtarget = MF.getSubtarget<SuperHSubtarget>();
  const SuperHRegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  bool HasFP = hasFP(MF);

  // R14 base
  int64_t FrameOffset = MF.getFrameInfo().getObjectOffset(FI);
  if (HasFP) {
    FrameReg = RegInfo->getFrameRegister();
    return StackOffset::getFixed(FrameOffset);
  }

  // R15 base
  FrameReg = RegInfo->getStackRegister(); // %sp
  return StackOffset::getFixed(FrameOffset + MF.getFrameInfo().getStackSize());
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
  return hasFP(MF) && !MFI.hasVarSizedObjects();
}