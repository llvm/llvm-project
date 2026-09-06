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
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdint>

#define DEBUG_TYPE "sh-framelowering"

static cl::opt<bool>
AccumOutgoingArgs("sh-accumulate-outgoing-args", cl::Hidden, cl::init(false),
          cl::desc("Reserve space for outgoing arguments in the function prologue."));

using namespace llvm;




//===--------------------------------------------------------------------------===//
//                                    Helpers
//===--------------------------------------------------------------------------===//

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

/// adjustFrameOffsetDown - Helper that adjusts frame offset
/// down. 
static void adjustFrameOffsetDown(int64_t &FrameOffset) {
  FrameOffset -= 4;
  if (FrameOffset < 0)
    FrameOffset = 0;
}

// Helper to emit stack pointer adjustment.
void SuperHFrameLowering::emitFrameAdjust(Register Base, MachineFunction &MF, MachineBasicBlock &MBB, 
                                          MachineBasicBlock::iterator MBBI, int32_t AdjValue) const {
  const SuperHInstrInfo &TII = *static_cast<const SuperHInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineInstr::MIFlag MFlag = AdjValue < 0 ? MachineInstr::FrameSetup : MachineInstr::FrameDestroy;
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  
  // No stack frame allocation neccesary.
  if (AdjValue == 0)
    return;

  // Check if the adjustment can fit in an 8-bit immediate.
  if (isInt<8>(AdjValue)) {

    // Fast path, emit a single immediate add.
    BuildMI(MBB, MBBI, DL, TII.get(SH::ADDI), Base)
      .addReg(Base)
      .addImm(AdjValue);
    return;
  }

  // TODO: Embed a constant instead?

  // Slow path, shift 8 bits at a time into r0.
  unsigned ToShift = getShiftAmt(AdjValue);

  // Empty R0 in case it had something.
  BuildMI(MBB, MBBI, DL, TII.get(SH::XOR), SH::R0)
    .addReg(SH::R0)
    .setMIFlag(MFlag);

  // Shift value in with the following pattern:
  //  or #(byte), r0
  //  shll8 r0
  for(unsigned i = 0; i < ToShift; i++) {
    BuildMI(MBB, MBBI, DL, TII.get(SH::ORI))
      .addImm((AdjValue >> (i*8)) & 0xFF)
      .setMIFlag(MFlag);
    BuildMI(MBB, MBBI, DL, TII.get(SH::SHLL8), SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);
  }

  // Finally negate and add to r15.
  //  neg r0, r0 (if negative displacement)
  //  add r0, <base>
  if (AdjValue < 0)
    BuildMI(MBB, MBBI, DL, TII.get(SH::NEG), SH::R0)
      .addReg(SH::R0)
      .addReg(SH::R0)
      .setMIFlag(MFlag);

  BuildMI(MBB, MBBI, DL, TII.get(SH::SUB), Base)
    .addReg(SH::R0, RegState::Kill)
    .addReg(Base)
    .setMIFlag(MFlag);
}




//===--------------------------------------------------------------------------===//
//                              Call-Frame Objects
//===--------------------------------------------------------------------------===//

StackOffset
SuperHFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const SuperHSubtarget &Subtarget = MF.getSubtarget<SuperHSubtarget>();
  const SuperHRegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  bool HasFP = hasFP(MF);

  // NOTE:  All the frame indices are relative to the stack/frame pointer
  //        post-offset. as such an extra adjustment is needed here.
  int64_t FrameOffset = MFI.getObjectOffset(FI);

  // R14 base
  if (HasFP) {

    // Adjust down to remove FP.
    FrameReg = RegInfo->getFrameRegister();
    return StackOffset::getFixed(-FrameOffset);
  }

  // R15 base
  FrameReg = RegInfo->getStackRegister(); // r15
  return StackOffset::getFixed(-FrameOffset);
}




//===--------------------------------------------------------------------------===//
//                          Prologue/Epilogue Emission
//===--------------------------------------------------------------------------===//

void SuperHFrameLowering::emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const {
  LLVM_DEBUG(dbgs() << "Emitting prologue for " << MF.getName() << "...\n");

  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  const MCContext &Ctx = MF.getContext();
  Register SP = RII.getStackRegister();
  Register FP = RII.getFrameRegister();
  Register GOT = RII.getGOTRegister();
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  bool HasFP = hasFP(MF);

  // Get stack frame size.
  int64_t StackSize = MFI.getStackSize();

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
  }

  // Create new stack frame.
  emitFrameAdjust(SP, MF, MBB, MBBI, -StackSize);
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
  LLVM_DEBUG(dbgs() << "Emitting epilogue for " << MF.getName() << "...\n");

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SuperHRegisterInfo &RII = *STI.getRegisterInfo();
  const SuperHSubtarget &STI = MF.getSubtarget<SuperHSubtarget>();
  const SuperHInstrInfo &TII = *STI.getInstrInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  Register SP = RII.getStackRegister();
  Register FP = RII.getFrameRegister();
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  bool HasFP = hasFP(MF);


  // Get stack frame size.
  int64_t StackSize = MFI.getStackSize();

  // Restore stack frame
  if (HasFP) {
    emitFrameAdjust(FP, MF, MBB, MBBI, StackSize);
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOV), SP)
      .addReg(FP)
      .setMIFlag(MachineInstr::FrameSetup);
  } else {
    emitFrameAdjust(SP, MF, MBB, MBBI, StackSize);
  }

  // Restore return address from stack (if needed.)
  if (MFI.hasCalls()) { 
    BuildMI(MBB, MBBI, DL, TII.get(SH::LDSMPR))
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameDestroy);
  }

  // Restore stack pointer
  if (HasFP) {
    BuildMI(MBB, MBBI, DL, TII.get(SH::MOVLP), FP)
      .addReg(SP)
      .setMIFlag(MachineInstr::FrameDestroy);
  }

  // TODO: Handle GOT
}




//===--------------------------------------------------------------------------===//
//                                Callee-Saves
//===--------------------------------------------------------------------------===//

void SuperHFrameLowering::determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                        RegScavenger *RS) const {
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

  for (const CalleeSavedInfo &I : reverse(CSI)) {
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




//===--------------------------------------------------------------------------===//
//                               Call-Frame Meta
//===--------------------------------------------------------------------------===//

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