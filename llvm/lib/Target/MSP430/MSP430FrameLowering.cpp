//===-- MSP430FrameLowering.cpp - MSP430 Frame Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MSP430FrameLowering.h"
#include "MSP430InstrInfo.h"
#include "MSP430MachineFunctionInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

bool MSP430FrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  return (MF.getTarget().Options.DisableFramePointerElim(MF) ||
          MF.getFrameInfo()->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken());
}

bool MSP430FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

void MSP430FrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  const MSP430InstrInfo &TII =
    *static_cast<const MSP430InstrInfo*>(MF.getTarget().getInstrInfo());

  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();

  uint64_t NumBytes = 0;
  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - MSP430FI->getCalleeSavedFrameSize();

    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(-NumBytes);

    // Save FPW into the appropriate stack slot...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::PUSH16r))
      .addReg(MSP430::FPW, RegState::Kill);

    // Update FPW with the new base value...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::MOV16rr), MSP430::FPW)
      .addReg(MSP430::SPW);

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineFunction::iterator I = llvm::next(MF.begin()), E = MF.end();
         I != E; ++I)
      I->addLiveIn(MSP430::FPW);

  } else
    NumBytes = StackSize - MSP430FI->getCalleeSavedFrameSize();

  // Skip the callee-saved push instructions.
  while (MBBI != MBB.end() && (MBBI->getOpcode() == MSP430::PUSH16r))
    ++MBBI;

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  if (NumBytes) { // adjust stack pointer: SPW -= numbytes
    // If there is an SUB16ri of SPW immediately before this instruction, merge
    // the two.
    //NumBytes -= mergeSPUpdates(MBB, MBBI, true);
    // If there is an ADD16ri or SUB16ri of SPW immediately after this
    // instruction, merge the two instructions.
    // mergeSPUpdatesDown(MBB, MBBI, &NumBytes);

    if (NumBytes) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL, TII.get(MSP430::SUB16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(NumBytes);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  }
}

void MSP430FrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  const MSP430InstrInfo &TII =
    *static_cast<const MSP430InstrInfo*>(MF.getTarget().getInstrInfo());

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();

  switch (RetOpcode) {
  case MSP430::RET:
  case MSP430::RETI: break;  // These are ok
  default:
    llvm_unreachable("Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t StackSize = MFI->getStackSize();
  unsigned CSSize = MSP430FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = 0;

  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - CSSize;

    // pop FPW.
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::POP16r), MSP430::FPW);
  } else
    NumBytes = StackSize - CSSize;

  // Skip the callee-saved pop instructions.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();
    if (Opc != MSP430::POP16r && !PI->isTerminator())
      break;
    --MBBI;
  }

  DL = MBBI->getDebugLoc();

  // If there is an ADD16ri or SUB16ri of SPW immediately before this
  // instruction, merge the two instructions.
  //if (NumBytes || MFI->hasVarSizedObjects())
  //  mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  if (MFI->hasVarSizedObjects()) {
    BuildMI(MBB, MBBI, DL,
            TII.get(MSP430::MOV16rr), MSP430::SPW).addReg(MSP430::FPW);
    if (CSSize) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL,
                TII.get(MSP430::SUB16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(CSSize);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  } else {
    // adjust stack pointer back: SPW += numbytes
    if (NumBytes) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL, TII.get(MSP430::ADD16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(NumBytes);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  }
}

// FIXME: Can we eleminate these in favour of generic code?
bool
MSP430FrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  MSP430MachineFunctionInfo *MFI = MF.getInfo<MSP430MachineFunctionInfo>();
  MFI->setCalleeSavedFrameSize(CSI.size() * 2);

  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    BuildMI(MBB, MI, DL, TII.get(MSP430::PUSH16r))
      .addReg(Reg, RegState::Kill);
  }
  return true;
}

bool
MSP430FrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                 MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i)
    BuildMI(MBB, MI, DL, TII.get(MSP430::POP16r), CSI[i].getReg());

  return true;
}

void
MSP430FrameLowering::processFunctionBeforeFrameFinalized(MachineFunction &MF)
                                                                         const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  // Create a frame entry for the FPW register that must be saved.
  if (TFI->hasFP(MF)) {
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(2, -4, true);
    (void)FrameIdx;
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for FPW register must be last in order to be found!");
  }
}
