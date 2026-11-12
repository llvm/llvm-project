//===-- MipsFrameLowering.cpp - Mips Frame Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MipsFrameLowering.h"
#include "MipsInstrInfo.h"
#include "MipsMachineFunction.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;


//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// The stack is allocated decrementing the stack pointer on
// the first instruction of a function prologue. Once decremented,
// all stack references are done thought a positive offset
// from the stack/frame pointer, so the stack is considering
// to grow up! Otherwise terrible hacks would have to be made
// to get this stack ABI compliant :)
//
//  The stack frame required by the ABI (after call):
//  Offset
//
//  0                 ----------
//  4                 Args to pass
//  .                 saved $GP  (used in PIC)
//  .                 Alloca allocations
//  .                 Local Area
//  .                 CPU "Callee Saved" Registers
//  .                 saved FP
//  .                 saved RA
//  .                 FPU "Callee Saved" Registers
//  StackSize         -----------
//
// Offset - offset from sp after stack allocation on function prologue
//
// The sp is the stack pointer subtracted/added from the stack size
// at the Prologue/Epilogue
//
// References to the previous stack (to obtain arguments) are done
// with offsets that exceeds the stack size: (stacksize+(4*(num_arg-1))
//
// Examples:
// - reference to the actual stack frame
//   for any local area var there is smt like : FI >= 0, StackOffset: 4
//     sw REGX, 4(SP)
//
// - reference to previous stack frame
//   suppose there's a load to the 5th arguments : FI < 0, StackOffset: 16.
//   The emitted instruction will be something like:
//     lw REGX, 16+StackSize(SP)
//
// Since the total stack size is unknown on LowerFormalArguments, all
// stack references (ObjectOffset) created to reference the function
// arguments, are negative numbers. This way, on eliminateFrameIndex it's
// possible to detect those references and the offsets are adjusted to
// their real location.
//
//===----------------------------------------------------------------------===//

const MipsFrameLowering *MipsFrameLowering::create(const MipsSubtarget &ST) {
  if (ST.inMips16Mode())
    return llvm::createMips16FrameLowering(ST);

  return llvm::createMipsSEFrameLowering(ST);
}

// hasFPImpl - Return true if the specified function should have a dedicated
// frame pointer register.  This is true if the function has variable sized
// allocas, if it needs dynamic stack realignment, if frame pointer elimination
// is disabled, or if the frame address is taken.
bool MipsFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MF.disableFramePointerElim() || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken() || TRI->hasStackRealignment(MF);
}

bool MipsFrameLowering::hasBP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MFI.hasVarSizedObjects() && TRI->hasStackRealignment(MF);
}

bool MipsFrameLowering::enableShrinkWrapping(const MachineFunction &MF) const {
  const MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  return !MipsFI->callsEhReturn() && !MipsFI->isISR();
}

// Estimate the size of the stack, including the incoming arguments. We need to
// account for register spills, local objects, reserved call frame and incoming
// arguments. This is required to determine the largest possible positive offset
// from $sp so that it can be determined if an emergency spill slot for stack
// addresses is required.
uint64_t MipsFrameLowering::estimateStackSize(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  int64_t IncomingArgEnd = 0;
  int64_t Offset = 0;

  // Iterate over fixed sized objects: non-negative offsets are incoming stack
  // arguments above the caller's $sp, whereas negative offsets are preallocated
  // in the local area.
  for (int I = MFI.getObjectIndexBegin(); I != 0; ++I) {
    if (MFI.getStackID(I) != TargetStackID::Default)
      continue;
    int64_t ObjOffset = MFI.getObjectOffset(I);
    if (ObjOffset >= 0) {
      IncomingArgEnd =
          std::max<int64_t>(IncomingArgEnd, ObjOffset + MFI.getObjectSize(I));
    } else {
      Offset = std::max<int64_t>(Offset, -ObjOffset);
    }
  }

  Align MaxAlign = MFI.getMaxAlign();

  // PEI places callee-saved register spill slots before local objects.
  if (MFI.isCalleeSavedInfoValid()) {
    for (unsigned I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I) {
      if (!MFI.isCalleeSavedObjectIndex(I) || MFI.isDeadObjectIndex(I) ||
          MFI.getStackID(I) != TargetStackID::Default)
        continue;
      Align Alignment = MFI.getObjectAlign(I);
      Offset = alignTo(Offset + MFI.getObjectSize(I), Alignment);
      MaxAlign = std::max(Alignment, MaxAlign);
    }
  } else {
    // Conservatively assume all callee-saved registers will be saved.
    for (const MCPhysReg *R = TRI.getCalleeSavedRegs(&MF); *R; ++R) {
      unsigned RegSize = TRI.getSpillSize(*TRI.getMinimalPhysRegClass(*R));
      Offset = alignTo(Offset + RegSize, RegSize);
    }
  }

  if (MFI.getUseLocalStackAllocationBlock()) {
    Align Alignment = MFI.getLocalFrameMaxAlign();
    Offset = alignTo(Offset, Alignment) + MFI.getLocalFrameSize();
    MaxAlign = std::max(Alignment, MaxAlign);
  }

  for (unsigned I = 0, E = MFI.getObjectIndexEnd(); I != E; ++I) {
    if (MFI.isCalleeSavedObjectIndex(I) || MFI.isDeadObjectIndex(I) ||
        MFI.getStackID(I) != TargetStackID::Default ||
        (MFI.isObjectPreAllocated(I) && MFI.getUseLocalStackAllocationBlock()))
      continue;
    Align Alignment = MFI.getObjectAlign(I);
    Offset = alignTo(Offset + MFI.getObjectSize(I), Alignment);
    MaxAlign = std::max(Alignment, MaxAlign);
  }

  if (MFI.adjustsStack() && hasReservedCallFrame(MF))
    Offset += MFI.getMaxCallFrameSize();

  Align StackAlign =
      (MFI.adjustsStack() || MFI.hasVarSizedObjects() ||
       (TRI.hasStackRealignment(MF) && MFI.getObjectIndexEnd() != 0))
          ? getStackAlign()
          : getTransientStackAlign();
  StackAlign = std::max(StackAlign, MaxAlign);
  Offset = alignTo(Offset, StackAlign);

  return IncomingArgEnd + Offset;
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions
MachineBasicBlock::iterator MipsFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  unsigned SP = STI.getABI().IsN64() ? Mips::SP_64 : Mips::SP;

  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();
    if (I->getOpcode() == Mips::ADJCALLSTACKDOWN)
      Amount = -Amount;

    STI.getInstrInfo()->adjustStackPtr(SP, Amount, MBB, I);
  }

  return MBB.erase(I);
}
