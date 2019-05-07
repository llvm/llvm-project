//===-- DPURegisterInfo.cpp - DPU Register Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the DPU implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "DPURegisterInfo.h"
#include "DPUTargetMachine.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define GET_REGINFO_ENUM
#define GET_REGINFO_TARGET_DESC

#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

#define DEBUG_TYPE "dpu-reg"

//===----------------------------------------------------------------------===//
// Generic register mapping:
// - Use a maximum of callees to reduce as much as possible the stack sizes.
// - Assign the following reserved registers:
//   - R23: saves the return address for a function call
//   - R22: the stack pointer
//   - R21: tracks the returned value of a function
//===----------------------------------------------------------------------===//

// NOTE: Order is coherent with the structure of Double Registers (R_ODD
// contains the LSBs, R_EVEN the MSBs). Also, we want the maximum of saved
// Double Registers to be aligned on 8 bytes (to use Store Double instruction),
// and because the register allocator starts from R0 to R16, we need D0 to be
// the closest Double Register from the Stack Pointer (which is defined to be
// aligned on 8 bytes)
static const MCPhysReg CalleeSavedRegs[] = {
    DPU::R16, DPU::D14, DPU::D12, DPU::D10,   DPU::D8, DPU::D6,
    DPU::D4,  DPU::D2,  DPU::D0,  DPU::RDFUN, 0
    /* Reserved and invisible DPU::R17r, DPU::R18r, DPU::R19r, DPU::RVALHI */
};

DPURegisterInfo::DPURegisterInfo() : DPUGenRegisterInfo(DPU::R0) {}

const MCPhysReg *
DPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CalleeSavedRegs;
}

BitVector DPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  // No reserved register in a DPU thread.
  BitVector reserved = BitVector(getNumRegs());
  reserved.set(DPU::RDFUN);
  reserved.set(DPU::STKP);
  reserved.set(DPU::RADD);
  reserved.set(DPU::ZERO);
  reserved.set(DPU::ONE);
  reserved.set(DPU::LNEG);
  reserved.set(DPU::MNEG);
  reserved.set(DPU::ID);
  reserved.set(DPU::ID2);
  reserved.set(DPU::ID4);
  reserved.set(DPU::ID8);
  reserved.set(DPU::R17);
  reserved.set(DPU::R18);
  reserved.set(DPU::R19);
  reserved.set(DPU::D16);
  reserved.set(DPU::D18);
  return reserved;
}

static int getFIOperandIndexFromInstruction(MachineInstr &MI) {
  unsigned int i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "searching for a frame index in an instruction that has no frame "
           "index operand!");
  }
  return (int)i;
}

// Adjusts the offset of every frame in an MI.
// This is mainly used for the following use case:
//  - A function is passed a pointer to an item in the stack (frame index N)
//  - First the stack pointer adjustment is achieved
//  - Then the arguments are passed in the form "addi Rx, STKP, N"
//  - Then call the function
// The problem is that STKP has moved, since the pointer was adjusted... As a
// consequence N is wrong at this step.
// One solution is to re-adjust all those N offsets.
static void adjustFrameIndexesWithNewOffset(MachineFunction &MF, int Offset) {
  for (auto eachFrame = MF.getFrameInfo().getObjectIndexBegin();
       eachFrame < MF.getFrameInfo().getObjectIndexEnd(); eachFrame++) {
    if (!MF.getFrameInfo().isDeadObjectIndex(eachFrame)) {
      int frameOffset = MF.getFrameInfo().getObjectOffset(eachFrame);
      MF.getFrameInfo().setObjectOffset(eachFrame, frameOffset + Offset);
    }
  }
}

unsigned int DPURegisterInfo::stackAdjustmentDependingOnOptimizationLevel(
    const MachineFunction &MF, unsigned int FrameSize) const {
  // In O0, the stack pointer is pushed above the stack before any function
  // call.
  unsigned int DebugAdjustment =
      (MF.getTarget().getOptLevel() == CodeGenOpt::None) ? 4 : 0;
  // If the stack size is not long-aligned, add 4 padding bytes
  unsigned int LongAlignment =
      (((FrameSize + DebugAdjustment) & 7) != 0) ? 4 : 0;
  return DebugAdjustment + LongAlignment;
}

void DPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected eliminate frame index with SPAdj == 0");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  DebugLoc DL = MI.getDebugLoc();

  LLVM_DEBUG({
    dbgs() << "DPU/Reg - eliminating frame index in instruction ";
    MI.dump();
    dbgs() << "\n";
  });

  int FIOperandIndex = getFIOperandIndexFromInstruction(MI);

  unsigned FrameReg = getFrameRegister(MF);
  int FrameIndex = MI.getOperand(FIOperandIndex).getIndex();

  switch (MI.getOpcode()) {
  case DPU::MOVEri: {
    // This is purely empirical: the generator creates move instructions
    // to save the registers.
    const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
    int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex);

    MI.getOperand(FIOperandIndex).ChangeToRegister(FrameReg, false);

    MachineBasicBlock &MBB = *MI.getParent();
    unsigned reg = MI.getOperand(FIOperandIndex - 1).getReg();
    BuildMI(MBB, ++II, DL, TII.get(DPU::ADDrri), reg)
        .addReg(reg)
        .addImm(Offset);
    break;
  }
  case DPU::ADDrri: {
    // This is necessarily a reference to an object in the stack. Please refer
    // to DPUISelDATToDAG::Select, to see how we transform things to this
    // beautiful piece of hack.
    int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex);
    LLVM_DEBUG(dbgs() << "DPU/Frame - adjusting frame object at index "
                      << std::to_string(FrameIndex) << " at offset "
                      << std::to_string(Offset) << "\n");
    MI.getOperand(FIOperandIndex).ChangeToImmediate(Offset);
    break;
  }
  case DPU::PUSH_STACK_POINTER:
  case DPU::STAIN_STACK: {
    unsigned int StackSize = MF.getFrameInfo().getStackSize();
    int Offset =
        StackSize + stackAdjustmentDependingOnOptimizationLevel(MF, StackSize);
    MI.getOperand(FIOperandIndex).ChangeToImmediate(Offset);
    break;
  }
  case DPU::ADJUST_STACK_BEFORE_CALL: {
    unsigned int StackSize = MF.getFrameInfo().getStackSize();
    int Offset =
        StackSize + stackAdjustmentDependingOnOptimizationLevel(MF, StackSize);
    LLVM_DEBUG({
      dbgs() << "DPU/Frame - stack size before call = "
             << std::to_string(Offset) << "\n";
      MF.getFrameInfo().dump(MF);
    });
    adjustFrameIndexesWithNewOffset(MF, -Offset);
    MI.getOperand(FIOperandIndex).ChangeToImmediate(Offset);
    break;
  }
  case DPU::ADJUST_STACK_AFTER_CALL: {
    unsigned int StackSize = MF.getFrameInfo().getStackSize();
    int Offset = -(StackSize +
                   stackAdjustmentDependingOnOptimizationLevel(MF, StackSize));
    LLVM_DEBUG({
      dbgs() << "DPU/Frame - stack size after call = " << std::to_string(Offset)
             << "\n";
      MF.getFrameInfo().dump(MF);
    });
    adjustFrameIndexesWithNewOffset(MF, -Offset);
    MI.getOperand(FIOperandIndex).ChangeToImmediate(Offset);
    break;
  }
  default: {
    int Offset = MF.getFrameInfo().getObjectOffset(FrameIndex) +
                 MI.getOperand(FIOperandIndex + 1).getImm();

    MI.getOperand(FIOperandIndex).ChangeToRegister(FrameReg, false);
    MI.getOperand(FIOperandIndex + 1).ChangeToImmediate(Offset);
  }
  }

  LLVM_DEBUG({
    dbgs() << "DPU/Reg - after frame index instruction is ";
    MI.dump();
    dbgs() << "\n";
  });
}

unsigned DPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return getFrameRegister();
}

unsigned DPURegisterInfo::getFrameRegister() { return R_STKP; }
