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
    DPU::D14, DPU::D12, DPU::D10, DPU::D8, DPU::D6,
    DPU::D4,  DPU::D2,  DPU::D0,  0
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
  reserved.set(DPU::R16);
  reserved.set(DPU::R17);
  reserved.set(DPU::R18);
  reserved.set(DPU::R19);
  reserved.set(DPU::D16);
  reserved.set(DPU::D18);
  return reserved;
}

void DPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected eliminate frame index with SPAdj == 0");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned FrameReg = getFrameRegister(MF);
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  int StackSize = MFI.getStackSize();
  int Offset = MFI.getObjectOffset(FrameIndex);
  if ((FrameIndex < 0) && (Offset >= -(STACK_SIZE_FOR_D22 + StackSize))) {
    // Call Args from DPUTargetLowering::LowerFormalArguments
    // We couldn't add the space for D22 in the lowering without seen the stack
    // reducing wrongly for no obvious reason. It represents the size in the
    // stack reserved manually in DPUFrameLowering::emitPrologue.
    Offset -= (STACK_SIZE_FOR_D22 + StackSize);
    MFI.setObjectOffset(FrameIndex, Offset);
  } else if (Offset >= 0) {
    Offset -= StackSize;
    MFI.setObjectOffset(FrameIndex, Offset);
  }

  LLVM_DEBUG({
      dbgs() << "DPU/Reg - eliminating frame index in instruction (index= "<< FrameIndex << ") ";
    MI.dump();
    dbgs() << "\n";
  });

  switch (MI.getOpcode()) {
  case DPU::ADDrri: {
    // This is necessarily a reference to an object in the stack. Please refer
    // to DPUISelDATToDAG::Select, to see how we generate this machineInstr
    LLVM_DEBUG(dbgs() << "DPU/Frame - adjusting frame object at index "
                      << std::to_string(FrameIndex) << " at offset "
                      << std::to_string(Offset) << "\n");
    MI.getOperand(FIOperandNum).ChangeToImmediate(Offset);
    break;
  }
  default: {
    Offset += MI.getOperand(FIOperandNum + 1).getImm();

    MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false);
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  }
  }

  LLVM_DEBUG({
    dbgs() << "DPU/Reg - after frame index instruction is ";
    MI.dump();
    dbgs() << "\n";
  });
}

unsigned DPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return DPU::STKP;
}
