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

#include "DPUMachineFunctionInfo.h"

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

DPURegisterInfo::DPURegisterInfo() : DPUGenRegisterInfo(DPU::R23) {}

const MCPhysReg *
DPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SaveList;
}

BitVector DPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector reserved = BitVector(getNumRegs());
  reserved.set(DPU::D22);
  reserved.set(DPU::R22);
  reserved.set(DPU::R23);
  reserved.set(DPU::ZERO);
  reserved.set(DPU::ONE);
  reserved.set(DPU::LNEG);
  reserved.set(DPU::MNEG);
  reserved.set(DPU::ID);
  reserved.set(DPU::ID2);
  reserved.set(DPU::ID4);
  reserved.set(DPU::ID8);
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
  DPUMachineFunctionInfo *FuncInfo= MF.getInfo<DPUMachineFunctionInfo>();
  int Offset = FuncInfo->getOffsetFromFrameIndex(FrameIndex);

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
  return DPU::R22;
}

const uint32_t *
DPURegisterInfo::getCallPreservedMask(const MachineFunction & /*MF*/,
                                      CallingConv::ID /*CC*/) const {
  return CSR_RegMask;
}
