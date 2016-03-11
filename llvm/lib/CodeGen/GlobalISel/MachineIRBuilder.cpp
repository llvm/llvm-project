//===-- llvm/CodeGen/GlobalISel/MachineIRBuilder.cpp - MIBuilder--*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the MachineIRBuidler class.
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

void MachineIRBuilder::setMF(MachineFunction &MF) {
  this->MF = &MF;
  this->MBB = nullptr;
  this->TII = MF.getSubtarget().getInstrInfo();
  this->DL = DebugLoc();
  this->MI = nullptr;
}

void MachineIRBuilder::setMBB(MachineBasicBlock &MBB, bool Beginning) {
  this->MBB = &MBB;
  Before = Beginning;
  assert(&getMF() == MBB.getParent() &&
         "Basic block is in a different function");
}

void MachineIRBuilder::setInstr(MachineInstr &MI, bool Before) {
  assert(MI.getParent() && "Instruction is not part of a basic block");
  setMBB(*MI.getParent());
  this->MI = &MI;
  this->Before = Before;
}

MachineBasicBlock::iterator MachineIRBuilder::getInsertPt() {
  if (MI) {
    if (Before)
      return MI;
    if (!MI->getNextNode())
      return getMBB().end();
    return MI->getNextNode();
  }
  return Before ? getMBB().begin() : getMBB().end();
}

//------------------------------------------------------------------------------
// Build instruction variants.
//------------------------------------------------------------------------------
MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, Type *Ty) {
  MachineInstr *NewMI = BuildMI(getMF(), DL, getTII().get(Opcode));
  if (Ty) {
    assert(isPreISelGenericOpcode(Opcode) &&
           "Only generic instruction can have a type");
    NewMI->setType(Ty);
  } else
    assert(!isPreISelGenericOpcode(Opcode) &&
           "Generic instruction must have a type");
  getMBB().insert(getInsertPt(), NewMI);
  return NewMI;
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, unsigned Res,
                                           unsigned Op0, unsigned Op1) {
  return buildInstr(Opcode, nullptr, Res, Op0, Op1);
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, Type *Ty,
                                           unsigned Res, unsigned Op0,
                                           unsigned Op1) {
  MachineInstr *NewMI = buildInstr(Opcode, Ty);
  MachineInstrBuilder(getMF(), NewMI)
      .addReg(Res, RegState::Define)
      .addReg(Op0)
      .addReg(Op1);
  return NewMI;
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, unsigned Res,
                                           unsigned Op0) {
  MachineInstr *NewMI = buildInstr(Opcode, nullptr);
  MachineInstrBuilder(getMF(), NewMI).addReg(Res, RegState::Define).addReg(Op0);
  return NewMI;
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode) {
  return buildInstr(Opcode, nullptr);
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, Type *Ty,
                                           MachineBasicBlock &BB) {
  MachineInstr *NewMI = buildInstr(Opcode, Ty);
  MachineInstrBuilder(getMF(), NewMI).addMBB(&BB);
  return NewMI;
}
