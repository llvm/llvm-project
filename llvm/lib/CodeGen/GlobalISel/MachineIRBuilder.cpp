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

void MachineIRBuilder::setFunction(MachineFunction &MF) {
  this->MF = &MF;
  this->MBB = nullptr;
  this->TII = MF.getSubtarget().getInstrInfo();
  this->DL = DebugLoc();
  this->MI = nullptr;
}

void MachineIRBuilder::setBasicBlock(MachineBasicBlock &MBB, bool Beginning) {
  this->MBB = &MBB;
  Before = Beginning;
  assert(&getMF() == MBB.getParent() &&
         "Basic block is in a different function");
}

void MachineIRBuilder::setInstr(MachineInstr &MI, bool Before) {
  assert(MI.getParent() && "Instruction is not part of a basic block");
  setBasicBlock(*MI.getParent());
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

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, unsigned Res,
                                           unsigned Op0, unsigned Op1) {
  return buildInstr(Opcode, nullptr, Res, Op0, Op1);
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode, Type *Ty,
                                           unsigned Res, unsigned Op0,
                                           unsigned Op1) {
  MachineInstr *NewMI =
      BuildMI(getMF(), DL, getTII().get(Opcode), Res).addReg(Op0).addReg(Op1);
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
                                           unsigned Op0) {
  assert(!isPreISelGenericOpcode(Opcode) &&
         "Generic instruction must have a type");

  MachineInstr *NewMI =
      BuildMI(getMF(), DL, getTII().get(Opcode), Res).addReg(Op0);
  getMBB().insert(getInsertPt(), NewMI);
  return NewMI;
}

MachineInstr *MachineIRBuilder::buildInstr(unsigned Opcode) {
  assert(!isPreISelGenericOpcode(Opcode) &&
         "Generic instruction must have a type");

  MachineInstr *NewMI = BuildMI(getMF(), DL, getTII().get(Opcode));
  getMBB().insert(getInsertPt(), NewMI);
  return NewMI;
}
