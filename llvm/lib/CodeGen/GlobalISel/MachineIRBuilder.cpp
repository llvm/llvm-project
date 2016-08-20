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

MachineInstrBuilder MachineIRBuilder::buildInstr(unsigned Opcode,
                                                  ArrayRef<LLT> Tys) {
  MachineInstrBuilder MIB = BuildMI(getMF(), DL, getTII().get(Opcode));
  if (Tys.size() > 0) {
    assert(isPreISelGenericOpcode(Opcode) &&
           "Only generic instruction can have a type");
    for (unsigned i = 0; i < Tys.size(); ++i)
      MIB->setType(Tys[i], i);
  } else
    assert(!isPreISelGenericOpcode(Opcode) &&
           "Generic instruction must have a type");
  getMBB().insert(getInsertPt(), MIB);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildFrameIndex(LLT Ty, unsigned Res,
                                                       int Idx) {
  return buildInstr(TargetOpcode::G_FRAME_INDEX, Ty)
      .addDef(Res)
      .addFrameIndex(Idx);
}

MachineInstrBuilder MachineIRBuilder::buildAdd(LLT Ty, unsigned Res,
                                                unsigned Op0, unsigned Op1) {
  return buildInstr(TargetOpcode::G_ADD, Ty)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildBr(MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BR, LLT::unsized()).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilder::buildCopy(unsigned Res, unsigned Op) {
  return buildInstr(TargetOpcode::COPY).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildConstant(LLT Ty, unsigned Res,
                                                    int64_t Val) {
  return buildInstr(TargetOpcode::G_CONSTANT, Ty).addDef(Res).addImm(Val);
}

MachineInstrBuilder MachineIRBuilder::buildFConstant(LLT Ty, unsigned Res,
                                                    const ConstantFP &Val) {
  return buildInstr(TargetOpcode::G_FCONSTANT, Ty).addDef(Res).addFPImm(&Val);
}

MachineInstrBuilder MachineIRBuilder::buildBrCond(LLT Ty, unsigned Tst,
                                                  MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BRCOND, Ty).addUse(Tst).addMBB(&Dest);
}


 MachineInstrBuilder MachineIRBuilder::buildLoad(LLT VTy, LLT PTy, unsigned Res,
                                                 unsigned Addr,
                                                 MachineMemOperand &MMO) {
  return buildInstr(TargetOpcode::G_LOAD, {VTy, PTy})
      .addDef(Res)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilder::buildStore(LLT VTy, LLT PTy,
                                                  unsigned Val, unsigned Addr,
                                                  MachineMemOperand &MMO) {
  return buildInstr(TargetOpcode::G_STORE, {VTy, PTy})
      .addUse(Val)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilder::buildUAdde(LLT Ty, unsigned Res,
                                                 unsigned CarryOut,
                                                 unsigned Op0, unsigned Op1,
                                                 unsigned CarryIn) {
  return buildInstr(TargetOpcode::G_UADDE, Ty)
      .addDef(Res)
      .addDef(CarryOut)
      .addUse(Op0)
      .addUse(Op1)
      .addUse(CarryIn);
}

MachineInstrBuilder MachineIRBuilder::buildAnyExtend(LLT Ty, unsigned Res,
                                                     unsigned Op) {
  return buildInstr(TargetOpcode::G_ANYEXTEND, Ty).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildExtract(ArrayRef<LLT> ResTys,
                                                   ArrayRef<unsigned> Results,
                                                   ArrayRef<uint64_t> Indices,
                                                   LLT SrcTy, unsigned Src) {
  assert(ResTys.size() == Results.size() && Results.size() == Indices.size() &&
         "inconsistent number of regs");
  assert(!Results.empty() && "invalid trivial extract");

  auto MIB = BuildMI(getMF(), DL, getTII().get(TargetOpcode::G_EXTRACT));
  for (unsigned i = 0; i < ResTys.size(); ++i)
    MIB->setType(LLT::scalar(ResTys[i].getSizeInBits()), i);
  MIB->setType(LLT::scalar(SrcTy.getSizeInBits()), ResTys.size());

  for (auto Res : Results)
    MIB.addDef(Res);

  MIB.addUse(Src);

  for (auto Idx : Indices)
    MIB.addImm(Idx);

  getMBB().insert(getInsertPt(), MIB);

  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildSequence(LLT ResTy, unsigned Res,
                                ArrayRef<LLT> OpTys,
                                ArrayRef<unsigned> Ops,
                                ArrayRef<unsigned> Indices) {
  assert(OpTys.size() == Ops.size() && Ops.size() == Indices.size() &&
         "incompatible args");
  assert(!Ops.empty() && "invalid trivial sequence");

  MachineInstrBuilder MIB =
      buildInstr(TargetOpcode::G_SEQUENCE, LLT::scalar(ResTy.getSizeInBits()));
  MIB.addDef(Res);
  for (unsigned i = 0; i < Ops.size(); ++i) {
    MIB.addUse(Ops[i]);
    MIB.addImm(Indices[i]);
    MIB->setType(LLT::scalar(OpTys[i].getSizeInBits()), MIB->getNumTypes());
  }
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildIntrinsic(ArrayRef<LLT> Tys,
                                                     Intrinsic::ID ID,
                                                     unsigned Res,
                                                     bool HasSideEffects) {
  auto MIB =
      buildInstr(HasSideEffects ? TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS
                                : TargetOpcode::G_INTRINSIC,
                 Tys);
  if (Res)
    MIB.addDef(Res);
  MIB.addIntrinsicID(ID);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildTrunc(LLT Ty, unsigned Res,
                                           unsigned Op) {
  return buildInstr(TargetOpcode::G_TRUNC, Ty).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildFPTrunc(LLT Ty, unsigned Res,
                                           unsigned Op) {
  return buildInstr(TargetOpcode::G_FPTRUNC, Ty).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildICmp(ArrayRef<LLT> Tys,
                                                CmpInst::Predicate Pred,
                                                unsigned Res, unsigned Op0,
                                                unsigned Op1) {
  return buildInstr(TargetOpcode::G_ICMP, Tys)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildFCmp(ArrayRef<LLT> Tys,
                                                CmpInst::Predicate Pred,
                                                unsigned Res, unsigned Op0,
                                                unsigned Op1) {
  return buildInstr(TargetOpcode::G_FCMP, Tys)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildSelect(LLT Ty, unsigned Res,
                                                  unsigned Tst,
                                                  unsigned Op0, unsigned Op1) {
  return buildInstr(TargetOpcode::G_SELECT, {Ty, LLT::scalar(1)})
      .addDef(Res)
      .addUse(Tst)
      .addUse(Op0)
      .addUse(Op1);
}
