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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOpcodes.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

void MachineIRBuilder::setMF(MachineFunction &MF) {
  this->MF = &MF;
  this->MBB = nullptr;
  this->MRI = &MF.getRegInfo();
  this->TII = MF.getSubtarget().getInstrInfo();
  this->DL = DebugLoc();
  this->MI = nullptr;
  this->InsertedInstr = nullptr;
}

void MachineIRBuilder::setMBB(MachineBasicBlock &MBB, bool Beginning) {
  this->MBB = &MBB;
  this->MI = nullptr;
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

void MachineIRBuilder::recordInsertions(
    std::function<void(MachineInstr *)> Inserted) {
  InsertedInstr = Inserted;
}

void MachineIRBuilder::stopRecordingInsertions() {
  InsertedInstr = nullptr;
}

//------------------------------------------------------------------------------
// Build instruction variants.
//------------------------------------------------------------------------------

MachineInstrBuilder MachineIRBuilder::buildInstr(unsigned Opcode) {
  return insertInstr(buildInstrNoInsert(Opcode));
}

MachineInstrBuilder MachineIRBuilder::buildInstrNoInsert(unsigned Opcode) {
  MachineInstrBuilder MIB = BuildMI(getMF(), DL, getTII().get(Opcode));
  return MIB;
}


MachineInstrBuilder MachineIRBuilder::insertInstr(MachineInstrBuilder MIB) {
  getMBB().insert(getInsertPt(), MIB);
  if (InsertedInstr)
    InsertedInstr(MIB);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildFrameIndex(unsigned Res, int Idx) {
  assert(MRI->getType(Res).isPointer() && "invalid operand type");
  return buildInstr(TargetOpcode::G_FRAME_INDEX)
      .addDef(Res)
      .addFrameIndex(Idx);
}

MachineInstrBuilder MachineIRBuilder::buildGlobalValue(unsigned Res,
                                                       const GlobalValue *GV) {
  assert(MRI->getType(Res).isPointer() && "invalid operand type");
  assert(MRI->getType(Res).getAddressSpace() ==
             GV->getType()->getAddressSpace() &&
         "address space mismatch");

  return buildInstr(TargetOpcode::G_GLOBAL_VALUE)
      .addDef(Res)
      .addGlobalAddress(GV);
}

MachineInstrBuilder MachineIRBuilder::buildAdd(unsigned Res, unsigned Op0,
                                               unsigned Op1) {
  assert((MRI->getType(Res).isScalar() || MRI->getType(Res).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");

  return buildInstr(TargetOpcode::G_ADD)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildGEP(unsigned Res, unsigned Op0,
                                               unsigned Op1) {
  assert(MRI->getType(Res).isPointer() &&
         MRI->getType(Res) == MRI->getType(Op0) && "type mismatch");
  assert(MRI->getType(Op1).isScalar()  && "invalid offset type");

  return buildInstr(TargetOpcode::G_GEP)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildSub(unsigned Res, unsigned Op0,
                                               unsigned Op1) {
  assert((MRI->getType(Res).isScalar() || MRI->getType(Res).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");

  return buildInstr(TargetOpcode::G_SUB)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildMul(unsigned Res, unsigned Op0,
                                               unsigned Op1) {
  assert((MRI->getType(Res).isScalar() || MRI->getType(Res).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");

  return buildInstr(TargetOpcode::G_MUL)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildBr(MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BR).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilder::buildCopy(unsigned Res, unsigned Op) {
  return buildInstr(TargetOpcode::COPY).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildConstant(unsigned Res, int64_t Val) {
  assert(MRI->getType(Res).isScalar() && "invalid operand type");

  return buildInstr(TargetOpcode::G_CONSTANT).addDef(Res).addImm(Val);
}

MachineInstrBuilder MachineIRBuilder::buildFConstant(unsigned Res,
                                                     const ConstantFP &Val) {
  assert(MRI->getType(Res).isScalar() && "invalid operand type");

  return buildInstr(TargetOpcode::G_FCONSTANT).addDef(Res).addFPImm(&Val);
}

MachineInstrBuilder MachineIRBuilder::buildBrCond(unsigned Tst,
                                                  MachineBasicBlock &Dest) {
  assert(MRI->getType(Tst).isScalar() && "invalid operand type");

  return buildInstr(TargetOpcode::G_BRCOND).addUse(Tst).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilder::buildLoad(unsigned Res, unsigned Addr,
                                                MachineMemOperand &MMO) {
  assert(MRI->getType(Res).isValid() && "invalid operand type");
  assert(MRI->getType(Addr).isPointer() && "invalid operand type");

  return buildInstr(TargetOpcode::G_LOAD)
      .addDef(Res)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilder::buildStore(unsigned Val, unsigned Addr,
                                                 MachineMemOperand &MMO) {
  assert(MRI->getType(Val).isValid() && "invalid operand type");
  assert(MRI->getType(Addr).isPointer() && "invalid operand type");

  return buildInstr(TargetOpcode::G_STORE)
      .addUse(Val)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilder::buildUAdde(unsigned Res,
                                                 unsigned CarryOut,
                                                 unsigned Op0, unsigned Op1,
                                                 unsigned CarryIn) {
  assert(MRI->getType(Res).isScalar() && "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");
  assert(MRI->getType(CarryOut).isScalar() && "invalid operand type");
  assert(MRI->getType(CarryOut) == MRI->getType(CarryIn) && "type mismatch");

  return buildInstr(TargetOpcode::G_UADDE)
      .addDef(Res)
      .addDef(CarryOut)
      .addUse(Op0)
      .addUse(Op1)
      .addUse(CarryIn);
}

MachineInstrBuilder MachineIRBuilder::buildAnyExt(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_ANYEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildSExt(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_SEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildZExt(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_ZEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildSExtOrTrunc(unsigned Res,
                                                       unsigned Op) {
  unsigned Opcode = TargetOpcode::COPY;
  if (MRI->getType(Res).getSizeInBits() > MRI->getType(Op).getSizeInBits())
    Opcode = TargetOpcode::G_SEXT;
  else if (MRI->getType(Res).getSizeInBits() < MRI->getType(Op).getSizeInBits())
    Opcode = TargetOpcode::G_TRUNC;

  return buildInstr(Opcode).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildExtract(ArrayRef<unsigned> Results,
                                                   ArrayRef<uint64_t> Indices,
                                                   unsigned Src) {
#ifndef NDEBUG
  assert(Results.size() == Indices.size() && "inconsistent number of regs");
  assert(!Results.empty() && "invalid trivial extract");
  assert(std::is_sorted(Indices.begin(), Indices.end()) &&
         "extract offsets must be in ascending order");

  assert(MRI->getType(Src).isValid() && "invalid operand type");
  for (auto Res : Results)
    assert(MRI->getType(Res).isValid() && "invalid operand type");
#endif

  auto MIB = BuildMI(getMF(), DL, getTII().get(TargetOpcode::G_EXTRACT));
  for (auto Res : Results)
    MIB.addDef(Res);

  MIB.addUse(Src);

  for (auto Idx : Indices)
    MIB.addImm(Idx);

  getMBB().insert(getInsertPt(), MIB);
  if (InsertedInstr)
    InsertedInstr(MIB);

  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildSequence(unsigned Res,
                                ArrayRef<unsigned> Ops,
                                ArrayRef<uint64_t> Indices) {
#ifndef NDEBUG
  assert(Ops.size() == Indices.size() && "incompatible args");
  assert(!Ops.empty() && "invalid trivial sequence");
  assert(std::is_sorted(Indices.begin(), Indices.end()) &&
         "sequence offsets must be in ascending order");

  assert(MRI->getType(Res).isValid() && "invalid operand type");
  for (auto Op : Ops)
    assert(MRI->getType(Op).isValid() && "invalid operand type");
#endif

  MachineInstrBuilder MIB = buildInstr(TargetOpcode::G_SEQUENCE);
  MIB.addDef(Res);
  for (unsigned i = 0; i < Ops.size(); ++i) {
    MIB.addUse(Ops[i]);
    MIB.addImm(Indices[i]);
  }
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildIntrinsic(Intrinsic::ID ID,
                                                     unsigned Res,
                                                     bool HasSideEffects) {
  auto MIB =
      buildInstr(HasSideEffects ? TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS
                                : TargetOpcode::G_INTRINSIC);
  if (Res)
    MIB.addDef(Res);
  MIB.addIntrinsicID(ID);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildTrunc(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, false);
  return buildInstr(TargetOpcode::G_TRUNC).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildFPTrunc(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, false);
  return buildInstr(TargetOpcode::G_FPTRUNC).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildICmp(CmpInst::Predicate Pred,
                                                unsigned Res, unsigned Op0,
                                                unsigned Op1) {
#ifndef NDEBUG
  assert(MRI->getType(Op0) == MRI->getType(Op0) && "type mismatch");
  assert(CmpInst::isIntPredicate(Pred) && "invalid predicate");
  if (MRI->getType(Op0).isScalar() || MRI->getType(Op0).isPointer())
    assert(MRI->getType(Res).isScalar() && "type mismatch");
  else
    assert(MRI->getType(Res).isVector() &&
           MRI->getType(Res).getNumElements() ==
               MRI->getType(Op0).getNumElements() &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_ICMP)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildFCmp(CmpInst::Predicate Pred,
                                                unsigned Res, unsigned Op0,
                                                unsigned Op1) {
#ifndef NDEBUG
  assert((MRI->getType(Op0).isScalar() || MRI->getType(Op0).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Op0) == MRI->getType(Op1) && "type mismatch");
  assert(CmpInst::isFPPredicate(Pred) && "invalid predicate");
  if (MRI->getType(Op0).isScalar())
    assert(MRI->getType(Res).isScalar() && "type mismatch");
  else
    assert(MRI->getType(Res).isVector() &&
           MRI->getType(Res).getNumElements() ==
               MRI->getType(Op0).getNumElements() &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_FCMP)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildSelect(unsigned Res, unsigned Tst,
                                                  unsigned Op0, unsigned Op1) {
#ifndef NDEBUG
  assert((MRI->getType(Res).isScalar() || MRI->getType(Res).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");
  if (MRI->getType(Res).isScalar())
    assert(MRI->getType(Tst).isScalar() && "type mismatch");
  else
    assert(MRI->getType(Tst).isVector() &&
           MRI->getType(Tst).getNumElements() ==
               MRI->getType(Op0).getNumElements() &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_SELECT)
      .addDef(Res)
      .addUse(Tst)
      .addUse(Op0)
      .addUse(Op1);
}

void MachineIRBuilder::validateTruncExt(unsigned Dst, unsigned Src,
                                        bool IsExtend) {
#ifndef NDEBUG
  LLT SrcTy = MRI->getType(Src);
  LLT DstTy = MRI->getType(Dst);

  if (DstTy.isVector()) {
    assert(SrcTy.isVector() && "mismatched cast between vecot and non-vector");
    assert(SrcTy.getNumElements() == DstTy.getNumElements() &&
           "different number of elements in a trunc/ext");
  } else
    assert(DstTy.isScalar() && SrcTy.isScalar() && "invalid extend/trunc");

  if (IsExtend)
    assert(DstTy.getSizeInBits() > SrcTy.getSizeInBits() &&
           "invalid narrowing extend");
  else
    assert(DstTy.getSizeInBits() < SrcTy.getSizeInBits() &&
           "invalid widening trunc");
#endif
}
