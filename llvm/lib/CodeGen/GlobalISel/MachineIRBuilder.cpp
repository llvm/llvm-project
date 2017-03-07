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
#include "llvm/IR/DebugInfo.h"
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
  this->II = MachineBasicBlock::iterator();
  this->InsertedInstr = nullptr;
}

void MachineIRBuilder::setMBB(MachineBasicBlock &MBB) {
  this->MBB = &MBB;
  this->II = MBB.end();
  assert(&getMF() == MBB.getParent() &&
         "Basic block is in a different function");
}

void MachineIRBuilder::setInstr(MachineInstr &MI) {
  assert(MI.getParent() && "Instruction is not part of a basic block");
  setMBB(*MI.getParent());
  this->II = MI.getIterator();
}

void MachineIRBuilder::setInsertPt(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator II) {
  assert(MBB.getParent() == &getMF() &&
         "Basic block is in a different function");
  this->MBB = &MBB;
  this->II = II;
}

void MachineIRBuilder::recordInsertions(
    std::function<void(MachineInstr *)> Inserted) {
  InsertedInstr = std::move(Inserted);
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

MachineInstrBuilder MachineIRBuilder::buildDirectDbgValue(
    unsigned Reg, const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return buildInstr(TargetOpcode::DBG_VALUE)
      .addReg(Reg, RegState::Debug)
      .addReg(0, RegState::Debug)
      .addMetadata(Variable)
      .addMetadata(Expr);
}

MachineInstrBuilder MachineIRBuilder::buildIndirectDbgValue(
    unsigned Reg, unsigned Offset, const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return buildInstr(TargetOpcode::DBG_VALUE)
      .addReg(Reg, RegState::Debug)
      .addImm(Offset)
      .addMetadata(Variable)
      .addMetadata(Expr);
}

MachineInstrBuilder MachineIRBuilder::buildFIDbgValue(int FI,
                                                      const MDNode *Variable,
                                                      const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  return buildInstr(TargetOpcode::DBG_VALUE)
      .addFrameIndex(FI)
      .addImm(0)
      .addMetadata(Variable)
      .addMetadata(Expr);
}

MachineInstrBuilder MachineIRBuilder::buildConstDbgValue(const Constant &C,
                                                         unsigned Offset,
                                                         const MDNode *Variable,
                                                         const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  auto MIB = buildInstr(TargetOpcode::DBG_VALUE);
  if (auto *CI = dyn_cast<ConstantInt>(&C)) {
    if (CI->getBitWidth() > 64)
      MIB.addCImm(CI);
    else
      MIB.addImm(CI->getZExtValue());
  } else
    MIB.addFPImm(&cast<ConstantFP>(C));

  return MIB.addImm(Offset).addMetadata(Variable).addMetadata(Expr);
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

MachineInstrBuilder MachineIRBuilder::buildPtrMask(unsigned Res, unsigned Op0,
                                                   uint32_t NumBits) {
  assert(MRI->getType(Res).isPointer() &&
         MRI->getType(Res) == MRI->getType(Op0) && "type mismatch");

  return buildInstr(TargetOpcode::G_PTR_MASK)
      .addDef(Res)
      .addUse(Op0)
      .addImm(NumBits);
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

MachineInstrBuilder MachineIRBuilder::buildAnd(unsigned Res, unsigned Op0,
                                               unsigned Op1) {
  assert((MRI->getType(Res).isScalar() || MRI->getType(Res).isVector()) &&
         "invalid operand type");
  assert(MRI->getType(Res) == MRI->getType(Op0) &&
         MRI->getType(Res) == MRI->getType(Op1) && "type mismatch");

  return buildInstr(TargetOpcode::G_AND)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilder::buildBr(MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BR).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilder::buildBrIndirect(unsigned Tgt) {
  return buildInstr(TargetOpcode::G_BRINDIRECT).addUse(Tgt);
}

MachineInstrBuilder MachineIRBuilder::buildCopy(unsigned Res, unsigned Op) {
  return buildInstr(TargetOpcode::COPY).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilder::buildConstant(unsigned Res,
                                                    const ConstantInt &Val) {
  LLT Ty = MRI->getType(Res);

  assert((Ty.isScalar() || Ty.isPointer()) && "invalid operand type");

  const ConstantInt *NewVal = &Val;
  if (Ty.getSizeInBits() != Val.getBitWidth())
    NewVal = ConstantInt::get(MF->getFunction()->getContext(),
                              Val.getValue().sextOrTrunc(Ty.getSizeInBits()));

  return buildInstr(TargetOpcode::G_CONSTANT).addDef(Res).addCImm(NewVal);
}

MachineInstrBuilder MachineIRBuilder::buildConstant(unsigned Res,
                                                    int64_t Val) {
  auto IntN = IntegerType::get(MF->getFunction()->getContext(),
                               MRI->getType(Res).getSizeInBits());
  ConstantInt *CI = ConstantInt::get(IntN, Val, true);
  return buildConstant(Res, *CI);
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

MachineInstrBuilder MachineIRBuilder::buildZExtOrTrunc(unsigned Res,
                                                       unsigned Op) {
  unsigned Opcode = TargetOpcode::COPY;
  if (MRI->getType(Res).getSizeInBits() > MRI->getType(Op).getSizeInBits())
    Opcode = TargetOpcode::G_ZEXT;
  else if (MRI->getType(Res).getSizeInBits() < MRI->getType(Op).getSizeInBits())
    Opcode = TargetOpcode::G_TRUNC;

  return buildInstr(Opcode).addDef(Res).addUse(Op);
}


MachineInstrBuilder MachineIRBuilder::buildCast(unsigned Dst, unsigned Src) {
  LLT SrcTy = MRI->getType(Src);
  LLT DstTy = MRI->getType(Dst);
  if (SrcTy == DstTy)
    return buildCopy(Dst, Src);

  unsigned Opcode;
  if (SrcTy.isPointer() && DstTy.isScalar())
    Opcode = TargetOpcode::G_PTRTOINT;
  else if (DstTy.isPointer() && SrcTy.isScalar())
    Opcode = TargetOpcode::G_INTTOPTR;
  else {
    assert(!SrcTy.isPointer() && !DstTy.isPointer() && "n G_ADDRCAST yet");
    Opcode = TargetOpcode::G_BITCAST;
  }

  return buildInstr(Opcode).addDef(Dst).addUse(Src);
}

MachineInstrBuilder MachineIRBuilder::buildExtract(unsigned Res, unsigned Src,
                                                   uint64_t Index) {
#ifndef NDEBUG
  assert(MRI->getType(Src).isValid() && "invalid operand type");
  assert(MRI->getType(Res).isValid() && "invalid operand type");
  assert(Index + MRI->getType(Res).getSizeInBits() <=
             MRI->getType(Src).getSizeInBits() &&
         "extracting off end of register");
#endif

  if (MRI->getType(Res).getSizeInBits() == MRI->getType(Src).getSizeInBits()) {
    assert(Index == 0 && "insertion past the end of a register");
    return buildCast(Res, Src);
  }

  return buildInstr(TargetOpcode::G_EXTRACT)
      .addDef(Res)
      .addUse(Src)
      .addImm(Index);
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

MachineInstrBuilder MachineIRBuilder::buildUndef(unsigned Res) {
  return buildInstr(TargetOpcode::IMPLICIT_DEF).addDef(Res);
}

MachineInstrBuilder MachineIRBuilder::buildMerge(unsigned Res,
                                                 ArrayRef<unsigned> Ops) {

#ifndef NDEBUG
  assert(!Ops.empty() && "invalid trivial sequence");
  LLT Ty = MRI->getType(Ops[0]);
  for (auto Reg : Ops)
    assert(MRI->getType(Reg) == Ty && "type mismatch in input list");
  assert(Ops.size() * MRI->getType(Ops[0]).getSizeInBits() ==
             MRI->getType(Res).getSizeInBits() &&
         "input operands do not cover output register");
#endif

  MachineInstrBuilder MIB = buildInstr(TargetOpcode::G_MERGE_VALUES);
  MIB.addDef(Res);
  for (unsigned i = 0; i < Ops.size(); ++i)
    MIB.addUse(Ops[i]);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildUnmerge(ArrayRef<unsigned> Res,
                                                   unsigned Op) {

#ifndef NDEBUG
  assert(!Res.empty() && "invalid trivial sequence");
  LLT Ty = MRI->getType(Res[0]);
  for (auto Reg : Res)
    assert(MRI->getType(Reg) == Ty && "type mismatch in input list");
  assert(Res.size() * MRI->getType(Res[0]).getSizeInBits() ==
             MRI->getType(Op).getSizeInBits() &&
         "input operands do not cover output register");
#endif

  MachineInstrBuilder MIB = buildInstr(TargetOpcode::G_UNMERGE_VALUES);
  for (unsigned i = 0; i < Res.size(); ++i)
    MIB.addDef(Res[i]);
  MIB.addUse(Op);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildInsert(unsigned Res, unsigned Src,
                                                  unsigned Op, unsigned Index) {
  if (MRI->getType(Res).getSizeInBits() == MRI->getType(Op).getSizeInBits()) {
    assert(Index == 0 && "insertion past the end of a register");
    return buildCast(Res, Op);
  }

  return buildInstr(TargetOpcode::G_INSERT)
      .addDef(Res)
      .addUse(Src)
      .addUse(Op)
      .addImm(Index);
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
  LLT ResTy = MRI->getType(Res);
  assert((ResTy.isScalar() || ResTy.isVector() || ResTy.isPointer()) &&
         "invalid operand type");
  assert(ResTy == MRI->getType(Op0) && ResTy == MRI->getType(Op1) &&
         "type mismatch");
  if (ResTy.isScalar() || ResTy.isPointer())
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
