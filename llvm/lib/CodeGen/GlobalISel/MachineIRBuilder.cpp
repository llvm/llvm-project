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
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfo.h"

using namespace llvm;

void MachineIRBuilderBase::setMF(MachineFunction &MF) {
  State.MF = &MF;
  State.MBB = nullptr;
  State.MRI = &MF.getRegInfo();
  State.TII = MF.getSubtarget().getInstrInfo();
  State.DL = DebugLoc();
  State.II = MachineBasicBlock::iterator();
  State.InsertedInstr = nullptr;
}

void MachineIRBuilderBase::setMBB(MachineBasicBlock &MBB) {
  State.MBB = &MBB;
  State.II = MBB.end();
  assert(&getMF() == MBB.getParent() &&
         "Basic block is in a different function");
}

void MachineIRBuilderBase::setInstr(MachineInstr &MI) {
  assert(MI.getParent() && "Instruction is not part of a basic block");
  setMBB(*MI.getParent());
  State.II = MI.getIterator();
}

void MachineIRBuilderBase::setInsertPt(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator II) {
  assert(MBB.getParent() == &getMF() &&
         "Basic block is in a different function");
  State.MBB = &MBB;
  State.II = II;
}

void MachineIRBuilderBase::recordInsertion(MachineInstr *InsertedInstr) const {
  if (State.InsertedInstr)
    State.InsertedInstr(InsertedInstr);
}

void MachineIRBuilderBase::recordInsertions(
    std::function<void(MachineInstr *)> Inserted) {
  State.InsertedInstr = std::move(Inserted);
}

void MachineIRBuilderBase::stopRecordingInsertions() {
  State.InsertedInstr = nullptr;
}

//------------------------------------------------------------------------------
// Build instruction variants.
//------------------------------------------------------------------------------

MachineInstrBuilder MachineIRBuilderBase::buildInstr(unsigned Opcode) {
  return insertInstr(buildInstrNoInsert(Opcode));
}

MachineInstrBuilder MachineIRBuilderBase::buildInstrNoInsert(unsigned Opcode) {
  MachineInstrBuilder MIB = BuildMI(getMF(), getDL(), getTII().get(Opcode));
  return MIB;
}

MachineInstrBuilder MachineIRBuilderBase::insertInstr(MachineInstrBuilder MIB) {
  getMBB().insert(getInsertPt(), MIB);
  recordInsertion(MIB);
  return MIB;
}

MachineInstrBuilder
MachineIRBuilderBase::buildDirectDbgValue(unsigned Reg, const MDNode *Variable,
                                          const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  return insertInstr(BuildMI(getMF(), getDL(),
                             getTII().get(TargetOpcode::DBG_VALUE),
                             /*IsIndirect*/ false, Reg, Variable, Expr));
}

MachineInstrBuilder MachineIRBuilderBase::buildIndirectDbgValue(
    unsigned Reg, const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  return insertInstr(BuildMI(getMF(), getDL(),
                             getTII().get(TargetOpcode::DBG_VALUE),
                             /*IsIndirect*/ true, Reg, Variable, Expr));
}

MachineInstrBuilder
MachineIRBuilderBase::buildFIDbgValue(int FI, const MDNode *Variable,
                                      const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  return buildInstr(TargetOpcode::DBG_VALUE)
      .addFrameIndex(FI)
      .addImm(0)
      .addMetadata(Variable)
      .addMetadata(Expr);
}

MachineInstrBuilder MachineIRBuilderBase::buildConstDbgValue(
    const Constant &C, const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  auto MIB = buildInstr(TargetOpcode::DBG_VALUE);
  if (auto *CI = dyn_cast<ConstantInt>(&C)) {
    if (CI->getBitWidth() > 64)
      MIB.addCImm(CI);
    else
      MIB.addImm(CI->getZExtValue());
  } else if (auto *CFP = dyn_cast<ConstantFP>(&C)) {
    MIB.addFPImm(CFP);
  } else {
    // Insert %noreg if we didn't find a usable constant and had to drop it.
    MIB.addReg(0U);
  }

  return MIB.addImm(0).addMetadata(Variable).addMetadata(Expr);
}

MachineInstrBuilder MachineIRBuilderBase::buildFrameIndex(unsigned Res,
                                                          int Idx) {
  assert(getMRI()->getType(Res).isPointer() && "invalid operand type");
  return buildInstr(TargetOpcode::G_FRAME_INDEX)
      .addDef(Res)
      .addFrameIndex(Idx);
}

MachineInstrBuilder
MachineIRBuilderBase::buildGlobalValue(unsigned Res, const GlobalValue *GV) {
  assert(getMRI()->getType(Res).isPointer() && "invalid operand type");
  assert(getMRI()->getType(Res).getAddressSpace() ==
             GV->getType()->getAddressSpace() &&
         "address space mismatch");

  return buildInstr(TargetOpcode::G_GLOBAL_VALUE)
      .addDef(Res)
      .addGlobalAddress(GV);
}

void MachineIRBuilderBase::validateBinaryOp(unsigned Res, unsigned Op0,
                                            unsigned Op1) {
  assert((getMRI()->getType(Res).isScalar() ||
          getMRI()->getType(Res).isVector()) &&
         "invalid operand type");
  assert(getMRI()->getType(Res) == getMRI()->getType(Op0) &&
         getMRI()->getType(Res) == getMRI()->getType(Op1) && "type mismatch");
}

MachineInstrBuilder MachineIRBuilderBase::buildGEP(unsigned Res, unsigned Op0,
                                                   unsigned Op1) {
  assert(getMRI()->getType(Res).isPointer() &&
         getMRI()->getType(Res) == getMRI()->getType(Op0) && "type mismatch");
  assert(getMRI()->getType(Op1).isScalar() && "invalid offset type");

  return buildInstr(TargetOpcode::G_GEP)
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
}

Optional<MachineInstrBuilder>
MachineIRBuilderBase::materializeGEP(unsigned &Res, unsigned Op0,
                                     const LLT &ValueTy, uint64_t Value) {
  assert(Res == 0 && "Res is a result argument");
  assert(ValueTy.isScalar()  && "invalid offset type");

  if (Value == 0) {
    Res = Op0;
    return None;
  }

  Res = getMRI()->createGenericVirtualRegister(getMRI()->getType(Op0));
  unsigned TmpReg = getMRI()->createGenericVirtualRegister(ValueTy);

  buildConstant(TmpReg, Value);
  return buildGEP(Res, Op0, TmpReg);
}

MachineInstrBuilder MachineIRBuilderBase::buildPtrMask(unsigned Res,
                                                       unsigned Op0,
                                                       uint32_t NumBits) {
  assert(getMRI()->getType(Res).isPointer() &&
         getMRI()->getType(Res) == getMRI()->getType(Op0) && "type mismatch");

  return buildInstr(TargetOpcode::G_PTR_MASK)
      .addDef(Res)
      .addUse(Op0)
      .addImm(NumBits);
}

MachineInstrBuilder MachineIRBuilderBase::buildBr(MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BR).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilderBase::buildBrIndirect(unsigned Tgt) {
  assert(getMRI()->getType(Tgt).isPointer() && "invalid branch destination");
  return buildInstr(TargetOpcode::G_BRINDIRECT).addUse(Tgt);
}

MachineInstrBuilder MachineIRBuilderBase::buildCopy(unsigned Res, unsigned Op) {
  assert(getMRI()->getType(Res) == LLT() || getMRI()->getType(Op) == LLT() ||
         getMRI()->getType(Res) == getMRI()->getType(Op));
  return buildInstr(TargetOpcode::COPY).addDef(Res).addUse(Op);
}

MachineInstrBuilder
MachineIRBuilderBase::buildConstant(unsigned Res, const ConstantInt &Val) {
  LLT Ty = getMRI()->getType(Res);

  assert((Ty.isScalar() || Ty.isPointer()) && "invalid operand type");

  const ConstantInt *NewVal = &Val;
  if (Ty.getSizeInBits() != Val.getBitWidth())
    NewVal = ConstantInt::get(getMF().getFunction().getContext(),
                              Val.getValue().sextOrTrunc(Ty.getSizeInBits()));

  return buildInstr(TargetOpcode::G_CONSTANT).addDef(Res).addCImm(NewVal);
}

MachineInstrBuilder MachineIRBuilderBase::buildConstant(unsigned Res,
                                                        int64_t Val) {
  auto IntN = IntegerType::get(getMF().getFunction().getContext(),
                               getMRI()->getType(Res).getSizeInBits());
  ConstantInt *CI = ConstantInt::get(IntN, Val, true);
  return buildConstant(Res, *CI);
}

MachineInstrBuilder
MachineIRBuilderBase::buildFConstant(unsigned Res, const ConstantFP &Val) {
  assert(getMRI()->getType(Res).isScalar() && "invalid operand type");

  return buildInstr(TargetOpcode::G_FCONSTANT).addDef(Res).addFPImm(&Val);
}

MachineInstrBuilder MachineIRBuilderBase::buildFConstant(unsigned Res,
                                                         double Val) {
  LLT DstTy = getMRI()->getType(Res);
  auto &Ctx = getMF().getFunction().getContext();
  auto *CFP =
      ConstantFP::get(Ctx, getAPFloatFromSize(Val, DstTy.getSizeInBits()));
  return buildFConstant(Res, *CFP);
}

MachineInstrBuilder MachineIRBuilderBase::buildBrCond(unsigned Tst,
                                                      MachineBasicBlock &Dest) {
  assert(getMRI()->getType(Tst).isScalar() && "invalid operand type");

  return buildInstr(TargetOpcode::G_BRCOND).addUse(Tst).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilderBase::buildLoad(unsigned Res, unsigned Addr,
                                                    MachineMemOperand &MMO) {
  return buildLoadInstr(TargetOpcode::G_LOAD, Res, Addr, MMO);
}

MachineInstrBuilder
MachineIRBuilderBase::buildLoadInstr(unsigned Opcode, unsigned Res,
                                     unsigned Addr, MachineMemOperand &MMO) {
  assert(getMRI()->getType(Res).isValid() && "invalid operand type");
  assert(getMRI()->getType(Addr).isPointer() && "invalid operand type");

  return buildInstr(Opcode)
      .addDef(Res)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilderBase::buildStore(unsigned Val,
                                                     unsigned Addr,
                                                     MachineMemOperand &MMO) {
  assert(getMRI()->getType(Val).isValid() && "invalid operand type");
  assert(getMRI()->getType(Addr).isPointer() && "invalid operand type");

  return buildInstr(TargetOpcode::G_STORE)
      .addUse(Val)
      .addUse(Addr)
      .addMemOperand(&MMO);
}

MachineInstrBuilder MachineIRBuilderBase::buildUAdde(unsigned Res,
                                                     unsigned CarryOut,
                                                     unsigned Op0, unsigned Op1,
                                                     unsigned CarryIn) {
  assert(getMRI()->getType(Res).isScalar() && "invalid operand type");
  assert(getMRI()->getType(Res) == getMRI()->getType(Op0) &&
         getMRI()->getType(Res) == getMRI()->getType(Op1) && "type mismatch");
  assert(getMRI()->getType(CarryOut).isScalar() && "invalid operand type");
  assert(getMRI()->getType(CarryOut) == getMRI()->getType(CarryIn) &&
         "type mismatch");

  return buildInstr(TargetOpcode::G_UADDE)
      .addDef(Res)
      .addDef(CarryOut)
      .addUse(Op0)
      .addUse(Op1)
      .addUse(CarryIn);
}

MachineInstrBuilder MachineIRBuilderBase::buildAnyExt(unsigned Res,
                                                      unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_ANYEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildSExt(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_SEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildZExt(unsigned Res, unsigned Op) {
  validateTruncExt(Res, Op, true);
  return buildInstr(TargetOpcode::G_ZEXT).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildExtOrTrunc(unsigned ExtOpc,
                                                          unsigned Res,
                                                          unsigned Op) {
  assert((TargetOpcode::G_ANYEXT == ExtOpc || TargetOpcode::G_ZEXT == ExtOpc ||
          TargetOpcode::G_SEXT == ExtOpc) &&
         "Expecting Extending Opc");
  assert(getMRI()->getType(Res).isScalar() ||
         getMRI()->getType(Res).isVector());
  assert(getMRI()->getType(Res).isScalar() == getMRI()->getType(Op).isScalar());

  unsigned Opcode = TargetOpcode::COPY;
  if (getMRI()->getType(Res).getSizeInBits() >
      getMRI()->getType(Op).getSizeInBits())
    Opcode = ExtOpc;
  else if (getMRI()->getType(Res).getSizeInBits() <
           getMRI()->getType(Op).getSizeInBits())
    Opcode = TargetOpcode::G_TRUNC;
  else
    assert(getMRI()->getType(Res) == getMRI()->getType(Op));

  return buildInstr(Opcode).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildSExtOrTrunc(unsigned Res,
                                                           unsigned Op) {
  return buildExtOrTrunc(TargetOpcode::G_SEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildZExtOrTrunc(unsigned Res,
                                                           unsigned Op) {
  return buildExtOrTrunc(TargetOpcode::G_ZEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildAnyExtOrTrunc(unsigned Res,
                                                             unsigned Op) {
  return buildExtOrTrunc(TargetOpcode::G_ANYEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildCast(unsigned Dst,
                                                    unsigned Src) {
  LLT SrcTy = getMRI()->getType(Src);
  LLT DstTy = getMRI()->getType(Dst);
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

MachineInstrBuilder
MachineIRBuilderBase::buildExtract(unsigned Res, unsigned Src, uint64_t Index) {
#ifndef NDEBUG
  assert(getMRI()->getType(Src).isValid() && "invalid operand type");
  assert(getMRI()->getType(Res).isValid() && "invalid operand type");
  assert(Index + getMRI()->getType(Res).getSizeInBits() <=
             getMRI()->getType(Src).getSizeInBits() &&
         "extracting off end of register");
#endif

  if (getMRI()->getType(Res).getSizeInBits() ==
      getMRI()->getType(Src).getSizeInBits()) {
    assert(Index == 0 && "insertion past the end of a register");
    return buildCast(Res, Src);
  }

  return buildInstr(TargetOpcode::G_EXTRACT)
      .addDef(Res)
      .addUse(Src)
      .addImm(Index);
}

void MachineIRBuilderBase::buildSequence(unsigned Res, ArrayRef<unsigned> Ops,
                                         ArrayRef<uint64_t> Indices) {
#ifndef NDEBUG
  assert(Ops.size() == Indices.size() && "incompatible args");
  assert(!Ops.empty() && "invalid trivial sequence");
  assert(std::is_sorted(Indices.begin(), Indices.end()) &&
         "sequence offsets must be in ascending order");

  assert(getMRI()->getType(Res).isValid() && "invalid operand type");
  for (auto Op : Ops)
    assert(getMRI()->getType(Op).isValid() && "invalid operand type");
#endif

  LLT ResTy = getMRI()->getType(Res);
  LLT OpTy = getMRI()->getType(Ops[0]);
  unsigned OpSize = OpTy.getSizeInBits();
  bool MaybeMerge = true;
  for (unsigned i = 0; i < Ops.size(); ++i) {
    if (getMRI()->getType(Ops[i]) != OpTy || Indices[i] != i * OpSize) {
      MaybeMerge = false;
      break;
    }
  }

  if (MaybeMerge && Ops.size() * OpSize == ResTy.getSizeInBits()) {
    buildMerge(Res, Ops);
    return;
  }

  unsigned ResIn = getMRI()->createGenericVirtualRegister(ResTy);
  buildUndef(ResIn);

  for (unsigned i = 0; i < Ops.size(); ++i) {
    unsigned ResOut = i + 1 == Ops.size()
                          ? Res
                          : getMRI()->createGenericVirtualRegister(ResTy);
    buildInsert(ResOut, ResIn, Ops[i], Indices[i]);
    ResIn = ResOut;
  }
}

MachineInstrBuilder MachineIRBuilderBase::buildUndef(unsigned Res) {
  return buildInstr(TargetOpcode::G_IMPLICIT_DEF).addDef(Res);
}

MachineInstrBuilder MachineIRBuilderBase::buildMerge(unsigned Res,
                                                     ArrayRef<unsigned> Ops) {

#ifndef NDEBUG
  assert(!Ops.empty() && "invalid trivial sequence");
  LLT Ty = getMRI()->getType(Ops[0]);
  for (auto Reg : Ops)
    assert(getMRI()->getType(Reg) == Ty && "type mismatch in input list");
  assert(Ops.size() * getMRI()->getType(Ops[0]).getSizeInBits() ==
             getMRI()->getType(Res).getSizeInBits() &&
         "input operands do not cover output register");
#endif

  if (Ops.size() == 1)
    return buildCast(Res, Ops[0]);

  MachineInstrBuilder MIB = buildInstr(TargetOpcode::G_MERGE_VALUES);
  MIB.addDef(Res);
  for (unsigned i = 0; i < Ops.size(); ++i)
    MIB.addUse(Ops[i]);
  return MIB;
}

MachineInstrBuilder MachineIRBuilderBase::buildUnmerge(ArrayRef<unsigned> Res,
                                                       unsigned Op) {

#ifndef NDEBUG
  assert(!Res.empty() && "invalid trivial sequence");
  LLT Ty = getMRI()->getType(Res[0]);
  for (auto Reg : Res)
    assert(getMRI()->getType(Reg) == Ty && "type mismatch in input list");
  assert(Res.size() * getMRI()->getType(Res[0]).getSizeInBits() ==
             getMRI()->getType(Op).getSizeInBits() &&
         "input operands do not cover output register");
#endif

  MachineInstrBuilder MIB = buildInstr(TargetOpcode::G_UNMERGE_VALUES);
  for (unsigned i = 0; i < Res.size(); ++i)
    MIB.addDef(Res[i]);
  MIB.addUse(Op);
  return MIB;
}

MachineInstrBuilder MachineIRBuilderBase::buildInsert(unsigned Res,
                                                      unsigned Src, unsigned Op,
                                                      unsigned Index) {
  assert(Index + getMRI()->getType(Op).getSizeInBits() <=
             getMRI()->getType(Res).getSizeInBits() &&
         "insertion past the end of a register");

  if (getMRI()->getType(Res).getSizeInBits() ==
      getMRI()->getType(Op).getSizeInBits()) {
    return buildCast(Res, Op);
  }

  return buildInstr(TargetOpcode::G_INSERT)
      .addDef(Res)
      .addUse(Src)
      .addUse(Op)
      .addImm(Index);
}

MachineInstrBuilder MachineIRBuilderBase::buildIntrinsic(Intrinsic::ID ID,
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

MachineInstrBuilder MachineIRBuilderBase::buildTrunc(unsigned Res,
                                                     unsigned Op) {
  validateTruncExt(Res, Op, false);
  return buildInstr(TargetOpcode::G_TRUNC).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildFPTrunc(unsigned Res,
                                                       unsigned Op) {
  validateTruncExt(Res, Op, false);
  return buildInstr(TargetOpcode::G_FPTRUNC).addDef(Res).addUse(Op);
}

MachineInstrBuilder MachineIRBuilderBase::buildICmp(CmpInst::Predicate Pred,
                                                    unsigned Res, unsigned Op0,
                                                    unsigned Op1) {
#ifndef NDEBUG
  assert(getMRI()->getType(Op0) == getMRI()->getType(Op0) && "type mismatch");
  assert(CmpInst::isIntPredicate(Pred) && "invalid predicate");
  if (getMRI()->getType(Op0).isScalar() || getMRI()->getType(Op0).isPointer())
    assert(getMRI()->getType(Res).isScalar() && "type mismatch");
  else
    assert(getMRI()->getType(Res).isVector() &&
           getMRI()->getType(Res).getNumElements() ==
               getMRI()->getType(Op0).getNumElements() &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_ICMP)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilderBase::buildFCmp(CmpInst::Predicate Pred,
                                                    unsigned Res, unsigned Op0,
                                                    unsigned Op1) {
#ifndef NDEBUG
  assert((getMRI()->getType(Op0).isScalar() ||
          getMRI()->getType(Op0).isVector()) &&
         "invalid operand type");
  assert(getMRI()->getType(Op0) == getMRI()->getType(Op1) && "type mismatch");
  assert(CmpInst::isFPPredicate(Pred) && "invalid predicate");
  if (getMRI()->getType(Op0).isScalar())
    assert(getMRI()->getType(Res).isScalar() && "type mismatch");
  else
    assert(getMRI()->getType(Res).isVector() &&
           getMRI()->getType(Res).getNumElements() ==
               getMRI()->getType(Op0).getNumElements() &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_FCMP)
      .addDef(Res)
      .addPredicate(Pred)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder MachineIRBuilderBase::buildSelect(unsigned Res,
                                                      unsigned Tst,
                                                      unsigned Op0,
                                                      unsigned Op1) {
#ifndef NDEBUG
  LLT ResTy = getMRI()->getType(Res);
  assert((ResTy.isScalar() || ResTy.isVector() || ResTy.isPointer()) &&
         "invalid operand type");
  assert(ResTy == getMRI()->getType(Op0) && ResTy == getMRI()->getType(Op1) &&
         "type mismatch");
  if (ResTy.isScalar() || ResTy.isPointer())
    assert(getMRI()->getType(Tst).isScalar() && "type mismatch");
  else
    assert((getMRI()->getType(Tst).isScalar() ||
            (getMRI()->getType(Tst).isVector() &&
             getMRI()->getType(Tst).getNumElements() ==
                 getMRI()->getType(Op0).getNumElements())) &&
           "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_SELECT)
      .addDef(Res)
      .addUse(Tst)
      .addUse(Op0)
      .addUse(Op1);
}

MachineInstrBuilder
MachineIRBuilderBase::buildInsertVectorElement(unsigned Res, unsigned Val,
                                               unsigned Elt, unsigned Idx) {
#ifndef NDEBUG
  LLT ResTy = getMRI()->getType(Res);
  LLT ValTy = getMRI()->getType(Val);
  LLT EltTy = getMRI()->getType(Elt);
  LLT IdxTy = getMRI()->getType(Idx);
  assert(ResTy.isVector() && ValTy.isVector() && "invalid operand type");
  assert(IdxTy.isScalar() && "invalid operand type");
  assert(ResTy.getNumElements() == ValTy.getNumElements() && "type mismatch");
  assert(ResTy.getElementType() == EltTy && "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_INSERT_VECTOR_ELT)
      .addDef(Res)
      .addUse(Val)
      .addUse(Elt)
      .addUse(Idx);
}

MachineInstrBuilder
MachineIRBuilderBase::buildExtractVectorElement(unsigned Res, unsigned Val,
                                                unsigned Idx) {
#ifndef NDEBUG
  LLT ResTy = getMRI()->getType(Res);
  LLT ValTy = getMRI()->getType(Val);
  LLT IdxTy = getMRI()->getType(Idx);
  assert(ValTy.isVector() && "invalid operand type");
  assert((ResTy.isScalar() || ResTy.isPointer()) && "invalid operand type");
  assert(IdxTy.isScalar() && "invalid operand type");
  assert(ValTy.getElementType() == ResTy && "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_EXTRACT_VECTOR_ELT)
      .addDef(Res)
      .addUse(Val)
      .addUse(Idx);
}

MachineInstrBuilder MachineIRBuilderBase::buildAtomicCmpXchgWithSuccess(
    unsigned OldValRes, unsigned SuccessRes, unsigned Addr, unsigned CmpVal,
    unsigned NewVal, MachineMemOperand &MMO) {
#ifndef NDEBUG
  LLT OldValResTy = getMRI()->getType(OldValRes);
  LLT SuccessResTy = getMRI()->getType(SuccessRes);
  LLT AddrTy = getMRI()->getType(Addr);
  LLT CmpValTy = getMRI()->getType(CmpVal);
  LLT NewValTy = getMRI()->getType(NewVal);
  assert(OldValResTy.isScalar() && "invalid operand type");
  assert(SuccessResTy.isScalar() && "invalid operand type");
  assert(AddrTy.isPointer() && "invalid operand type");
  assert(CmpValTy.isValid() && "invalid operand type");
  assert(NewValTy.isValid() && "invalid operand type");
  assert(OldValResTy == CmpValTy && "type mismatch");
  assert(OldValResTy == NewValTy && "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_ATOMIC_CMPXCHG_WITH_SUCCESS)
      .addDef(OldValRes)
      .addDef(SuccessRes)
      .addUse(Addr)
      .addUse(CmpVal)
      .addUse(NewVal)
      .addMemOperand(&MMO);
}

MachineInstrBuilder
MachineIRBuilderBase::buildAtomicCmpXchg(unsigned OldValRes, unsigned Addr,
                                         unsigned CmpVal, unsigned NewVal,
                                         MachineMemOperand &MMO) {
#ifndef NDEBUG
  LLT OldValResTy = getMRI()->getType(OldValRes);
  LLT AddrTy = getMRI()->getType(Addr);
  LLT CmpValTy = getMRI()->getType(CmpVal);
  LLT NewValTy = getMRI()->getType(NewVal);
  assert(OldValResTy.isScalar() && "invalid operand type");
  assert(AddrTy.isPointer() && "invalid operand type");
  assert(CmpValTy.isValid() && "invalid operand type");
  assert(NewValTy.isValid() && "invalid operand type");
  assert(OldValResTy == CmpValTy && "type mismatch");
  assert(OldValResTy == NewValTy && "type mismatch");
#endif

  return buildInstr(TargetOpcode::G_ATOMIC_CMPXCHG)
      .addDef(OldValRes)
      .addUse(Addr)
      .addUse(CmpVal)
      .addUse(NewVal)
      .addMemOperand(&MMO);
}

MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMW(unsigned Opcode, unsigned OldValRes,
                                     unsigned Addr, unsigned Val,
                                     MachineMemOperand &MMO) {
#ifndef NDEBUG
  LLT OldValResTy = getMRI()->getType(OldValRes);
  LLT AddrTy = getMRI()->getType(Addr);
  LLT ValTy = getMRI()->getType(Val);
  assert(OldValResTy.isScalar() && "invalid operand type");
  assert(AddrTy.isPointer() && "invalid operand type");
  assert(ValTy.isValid() && "invalid operand type");
  assert(OldValResTy == ValTy && "type mismatch");
#endif

  return buildInstr(Opcode)
      .addDef(OldValRes)
      .addUse(Addr)
      .addUse(Val)
      .addMemOperand(&MMO);
}

MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWXchg(unsigned OldValRes, unsigned Addr,
                                         unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_XCHG, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWAdd(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_ADD, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWSub(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_SUB, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWAnd(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_AND, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWNand(unsigned OldValRes, unsigned Addr,
                                         unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_NAND, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWOr(unsigned OldValRes, unsigned Addr,
                                       unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_OR, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWXor(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_XOR, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWMax(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_MAX, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWMin(unsigned OldValRes, unsigned Addr,
                                        unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_MIN, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWUmax(unsigned OldValRes, unsigned Addr,
                                         unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_UMAX, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilderBase::buildAtomicRMWUmin(unsigned OldValRes, unsigned Addr,
                                         unsigned Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_UMIN, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilderBase::buildBlockAddress(unsigned Res, const BlockAddress *BA) {
#ifndef NDEBUG
  assert(getMRI()->getType(Res).isPointer() && "invalid res type");
#endif

  return buildInstr(TargetOpcode::G_BLOCK_ADDR).addDef(Res).addBlockAddress(BA);
}

void MachineIRBuilderBase::validateTruncExt(unsigned Dst, unsigned Src,
                                            bool IsExtend) {
#ifndef NDEBUG
  LLT SrcTy = getMRI()->getType(Src);
  LLT DstTy = getMRI()->getType(Dst);

  if (DstTy.isVector()) {
    assert(SrcTy.isVector() && "mismatched cast between vector and non-vector");
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
