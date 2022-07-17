//===-- llvm/CodeGen/GlobalISel/MachineIRBuilder.cpp - MIBuilder--*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

void MachineIRBuilder::setMF(MachineFunction &MF) {
  State.MF = &MF;
  State.MBB = nullptr;
  State.MRI = &MF.getRegInfo();
  State.TII = MF.getSubtarget().getInstrInfo();
  State.DL = DebugLoc();
  State.II = MachineBasicBlock::iterator();
  State.Observer = nullptr;
}

//------------------------------------------------------------------------------
// Build instruction variants.
//------------------------------------------------------------------------------

MachineInstrBuilder MachineIRBuilder::buildInstrNoInsert(unsigned Opcode) {
  MachineInstrBuilder MIB = BuildMI(getMF(), getDL(), getTII().get(Opcode));
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::insertInstr(MachineInstrBuilder MIB) {
  getMBB().insert(getInsertPt(), MIB);
  recordInsertion(MIB);
  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildDirectDbgValue(Register Reg, const MDNode *Variable,
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

MachineInstrBuilder
MachineIRBuilder::buildIndirectDbgValue(Register Reg, const MDNode *Variable,
                                        const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  return insertInstr(BuildMI(getMF(), getDL(),
                             getTII().get(TargetOpcode::DBG_VALUE),
                             /*IsIndirect*/ true, Reg, Variable, Expr));
}

MachineInstrBuilder MachineIRBuilder::buildFIDbgValue(int FI,
                                                      const MDNode *Variable,
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

MachineInstrBuilder MachineIRBuilder::buildConstDbgValue(const Constant &C,
                                                         const MDNode *Variable,
                                                         const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(
      cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(getDL()) &&
      "Expected inlined-at fields to agree");
  auto MIB = buildInstrNoInsert(TargetOpcode::DBG_VALUE);
  if (auto *CI = dyn_cast<ConstantInt>(&C)) {
    if (CI->getBitWidth() > 64)
      MIB.addCImm(CI);
    else
      MIB.addImm(CI->getZExtValue());
  } else if (auto *CFP = dyn_cast<ConstantFP>(&C)) {
    MIB.addFPImm(CFP);
  } else {
    // Insert $noreg if we didn't find a usable constant and had to drop it.
    MIB.addReg(Register());
  }

  MIB.addImm(0).addMetadata(Variable).addMetadata(Expr);
  return insertInstr(MIB);
}

MachineInstrBuilder MachineIRBuilder::buildDbgLabel(const MDNode *Label) {
  assert(isa<DILabel>(Label) && "not a label");
  assert(cast<DILabel>(Label)->isValidLocationForIntrinsic(State.DL) &&
         "Expected inlined-at fields to agree");
  auto MIB = buildInstr(TargetOpcode::DBG_LABEL);

  return MIB.addMetadata(Label);
}

MachineInstrBuilder MachineIRBuilder::buildDynStackAlloc(const DstOp &Res,
                                                         const SrcOp &Size,
                                                         Align Alignment) {
  assert(Res.getLLTTy(*getMRI()).isPointer() && "expected ptr dst type");
  auto MIB = buildInstr(TargetOpcode::G_DYN_STACKALLOC);
  Res.addDefToMIB(*getMRI(), MIB);
  Size.addSrcToMIB(MIB);
  MIB.addImm(Alignment.value());
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildFrameIndex(const DstOp &Res,
                                                      int Idx) {
  assert(Res.getLLTTy(*getMRI()).isPointer() && "invalid operand type");
  auto MIB = buildInstr(TargetOpcode::G_FRAME_INDEX);
  Res.addDefToMIB(*getMRI(), MIB);
  MIB.addFrameIndex(Idx);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildGlobalValue(const DstOp &Res,
                                                       const GlobalValue *GV) {
  assert(Res.getLLTTy(*getMRI()).isPointer() && "invalid operand type");
  assert(Res.getLLTTy(*getMRI()).getAddressSpace() ==
             GV->getType()->getAddressSpace() &&
         "address space mismatch");

  auto MIB = buildInstr(TargetOpcode::G_GLOBAL_VALUE);
  Res.addDefToMIB(*getMRI(), MIB);
  MIB.addGlobalAddress(GV);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildJumpTable(const LLT PtrTy,
                                                     unsigned JTI) {
  return buildInstr(TargetOpcode::G_JUMP_TABLE, {PtrTy}, {})
      .addJumpTableIndex(JTI);
}

void MachineIRBuilder::validateUnaryOp(const LLT Res, const LLT Op0) {
  assert((Res.isScalar() || Res.isVector()) && "invalid operand type");
  assert((Res == Op0) && "type mismatch");
}

void MachineIRBuilder::validateBinaryOp(const LLT Res, const LLT Op0,
                                        const LLT Op1) {
  assert((Res.isScalar() || Res.isVector()) && "invalid operand type");
  assert((Res == Op0 && Res == Op1) && "type mismatch");
}

void MachineIRBuilder::validateShiftOp(const LLT Res, const LLT Op0,
                                       const LLT Op1) {
  assert((Res.isScalar() || Res.isVector()) && "invalid operand type");
  assert((Res == Op0) && "type mismatch");
}

MachineInstrBuilder MachineIRBuilder::buildPtrAdd(const DstOp &Res,
                                                  const SrcOp &Op0,
                                                  const SrcOp &Op1) {
  assert(Res.getLLTTy(*getMRI()).getScalarType().isPointer() &&
         Res.getLLTTy(*getMRI()) == Op0.getLLTTy(*getMRI()) && "type mismatch");
  assert(Op1.getLLTTy(*getMRI()).getScalarType().isScalar() && "invalid offset type");

  return buildInstr(TargetOpcode::G_PTR_ADD, {Res}, {Op0, Op1});
}

Optional<MachineInstrBuilder>
MachineIRBuilder::materializePtrAdd(Register &Res, Register Op0,
                                    const LLT ValueTy, uint64_t Value) {
  assert(Res == 0 && "Res is a result argument");
  assert(ValueTy.isScalar()  && "invalid offset type");

  if (Value == 0) {
    Res = Op0;
    return None;
  }

  Res = getMRI()->createGenericVirtualRegister(getMRI()->getType(Op0));
  auto Cst = buildConstant(ValueTy, Value);
  return buildPtrAdd(Res, Op0, Cst.getReg(0));
}

MachineInstrBuilder MachineIRBuilder::buildMaskLowPtrBits(const DstOp &Res,
                                                          const SrcOp &Op0,
                                                          uint32_t NumBits) {
  LLT PtrTy = Res.getLLTTy(*getMRI());
  LLT MaskTy = LLT::scalar(PtrTy.getSizeInBits());
  Register MaskReg = getMRI()->createGenericVirtualRegister(MaskTy);
  buildConstant(MaskReg, maskTrailingZeros<uint64_t>(NumBits));
  return buildPtrMask(Res, Op0, MaskReg);
}

MachineInstrBuilder
MachineIRBuilder::buildPadVectorWithUndefElements(const DstOp &Res,
                                                  const SrcOp &Op0) {
  LLT ResTy = Res.getLLTTy(*getMRI());
  LLT Op0Ty = Op0.getLLTTy(*getMRI());

  assert((ResTy.isVector() && Op0Ty.isVector()) && "Non vector type");
  assert((ResTy.getElementType() == Op0Ty.getElementType()) &&
         "Different vector element types");
  assert((ResTy.getNumElements() > Op0Ty.getNumElements()) &&
         "Op0 has more elements");

  auto Unmerge = buildUnmerge(Op0Ty.getElementType(), Op0);
  SmallVector<Register, 8> Regs;
  for (auto Op : Unmerge.getInstr()->defs())
    Regs.push_back(Op.getReg());
  Register Undef = buildUndef(Op0Ty.getElementType()).getReg(0);
  unsigned NumberOfPadElts = ResTy.getNumElements() - Regs.size();
  for (unsigned i = 0; i < NumberOfPadElts; ++i)
    Regs.push_back(Undef);
  return buildMerge(Res, Regs);
}

MachineInstrBuilder
MachineIRBuilder::buildDeleteTrailingVectorElements(const DstOp &Res,
                                                    const SrcOp &Op0) {
  LLT ResTy = Res.getLLTTy(*getMRI());
  LLT Op0Ty = Op0.getLLTTy(*getMRI());

  assert((ResTy.isVector() && Op0Ty.isVector()) && "Non vector type");
  assert((ResTy.getElementType() == Op0Ty.getElementType()) &&
         "Different vector element types");
  assert((ResTy.getNumElements() < Op0Ty.getNumElements()) &&
         "Op0 has fewer elements");

  SmallVector<Register, 8> Regs;
  auto Unmerge = buildUnmerge(Op0Ty.getElementType(), Op0);
  for (unsigned i = 0; i < ResTy.getNumElements(); ++i)
    Regs.push_back(Unmerge.getReg(i));
  return buildMerge(Res, Regs);
}

MachineInstrBuilder MachineIRBuilder::buildBr(MachineBasicBlock &Dest) {
  return buildInstr(TargetOpcode::G_BR).addMBB(&Dest);
}

MachineInstrBuilder MachineIRBuilder::buildBrIndirect(Register Tgt) {
  assert(getMRI()->getType(Tgt).isPointer() && "invalid branch destination");
  return buildInstr(TargetOpcode::G_BRINDIRECT).addUse(Tgt);
}

MachineInstrBuilder MachineIRBuilder::buildBrJT(Register TablePtr,
                                                unsigned JTI,
                                                Register IndexReg) {
  assert(getMRI()->getType(TablePtr).isPointer() &&
         "Table reg must be a pointer");
  return buildInstr(TargetOpcode::G_BRJT)
      .addUse(TablePtr)
      .addJumpTableIndex(JTI)
      .addUse(IndexReg);
}

MachineInstrBuilder MachineIRBuilder::buildCopy(const DstOp &Res,
                                                const SrcOp &Op) {
  return buildInstr(TargetOpcode::COPY, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildConstant(const DstOp &Res,
                                                    const ConstantInt &Val) {
  LLT Ty = Res.getLLTTy(*getMRI());
  LLT EltTy = Ty.getScalarType();
  assert(EltTy.getScalarSizeInBits() == Val.getBitWidth() &&
         "creating constant with the wrong size");

  if (Ty.isVector()) {
    auto Const = buildInstr(TargetOpcode::G_CONSTANT)
    .addDef(getMRI()->createGenericVirtualRegister(EltTy))
    .addCImm(&Val);
    return buildSplatVector(Res, Const);
  }

  auto Const = buildInstr(TargetOpcode::G_CONSTANT);
  Const->setDebugLoc(DebugLoc());
  Res.addDefToMIB(*getMRI(), Const);
  Const.addCImm(&Val);
  return Const;
}

MachineInstrBuilder MachineIRBuilder::buildConstant(const DstOp &Res,
                                                    int64_t Val) {
  auto IntN = IntegerType::get(getMF().getFunction().getContext(),
                               Res.getLLTTy(*getMRI()).getScalarSizeInBits());
  ConstantInt *CI = ConstantInt::get(IntN, Val, true);
  return buildConstant(Res, *CI);
}

MachineInstrBuilder MachineIRBuilder::buildFConstant(const DstOp &Res,
                                                     const ConstantFP &Val) {
  LLT Ty = Res.getLLTTy(*getMRI());
  LLT EltTy = Ty.getScalarType();

  assert(APFloat::getSizeInBits(Val.getValueAPF().getSemantics())
         == EltTy.getSizeInBits() &&
         "creating fconstant with the wrong size");

  assert(!Ty.isPointer() && "invalid operand type");

  if (Ty.isVector()) {
    auto Const = buildInstr(TargetOpcode::G_FCONSTANT)
    .addDef(getMRI()->createGenericVirtualRegister(EltTy))
    .addFPImm(&Val);

    return buildSplatVector(Res, Const);
  }

  auto Const = buildInstr(TargetOpcode::G_FCONSTANT);
  Const->setDebugLoc(DebugLoc());
  Res.addDefToMIB(*getMRI(), Const);
  Const.addFPImm(&Val);
  return Const;
}

MachineInstrBuilder MachineIRBuilder::buildConstant(const DstOp &Res,
                                                    const APInt &Val) {
  ConstantInt *CI = ConstantInt::get(getMF().getFunction().getContext(), Val);
  return buildConstant(Res, *CI);
}

MachineInstrBuilder MachineIRBuilder::buildFConstant(const DstOp &Res,
                                                     double Val) {
  LLT DstTy = Res.getLLTTy(*getMRI());
  auto &Ctx = getMF().getFunction().getContext();
  auto *CFP =
      ConstantFP::get(Ctx, getAPFloatFromSize(Val, DstTy.getScalarSizeInBits()));
  return buildFConstant(Res, *CFP);
}

MachineInstrBuilder MachineIRBuilder::buildFConstant(const DstOp &Res,
                                                     const APFloat &Val) {
  auto &Ctx = getMF().getFunction().getContext();
  auto *CFP = ConstantFP::get(Ctx, Val);
  return buildFConstant(Res, *CFP);
}

MachineInstrBuilder MachineIRBuilder::buildBrCond(const SrcOp &Tst,
                                                  MachineBasicBlock &Dest) {
  assert(Tst.getLLTTy(*getMRI()).isScalar() && "invalid operand type");

  auto MIB = buildInstr(TargetOpcode::G_BRCOND);
  Tst.addSrcToMIB(MIB);
  MIB.addMBB(&Dest);
  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildLoad(const DstOp &Dst, const SrcOp &Addr,
                            MachinePointerInfo PtrInfo, Align Alignment,
                            MachineMemOperand::Flags MMOFlags,
                            const AAMDNodes &AAInfo) {
  MMOFlags |= MachineMemOperand::MOLoad;
  assert((MMOFlags & MachineMemOperand::MOStore) == 0);

  LLT Ty = Dst.getLLTTy(*getMRI());
  MachineMemOperand *MMO =
      getMF().getMachineMemOperand(PtrInfo, MMOFlags, Ty, Alignment, AAInfo);
  return buildLoad(Dst, Addr, *MMO);
}

MachineInstrBuilder MachineIRBuilder::buildLoadInstr(unsigned Opcode,
                                                     const DstOp &Res,
                                                     const SrcOp &Addr,
                                                     MachineMemOperand &MMO) {
  assert(Res.getLLTTy(*getMRI()).isValid() && "invalid operand type");
  assert(Addr.getLLTTy(*getMRI()).isPointer() && "invalid operand type");

  auto MIB = buildInstr(Opcode);
  Res.addDefToMIB(*getMRI(), MIB);
  Addr.addSrcToMIB(MIB);
  MIB.addMemOperand(&MMO);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildLoadFromOffset(
  const DstOp &Dst, const SrcOp &BasePtr,
  MachineMemOperand &BaseMMO, int64_t Offset) {
  LLT LoadTy = Dst.getLLTTy(*getMRI());
  MachineMemOperand *OffsetMMO =
      getMF().getMachineMemOperand(&BaseMMO, Offset, LoadTy);

  if (Offset == 0) // This may be a size or type changing load.
    return buildLoad(Dst, BasePtr, *OffsetMMO);

  LLT PtrTy = BasePtr.getLLTTy(*getMRI());
  LLT OffsetTy = LLT::scalar(PtrTy.getSizeInBits());
  auto ConstOffset = buildConstant(OffsetTy, Offset);
  auto Ptr = buildPtrAdd(PtrTy, BasePtr, ConstOffset);
  return buildLoad(Dst, Ptr, *OffsetMMO);
}

MachineInstrBuilder MachineIRBuilder::buildStore(const SrcOp &Val,
                                                 const SrcOp &Addr,
                                                 MachineMemOperand &MMO) {
  assert(Val.getLLTTy(*getMRI()).isValid() && "invalid operand type");
  assert(Addr.getLLTTy(*getMRI()).isPointer() && "invalid operand type");

  auto MIB = buildInstr(TargetOpcode::G_STORE);
  Val.addSrcToMIB(MIB);
  Addr.addSrcToMIB(MIB);
  MIB.addMemOperand(&MMO);
  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildStore(const SrcOp &Val, const SrcOp &Addr,
                             MachinePointerInfo PtrInfo, Align Alignment,
                             MachineMemOperand::Flags MMOFlags,
                             const AAMDNodes &AAInfo) {
  MMOFlags |= MachineMemOperand::MOStore;
  assert((MMOFlags & MachineMemOperand::MOLoad) == 0);

  LLT Ty = Val.getLLTTy(*getMRI());
  MachineMemOperand *MMO =
      getMF().getMachineMemOperand(PtrInfo, MMOFlags, Ty, Alignment, AAInfo);
  return buildStore(Val, Addr, *MMO);
}

MachineInstrBuilder MachineIRBuilder::buildAnyExt(const DstOp &Res,
                                                  const SrcOp &Op) {
  return buildInstr(TargetOpcode::G_ANYEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildSExt(const DstOp &Res,
                                                const SrcOp &Op) {
  return buildInstr(TargetOpcode::G_SEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildZExt(const DstOp &Res,
                                                const SrcOp &Op) {
  return buildInstr(TargetOpcode::G_ZEXT, Res, Op);
}

unsigned MachineIRBuilder::getBoolExtOp(bool IsVec, bool IsFP) const {
  const auto *TLI = getMF().getSubtarget().getTargetLowering();
  switch (TLI->getBooleanContents(IsVec, IsFP)) {
  case TargetLoweringBase::ZeroOrNegativeOneBooleanContent:
    return TargetOpcode::G_SEXT;
  case TargetLoweringBase::ZeroOrOneBooleanContent:
    return TargetOpcode::G_ZEXT;
  default:
    return TargetOpcode::G_ANYEXT;
  }
}

MachineInstrBuilder MachineIRBuilder::buildBoolExt(const DstOp &Res,
                                                   const SrcOp &Op,
                                                   bool IsFP) {
  unsigned ExtOp = getBoolExtOp(getMRI()->getType(Op.getReg()).isVector(), IsFP);
  return buildInstr(ExtOp, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildBoolExtInReg(const DstOp &Res,
                                                        const SrcOp &Op,
                                                        bool IsVector,
                                                        bool IsFP) {
  const auto *TLI = getMF().getSubtarget().getTargetLowering();
  switch (TLI->getBooleanContents(IsVector, IsFP)) {
  case TargetLoweringBase::ZeroOrNegativeOneBooleanContent:
    return buildSExtInReg(Res, Op, 1);
  case TargetLoweringBase::ZeroOrOneBooleanContent:
    return buildZExtInReg(Res, Op, 1);
  case TargetLoweringBase::UndefinedBooleanContent:
    return buildCopy(Res, Op);
  }

  llvm_unreachable("unexpected BooleanContent");
}

MachineInstrBuilder MachineIRBuilder::buildExtOrTrunc(unsigned ExtOpc,
                                                      const DstOp &Res,
                                                      const SrcOp &Op) {
  assert((TargetOpcode::G_ANYEXT == ExtOpc || TargetOpcode::G_ZEXT == ExtOpc ||
          TargetOpcode::G_SEXT == ExtOpc) &&
         "Expecting Extending Opc");
  assert(Res.getLLTTy(*getMRI()).isScalar() ||
         Res.getLLTTy(*getMRI()).isVector());
  assert(Res.getLLTTy(*getMRI()).isScalar() ==
         Op.getLLTTy(*getMRI()).isScalar());

  unsigned Opcode = TargetOpcode::COPY;
  if (Res.getLLTTy(*getMRI()).getSizeInBits() >
      Op.getLLTTy(*getMRI()).getSizeInBits())
    Opcode = ExtOpc;
  else if (Res.getLLTTy(*getMRI()).getSizeInBits() <
           Op.getLLTTy(*getMRI()).getSizeInBits())
    Opcode = TargetOpcode::G_TRUNC;
  else
    assert(Res.getLLTTy(*getMRI()) == Op.getLLTTy(*getMRI()));

  return buildInstr(Opcode, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildSExtOrTrunc(const DstOp &Res,
                                                       const SrcOp &Op) {
  return buildExtOrTrunc(TargetOpcode::G_SEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildZExtOrTrunc(const DstOp &Res,
                                                       const SrcOp &Op) {
  return buildExtOrTrunc(TargetOpcode::G_ZEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildAnyExtOrTrunc(const DstOp &Res,
                                                         const SrcOp &Op) {
  return buildExtOrTrunc(TargetOpcode::G_ANYEXT, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildZExtInReg(const DstOp &Res,
                                                     const SrcOp &Op,
                                                     int64_t ImmOp) {
  LLT ResTy = Res.getLLTTy(*getMRI());
  auto Mask = buildConstant(
      ResTy, APInt::getLowBitsSet(ResTy.getScalarSizeInBits(), ImmOp));
  return buildAnd(Res, Op, Mask);
}

MachineInstrBuilder MachineIRBuilder::buildCast(const DstOp &Dst,
                                                const SrcOp &Src) {
  LLT SrcTy = Src.getLLTTy(*getMRI());
  LLT DstTy = Dst.getLLTTy(*getMRI());
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

  return buildInstr(Opcode, Dst, Src);
}

MachineInstrBuilder MachineIRBuilder::buildExtract(const DstOp &Dst,
                                                   const SrcOp &Src,
                                                   uint64_t Index) {
  LLT SrcTy = Src.getLLTTy(*getMRI());
  LLT DstTy = Dst.getLLTTy(*getMRI());

#ifndef NDEBUG
  assert(SrcTy.isValid() && "invalid operand type");
  assert(DstTy.isValid() && "invalid operand type");
  assert(Index + DstTy.getSizeInBits() <= SrcTy.getSizeInBits() &&
         "extracting off end of register");
#endif

  if (DstTy.getSizeInBits() == SrcTy.getSizeInBits()) {
    assert(Index == 0 && "insertion past the end of a register");
    return buildCast(Dst, Src);
  }

  auto Extract = buildInstr(TargetOpcode::G_EXTRACT);
  Dst.addDefToMIB(*getMRI(), Extract);
  Src.addSrcToMIB(Extract);
  Extract.addImm(Index);
  return Extract;
}

MachineInstrBuilder MachineIRBuilder::buildUndef(const DstOp &Res) {
  return buildInstr(TargetOpcode::G_IMPLICIT_DEF, {Res}, {});
}

MachineInstrBuilder MachineIRBuilder::buildMerge(const DstOp &Res,
                                                 ArrayRef<Register> Ops) {
  // Unfortunately to convert from ArrayRef<LLT> to ArrayRef<SrcOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<SrcOp, 8> TmpVec(Ops.begin(), Ops.end());
  assert(TmpVec.size() > 1);
  return buildInstr(TargetOpcode::G_MERGE_VALUES, Res, TmpVec);
}

MachineInstrBuilder
MachineIRBuilder::buildMerge(const DstOp &Res,
                             std::initializer_list<SrcOp> Ops) {
  assert(Ops.size() > 1);
  return buildInstr(TargetOpcode::G_MERGE_VALUES, Res, Ops);
}

MachineInstrBuilder MachineIRBuilder::buildUnmerge(ArrayRef<LLT> Res,
                                                   const SrcOp &Op) {
  // Unfortunately to convert from ArrayRef<LLT> to ArrayRef<DstOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<DstOp, 8> TmpVec(Res.begin(), Res.end());
  assert(TmpVec.size() > 1);
  return buildInstr(TargetOpcode::G_UNMERGE_VALUES, TmpVec, Op);
}

MachineInstrBuilder MachineIRBuilder::buildUnmerge(LLT Res,
                                                   const SrcOp &Op) {
  unsigned NumReg = Op.getLLTTy(*getMRI()).getSizeInBits() / Res.getSizeInBits();
  SmallVector<DstOp, 8> TmpVec(NumReg, Res);
  return buildInstr(TargetOpcode::G_UNMERGE_VALUES, TmpVec, Op);
}

MachineInstrBuilder MachineIRBuilder::buildUnmerge(ArrayRef<Register> Res,
                                                   const SrcOp &Op) {
  // Unfortunately to convert from ArrayRef<Register> to ArrayRef<DstOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<DstOp, 8> TmpVec(Res.begin(), Res.end());
  assert(TmpVec.size() > 1);
  return buildInstr(TargetOpcode::G_UNMERGE_VALUES, TmpVec, Op);
}

MachineInstrBuilder MachineIRBuilder::buildBuildVector(const DstOp &Res,
                                                       ArrayRef<Register> Ops) {
  // Unfortunately to convert from ArrayRef<Register> to ArrayRef<SrcOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<SrcOp, 8> TmpVec(Ops.begin(), Ops.end());
  return buildInstr(TargetOpcode::G_BUILD_VECTOR, Res, TmpVec);
}

MachineInstrBuilder
MachineIRBuilder::buildBuildVectorConstant(const DstOp &Res,
                                           ArrayRef<APInt> Ops) {
  SmallVector<SrcOp> TmpVec;
  TmpVec.reserve(Ops.size());
  LLT EltTy = Res.getLLTTy(*getMRI()).getElementType();
  for (const auto &Op : Ops)
    TmpVec.push_back(buildConstant(EltTy, Op));
  return buildInstr(TargetOpcode::G_BUILD_VECTOR, Res, TmpVec);
}

MachineInstrBuilder MachineIRBuilder::buildSplatVector(const DstOp &Res,
                                                       const SrcOp &Src) {
  SmallVector<SrcOp, 8> TmpVec(Res.getLLTTy(*getMRI()).getNumElements(), Src);
  return buildInstr(TargetOpcode::G_BUILD_VECTOR, Res, TmpVec);
}

MachineInstrBuilder
MachineIRBuilder::buildBuildVectorTrunc(const DstOp &Res,
                                        ArrayRef<Register> Ops) {
  // Unfortunately to convert from ArrayRef<Register> to ArrayRef<SrcOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<SrcOp, 8> TmpVec(Ops.begin(), Ops.end());
  return buildInstr(TargetOpcode::G_BUILD_VECTOR_TRUNC, Res, TmpVec);
}

MachineInstrBuilder MachineIRBuilder::buildShuffleSplat(const DstOp &Res,
                                                        const SrcOp &Src) {
  LLT DstTy = Res.getLLTTy(*getMRI());
  assert(Src.getLLTTy(*getMRI()) == DstTy.getElementType() &&
         "Expected Src to match Dst elt ty");
  auto UndefVec = buildUndef(DstTy);
  auto Zero = buildConstant(LLT::scalar(64), 0);
  auto InsElt = buildInsertVectorElement(DstTy, UndefVec, Src, Zero);
  SmallVector<int, 16> ZeroMask(DstTy.getNumElements());
  return buildShuffleVector(DstTy, InsElt, UndefVec, ZeroMask);
}

MachineInstrBuilder MachineIRBuilder::buildShuffleVector(const DstOp &Res,
                                                         const SrcOp &Src1,
                                                         const SrcOp &Src2,
                                                         ArrayRef<int> Mask) {
  LLT DstTy = Res.getLLTTy(*getMRI());
  LLT Src1Ty = Src1.getLLTTy(*getMRI());
  LLT Src2Ty = Src2.getLLTTy(*getMRI());
  assert((size_t)(Src1Ty.getNumElements() + Src2Ty.getNumElements()) >=
         Mask.size());
  assert(DstTy.getElementType() == Src1Ty.getElementType() &&
         DstTy.getElementType() == Src2Ty.getElementType());
  (void)DstTy;
  (void)Src1Ty;
  (void)Src2Ty;
  ArrayRef<int> MaskAlloc = getMF().allocateShuffleMask(Mask);
  return buildInstr(TargetOpcode::G_SHUFFLE_VECTOR, {Res}, {Src1, Src2})
      .addShuffleMask(MaskAlloc);
}

MachineInstrBuilder
MachineIRBuilder::buildConcatVectors(const DstOp &Res, ArrayRef<Register> Ops) {
  // Unfortunately to convert from ArrayRef<Register> to ArrayRef<SrcOp>,
  // we need some temporary storage for the DstOp objects. Here we use a
  // sufficiently large SmallVector to not go through the heap.
  SmallVector<SrcOp, 8> TmpVec(Ops.begin(), Ops.end());
  return buildInstr(TargetOpcode::G_CONCAT_VECTORS, Res, TmpVec);
}

MachineInstrBuilder MachineIRBuilder::buildInsert(const DstOp &Res,
                                                  const SrcOp &Src,
                                                  const SrcOp &Op,
                                                  unsigned Index) {
  assert(Index + Op.getLLTTy(*getMRI()).getSizeInBits() <=
             Res.getLLTTy(*getMRI()).getSizeInBits() &&
         "insertion past the end of a register");

  if (Res.getLLTTy(*getMRI()).getSizeInBits() ==
      Op.getLLTTy(*getMRI()).getSizeInBits()) {
    return buildCast(Res, Op);
  }

  return buildInstr(TargetOpcode::G_INSERT, Res, {Src, Op, uint64_t(Index)});
}

MachineInstrBuilder MachineIRBuilder::buildIntrinsic(Intrinsic::ID ID,
                                                     ArrayRef<Register> ResultRegs,
                                                     bool HasSideEffects) {
  auto MIB =
      buildInstr(HasSideEffects ? TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS
                                : TargetOpcode::G_INTRINSIC);
  for (unsigned ResultReg : ResultRegs)
    MIB.addDef(ResultReg);
  MIB.addIntrinsicID(ID);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildIntrinsic(Intrinsic::ID ID,
                                                     ArrayRef<DstOp> Results,
                                                     bool HasSideEffects) {
  auto MIB =
      buildInstr(HasSideEffects ? TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS
                                : TargetOpcode::G_INTRINSIC);
  for (DstOp Result : Results)
    Result.addDefToMIB(*getMRI(), MIB);
  MIB.addIntrinsicID(ID);
  return MIB;
}

MachineInstrBuilder MachineIRBuilder::buildTrunc(const DstOp &Res,
                                                 const SrcOp &Op) {
  return buildInstr(TargetOpcode::G_TRUNC, Res, Op);
}

MachineInstrBuilder MachineIRBuilder::buildFPTrunc(const DstOp &Res,
                                                   const SrcOp &Op,
                                                   Optional<unsigned> Flags) {
  return buildInstr(TargetOpcode::G_FPTRUNC, Res, Op, Flags);
}

MachineInstrBuilder MachineIRBuilder::buildICmp(CmpInst::Predicate Pred,
                                                const DstOp &Res,
                                                const SrcOp &Op0,
                                                const SrcOp &Op1) {
  return buildInstr(TargetOpcode::G_ICMP, Res, {Pred, Op0, Op1});
}

MachineInstrBuilder MachineIRBuilder::buildFCmp(CmpInst::Predicate Pred,
                                                const DstOp &Res,
                                                const SrcOp &Op0,
                                                const SrcOp &Op1,
                                                Optional<unsigned> Flags) {

  return buildInstr(TargetOpcode::G_FCMP, Res, {Pred, Op0, Op1}, Flags);
}

MachineInstrBuilder MachineIRBuilder::buildSelect(const DstOp &Res,
                                                  const SrcOp &Tst,
                                                  const SrcOp &Op0,
                                                  const SrcOp &Op1,
                                                  Optional<unsigned> Flags) {

  return buildInstr(TargetOpcode::G_SELECT, {Res}, {Tst, Op0, Op1}, Flags);
}

MachineInstrBuilder
MachineIRBuilder::buildInsertVectorElement(const DstOp &Res, const SrcOp &Val,
                                           const SrcOp &Elt, const SrcOp &Idx) {
  return buildInstr(TargetOpcode::G_INSERT_VECTOR_ELT, Res, {Val, Elt, Idx});
}

MachineInstrBuilder
MachineIRBuilder::buildExtractVectorElement(const DstOp &Res, const SrcOp &Val,
                                            const SrcOp &Idx) {
  return buildInstr(TargetOpcode::G_EXTRACT_VECTOR_ELT, Res, {Val, Idx});
}

MachineInstrBuilder MachineIRBuilder::buildAtomicCmpXchgWithSuccess(
    Register OldValRes, Register SuccessRes, Register Addr, Register CmpVal,
    Register NewVal, MachineMemOperand &MMO) {
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
MachineIRBuilder::buildAtomicCmpXchg(Register OldValRes, Register Addr,
                                     Register CmpVal, Register NewVal,
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

MachineInstrBuilder MachineIRBuilder::buildAtomicRMW(
  unsigned Opcode, const DstOp &OldValRes,
  const SrcOp &Addr, const SrcOp &Val,
  MachineMemOperand &MMO) {

#ifndef NDEBUG
  LLT OldValResTy = OldValRes.getLLTTy(*getMRI());
  LLT AddrTy = Addr.getLLTTy(*getMRI());
  LLT ValTy = Val.getLLTTy(*getMRI());
  assert(OldValResTy.isScalar() && "invalid operand type");
  assert(AddrTy.isPointer() && "invalid operand type");
  assert(ValTy.isValid() && "invalid operand type");
  assert(OldValResTy == ValTy && "type mismatch");
  assert(MMO.isAtomic() && "not atomic mem operand");
#endif

  auto MIB = buildInstr(Opcode);
  OldValRes.addDefToMIB(*getMRI(), MIB);
  Addr.addSrcToMIB(MIB);
  Val.addSrcToMIB(MIB);
  MIB.addMemOperand(&MMO);
  return MIB;
}

MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWXchg(Register OldValRes, Register Addr,
                                     Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_XCHG, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWAdd(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_ADD, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWSub(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_SUB, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWAnd(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_AND, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWNand(Register OldValRes, Register Addr,
                                     Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_NAND, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder MachineIRBuilder::buildAtomicRMWOr(Register OldValRes,
                                                       Register Addr,
                                                       Register Val,
                                                       MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_OR, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWXor(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_XOR, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWMax(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_MAX, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWMin(Register OldValRes, Register Addr,
                                    Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_MIN, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWUmax(Register OldValRes, Register Addr,
                                     Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_UMAX, OldValRes, Addr, Val,
                        MMO);
}
MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWUmin(Register OldValRes, Register Addr,
                                     Register Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_UMIN, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWFAdd(
  const DstOp &OldValRes, const SrcOp &Addr, const SrcOp &Val,
  MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_FADD, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWFSub(const DstOp &OldValRes, const SrcOp &Addr, const SrcOp &Val,
                                     MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_FSUB, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWFMax(const DstOp &OldValRes, const SrcOp &Addr,
                                     const SrcOp &Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_FMAX, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilder::buildAtomicRMWFMin(const DstOp &OldValRes, const SrcOp &Addr,
                                     const SrcOp &Val, MachineMemOperand &MMO) {
  return buildAtomicRMW(TargetOpcode::G_ATOMICRMW_FMIN, OldValRes, Addr, Val,
                        MMO);
}

MachineInstrBuilder
MachineIRBuilder::buildFence(unsigned Ordering, unsigned Scope) {
  return buildInstr(TargetOpcode::G_FENCE)
    .addImm(Ordering)
    .addImm(Scope);
}

MachineInstrBuilder
MachineIRBuilder::buildBlockAddress(Register Res, const BlockAddress *BA) {
#ifndef NDEBUG
  assert(getMRI()->getType(Res).isPointer() && "invalid res type");
#endif

  return buildInstr(TargetOpcode::G_BLOCK_ADDR).addDef(Res).addBlockAddress(BA);
}

void MachineIRBuilder::validateTruncExt(const LLT DstTy, const LLT SrcTy,
                                        bool IsExtend) {
#ifndef NDEBUG
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

void MachineIRBuilder::validateSelectOp(const LLT ResTy, const LLT TstTy,
                                        const LLT Op0Ty, const LLT Op1Ty) {
#ifndef NDEBUG
  assert((ResTy.isScalar() || ResTy.isVector() || ResTy.isPointer()) &&
         "invalid operand type");
  assert((ResTy == Op0Ty && ResTy == Op1Ty) && "type mismatch");
  if (ResTy.isScalar() || ResTy.isPointer())
    assert(TstTy.isScalar() && "type mismatch");
  else
    assert((TstTy.isScalar() ||
            (TstTy.isVector() &&
             TstTy.getNumElements() == Op0Ty.getNumElements())) &&
           "type mismatch");
#endif
}

MachineInstrBuilder MachineIRBuilder::buildInstr(unsigned Opc,
                                                 ArrayRef<DstOp> DstOps,
                                                 ArrayRef<SrcOp> SrcOps,
                                                 Optional<unsigned> Flags) {
  switch (Opc) {
  default:
    break;
  case TargetOpcode::G_SELECT: {
    assert(DstOps.size() == 1 && "Invalid select");
    assert(SrcOps.size() == 3 && "Invalid select");
    validateSelectOp(
        DstOps[0].getLLTTy(*getMRI()), SrcOps[0].getLLTTy(*getMRI()),
        SrcOps[1].getLLTTy(*getMRI()), SrcOps[2].getLLTTy(*getMRI()));
    break;
  }
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_ABS:
    // All these are unary ops.
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 1 && "Invalid Srcs");
    validateUnaryOp(DstOps[0].getLLTTy(*getMRI()),
                    SrcOps[0].getLLTTy(*getMRI()));
    break;
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_UADDSAT:
  case TargetOpcode::G_SADDSAT:
  case TargetOpcode::G_USUBSAT:
  case TargetOpcode::G_SSUBSAT: {
    // All these are binary ops.
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 2 && "Invalid Srcs");
    validateBinaryOp(DstOps[0].getLLTTy(*getMRI()),
                     SrcOps[0].getLLTTy(*getMRI()),
                     SrcOps[1].getLLTTy(*getMRI()));
    break;
  }
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_USHLSAT:
  case TargetOpcode::G_SSHLSAT: {
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 2 && "Invalid Srcs");
    validateShiftOp(DstOps[0].getLLTTy(*getMRI()),
                    SrcOps[0].getLLTTy(*getMRI()),
                    SrcOps[1].getLLTTy(*getMRI()));
    break;
  }
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_ANYEXT:
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 1 && "Invalid Srcs");
    validateTruncExt(DstOps[0].getLLTTy(*getMRI()),
                     SrcOps[0].getLLTTy(*getMRI()), true);
    break;
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::G_FPTRUNC: {
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 1 && "Invalid Srcs");
    validateTruncExt(DstOps[0].getLLTTy(*getMRI()),
                     SrcOps[0].getLLTTy(*getMRI()), false);
    break;
  }
  case TargetOpcode::G_BITCAST: {
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(SrcOps.size() == 1 && "Invalid Srcs");
    assert(DstOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
           SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() && "invalid bitcast");
    break;
  }
  case TargetOpcode::COPY:
    assert(DstOps.size() == 1 && "Invalid Dst");
    // If the caller wants to add a subreg source it has to be done separately
    // so we may not have any SrcOps at this point yet.
    break;
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_ICMP: {
    assert(DstOps.size() == 1 && "Invalid Dst Operands");
    assert(SrcOps.size() == 3 && "Invalid Src Operands");
    // For F/ICMP, the first src operand is the predicate, followed by
    // the two comparands.
    assert(SrcOps[0].getSrcOpKind() == SrcOp::SrcType::Ty_Predicate &&
           "Expecting predicate");
    assert([&]() -> bool {
      CmpInst::Predicate Pred = SrcOps[0].getPredicate();
      return Opc == TargetOpcode::G_ICMP ? CmpInst::isIntPredicate(Pred)
                                         : CmpInst::isFPPredicate(Pred);
    }() && "Invalid predicate");
    assert(SrcOps[1].getLLTTy(*getMRI()) == SrcOps[2].getLLTTy(*getMRI()) &&
           "Type mismatch");
    assert([&]() -> bool {
      LLT Op0Ty = SrcOps[1].getLLTTy(*getMRI());
      LLT DstTy = DstOps[0].getLLTTy(*getMRI());
      if (Op0Ty.isScalar() || Op0Ty.isPointer())
        return DstTy.isScalar();
      else
        return DstTy.isVector() &&
               DstTy.getNumElements() == Op0Ty.getNumElements();
    }() && "Type Mismatch");
    break;
  }
  case TargetOpcode::G_UNMERGE_VALUES: {
    assert(!DstOps.empty() && "Invalid trivial sequence");
    assert(SrcOps.size() == 1 && "Invalid src for Unmerge");
    assert(llvm::all_of(DstOps,
                        [&, this](const DstOp &Op) {
                          return Op.getLLTTy(*getMRI()) ==
                                 DstOps[0].getLLTTy(*getMRI());
                        }) &&
           "type mismatch in output list");
    assert((TypeSize::ScalarTy)DstOps.size() *
                   DstOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
               SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() &&
           "input operands do not cover output register");
    break;
  }
  case TargetOpcode::G_MERGE_VALUES: {
    assert(!SrcOps.empty() && "invalid trivial sequence");
    assert(DstOps.size() == 1 && "Invalid Dst");
    assert(llvm::all_of(SrcOps,
                        [&, this](const SrcOp &Op) {
                          return Op.getLLTTy(*getMRI()) ==
                                 SrcOps[0].getLLTTy(*getMRI());
                        }) &&
           "type mismatch in input list");
    assert((TypeSize::ScalarTy)SrcOps.size() *
                   SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
               DstOps[0].getLLTTy(*getMRI()).getSizeInBits() &&
           "input operands do not cover output register");
    if (SrcOps.size() == 1)
      return buildCast(DstOps[0], SrcOps[0]);
    if (DstOps[0].getLLTTy(*getMRI()).isVector()) {
      if (SrcOps[0].getLLTTy(*getMRI()).isVector())
        return buildInstr(TargetOpcode::G_CONCAT_VECTORS, DstOps, SrcOps);
      return buildInstr(TargetOpcode::G_BUILD_VECTOR, DstOps, SrcOps);
    }
    break;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    assert(DstOps.size() == 1 && "Invalid Dst size");
    assert(SrcOps.size() == 2 && "Invalid Src size");
    assert(SrcOps[0].getLLTTy(*getMRI()).isVector() && "Invalid operand type");
    assert((DstOps[0].getLLTTy(*getMRI()).isScalar() ||
            DstOps[0].getLLTTy(*getMRI()).isPointer()) &&
           "Invalid operand type");
    assert(SrcOps[1].getLLTTy(*getMRI()).isScalar() && "Invalid operand type");
    assert(SrcOps[0].getLLTTy(*getMRI()).getElementType() ==
               DstOps[0].getLLTTy(*getMRI()) &&
           "Type mismatch");
    break;
  }
  case TargetOpcode::G_INSERT_VECTOR_ELT: {
    assert(DstOps.size() == 1 && "Invalid dst size");
    assert(SrcOps.size() == 3 && "Invalid src size");
    assert(DstOps[0].getLLTTy(*getMRI()).isVector() &&
           SrcOps[0].getLLTTy(*getMRI()).isVector() && "Invalid operand type");
    assert(DstOps[0].getLLTTy(*getMRI()).getElementType() ==
               SrcOps[1].getLLTTy(*getMRI()) &&
           "Type mismatch");
    assert(SrcOps[2].getLLTTy(*getMRI()).isScalar() && "Invalid index");
    assert(DstOps[0].getLLTTy(*getMRI()).getNumElements() ==
               SrcOps[0].getLLTTy(*getMRI()).getNumElements() &&
           "Type mismatch");
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    assert((!SrcOps.empty() || SrcOps.size() < 2) &&
           "Must have at least 2 operands");
    assert(DstOps.size() == 1 && "Invalid DstOps");
    assert(DstOps[0].getLLTTy(*getMRI()).isVector() &&
           "Res type must be a vector");
    assert(llvm::all_of(SrcOps,
                        [&, this](const SrcOp &Op) {
                          return Op.getLLTTy(*getMRI()) ==
                                 SrcOps[0].getLLTTy(*getMRI());
                        }) &&
           "type mismatch in input list");
    assert((TypeSize::ScalarTy)SrcOps.size() *
                   SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
               DstOps[0].getLLTTy(*getMRI()).getSizeInBits() &&
           "input scalars do not exactly cover the output vector register");
    break;
  }
  case TargetOpcode::G_BUILD_VECTOR_TRUNC: {
    assert((!SrcOps.empty() || SrcOps.size() < 2) &&
           "Must have at least 2 operands");
    assert(DstOps.size() == 1 && "Invalid DstOps");
    assert(DstOps[0].getLLTTy(*getMRI()).isVector() &&
           "Res type must be a vector");
    assert(llvm::all_of(SrcOps,
                        [&, this](const SrcOp &Op) {
                          return Op.getLLTTy(*getMRI()) ==
                                 SrcOps[0].getLLTTy(*getMRI());
                        }) &&
           "type mismatch in input list");
    if (SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
        DstOps[0].getLLTTy(*getMRI()).getElementType().getSizeInBits())
      return buildInstr(TargetOpcode::G_BUILD_VECTOR, DstOps, SrcOps);
    break;
  }
  case TargetOpcode::G_CONCAT_VECTORS: {
    assert(DstOps.size() == 1 && "Invalid DstOps");
    assert((!SrcOps.empty() || SrcOps.size() < 2) &&
           "Must have at least 2 operands");
    assert(llvm::all_of(SrcOps,
                        [&, this](const SrcOp &Op) {
                          return (Op.getLLTTy(*getMRI()).isVector() &&
                                  Op.getLLTTy(*getMRI()) ==
                                      SrcOps[0].getLLTTy(*getMRI()));
                        }) &&
           "type mismatch in input list");
    assert((TypeSize::ScalarTy)SrcOps.size() *
                   SrcOps[0].getLLTTy(*getMRI()).getSizeInBits() ==
               DstOps[0].getLLTTy(*getMRI()).getSizeInBits() &&
           "input vectors do not exactly cover the output vector register");
    break;
  }
  case TargetOpcode::G_UADDE: {
    assert(DstOps.size() == 2 && "Invalid no of dst operands");
    assert(SrcOps.size() == 3 && "Invalid no of src operands");
    assert(DstOps[0].getLLTTy(*getMRI()).isScalar() && "Invalid operand");
    assert((DstOps[0].getLLTTy(*getMRI()) == SrcOps[0].getLLTTy(*getMRI())) &&
           (DstOps[0].getLLTTy(*getMRI()) == SrcOps[1].getLLTTy(*getMRI())) &&
           "Invalid operand");
    assert(DstOps[1].getLLTTy(*getMRI()).isScalar() && "Invalid operand");
    assert(DstOps[1].getLLTTy(*getMRI()) == SrcOps[2].getLLTTy(*getMRI()) &&
           "type mismatch");
    break;
  }
  }

  auto MIB = buildInstr(Opc);
  for (const DstOp &Op : DstOps)
    Op.addDefToMIB(*getMRI(), MIB);
  for (const SrcOp &Op : SrcOps)
    Op.addSrcToMIB(MIB);
  if (Flags)
    MIB->setFlags(*Flags);
  return MIB;
}
