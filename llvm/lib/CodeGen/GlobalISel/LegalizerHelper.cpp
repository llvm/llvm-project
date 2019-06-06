//===-- llvm/CodeGen/GlobalISel/LegalizerHelper.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizerHelper class to legalize
/// individual instructions and the LegalizeMachineIR wrapper pass for the
/// primary legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "legalizer"

using namespace llvm;
using namespace LegalizeActions;

/// Try to break down \p OrigTy into \p NarrowTy sized pieces.
///
/// Returns the number of \p NarrowTy elements needed to reconstruct \p OrigTy,
/// with any leftover piece as type \p LeftoverTy
///
/// Returns -1 in the first element of the pair if the breakdown is not
/// satisfiable.
static std::pair<int, int>
getNarrowTypeBreakDown(LLT OrigTy, LLT NarrowTy, LLT &LeftoverTy) {
  assert(!LeftoverTy.isValid() && "this is an out argument");

  unsigned Size = OrigTy.getSizeInBits();
  unsigned NarrowSize = NarrowTy.getSizeInBits();
  unsigned NumParts = Size / NarrowSize;
  unsigned LeftoverSize = Size - NumParts * NarrowSize;
  assert(Size > NarrowSize);

  if (LeftoverSize == 0)
    return {NumParts, 0};

  if (NarrowTy.isVector()) {
    unsigned EltSize = OrigTy.getScalarSizeInBits();
    if (LeftoverSize % EltSize != 0)
      return {-1, -1};
    LeftoverTy = LLT::scalarOrVector(LeftoverSize / EltSize, EltSize);
  } else {
    LeftoverTy = LLT::scalar(LeftoverSize);
  }

  int NumLeftover = LeftoverSize / LeftoverTy.getSizeInBits();
  return std::make_pair(NumParts, NumLeftover);
}

LegalizerHelper::LegalizerHelper(MachineFunction &MF,
                                 GISelChangeObserver &Observer,
                                 MachineIRBuilder &Builder)
    : MIRBuilder(Builder), MRI(MF.getRegInfo()),
      LI(*MF.getSubtarget().getLegalizerInfo()), Observer(Observer) {
  MIRBuilder.setMF(MF);
  MIRBuilder.setChangeObserver(Observer);
}

LegalizerHelper::LegalizerHelper(MachineFunction &MF, const LegalizerInfo &LI,
                                 GISelChangeObserver &Observer,
                                 MachineIRBuilder &B)
    : MIRBuilder(B), MRI(MF.getRegInfo()), LI(LI), Observer(Observer) {
  MIRBuilder.setMF(MF);
  MIRBuilder.setChangeObserver(Observer);
}
LegalizerHelper::LegalizeResult
LegalizerHelper::legalizeInstrStep(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "Legalizing: "; MI.print(dbgs()));

  auto Step = LI.getAction(MI, MRI);
  switch (Step.Action) {
  case Legal:
    LLVM_DEBUG(dbgs() << ".. Already legal\n");
    return AlreadyLegal;
  case Libcall:
    LLVM_DEBUG(dbgs() << ".. Convert to libcall\n");
    return libcall(MI);
  case NarrowScalar:
    LLVM_DEBUG(dbgs() << ".. Narrow scalar\n");
    return narrowScalar(MI, Step.TypeIdx, Step.NewType);
  case WidenScalar:
    LLVM_DEBUG(dbgs() << ".. Widen scalar\n");
    return widenScalar(MI, Step.TypeIdx, Step.NewType);
  case Lower:
    LLVM_DEBUG(dbgs() << ".. Lower\n");
    return lower(MI, Step.TypeIdx, Step.NewType);
  case FewerElements:
    LLVM_DEBUG(dbgs() << ".. Reduce number of elements\n");
    return fewerElementsVector(MI, Step.TypeIdx, Step.NewType);
  case MoreElements:
    LLVM_DEBUG(dbgs() << ".. Increase number of elements\n");
    return moreElementsVector(MI, Step.TypeIdx, Step.NewType);
  case Custom:
    LLVM_DEBUG(dbgs() << ".. Custom legalization\n");
    return LI.legalizeCustom(MI, MRI, MIRBuilder, Observer) ? Legalized
                                                            : UnableToLegalize;
  default:
    LLVM_DEBUG(dbgs() << ".. Unable to legalize\n");
    return UnableToLegalize;
  }
}

void LegalizerHelper::extractParts(unsigned Reg, LLT Ty, int NumParts,
                                   SmallVectorImpl<unsigned> &VRegs) {
  for (int i = 0; i < NumParts; ++i)
    VRegs.push_back(MRI.createGenericVirtualRegister(Ty));
  MIRBuilder.buildUnmerge(VRegs, Reg);
}

bool LegalizerHelper::extractParts(unsigned Reg, LLT RegTy,
                                   LLT MainTy, LLT &LeftoverTy,
                                   SmallVectorImpl<unsigned> &VRegs,
                                   SmallVectorImpl<unsigned> &LeftoverRegs) {
  assert(!LeftoverTy.isValid() && "this is an out argument");

  unsigned RegSize = RegTy.getSizeInBits();
  unsigned MainSize = MainTy.getSizeInBits();
  unsigned NumParts = RegSize / MainSize;
  unsigned LeftoverSize = RegSize - NumParts * MainSize;

  // Use an unmerge when possible.
  if (LeftoverSize == 0) {
    for (unsigned I = 0; I < NumParts; ++I)
      VRegs.push_back(MRI.createGenericVirtualRegister(MainTy));
    MIRBuilder.buildUnmerge(VRegs, Reg);
    return true;
  }

  if (MainTy.isVector()) {
    unsigned EltSize = MainTy.getScalarSizeInBits();
    if (LeftoverSize % EltSize != 0)
      return false;
    LeftoverTy = LLT::scalarOrVector(LeftoverSize / EltSize, EltSize);
  } else {
    LeftoverTy = LLT::scalar(LeftoverSize);
  }

  // For irregular sizes, extract the individual parts.
  for (unsigned I = 0; I != NumParts; ++I) {
    unsigned NewReg = MRI.createGenericVirtualRegister(MainTy);
    VRegs.push_back(NewReg);
    MIRBuilder.buildExtract(NewReg, Reg, MainSize * I);
  }

  for (unsigned Offset = MainSize * NumParts; Offset < RegSize;
       Offset += LeftoverSize) {
    unsigned NewReg = MRI.createGenericVirtualRegister(LeftoverTy);
    LeftoverRegs.push_back(NewReg);
    MIRBuilder.buildExtract(NewReg, Reg, Offset);
  }

  return true;
}

void LegalizerHelper::insertParts(unsigned DstReg,
                                  LLT ResultTy, LLT PartTy,
                                  ArrayRef<unsigned> PartRegs,
                                  LLT LeftoverTy,
                                  ArrayRef<unsigned> LeftoverRegs) {
  if (!LeftoverTy.isValid()) {
    assert(LeftoverRegs.empty());

    if (!ResultTy.isVector()) {
      MIRBuilder.buildMerge(DstReg, PartRegs);
      return;
    }

    if (PartTy.isVector())
      MIRBuilder.buildConcatVectors(DstReg, PartRegs);
    else
      MIRBuilder.buildBuildVector(DstReg, PartRegs);
    return;
  }

  unsigned PartSize = PartTy.getSizeInBits();
  unsigned LeftoverPartSize = LeftoverTy.getSizeInBits();

  unsigned CurResultReg = MRI.createGenericVirtualRegister(ResultTy);
  MIRBuilder.buildUndef(CurResultReg);

  unsigned Offset = 0;
  for (unsigned PartReg : PartRegs) {
    unsigned NewResultReg = MRI.createGenericVirtualRegister(ResultTy);
    MIRBuilder.buildInsert(NewResultReg, CurResultReg, PartReg, Offset);
    CurResultReg = NewResultReg;
    Offset += PartSize;
  }

  for (unsigned I = 0, E = LeftoverRegs.size(); I != E; ++I) {
    // Use the original output register for the final insert to avoid a copy.
    unsigned NewResultReg = (I + 1 == E) ?
      DstReg : MRI.createGenericVirtualRegister(ResultTy);

    MIRBuilder.buildInsert(NewResultReg, CurResultReg, LeftoverRegs[I], Offset);
    CurResultReg = NewResultReg;
    Offset += LeftoverPartSize;
  }
}

static RTLIB::Libcall getRTLibDesc(unsigned Opcode, unsigned Size) {
  switch (Opcode) {
  case TargetOpcode::G_SDIV:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::SDIV_I64 : RTLIB::SDIV_I32;
  case TargetOpcode::G_UDIV:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::UDIV_I64 : RTLIB::UDIV_I32;
  case TargetOpcode::G_SREM:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::SREM_I64 : RTLIB::SREM_I32;
  case TargetOpcode::G_UREM:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::UREM_I64 : RTLIB::UREM_I32;
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
    assert(Size == 32 && "Unsupported size");
    return RTLIB::CTLZ_I32;
  case TargetOpcode::G_FADD:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::ADD_F64 : RTLIB::ADD_F32;
  case TargetOpcode::G_FSUB:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::SUB_F64 : RTLIB::SUB_F32;
  case TargetOpcode::G_FMUL:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::MUL_F64 : RTLIB::MUL_F32;
  case TargetOpcode::G_FDIV:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::DIV_F64 : RTLIB::DIV_F32;
  case TargetOpcode::G_FEXP:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::EXP_F64 : RTLIB::EXP_F32;
  case TargetOpcode::G_FEXP2:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::EXP2_F64 : RTLIB::EXP2_F32;
  case TargetOpcode::G_FREM:
    return Size == 64 ? RTLIB::REM_F64 : RTLIB::REM_F32;
  case TargetOpcode::G_FPOW:
    return Size == 64 ? RTLIB::POW_F64 : RTLIB::POW_F32;
  case TargetOpcode::G_FMA:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::FMA_F64 : RTLIB::FMA_F32;
  case TargetOpcode::G_FSIN:
    assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");
    return Size == 128 ? RTLIB::SIN_F128
                       : Size == 64 ? RTLIB::SIN_F64 : RTLIB::SIN_F32;
  case TargetOpcode::G_FCOS:
    assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");
    return Size == 128 ? RTLIB::COS_F128
                       : Size == 64 ? RTLIB::COS_F64 : RTLIB::COS_F32;
  case TargetOpcode::G_FLOG10:
    assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");
    return Size == 128 ? RTLIB::LOG10_F128
                       : Size == 64 ? RTLIB::LOG10_F64 : RTLIB::LOG10_F32;
  case TargetOpcode::G_FLOG:
    assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");
    return Size == 128 ? RTLIB::LOG_F128
                       : Size == 64 ? RTLIB::LOG_F64 : RTLIB::LOG_F32;
  case TargetOpcode::G_FLOG2:
    assert((Size == 32 || Size == 64 || Size == 128) && "Unsupported size");
    return Size == 128 ? RTLIB::LOG2_F128
                       : Size == 64 ? RTLIB::LOG2_F64 : RTLIB::LOG2_F32;
  case TargetOpcode::G_FCEIL:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::CEIL_F64 : RTLIB::CEIL_F32;
  case TargetOpcode::G_FFLOOR:
    assert((Size == 32 || Size == 64) && "Unsupported size");
    return Size == 64 ? RTLIB::FLOOR_F64 : RTLIB::FLOOR_F32;
  }
  llvm_unreachable("Unknown libcall function");
}

LegalizerHelper::LegalizeResult
llvm::createLibcall(MachineIRBuilder &MIRBuilder, RTLIB::Libcall Libcall,
                    const CallLowering::ArgInfo &Result,
                    ArrayRef<CallLowering::ArgInfo> Args) {
  auto &CLI = *MIRBuilder.getMF().getSubtarget().getCallLowering();
  auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
  const char *Name = TLI.getLibcallName(Libcall);

  MIRBuilder.getMF().getFrameInfo().setHasCalls(true);
  if (!CLI.lowerCall(MIRBuilder, TLI.getLibcallCallingConv(Libcall),
                     MachineOperand::CreateES(Name), Result, Args))
    return LegalizerHelper::UnableToLegalize;

  return LegalizerHelper::Legalized;
}

// Useful for libcalls where all operands have the same type.
static LegalizerHelper::LegalizeResult
simpleLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, unsigned Size,
              Type *OpType) {
  auto Libcall = getRTLibDesc(MI.getOpcode(), Size);

  SmallVector<CallLowering::ArgInfo, 3> Args;
  for (unsigned i = 1; i < MI.getNumOperands(); i++)
    Args.push_back({MI.getOperand(i).getReg(), OpType});
  return createLibcall(MIRBuilder, Libcall, {MI.getOperand(0).getReg(), OpType},
                       Args);
}

static RTLIB::Libcall getConvRTLibDesc(unsigned Opcode, Type *ToType,
                                       Type *FromType) {
  auto ToMVT = MVT::getVT(ToType);
  auto FromMVT = MVT::getVT(FromType);

  switch (Opcode) {
  case TargetOpcode::G_FPEXT:
    return RTLIB::getFPEXT(FromMVT, ToMVT);
  case TargetOpcode::G_FPTRUNC:
    return RTLIB::getFPROUND(FromMVT, ToMVT);
  case TargetOpcode::G_FPTOSI:
    return RTLIB::getFPTOSINT(FromMVT, ToMVT);
  case TargetOpcode::G_FPTOUI:
    return RTLIB::getFPTOUINT(FromMVT, ToMVT);
  case TargetOpcode::G_SITOFP:
    return RTLIB::getSINTTOFP(FromMVT, ToMVT);
  case TargetOpcode::G_UITOFP:
    return RTLIB::getUINTTOFP(FromMVT, ToMVT);
  }
  llvm_unreachable("Unsupported libcall function");
}

static LegalizerHelper::LegalizeResult
conversionLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, Type *ToType,
                  Type *FromType) {
  RTLIB::Libcall Libcall = getConvRTLibDesc(MI.getOpcode(), ToType, FromType);
  return createLibcall(MIRBuilder, Libcall, {MI.getOperand(0).getReg(), ToType},
                       {{MI.getOperand(1).getReg(), FromType}});
}

LegalizerHelper::LegalizeResult
LegalizerHelper::libcall(MachineInstr &MI) {
  LLT LLTy = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = LLTy.getSizeInBits();
  auto &Ctx = MIRBuilder.getMF().getFunction().getContext();

  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF: {
    Type *HLTy = IntegerType::get(Ctx, Size);
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR: {
    if (Size > 64) {
      LLVM_DEBUG(dbgs() << "Size " << Size << " too large to legalize.\n");
      return UnableToLegalize;
    }
    Type *HLTy = Size == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx);
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FPEXT: {
    // FIXME: Support other floating point types (half, fp128 etc)
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (ToSize != 64 || FromSize != 32)
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder, Type::getDoubleTy(Ctx), Type::getFloatTy(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FPTRUNC: {
    // FIXME: Support other floating point types (half, fp128 etc)
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (ToSize != 32 || FromSize != 64)
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder, Type::getFloatTy(Ctx), Type::getDoubleTy(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI: {
    // FIXME: Support other types
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (ToSize != 32 || (FromSize != 32 && FromSize != 64))
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder, Type::getInt32Ty(Ctx),
        FromSize == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP: {
    // FIXME: Support other types
    unsigned FromSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned ToSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (FromSize != 32 || (ToSize != 32 && ToSize != 64))
      return UnableToLegalize;
    LegalizeResult Status = conversionLibcall(
        MI, MIRBuilder,
        ToSize == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx),
        Type::getInt32Ty(Ctx));
    if (Status != Legalized)
      return Status;
    break;
  }
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult LegalizerHelper::narrowScalar(MachineInstr &MI,
                                                              unsigned TypeIdx,
                                                              LLT NarrowTy) {
  MIRBuilder.setInstr(MI);

  uint64_t SizeOp0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_IMPLICIT_DEF: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;

    SmallVector<unsigned, 2> DstRegs;
    for (int i = 0; i < NumParts; ++i)
      DstRegs.push_back(
          MIRBuilder.buildUndef(NarrowTy)->getOperand(0).getReg());

    unsigned DstReg = MI.getOperand(0).getReg();
    if(MRI.getType(DstReg).isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    const APInt &Val = MI.getOperand(1).getCImm()->getValue();
    unsigned TotalSize = Ty.getSizeInBits();
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    int NumParts = TotalSize / NarrowSize;

    SmallVector<unsigned, 4> PartRegs;
    for (int I = 0; I != NumParts; ++I) {
      unsigned Offset = I * NarrowSize;
      auto K = MIRBuilder.buildConstant(NarrowTy,
                                        Val.lshr(Offset).trunc(NarrowSize));
      PartRegs.push_back(K.getReg(0));
    }

    LLT LeftoverTy;
    unsigned LeftoverBits = TotalSize - NumParts * NarrowSize;
    SmallVector<unsigned, 1> LeftoverRegs;
    if (LeftoverBits != 0) {
      LeftoverTy = LLT::scalar(LeftoverBits);
      auto K = MIRBuilder.buildConstant(
        LeftoverTy,
        Val.lshr(NumParts * NarrowSize).trunc(LeftoverBits));
      LeftoverRegs.push_back(K.getReg(0));
    }

    insertParts(MI.getOperand(0).getReg(),
                Ty, NarrowTy, PartRegs, LeftoverTy, LeftoverRegs);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_ADD: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    // Expand in terms of carry-setting/consuming G_ADDE instructions.
    int NumParts = SizeOp0 / NarrowTy.getSizeInBits();

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    unsigned CarryIn = MRI.createGenericVirtualRegister(LLT::scalar(1));
    MIRBuilder.buildConstant(CarryIn, 0);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      unsigned CarryOut = MRI.createGenericVirtualRegister(LLT::scalar(1));

      MIRBuilder.buildUAdde(DstReg, CarryOut, Src1Regs[i],
                            Src2Regs[i], CarryIn);

      DstRegs.push_back(DstReg);
      CarryIn = CarryOut;
    }
    unsigned DstReg = MI.getOperand(0).getReg();
    if(MRI.getType(DstReg).isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SUB: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowTy.getSizeInBits();

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
    unsigned BorrowOut = MRI.createGenericVirtualRegister(LLT::scalar(1));
    MIRBuilder.buildInstr(TargetOpcode::G_USUBO, {DstReg, BorrowOut},
                          {Src1Regs[0], Src2Regs[0]});
    DstRegs.push_back(DstReg);
    unsigned BorrowIn = BorrowOut;
    for (int i = 1; i < NumParts; ++i) {
      DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      BorrowOut = MRI.createGenericVirtualRegister(LLT::scalar(1));

      MIRBuilder.buildInstr(TargetOpcode::G_USUBE, {DstReg, BorrowOut},
                            {Src1Regs[i], Src2Regs[i], BorrowIn});

      DstRegs.push_back(DstReg);
      BorrowIn = BorrowOut;
    }
    MIRBuilder.buildMerge(MI.getOperand(0).getReg(), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_UMULH:
    return narrowScalarMul(MI, NarrowTy);
  case TargetOpcode::G_EXTRACT:
    return narrowScalarExtract(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_INSERT:
    return narrowScalarInsert(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_LOAD: {
    const auto &MMO = **MI.memoperands_begin();
    unsigned DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy.isVector())
      return UnableToLegalize;

    if (8 * MMO.getSize() != DstTy.getSizeInBits()) {
      unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildLoad(TmpReg, MI.getOperand(1).getReg(), MMO);
      MIRBuilder.buildAnyExt(DstReg, TmpReg);
      MI.eraseFromParent();
      return Legalized;
    }

    return reduceLoadStoreWidth(MI, TypeIdx, NarrowTy);
  }
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    bool ZExt = MI.getOpcode() == TargetOpcode::G_ZEXTLOAD;
    unsigned DstReg = MI.getOperand(0).getReg();
    unsigned PtrReg = MI.getOperand(1).getReg();

    unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
    auto &MMO = **MI.memoperands_begin();
    if (MMO.getSizeInBits() == NarrowSize) {
      MIRBuilder.buildLoad(TmpReg, PtrReg, MMO);
    } else {
      unsigned ExtLoad = ZExt ? TargetOpcode::G_ZEXTLOAD
        : TargetOpcode::G_SEXTLOAD;
      MIRBuilder.buildInstr(ExtLoad)
        .addDef(TmpReg)
        .addUse(PtrReg)
        .addMemOperand(&MMO);
    }

    if (ZExt)
      MIRBuilder.buildZExt(DstReg, TmpReg);
    else
      MIRBuilder.buildSExt(DstReg, TmpReg);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    const auto &MMO = **MI.memoperands_begin();

    unsigned SrcReg = MI.getOperand(0).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    if (SrcTy.isVector())
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowSize;
    unsigned HandledSize = NumParts * NarrowTy.getSizeInBits();
    unsigned LeftoverBits = SrcTy.getSizeInBits() - HandledSize;
    if (SrcTy.isVector() && LeftoverBits != 0)
      return UnableToLegalize;

    if (8 * MMO.getSize() != SrcTy.getSizeInBits()) {
      unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildTrunc(TmpReg, SrcReg);
      MIRBuilder.buildStore(TmpReg, MI.getOperand(1).getReg(), MMO);
      MI.eraseFromParent();
      return Legalized;
    }

    return reduceLoadStoreWidth(MI, 0, NarrowTy);
  }
  case TargetOpcode::G_SELECT:
    return narrowScalarSelect(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    // Legalize bitwise operation:
    // A = BinOp<Ty> B, C
    // into:
    // B1, ..., BN = G_UNMERGE_VALUES B
    // C1, ..., CN = G_UNMERGE_VALUES C
    // A1 = BinOp<Ty/N> B1, C2
    // ...
    // AN = BinOp<Ty/N> BN, CN
    // A = G_MERGE_VALUES A1, ..., AN
    return narrowScalarBasic(MI, TypeIdx, NarrowTy);
  }
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR:
    return narrowScalarShift(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP:
    if (TypeIdx != 0)
      return UnableToLegalize; // TODO

    Observer.changingInstr(MI);
    narrowScalarDst(MI, NarrowTy, 0, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INTTOPTR:
    if (TypeIdx != 1)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    narrowScalarSrc(MI, NarrowTy, 1);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PTRTOINT:
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    narrowScalarDst(MI, NarrowTy, 0, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }
}

void LegalizerHelper::widenScalarSrc(MachineInstr &MI, LLT WideTy,
                                     unsigned OpIdx, unsigned ExtOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  auto ExtB = MIRBuilder.buildInstr(ExtOpcode, {WideTy}, {MO.getReg()});
  MO.setReg(ExtB->getOperand(0).getReg());
}

void LegalizerHelper::narrowScalarSrc(MachineInstr &MI, LLT NarrowTy,
                                      unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  auto ExtB = MIRBuilder.buildInstr(TargetOpcode::G_TRUNC, {NarrowTy},
                                    {MO.getReg()});
  MO.setReg(ExtB->getOperand(0).getReg());
}

void LegalizerHelper::widenScalarDst(MachineInstr &MI, LLT WideTy,
                                     unsigned OpIdx, unsigned TruncOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildInstr(TruncOpcode, {MO.getReg()}, {DstExt});
  MO.setReg(DstExt);
}

void LegalizerHelper::narrowScalarDst(MachineInstr &MI, LLT NarrowTy,
                                      unsigned OpIdx, unsigned ExtOpcode) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  unsigned DstTrunc = MRI.createGenericVirtualRegister(NarrowTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildInstr(ExtOpcode, {MO.getReg()}, {DstTrunc});
  MO.setReg(DstTrunc);
}

void LegalizerHelper::moreElementsVectorDst(MachineInstr &MI, LLT WideTy,
                                            unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());
  MIRBuilder.buildExtract(MO.getReg(), DstExt, 0);
  MO.setReg(DstExt);
}

void LegalizerHelper::moreElementsVectorSrc(MachineInstr &MI, LLT MoreTy,
                                            unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);

  LLT OldTy = MRI.getType(MO.getReg());
  unsigned OldElts = OldTy.getNumElements();
  unsigned NewElts = MoreTy.getNumElements();

  unsigned NumParts = NewElts / OldElts;

  // Use concat_vectors if the result is a multiple of the number of elements.
  if (NumParts * OldElts == NewElts) {
    SmallVector<unsigned, 8> Parts;
    Parts.push_back(MO.getReg());

    unsigned ImpDef = MIRBuilder.buildUndef(OldTy).getReg(0);
    for (unsigned I = 1; I != NumParts; ++I)
      Parts.push_back(ImpDef);

    auto Concat = MIRBuilder.buildConcatVectors(MoreTy, Parts);
    MO.setReg(Concat.getReg(0));
    return;
  }

  unsigned MoreReg = MRI.createGenericVirtualRegister(MoreTy);
  unsigned ImpDef = MIRBuilder.buildUndef(MoreTy).getReg(0);
  MIRBuilder.buildInsert(MoreReg, ImpDef, MO.getReg(), 0);
  MO.setReg(MoreReg);
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarMergeValues(MachineInstr &MI, unsigned TypeIdx,
                                        LLT WideTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  unsigned DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (!DstTy.isScalar())
    return UnableToLegalize;

  unsigned NumOps = MI.getNumOperands();
  unsigned NumSrc = MI.getNumOperands() - 1;
  unsigned PartSize = DstTy.getSizeInBits() / NumSrc;

  unsigned Src1 = MI.getOperand(1).getReg();
  unsigned ResultReg = MIRBuilder.buildZExt(DstTy, Src1)->getOperand(0).getReg();

  for (unsigned I = 2; I != NumOps; ++I) {
    const unsigned Offset = (I - 1) * PartSize;

    unsigned SrcReg = MI.getOperand(I).getReg();
    assert(MRI.getType(SrcReg) == LLT::scalar(PartSize));

    auto ZextInput = MIRBuilder.buildZExt(DstTy, SrcReg);

    unsigned NextResult = I + 1 == NumOps ? DstReg :
      MRI.createGenericVirtualRegister(DstTy);

    auto ShiftAmt = MIRBuilder.buildConstant(DstTy, Offset);
    auto Shl = MIRBuilder.buildShl(DstTy, ZextInput, ShiftAmt);
    MIRBuilder.buildOr(NextResult, ResultReg, Shl);
    ResultReg = NextResult;
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarUnmergeValues(MachineInstr &MI, unsigned TypeIdx,
                                          LLT WideTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  unsigned NumDst = MI.getNumOperands() - 1;
  unsigned SrcReg = MI.getOperand(NumDst).getReg();
  LLT SrcTy = MRI.getType(SrcReg);
  if (!SrcTy.isScalar())
    return UnableToLegalize;

  unsigned Dst0Reg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst0Reg);
  if (!DstTy.isScalar())
    return UnableToLegalize;

  unsigned NewSrcSize = NumDst * WideTy.getSizeInBits();
  LLT NewSrcTy = LLT::scalar(NewSrcSize);
  unsigned SizeDiff = WideTy.getSizeInBits() - DstTy.getSizeInBits();

  auto WideSrc = MIRBuilder.buildZExt(NewSrcTy, SrcReg);

  for (unsigned I = 1; I != NumDst; ++I) {
    auto ShiftAmt = MIRBuilder.buildConstant(NewSrcTy, SizeDiff * I);
    auto Shl = MIRBuilder.buildShl(NewSrcTy, WideSrc, ShiftAmt);
    WideSrc = MIRBuilder.buildOr(NewSrcTy, WideSrc, Shl);
  }

  Observer.changingInstr(MI);

  MI.getOperand(NumDst).setReg(WideSrc->getOperand(0).getReg());
  for (unsigned I = 0; I != NumDst; ++I)
    widenScalarDst(MI, WideTy, I);

  Observer.changedInstr(MI);

  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarExtract(MachineInstr &MI, unsigned TypeIdx,
                                    LLT WideTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SrcReg = MI.getOperand(1).getReg();
  LLT SrcTy = MRI.getType(SrcReg);

  LLT DstTy = MRI.getType(DstReg);
  unsigned Offset = MI.getOperand(2).getImm();

  if (TypeIdx == 0) {
    if (SrcTy.isVector() || DstTy.isVector())
      return UnableToLegalize;

    SrcOp Src(SrcReg);
    if (SrcTy.isPointer()) {
      // Extracts from pointers can be handled only if they are really just
      // simple integers.
      const DataLayout &DL = MIRBuilder.getDataLayout();
      if (DL.isNonIntegralAddressSpace(SrcTy.getAddressSpace()))
        return UnableToLegalize;

      LLT SrcAsIntTy = LLT::scalar(SrcTy.getSizeInBits());
      Src = MIRBuilder.buildPtrToInt(SrcAsIntTy, Src);
      SrcTy = SrcAsIntTy;
    }

    if (DstTy.isPointer())
      return UnableToLegalize;

    if (Offset == 0) {
      // Avoid a shift in the degenerate case.
      MIRBuilder.buildTrunc(DstReg,
                            MIRBuilder.buildAnyExtOrTrunc(WideTy, Src));
      MI.eraseFromParent();
      return Legalized;
    }

    // Do a shift in the source type.
    LLT ShiftTy = SrcTy;
    if (WideTy.getSizeInBits() > SrcTy.getSizeInBits()) {
      Src = MIRBuilder.buildAnyExt(WideTy, Src);
      ShiftTy = WideTy;
    } else if (WideTy.getSizeInBits() > SrcTy.getSizeInBits())
      return UnableToLegalize;

    auto LShr = MIRBuilder.buildLShr(
      ShiftTy, Src, MIRBuilder.buildConstant(ShiftTy, Offset));
    MIRBuilder.buildTrunc(DstReg, LShr);
    MI.eraseFromParent();
    return Legalized;
  }

  if (SrcTy.isScalar()) {
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }

  if (!SrcTy.isVector())
    return UnableToLegalize;

  if (DstTy != SrcTy.getElementType())
    return UnableToLegalize;

  if (Offset % SrcTy.getScalarSizeInBits() != 0)
    return UnableToLegalize;

  Observer.changingInstr(MI);
  widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);

  MI.getOperand(2).setImm((WideTy.getSizeInBits() / SrcTy.getSizeInBits()) *
                          Offset);
  widenScalarDst(MI, WideTy.getScalarType(), 0);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalarInsert(MachineInstr &MI, unsigned TypeIdx,
                                   LLT WideTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;
  Observer.changingInstr(MI);
  widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
  widenScalarDst(MI, WideTy);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx, LLT WideTy) {
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_EXTRACT:
    return widenScalarExtract(MI, TypeIdx, WideTy);
  case TargetOpcode::G_INSERT:
    return widenScalarInsert(MI, TypeIdx, WideTy);
  case TargetOpcode::G_MERGE_VALUES:
    return widenScalarMergeValues(MI, TypeIdx, WideTy);
  case TargetOpcode::G_UNMERGE_VALUES:
    return widenScalarUnmergeValues(MI, TypeIdx, WideTy);
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_USUBO: {
    if (TypeIdx == 1)
      return UnableToLegalize; // TODO
    auto LHSZext = MIRBuilder.buildInstr(TargetOpcode::G_ZEXT, {WideTy},
                                         {MI.getOperand(2).getReg()});
    auto RHSZext = MIRBuilder.buildInstr(TargetOpcode::G_ZEXT, {WideTy},
                                         {MI.getOperand(3).getReg()});
    unsigned Opcode = MI.getOpcode() == TargetOpcode::G_UADDO
                          ? TargetOpcode::G_ADD
                          : TargetOpcode::G_SUB;
    // Do the arithmetic in the larger type.
    auto NewOp = MIRBuilder.buildInstr(Opcode, {WideTy}, {LHSZext, RHSZext});
    LLT OrigTy = MRI.getType(MI.getOperand(0).getReg());
    APInt Mask = APInt::getAllOnesValue(OrigTy.getSizeInBits());
    auto AndOp = MIRBuilder.buildInstr(
        TargetOpcode::G_AND, {WideTy},
        {NewOp, MIRBuilder.buildConstant(WideTy, Mask.getZExtValue())});
    // There is no overflow if the AndOp is the same as NewOp.
    MIRBuilder.buildICmp(CmpInst::ICMP_NE, MI.getOperand(1).getReg(), NewOp,
                         AndOp);
    // Now trunc the NewOp to the original result.
    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), NewOp);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP: {
    if (TypeIdx == 0) {
      Observer.changingInstr(MI);
      widenScalarDst(MI, WideTy, 0);
      Observer.changedInstr(MI);
      return Legalized;
    }

    unsigned SrcReg = MI.getOperand(1).getReg();

    // First ZEXT the input.
    auto MIBSrc = MIRBuilder.buildZExt(WideTy, SrcReg);
    LLT CurTy = MRI.getType(SrcReg);
    if (MI.getOpcode() == TargetOpcode::G_CTTZ) {
      // The count is the same in the larger type except if the original
      // value was zero.  This can be handled by setting the bit just off
      // the top of the original type.
      auto TopBit =
          APInt::getOneBitSet(WideTy.getSizeInBits(), CurTy.getSizeInBits());
      MIBSrc = MIRBuilder.buildOr(
        WideTy, MIBSrc, MIRBuilder.buildConstant(WideTy, TopBit));
    }

    // Perform the operation at the larger size.
    auto MIBNewOp = MIRBuilder.buildInstr(MI.getOpcode(), {WideTy}, {MIBSrc});
    // This is already the correct result for CTPOP and CTTZs
    if (MI.getOpcode() == TargetOpcode::G_CTLZ ||
        MI.getOpcode() == TargetOpcode::G_CTLZ_ZERO_UNDEF) {
      // The correct result is NewOp - (Difference in widety and current ty).
      unsigned SizeDiff = WideTy.getSizeInBits() - CurTy.getSizeInBits();
      MIBNewOp = MIRBuilder.buildInstr(
          TargetOpcode::G_SUB, {WideTy},
          {MIBNewOp, MIRBuilder.buildConstant(WideTy, SizeDiff)});
    }

    MIRBuilder.buildZExtOrTrunc(MI.getOperand(0), MIBNewOp);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_BSWAP: {
    Observer.changingInstr(MI);
    unsigned DstReg = MI.getOperand(0).getReg();

    unsigned ShrReg = MRI.createGenericVirtualRegister(WideTy);
    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    unsigned ShiftAmtReg = MRI.createGenericVirtualRegister(WideTy);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);

    MI.getOperand(0).setReg(DstExt);

    MIRBuilder.setInsertPt(MIRBuilder.getMBB(), ++MIRBuilder.getInsertPt());

    LLT Ty = MRI.getType(DstReg);
    unsigned DiffBits = WideTy.getScalarSizeInBits() - Ty.getScalarSizeInBits();
    MIRBuilder.buildConstant(ShiftAmtReg, DiffBits);
    MIRBuilder.buildInstr(TargetOpcode::G_LSHR)
      .addDef(ShrReg)
      .addUse(DstExt)
      .addUse(ShiftAmtReg);

    MIRBuilder.buildTrunc(DstReg, ShrReg);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SUB:
    // Perform operation at larger width (any extension is fines here, high bits
    // don't affect the result) and then truncate the result back to the
    // original type.
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ANYEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SHL:
    Observer.changingInstr(MI);

    if (TypeIdx == 0) {
      widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
      widenScalarDst(MI, WideTy);
    } else {
      assert(TypeIdx == 1);
      // The "number of bits to shift" operand must preserve its value as an
      // unsigned integer:
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    }

    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_SEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR:
    Observer.changingInstr(MI);

    if (TypeIdx == 0) {
      unsigned CvtOp = MI.getOpcode() == TargetOpcode::G_ASHR ?
        TargetOpcode::G_SEXT : TargetOpcode::G_ZEXT;

      widenScalarSrc(MI, WideTy, 1, CvtOp);
      widenScalarDst(MI, WideTy);
    } else {
      assert(TypeIdx == 1);
      // The "number of bits to shift" operand must preserve its value as an
      // unsigned integer:
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    }

    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ZEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SELECT:
    Observer.changingInstr(MI);
    if (TypeIdx == 0) {
      // Perform operation at larger width (any extension is fine here, high
      // bits don't affect the result) and then truncate the result back to the
      // original type.
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_ANYEXT);
      widenScalarSrc(MI, WideTy, 3, TargetOpcode::G_ANYEXT);
      widenScalarDst(MI, WideTy);
    } else {
      bool IsVec = MRI.getType(MI.getOperand(1).getReg()).isVector();
      // Explicit extension is required here since high bits affect the result.
      widenScalarSrc(MI, WideTy, 1, MIRBuilder.getBoolExtOp(IsVec, false));
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_SITOFP:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_UITOFP:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD:
    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_STORE: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    if (!isPowerOf2_32(Ty.getSizeInBits()))
      return UnableToLegalize;

    Observer.changingInstr(MI);

    unsigned ExtType = Ty.getScalarSizeInBits() == 1 ?
      TargetOpcode::G_ZEXT : TargetOpcode::G_ANYEXT;
    widenScalarSrc(MI, WideTy, 0, ExtType);

    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    MachineOperand &SrcMO = MI.getOperand(1);
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    const APInt &Val = SrcMO.getCImm()->getValue().sext(WideTy.getSizeInBits());
    Observer.changingInstr(MI);
    SrcMO.setCImm(ConstantInt::get(Ctx, Val));

    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_FCONSTANT: {
    MachineOperand &SrcMO = MI.getOperand(1);
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    APFloat Val = SrcMO.getFPImm()->getValueAPF();
    bool LosesInfo;
    switch (WideTy.getSizeInBits()) {
    case 32:
      Val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                  &LosesInfo);
      break;
    case 64:
      Val.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &LosesInfo);
      break;
    default:
      return UnableToLegalize;
    }

    assert(!LosesInfo && "extend should always be lossless");

    Observer.changingInstr(MI);
    SrcMO.setFPImm(ConstantFP::get(Ctx, Val));

    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_FPTRUNC);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_IMPLICIT_DEF: {
    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_BRCOND:
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 0, MIRBuilder.getBoolExtOp(false, false));
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_FCMP:
    Observer.changingInstr(MI);
    if (TypeIdx == 0)
      widenScalarDst(MI, WideTy);
    else {
      widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_FPEXT);
      widenScalarSrc(MI, WideTy, 3, TargetOpcode::G_FPEXT);
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_ICMP:
    Observer.changingInstr(MI);
    if (TypeIdx == 0)
      widenScalarDst(MI, WideTy);
    else {
      unsigned ExtOpcode = CmpInst::isSigned(static_cast<CmpInst::Predicate>(
                               MI.getOperand(1).getPredicate()))
                               ? TargetOpcode::G_SEXT
                               : TargetOpcode::G_ZEXT;
      widenScalarSrc(MI, WideTy, 2, ExtOpcode);
      widenScalarSrc(MI, WideTy, 3, ExtOpcode);
    }
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_GEP:
    assert(TypeIdx == 1 && "unable to legalize pointer of GEP");
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_PHI: {
    assert(TypeIdx == 0 && "Expecting only Idx 0");

    Observer.changingInstr(MI);
    for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
      MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
      MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());
      widenScalarSrc(MI, WideTy, I, TargetOpcode::G_ANYEXT);
    }

    MachineBasicBlock &MBB = *MI.getParent();
    MIRBuilder.setInsertPt(MBB, --MBB.getFirstNonPHI());
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    if (TypeIdx == 0) {
      unsigned VecReg = MI.getOperand(1).getReg();
      LLT VecTy = MRI.getType(VecReg);
      Observer.changingInstr(MI);

      widenScalarSrc(MI, LLT::vector(VecTy.getNumElements(),
                                     WideTy.getSizeInBits()),
                     1, TargetOpcode::G_SEXT);

      widenScalarDst(MI, WideTy, 0);
      Observer.changedInstr(MI);
      return Legalized;
    }

    if (TypeIdx != 2)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 2, TargetOpcode::G_SEXT);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FCANONICALIZE:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_ROUND:
    assert(TypeIdx == 0);
    Observer.changingInstr(MI);

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I)
      widenScalarSrc(MI, WideTy, I, TargetOpcode::G_FPEXT);

    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_FPTRUNC);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INTTOPTR:
    if (TypeIdx != 1)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ZEXT);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PTRTOINT:
    if (TypeIdx != 0)
      return UnableToLegalize;

    Observer.changingInstr(MI);
    widenScalarDst(MI, WideTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lower(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  using namespace TargetOpcode;
  MIRBuilder.setInstr(MI);

  switch(MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM: {
    unsigned QuotReg = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildInstr(MI.getOpcode() == G_SREM ? G_SDIV : G_UDIV)
        .addDef(QuotReg)
        .addUse(MI.getOperand(1).getReg())
        .addUse(MI.getOperand(2).getReg());

    unsigned ProdReg = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildMul(ProdReg, QuotReg, MI.getOperand(2).getReg());
    MIRBuilder.buildSub(MI.getOperand(0).getReg(), MI.getOperand(1).getReg(),
                        ProdReg);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SMULO:
  case TargetOpcode::G_UMULO: {
    // Generate G_UMULH/G_SMULH to check for overflow and a normal G_MUL for the
    // result.
    unsigned Res = MI.getOperand(0).getReg();
    unsigned Overflow = MI.getOperand(1).getReg();
    unsigned LHS = MI.getOperand(2).getReg();
    unsigned RHS = MI.getOperand(3).getReg();

    MIRBuilder.buildMul(Res, LHS, RHS);

    unsigned Opcode = MI.getOpcode() == TargetOpcode::G_SMULO
                          ? TargetOpcode::G_SMULH
                          : TargetOpcode::G_UMULH;

    unsigned HiPart = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildInstr(Opcode)
      .addDef(HiPart)
      .addUse(LHS)
      .addUse(RHS);

    unsigned Zero = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildConstant(Zero, 0);

    // For *signed* multiply, overflow is detected by checking:
    // (hi != (lo >> bitwidth-1))
    if (Opcode == TargetOpcode::G_SMULH) {
      unsigned Shifted = MRI.createGenericVirtualRegister(Ty);
      unsigned ShiftAmt = MRI.createGenericVirtualRegister(Ty);
      MIRBuilder.buildConstant(ShiftAmt, Ty.getSizeInBits() - 1);
      MIRBuilder.buildInstr(TargetOpcode::G_ASHR)
        .addDef(Shifted)
        .addUse(Res)
        .addUse(ShiftAmt);
      MIRBuilder.buildICmp(CmpInst::ICMP_NE, Overflow, HiPart, Shifted);
    } else {
      MIRBuilder.buildICmp(CmpInst::ICMP_NE, Overflow, HiPart, Zero);
    }
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FNEG: {
    // TODO: Handle vector types once we are able to
    // represent them.
    if (Ty.isVector())
      return UnableToLegalize;
    unsigned Res = MI.getOperand(0).getReg();
    Type *ZeroTy;
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();
    switch (Ty.getSizeInBits()) {
    case 16:
      ZeroTy = Type::getHalfTy(Ctx);
      break;
    case 32:
      ZeroTy = Type::getFloatTy(Ctx);
      break;
    case 64:
      ZeroTy = Type::getDoubleTy(Ctx);
      break;
    case 128:
      ZeroTy = Type::getFP128Ty(Ctx);
      break;
    default:
      llvm_unreachable("unexpected floating-point type");
    }
    ConstantFP &ZeroForNegation =
        *cast<ConstantFP>(ConstantFP::getZeroValueForNegation(ZeroTy));
    auto Zero = MIRBuilder.buildFConstant(Ty, ZeroForNegation);
    unsigned SubByReg = MI.getOperand(1).getReg();
    unsigned ZeroReg = Zero->getOperand(0).getReg();
    MachineInstr *SrcMI = MRI.getVRegDef(SubByReg);
    MIRBuilder.buildInstr(TargetOpcode::G_FSUB, {Res}, {ZeroReg, SubByReg},
                          SrcMI->getFlags());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FSUB: {
    // Lower (G_FSUB LHS, RHS) to (G_FADD LHS, (G_FNEG RHS)).
    // First, check if G_FNEG is marked as Lower. If so, we may
    // end up with an infinite loop as G_FSUB is used to legalize G_FNEG.
    if (LI.getAction({G_FNEG, {Ty}}).Action == Lower)
      return UnableToLegalize;
    unsigned Res = MI.getOperand(0).getReg();
    unsigned LHS = MI.getOperand(1).getReg();
    unsigned RHS = MI.getOperand(2).getReg();
    unsigned Neg = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildInstr(TargetOpcode::G_FNEG).addDef(Neg).addUse(RHS);
    MIRBuilder.buildInstr(TargetOpcode::G_FADD, {Res}, {LHS, Neg}, MI.getFlags());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_ATOMIC_CMPXCHG_WITH_SUCCESS: {
    unsigned OldValRes = MI.getOperand(0).getReg();
    unsigned SuccessRes = MI.getOperand(1).getReg();
    unsigned Addr = MI.getOperand(2).getReg();
    unsigned CmpVal = MI.getOperand(3).getReg();
    unsigned NewVal = MI.getOperand(4).getReg();
    MIRBuilder.buildAtomicCmpXchg(OldValRes, Addr, CmpVal, NewVal,
                                  **MI.memoperands_begin());
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, SuccessRes, OldValRes, CmpVal);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD: {
    // Lower to a memory-width G_LOAD and a G_SEXT/G_ZEXT/G_ANYEXT
    unsigned DstReg = MI.getOperand(0).getReg();
    unsigned PtrReg = MI.getOperand(1).getReg();
    LLT DstTy = MRI.getType(DstReg);
    auto &MMO = **MI.memoperands_begin();

    if (DstTy.getSizeInBits() == MMO.getSize() /* in bytes */ * 8) {
      // In the case of G_LOAD, this was a non-extending load already and we're
      // about to lower to the same instruction.
      if (MI.getOpcode() == TargetOpcode::G_LOAD)
          return UnableToLegalize;
      MIRBuilder.buildLoad(DstReg, PtrReg, MMO);
      MI.eraseFromParent();
      return Legalized;
    }

    if (DstTy.isScalar()) {
      unsigned TmpReg =
          MRI.createGenericVirtualRegister(LLT::scalar(MMO.getSizeInBits()));
      MIRBuilder.buildLoad(TmpReg, PtrReg, MMO);
      switch (MI.getOpcode()) {
      default:
        llvm_unreachable("Unexpected opcode");
      case TargetOpcode::G_LOAD:
        MIRBuilder.buildAnyExt(DstReg, TmpReg);
        break;
      case TargetOpcode::G_SEXTLOAD:
        MIRBuilder.buildSExt(DstReg, TmpReg);
        break;
      case TargetOpcode::G_ZEXTLOAD:
        MIRBuilder.buildZExt(DstReg, TmpReg);
        break;
      }
      MI.eraseFromParent();
      return Legalized;
    }

    return UnableToLegalize;
  }
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTPOP:
    return lowerBitCount(MI, TypeIdx, Ty);
  case G_UADDO: {
    unsigned Res = MI.getOperand(0).getReg();
    unsigned CarryOut = MI.getOperand(1).getReg();
    unsigned LHS = MI.getOperand(2).getReg();
    unsigned RHS = MI.getOperand(3).getReg();

    MIRBuilder.buildAdd(Res, LHS, RHS);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CarryOut, Res, RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_UADDE: {
    unsigned Res = MI.getOperand(0).getReg();
    unsigned CarryOut = MI.getOperand(1).getReg();
    unsigned LHS = MI.getOperand(2).getReg();
    unsigned RHS = MI.getOperand(3).getReg();
    unsigned CarryIn = MI.getOperand(4).getReg();

    unsigned TmpRes = MRI.createGenericVirtualRegister(Ty);
    unsigned ZExtCarryIn = MRI.createGenericVirtualRegister(Ty);

    MIRBuilder.buildAdd(TmpRes, LHS, RHS);
    MIRBuilder.buildZExt(ZExtCarryIn, CarryIn);
    MIRBuilder.buildAdd(Res, TmpRes, ZExtCarryIn);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CarryOut, Res, LHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_USUBO: {
    unsigned Res = MI.getOperand(0).getReg();
    unsigned BorrowOut = MI.getOperand(1).getReg();
    unsigned LHS = MI.getOperand(2).getReg();
    unsigned RHS = MI.getOperand(3).getReg();

    MIRBuilder.buildSub(Res, LHS, RHS);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, BorrowOut, LHS, RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_USUBE: {
    unsigned Res = MI.getOperand(0).getReg();
    unsigned BorrowOut = MI.getOperand(1).getReg();
    unsigned LHS = MI.getOperand(2).getReg();
    unsigned RHS = MI.getOperand(3).getReg();
    unsigned BorrowIn = MI.getOperand(4).getReg();

    unsigned TmpRes = MRI.createGenericVirtualRegister(Ty);
    unsigned ZExtBorrowIn = MRI.createGenericVirtualRegister(Ty);
    unsigned LHS_EQ_RHS = MRI.createGenericVirtualRegister(LLT::scalar(1));
    unsigned LHS_ULT_RHS = MRI.createGenericVirtualRegister(LLT::scalar(1));

    MIRBuilder.buildSub(TmpRes, LHS, RHS);
    MIRBuilder.buildZExt(ZExtBorrowIn, BorrowIn);
    MIRBuilder.buildSub(Res, TmpRes, ZExtBorrowIn);
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, LHS_EQ_RHS, LHS, RHS);
    MIRBuilder.buildICmp(CmpInst::ICMP_ULT, LHS_ULT_RHS, LHS, RHS);
    MIRBuilder.buildSelect(BorrowOut, LHS_EQ_RHS, BorrowIn, LHS_ULT_RHS);

    MI.eraseFromParent();
    return Legalized;
  }
  case G_UITOFP:
    return lowerUITOFP(MI, TypeIdx, Ty);
  case G_SITOFP:
    return lowerSITOFP(MI, TypeIdx, Ty);
  }
}

LegalizerHelper::LegalizeResult LegalizerHelper::fewerElementsVectorImplicitDef(
    MachineInstr &MI, unsigned TypeIdx, LLT NarrowTy) {
  SmallVector<unsigned, 2> DstRegs;

  unsigned NarrowSize = NarrowTy.getSizeInBits();
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned Size = MRI.getType(DstReg).getSizeInBits();
  int NumParts = Size / NarrowSize;
  // FIXME: Don't know how to handle the situation where the small vectors
  // aren't all the same size yet.
  if (Size % NarrowSize != 0)
    return UnableToLegalize;

  for (int i = 0; i < NumParts; ++i) {
    unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildUndef(TmpReg);
    DstRegs.push_back(TmpReg);
  }

  if (NarrowTy.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorBasic(MachineInstr &MI, unsigned TypeIdx,
                                          LLT NarrowTy) {
  const unsigned Opc = MI.getOpcode();
  const unsigned NumOps = MI.getNumOperands() - 1;
  const unsigned NarrowSize = NarrowTy.getSizeInBits();
  const unsigned DstReg = MI.getOperand(0).getReg();
  const unsigned Flags = MI.getFlags();
  const LLT DstTy = MRI.getType(DstReg);
  const unsigned Size = DstTy.getSizeInBits();
  const int NumParts = Size / NarrowSize;
  const LLT EltTy = DstTy.getElementType();
  const unsigned EltSize = EltTy.getSizeInBits();
  const unsigned BitsForNumParts = NarrowSize * NumParts;

  // Check if we have any leftovers. If we do, then only handle the case where
  // the leftover is one element.
  if (BitsForNumParts != Size && BitsForNumParts + EltSize != Size)
    return UnableToLegalize;

  if (BitsForNumParts != Size) {
    unsigned AccumDstReg = MRI.createGenericVirtualRegister(DstTy);
    MIRBuilder.buildUndef(AccumDstReg);

    // Handle the pieces which evenly divide into the requested type with
    // extract/op/insert sequence.
    for (unsigned Offset = 0; Offset < BitsForNumParts; Offset += NarrowSize) {
      SmallVector<SrcOp, 4> SrcOps;
      for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
        unsigned PartOpReg = MRI.createGenericVirtualRegister(NarrowTy);
        MIRBuilder.buildExtract(PartOpReg, MI.getOperand(I).getReg(), Offset);
        SrcOps.push_back(PartOpReg);
      }

      unsigned PartDstReg = MRI.createGenericVirtualRegister(NarrowTy);
      MIRBuilder.buildInstr(Opc, {PartDstReg}, SrcOps, Flags);

      unsigned PartInsertReg = MRI.createGenericVirtualRegister(DstTy);
      MIRBuilder.buildInsert(PartInsertReg, AccumDstReg, PartDstReg, Offset);
      AccumDstReg = PartInsertReg;
    }

    // Handle the remaining element sized leftover piece.
    SmallVector<SrcOp, 4> SrcOps;
    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
      unsigned PartOpReg = MRI.createGenericVirtualRegister(EltTy);
      MIRBuilder.buildExtract(PartOpReg, MI.getOperand(I).getReg(),
                              BitsForNumParts);
      SrcOps.push_back(PartOpReg);
    }

    unsigned PartDstReg = MRI.createGenericVirtualRegister(EltTy);
    MIRBuilder.buildInstr(Opc, {PartDstReg}, SrcOps, Flags);
    MIRBuilder.buildInsert(DstReg, AccumDstReg, PartDstReg, BitsForNumParts);
    MI.eraseFromParent();

    return Legalized;
  }

  SmallVector<unsigned, 2> DstRegs, Src0Regs, Src1Regs, Src2Regs;

  extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src0Regs);

  if (NumOps >= 2)
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src1Regs);

  if (NumOps >= 3)
    extractParts(MI.getOperand(3).getReg(), NarrowTy, NumParts, Src2Regs);

  for (int i = 0; i < NumParts; ++i) {
    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);

    if (NumOps == 1)
      MIRBuilder.buildInstr(Opc, {DstReg}, {Src0Regs[i]}, Flags);
    else if (NumOps == 2) {
      MIRBuilder.buildInstr(Opc, {DstReg}, {Src0Regs[i], Src1Regs[i]}, Flags);
    } else if (NumOps == 3) {
      MIRBuilder.buildInstr(Opc, {DstReg},
                            {Src0Regs[i], Src1Regs[i], Src2Regs[i]}, Flags);
    }

    DstRegs.push_back(DstReg);
  }

  if (NarrowTy.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

// Handle splitting vector operations which need to have the same number of
// elements in each type index, but each type index may have a different element
// type.
//
// e.g.  <4 x s64> = G_SHL <4 x s64>, <4 x s32> ->
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//
// Also handles some irregular breakdown cases, e.g.
// e.g.  <3 x s64> = G_SHL <3 x s64>, <3 x s32> ->
//       <2 x s64> = G_SHL <2 x s64>, <2 x s32>
//             s64 = G_SHL s64, s32
LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorMultiEltType(
  MachineInstr &MI, unsigned TypeIdx, LLT NarrowTyArg) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  const LLT NarrowTy0 = NarrowTyArg;
  const unsigned NewNumElts =
      NarrowTy0.isVector() ? NarrowTy0.getNumElements() : 1;

  const unsigned DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT LeftoverTy0;

  int NumParts, NumLeftover;
  // All of the operands need to have the same number of elements, so if we can
  // determine a type breakdown for the result type, we can for all of the
  // source types.
  std::tie(NumParts, NumLeftover)
    = getNarrowTypeBreakDown(DstTy, NarrowTy0, LeftoverTy0);
  if (NumParts < 0)
    return UnableToLegalize;

  SmallVector<MachineInstrBuilder, 4> NewInsts;

  SmallVector<unsigned, 4> DstRegs, LeftoverDstRegs;
  SmallVector<unsigned, 4> PartRegs, LeftoverRegs;

  for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
    LLT LeftoverTy;
    unsigned SrcReg = MI.getOperand(I).getReg();
    LLT SrcTyI = MRI.getType(SrcReg);
    LLT NarrowTyI = LLT::scalarOrVector(NewNumElts, SrcTyI.getScalarType());
    LLT LeftoverTyI;

    // Split this operand into the requested typed registers, and any leftover
    // required to reproduce the original type.
    if (!extractParts(SrcReg, SrcTyI, NarrowTyI, LeftoverTyI, PartRegs,
                      LeftoverRegs))
      return UnableToLegalize;

    if (I == 1) {
      // For the first operand, create an instruction for each part and setup
      // the result.
      for (unsigned PartReg : PartRegs) {
        unsigned PartDstReg = MRI.createGenericVirtualRegister(NarrowTy0);
        NewInsts.push_back(MIRBuilder.buildInstrNoInsert(MI.getOpcode())
                               .addDef(PartDstReg)
                               .addUse(PartReg));
        DstRegs.push_back(PartDstReg);
      }

      for (unsigned LeftoverReg : LeftoverRegs) {
        unsigned PartDstReg = MRI.createGenericVirtualRegister(LeftoverTy0);
        NewInsts.push_back(MIRBuilder.buildInstrNoInsert(MI.getOpcode())
                               .addDef(PartDstReg)
                               .addUse(LeftoverReg));
        LeftoverDstRegs.push_back(PartDstReg);
      }
    } else {
      assert(NewInsts.size() == PartRegs.size() + LeftoverRegs.size());

      // Add the newly created operand splits to the existing instructions. The
      // odd-sized pieces are ordered after the requested NarrowTyArg sized
      // pieces.
      unsigned InstCount = 0;
      for (unsigned J = 0, JE = PartRegs.size(); J != JE; ++J)
        NewInsts[InstCount++].addUse(PartRegs[J]);
      for (unsigned J = 0, JE = LeftoverRegs.size(); J != JE; ++J)
        NewInsts[InstCount++].addUse(LeftoverRegs[J]);
    }

    PartRegs.clear();
    LeftoverRegs.clear();
  }

  // Insert the newly built operations and rebuild the result register.
  for (auto &MIB : NewInsts)
    MIRBuilder.insertInstr(MIB);

  insertParts(DstReg, DstTy, NarrowTy0, DstRegs, LeftoverTy0, LeftoverDstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorCasts(MachineInstr &MI, unsigned TypeIdx,
                                          LLT NarrowTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);

  LLT NarrowTy0 = NarrowTy;
  LLT NarrowTy1;
  unsigned NumParts;

  if (NarrowTy.isVector()) {
    // Uneven breakdown not handled.
    NumParts = DstTy.getNumElements() / NarrowTy.getNumElements();
    if (NumParts * NarrowTy.getNumElements() != DstTy.getNumElements())
      return UnableToLegalize;

    NarrowTy1 = LLT::vector(NumParts, SrcTy.getElementType().getSizeInBits());
  } else {
    NumParts = DstTy.getNumElements();
    NarrowTy1 = SrcTy.getElementType();
  }

  SmallVector<unsigned, 4> SrcRegs, DstRegs;
  extractParts(SrcReg, NarrowTy1, NumParts, SrcRegs);

  for (unsigned I = 0; I < NumParts; ++I) {
    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    MachineInstr *NewInst = MIRBuilder.buildInstr(MI.getOpcode())
      .addDef(DstReg)
      .addUse(SrcRegs[I]);

    NewInst->setFlags(MI.getFlags());
    DstRegs.push_back(DstReg);
  }

  if (NarrowTy.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorCmp(MachineInstr &MI, unsigned TypeIdx,
                                        LLT NarrowTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned Src0Reg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(Src0Reg);

  unsigned NumParts;
  LLT NarrowTy0, NarrowTy1;

  if (TypeIdx == 0) {
    unsigned NewElts = NarrowTy.isVector() ? NarrowTy.getNumElements() : 1;
    unsigned OldElts = DstTy.getNumElements();

    NarrowTy0 = NarrowTy;
    NumParts = NarrowTy.isVector() ? (OldElts / NewElts) : DstTy.getNumElements();
    NarrowTy1 = NarrowTy.isVector() ?
      LLT::vector(NarrowTy.getNumElements(), SrcTy.getScalarSizeInBits()) :
      SrcTy.getElementType();

  } else {
    unsigned NewElts = NarrowTy.isVector() ? NarrowTy.getNumElements() : 1;
    unsigned OldElts = SrcTy.getNumElements();

    NumParts = NarrowTy.isVector() ? (OldElts / NewElts) :
      NarrowTy.getNumElements();
    NarrowTy0 = LLT::vector(NarrowTy.getNumElements(),
                            DstTy.getScalarSizeInBits());
    NarrowTy1 = NarrowTy;
  }

  // FIXME: Don't know how to handle the situation where the small vectors
  // aren't all the same size yet.
  if (NarrowTy1.isVector() &&
      NarrowTy1.getNumElements() * NumParts != DstTy.getNumElements())
    return UnableToLegalize;

  CmpInst::Predicate Pred
    = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());

  SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
  extractParts(MI.getOperand(2).getReg(), NarrowTy1, NumParts, Src1Regs);
  extractParts(MI.getOperand(3).getReg(), NarrowTy1, NumParts, Src2Regs);

  for (unsigned I = 0; I < NumParts; ++I) {
    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    DstRegs.push_back(DstReg);

    if (MI.getOpcode() == TargetOpcode::G_ICMP)
      MIRBuilder.buildICmp(Pred, DstReg, Src1Regs[I], Src2Regs[I]);
    else {
      MachineInstr *NewCmp
        = MIRBuilder.buildFCmp(Pred, DstReg, Src1Regs[I], Src2Regs[I]);
      NewCmp->setFlags(MI.getFlags());
    }
  }

  if (NarrowTy1.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorSelect(MachineInstr &MI, unsigned TypeIdx,
                                           LLT NarrowTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned CondReg = MI.getOperand(1).getReg();

  unsigned NumParts = 0;
  LLT NarrowTy0, NarrowTy1;

  LLT DstTy = MRI.getType(DstReg);
  LLT CondTy = MRI.getType(CondReg);
  unsigned Size = DstTy.getSizeInBits();

  assert(TypeIdx == 0 || CondTy.isVector());

  if (TypeIdx == 0) {
    NarrowTy0 = NarrowTy;
    NarrowTy1 = CondTy;

    unsigned NarrowSize = NarrowTy0.getSizeInBits();
    // FIXME: Don't know how to handle the situation where the small vectors
    // aren't all the same size yet.
    if (Size % NarrowSize != 0)
      return UnableToLegalize;

    NumParts = Size / NarrowSize;

    // Need to break down the condition type
    if (CondTy.isVector()) {
      if (CondTy.getNumElements() == NumParts)
        NarrowTy1 = CondTy.getElementType();
      else
        NarrowTy1 = LLT::vector(CondTy.getNumElements() / NumParts,
                                CondTy.getScalarSizeInBits());
    }
  } else {
    NumParts = CondTy.getNumElements();
    if (NarrowTy.isVector()) {
      // TODO: Handle uneven breakdown.
      if (NumParts * NarrowTy.getNumElements() != CondTy.getNumElements())
        return UnableToLegalize;

      return UnableToLegalize;
    } else {
      NarrowTy0 = DstTy.getElementType();
      NarrowTy1 = NarrowTy;
    }
  }

  SmallVector<unsigned, 2> DstRegs, Src0Regs, Src1Regs, Src2Regs;
  if (CondTy.isVector())
    extractParts(MI.getOperand(1).getReg(), NarrowTy1, NumParts, Src0Regs);

  extractParts(MI.getOperand(2).getReg(), NarrowTy0, NumParts, Src1Regs);
  extractParts(MI.getOperand(3).getReg(), NarrowTy0, NumParts, Src2Regs);

  for (unsigned i = 0; i < NumParts; ++i) {
    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy0);
    MIRBuilder.buildSelect(DstReg, CondTy.isVector() ? Src0Regs[i] : CondReg,
                           Src1Regs[i], Src2Regs[i]);
    DstRegs.push_back(DstReg);
  }

  if (NarrowTy0.isVector())
    MIRBuilder.buildConcatVectors(DstReg, DstRegs);
  else
    MIRBuilder.buildBuildVector(DstReg, DstRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVectorPhi(MachineInstr &MI, unsigned TypeIdx,
                                        LLT NarrowTy) {
  const unsigned DstReg = MI.getOperand(0).getReg();
  LLT PhiTy = MRI.getType(DstReg);
  LLT LeftoverTy;

  // All of the operands need to have the same number of elements, so if we can
  // determine a type breakdown for the result type, we can for all of the
  // source types.
  int NumParts, NumLeftover;
  std::tie(NumParts, NumLeftover)
    = getNarrowTypeBreakDown(PhiTy, NarrowTy, LeftoverTy);
  if (NumParts < 0)
    return UnableToLegalize;

  SmallVector<unsigned, 4> DstRegs, LeftoverDstRegs;
  SmallVector<MachineInstrBuilder, 4> NewInsts;

  const int TotalNumParts = NumParts + NumLeftover;

  // Insert the new phis in the result block first.
  for (int I = 0; I != TotalNumParts; ++I) {
    LLT Ty = I < NumParts ? NarrowTy : LeftoverTy;
    unsigned PartDstReg = MRI.createGenericVirtualRegister(Ty);
    NewInsts.push_back(MIRBuilder.buildInstr(TargetOpcode::G_PHI)
                       .addDef(PartDstReg));
    if (I < NumParts)
      DstRegs.push_back(PartDstReg);
    else
      LeftoverDstRegs.push_back(PartDstReg);
  }

  MachineBasicBlock *MBB = MI.getParent();
  MIRBuilder.setInsertPt(*MBB, MBB->getFirstNonPHI());
  insertParts(DstReg, PhiTy, NarrowTy, DstRegs, LeftoverTy, LeftoverDstRegs);

  SmallVector<unsigned, 4> PartRegs, LeftoverRegs;

  // Insert code to extract the incoming values in each predecessor block.
  for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
    PartRegs.clear();
    LeftoverRegs.clear();

    unsigned SrcReg = MI.getOperand(I).getReg();
    MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
    MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());

    LLT Unused;
    if (!extractParts(SrcReg, PhiTy, NarrowTy, Unused, PartRegs,
                      LeftoverRegs))
      return UnableToLegalize;

    // Add the newly created operand splits to the existing instructions. The
    // odd-sized pieces are ordered after the requested NarrowTyArg sized
    // pieces.
    for (int J = 0; J != TotalNumParts; ++J) {
      MachineInstrBuilder MIB = NewInsts[J];
      MIB.addUse(J < NumParts ? PartRegs[J] : LeftoverRegs[J - NumParts]);
      MIB.addMBB(&OpMBB);
    }
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::reduceLoadStoreWidth(MachineInstr &MI, unsigned TypeIdx,
                                      LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;

  MachineMemOperand *MMO = *MI.memoperands_begin();

  // This implementation doesn't work for atomics. Give up instead of doing
  // something invalid.
  if (MMO->getOrdering() != AtomicOrdering::NotAtomic ||
      MMO->getFailureOrdering() != AtomicOrdering::NotAtomic)
    return UnableToLegalize;

  bool IsLoad = MI.getOpcode() == TargetOpcode::G_LOAD;
  unsigned ValReg = MI.getOperand(0).getReg();
  unsigned AddrReg = MI.getOperand(1).getReg();
  LLT ValTy = MRI.getType(ValReg);

  int NumParts = -1;
  int NumLeftover = -1;
  LLT LeftoverTy;
  SmallVector<unsigned, 8> NarrowRegs, NarrowLeftoverRegs;
  if (IsLoad) {
    std::tie(NumParts, NumLeftover) = getNarrowTypeBreakDown(ValTy, NarrowTy, LeftoverTy);
  } else {
    if (extractParts(ValReg, ValTy, NarrowTy, LeftoverTy, NarrowRegs,
                     NarrowLeftoverRegs)) {
      NumParts = NarrowRegs.size();
      NumLeftover = NarrowLeftoverRegs.size();
    }
  }

  if (NumParts == -1)
    return UnableToLegalize;

  const LLT OffsetTy = LLT::scalar(MRI.getType(AddrReg).getScalarSizeInBits());

  unsigned TotalSize = ValTy.getSizeInBits();

  // Split the load/store into PartTy sized pieces starting at Offset. If this
  // is a load, return the new registers in ValRegs. For a store, each elements
  // of ValRegs should be PartTy. Returns the next offset that needs to be
  // handled.
  auto splitTypePieces = [=](LLT PartTy, SmallVectorImpl<unsigned> &ValRegs,
                             unsigned Offset) -> unsigned {
    MachineFunction &MF = MIRBuilder.getMF();
    unsigned PartSize = PartTy.getSizeInBits();
    for (unsigned Idx = 0, E = NumParts; Idx != E && Offset < TotalSize;
         Offset += PartSize, ++Idx) {
      unsigned ByteSize = PartSize / 8;
      unsigned ByteOffset = Offset / 8;
      unsigned NewAddrReg = 0;

      MIRBuilder.materializeGEP(NewAddrReg, AddrReg, OffsetTy, ByteOffset);

      MachineMemOperand *NewMMO =
        MF.getMachineMemOperand(MMO, ByteOffset, ByteSize);

      if (IsLoad) {
        unsigned Dst = MRI.createGenericVirtualRegister(PartTy);
        ValRegs.push_back(Dst);
        MIRBuilder.buildLoad(Dst, NewAddrReg, *NewMMO);
      } else {
        MIRBuilder.buildStore(ValRegs[Idx], NewAddrReg, *NewMMO);
      }
    }

    return Offset;
  };

  unsigned HandledOffset = splitTypePieces(NarrowTy, NarrowRegs, 0);

  // Handle the rest of the register if this isn't an even type breakdown.
  if (LeftoverTy.isValid())
    splitTypePieces(LeftoverTy, NarrowLeftoverRegs, HandledOffset);

  if (IsLoad) {
    insertParts(ValReg, ValTy, NarrowTy, NarrowRegs,
                LeftoverTy, NarrowLeftoverRegs);
  }

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy) {
  using namespace TargetOpcode;

  MIRBuilder.setInstr(MI);
  switch (MI.getOpcode()) {
  case G_IMPLICIT_DEF:
    return fewerElementsVectorImplicitDef(MI, TypeIdx, NarrowTy);
  case G_AND:
  case G_OR:
  case G_XOR:
  case G_ADD:
  case G_SUB:
  case G_MUL:
  case G_SMULH:
  case G_UMULH:
  case G_FADD:
  case G_FMUL:
  case G_FSUB:
  case G_FNEG:
  case G_FABS:
  case G_FCANONICALIZE:
  case G_FDIV:
  case G_FREM:
  case G_FMA:
  case G_FPOW:
  case G_FEXP:
  case G_FEXP2:
  case G_FLOG:
  case G_FLOG2:
  case G_FLOG10:
  case G_FNEARBYINT:
  case G_FCEIL:
  case G_FFLOOR:
  case G_FRINT:
  case G_INTRINSIC_ROUND:
  case G_INTRINSIC_TRUNC:
  case G_FCOS:
  case G_FSIN:
  case G_FSQRT:
  case G_BSWAP:
  case G_SDIV:
  case G_SMIN:
  case G_SMAX:
  case G_UMIN:
  case G_UMAX:
    return fewerElementsVectorBasic(MI, TypeIdx, NarrowTy);
  case G_SHL:
  case G_LSHR:
  case G_ASHR:
  case G_CTLZ:
  case G_CTLZ_ZERO_UNDEF:
  case G_CTTZ:
  case G_CTTZ_ZERO_UNDEF:
  case G_CTPOP:
  case G_FCOPYSIGN:
    return fewerElementsVectorMultiEltType(MI, TypeIdx, NarrowTy);
  case G_ZEXT:
  case G_SEXT:
  case G_ANYEXT:
  case G_FPEXT:
  case G_FPTRUNC:
  case G_SITOFP:
  case G_UITOFP:
  case G_FPTOSI:
  case G_FPTOUI:
  case G_INTTOPTR:
  case G_PTRTOINT:
  case G_ADDRSPACE_CAST:
    return fewerElementsVectorCasts(MI, TypeIdx, NarrowTy);
  case G_ICMP:
  case G_FCMP:
    return fewerElementsVectorCmp(MI, TypeIdx, NarrowTy);
  case G_SELECT:
    return fewerElementsVectorSelect(MI, TypeIdx, NarrowTy);
  case G_PHI:
    return fewerElementsVectorPhi(MI, TypeIdx, NarrowTy);
  case G_LOAD:
  case G_STORE:
    return reduceLoadStoreWidth(MI, TypeIdx, NarrowTy);
  default:
    return UnableToLegalize;
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarShiftByConstant(MachineInstr &MI, const APInt &Amt,
                                             const LLT HalfTy, const LLT AmtTy) {

  unsigned InL = MRI.createGenericVirtualRegister(HalfTy);
  unsigned InH = MRI.createGenericVirtualRegister(HalfTy);
  MIRBuilder.buildUnmerge({InL, InH}, MI.getOperand(1).getReg());

  if (Amt.isNullValue()) {
    MIRBuilder.buildMerge(MI.getOperand(0).getReg(), {InL, InH});
    MI.eraseFromParent();
    return Legalized;
  }

  LLT NVT = HalfTy;
  unsigned NVTBits = HalfTy.getSizeInBits();
  unsigned VTBits = 2 * NVTBits;

  SrcOp Lo(0), Hi(0);
  if (MI.getOpcode() == TargetOpcode::G_SHL) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildConstant(NVT, 0);
      Hi = MIRBuilder.buildShl(NVT, InL,
                               MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
    } else if (Amt == NVTBits) {
      Lo = MIRBuilder.buildConstant(NVT, 0);
      Hi = InL;
    } else {
      Lo = MIRBuilder.buildShl(NVT, InL, MIRBuilder.buildConstant(AmtTy, Amt));
      auto OrLHS =
          MIRBuilder.buildShl(NVT, InH, MIRBuilder.buildConstant(AmtTy, Amt));
      auto OrRHS = MIRBuilder.buildLShr(
          NVT, InL, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));
      Hi = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
    }
  } else if (MI.getOpcode() == TargetOpcode::G_LSHR) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildLShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
      Hi = MIRBuilder.buildConstant(NVT, 0);
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = MIRBuilder.buildConstant(NVT, 0);
    } else {
      auto ShiftAmtConst = MIRBuilder.buildConstant(AmtTy, Amt);

      auto OrLHS = MIRBuilder.buildLShr(NVT, InL, ShiftAmtConst);
      auto OrRHS = MIRBuilder.buildShl(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));

      Lo = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
      Hi = MIRBuilder.buildLShr(NVT, InH, ShiftAmtConst);
    }
  } else {
    if (Amt.ugt(VTBits)) {
      Hi = Lo = MIRBuilder.buildAShr(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else if (Amt.ugt(NVTBits)) {
      Lo = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, Amt - NVTBits));
      Hi = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = MIRBuilder.buildAShr(NVT, InH,
                                MIRBuilder.buildConstant(AmtTy, NVTBits - 1));
    } else {
      auto ShiftAmtConst = MIRBuilder.buildConstant(AmtTy, Amt);

      auto OrLHS = MIRBuilder.buildLShr(NVT, InL, ShiftAmtConst);
      auto OrRHS = MIRBuilder.buildShl(
          NVT, InH, MIRBuilder.buildConstant(AmtTy, -Amt + NVTBits));

      Lo = MIRBuilder.buildOr(NVT, OrLHS, OrRHS);
      Hi = MIRBuilder.buildAShr(NVT, InH, ShiftAmtConst);
    }
  }

  MIRBuilder.buildMerge(MI.getOperand(0).getReg(), {Lo.getReg(), Hi.getReg()});
  MI.eraseFromParent();

  return Legalized;
}

// TODO: Optimize if constant shift amount.
LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarShift(MachineInstr &MI, unsigned TypeIdx,
                                   LLT RequestedTy) {
  if (TypeIdx == 1) {
    Observer.changingInstr(MI);
    narrowScalarSrc(MI, RequestedTy, 2);
    Observer.changedInstr(MI);
    return Legalized;
  }

  unsigned DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (DstTy.isVector())
    return UnableToLegalize;

  unsigned Amt = MI.getOperand(2).getReg();
  LLT ShiftAmtTy = MRI.getType(Amt);
  const unsigned DstEltSize = DstTy.getScalarSizeInBits();
  if (DstEltSize % 2 != 0)
    return UnableToLegalize;

  // Ignore the input type. We can only go to exactly half the size of the
  // input. If that isn't small enough, the resulting pieces will be further
  // legalized.
  const unsigned NewBitSize = DstEltSize / 2;
  const LLT HalfTy = LLT::scalar(NewBitSize);
  const LLT CondTy = LLT::scalar(1);

  if (const MachineInstr *KShiftAmt =
          getOpcodeDef(TargetOpcode::G_CONSTANT, Amt, MRI)) {
    return narrowScalarShiftByConstant(
        MI, KShiftAmt->getOperand(1).getCImm()->getValue(), HalfTy, ShiftAmtTy);
  }

  // TODO: Expand with known bits.

  // Handle the fully general expansion by an unknown amount.
  auto NewBits = MIRBuilder.buildConstant(ShiftAmtTy, NewBitSize);

  unsigned InL = MRI.createGenericVirtualRegister(HalfTy);
  unsigned InH = MRI.createGenericVirtualRegister(HalfTy);
  MIRBuilder.buildUnmerge({InL, InH}, MI.getOperand(1).getReg());

  auto AmtExcess = MIRBuilder.buildSub(ShiftAmtTy, Amt, NewBits);
  auto AmtLack = MIRBuilder.buildSub(ShiftAmtTy, NewBits, Amt);

  auto Zero = MIRBuilder.buildConstant(ShiftAmtTy, 0);
  auto IsShort = MIRBuilder.buildICmp(ICmpInst::ICMP_ULT, CondTy, Amt, NewBits);
  auto IsZero = MIRBuilder.buildICmp(ICmpInst::ICMP_EQ, CondTy, Amt, Zero);

  unsigned ResultRegs[2];
  switch (MI.getOpcode()) {
  case TargetOpcode::G_SHL: {
    // Short: ShAmt < NewBitSize
    auto LoS = MIRBuilder.buildShl(HalfTy, InH, Amt);

    auto OrLHS = MIRBuilder.buildShl(HalfTy, InH, Amt);
    auto OrRHS = MIRBuilder.buildLShr(HalfTy, InL, AmtLack);
    auto HiS = MIRBuilder.buildOr(HalfTy, OrLHS, OrRHS);

    // Long: ShAmt >= NewBitSize
    auto LoL = MIRBuilder.buildConstant(HalfTy, 0);         // Lo part is zero.
    auto HiL = MIRBuilder.buildShl(HalfTy, InL, AmtExcess); // Hi from Lo part.

    auto Lo = MIRBuilder.buildSelect(HalfTy, IsShort, LoS, LoL);
    auto Hi = MIRBuilder.buildSelect(
        HalfTy, IsZero, InH, MIRBuilder.buildSelect(HalfTy, IsShort, HiS, HiL));

    ResultRegs[0] = Lo.getReg(0);
    ResultRegs[1] = Hi.getReg(0);
    break;
  }
  case TargetOpcode::G_LSHR: {
    // Short: ShAmt < NewBitSize
    auto HiS = MIRBuilder.buildLShr(HalfTy, InH, Amt);

    auto OrLHS = MIRBuilder.buildLShr(HalfTy, InL, Amt);
    auto OrRHS = MIRBuilder.buildShl(HalfTy, InH, AmtLack);
    auto LoS = MIRBuilder.buildOr(HalfTy, OrLHS, OrRHS);

    // Long: ShAmt >= NewBitSize
    auto HiL = MIRBuilder.buildConstant(HalfTy, 0);          // Hi part is zero.
    auto LoL = MIRBuilder.buildLShr(HalfTy, InH, AmtExcess); // Lo from Hi part.

    auto Lo = MIRBuilder.buildSelect(
        HalfTy, IsZero, InL, MIRBuilder.buildSelect(HalfTy, IsShort, LoS, LoL));
    auto Hi = MIRBuilder.buildSelect(HalfTy, IsShort, HiS, HiL);

    ResultRegs[0] = Lo.getReg(0);
    ResultRegs[1] = Hi.getReg(0);
    break;
  }
  case TargetOpcode::G_ASHR: {
    // Short: ShAmt < NewBitSize
    auto HiS = MIRBuilder.buildAShr(HalfTy, InH, Amt);

    auto OrLHS = MIRBuilder.buildLShr(HalfTy, InL, Amt);
    auto OrRHS = MIRBuilder.buildLShr(HalfTy, InH, AmtLack);
    auto LoS = MIRBuilder.buildOr(HalfTy, OrLHS, OrRHS);

    // Long: ShAmt >= NewBitSize

    // Sign of Hi part.
    auto HiL = MIRBuilder.buildAShr(
        HalfTy, InH, MIRBuilder.buildConstant(ShiftAmtTy, NewBitSize - 1));

    auto LoL = MIRBuilder.buildAShr(HalfTy, InH, AmtExcess); // Lo from Hi part.

    auto Lo = MIRBuilder.buildSelect(
        HalfTy, IsZero, InL, MIRBuilder.buildSelect(HalfTy, IsShort, LoS, LoL));

    auto Hi = MIRBuilder.buildSelect(HalfTy, IsShort, HiS, HiL);

    ResultRegs[0] = Lo.getReg(0);
    ResultRegs[1] = Hi.getReg(0);
    break;
  }
  default:
    llvm_unreachable("not a shift");
  }

  MIRBuilder.buildMerge(DstReg, ResultRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::moreElementsVectorPhi(MachineInstr &MI, unsigned TypeIdx,
                                       LLT MoreTy) {
  assert(TypeIdx == 0 && "Expecting only Idx 0");

  Observer.changingInstr(MI);
  for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
    MachineBasicBlock &OpMBB = *MI.getOperand(I + 1).getMBB();
    MIRBuilder.setInsertPt(OpMBB, OpMBB.getFirstTerminator());
    moreElementsVectorSrc(MI, MoreTy, I);
  }

  MachineBasicBlock &MBB = *MI.getParent();
  MIRBuilder.setInsertPt(MBB, --MBB.getFirstNonPHI());
  moreElementsVectorDst(MI, MoreTy, 0);
  Observer.changedInstr(MI);
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::moreElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                    LLT MoreTy) {
  MIRBuilder.setInstr(MI);
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case TargetOpcode::G_IMPLICIT_DEF: {
    Observer.changingInstr(MI);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX: {
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    moreElementsVectorSrc(MI, MoreTy, 2);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_EXTRACT:
    if (TypeIdx != 1)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_INSERT:
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 1);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_SELECT:
    if (TypeIdx != 0)
      return UnableToLegalize;
    if (MRI.getType(MI.getOperand(1).getReg()).isVector())
      return UnableToLegalize;

    Observer.changingInstr(MI);
    moreElementsVectorSrc(MI, MoreTy, 2);
    moreElementsVectorSrc(MI, MoreTy, 3);
    moreElementsVectorDst(MI, MoreTy, 0);
    Observer.changedInstr(MI);
    return Legalized;
  case TargetOpcode::G_PHI:
    return moreElementsVectorPhi(MI, TypeIdx, MoreTy);
  default:
    return UnableToLegalize;
  }
}

void LegalizerHelper::multiplyRegisters(SmallVectorImpl<unsigned> &DstRegs,
                                        ArrayRef<unsigned> Src1Regs,
                                        ArrayRef<unsigned> Src2Regs,
                                        LLT NarrowTy) {
  MachineIRBuilder &B = MIRBuilder;
  unsigned SrcParts = Src1Regs.size();
  unsigned DstParts = DstRegs.size();

  unsigned DstIdx = 0; // Low bits of the result.
  unsigned FactorSum =
      B.buildMul(NarrowTy, Src1Regs[DstIdx], Src2Regs[DstIdx]).getReg(0);
  DstRegs[DstIdx] = FactorSum;

  unsigned CarrySumPrevDstIdx;
  SmallVector<unsigned, 4> Factors;

  for (DstIdx = 1; DstIdx < DstParts; DstIdx++) {
    // Collect low parts of muls for DstIdx.
    for (unsigned i = DstIdx + 1 < SrcParts ? 0 : DstIdx - SrcParts + 1;
         i <= std::min(DstIdx, SrcParts - 1); ++i) {
      MachineInstrBuilder Mul =
          B.buildMul(NarrowTy, Src1Regs[DstIdx - i], Src2Regs[i]);
      Factors.push_back(Mul.getReg(0));
    }
    // Collect high parts of muls from previous DstIdx.
    for (unsigned i = DstIdx < SrcParts ? 0 : DstIdx - SrcParts;
         i <= std::min(DstIdx - 1, SrcParts - 1); ++i) {
      MachineInstrBuilder Umulh =
          B.buildUMulH(NarrowTy, Src1Regs[DstIdx - 1 - i], Src2Regs[i]);
      Factors.push_back(Umulh.getReg(0));
    }
    // Add CarrySum from additons calculated for previous DstIdx.
    if (DstIdx != 1) {
      Factors.push_back(CarrySumPrevDstIdx);
    }

    unsigned CarrySum = 0;
    // Add all factors and accumulate all carries into CarrySum.
    if (DstIdx != DstParts - 1) {
      MachineInstrBuilder Uaddo =
          B.buildUAddo(NarrowTy, LLT::scalar(1), Factors[0], Factors[1]);
      FactorSum = Uaddo.getReg(0);
      CarrySum = B.buildZExt(NarrowTy, Uaddo.getReg(1)).getReg(0);
      for (unsigned i = 2; i < Factors.size(); ++i) {
        MachineInstrBuilder Uaddo =
            B.buildUAddo(NarrowTy, LLT::scalar(1), FactorSum, Factors[i]);
        FactorSum = Uaddo.getReg(0);
        MachineInstrBuilder Carry = B.buildZExt(NarrowTy, Uaddo.getReg(1));
        CarrySum = B.buildAdd(NarrowTy, CarrySum, Carry).getReg(0);
      }
    } else {
      // Since value for the next index is not calculated, neither is CarrySum.
      FactorSum = B.buildAdd(NarrowTy, Factors[0], Factors[1]).getReg(0);
      for (unsigned i = 2; i < Factors.size(); ++i)
        FactorSum = B.buildAdd(NarrowTy, FactorSum, Factors[i]).getReg(0);
    }

    CarrySumPrevDstIdx = CarrySum;
    DstRegs[DstIdx] = FactorSum;
    Factors.clear();
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarMul(MachineInstr &MI, LLT NarrowTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned Src1 = MI.getOperand(1).getReg();
  unsigned Src2 = MI.getOperand(2).getReg();

  LLT Ty = MRI.getType(DstReg);
  if (Ty.isVector())
    return UnableToLegalize;

  unsigned SrcSize = MRI.getType(Src1).getSizeInBits();
  unsigned DstSize = Ty.getSizeInBits();
  unsigned NarrowSize = NarrowTy.getSizeInBits();
  if (DstSize % NarrowSize != 0 || SrcSize % NarrowSize != 0)
    return UnableToLegalize;

  unsigned NumDstParts = DstSize / NarrowSize;
  unsigned NumSrcParts = SrcSize / NarrowSize;
  bool IsMulHigh = MI.getOpcode() == TargetOpcode::G_UMULH;
  unsigned DstTmpParts = NumDstParts * (IsMulHigh ? 2 : 1);

  SmallVector<unsigned, 2> Src1Parts, Src2Parts, DstTmpRegs;
  extractParts(Src1, NarrowTy, NumSrcParts, Src1Parts);
  extractParts(Src2, NarrowTy, NumSrcParts, Src2Parts);
  DstTmpRegs.resize(DstTmpParts);
  multiplyRegisters(DstTmpRegs, Src1Parts, Src2Parts, NarrowTy);

  // Take only high half of registers if this is high mul.
  ArrayRef<unsigned> DstRegs(
      IsMulHigh ? &DstTmpRegs[DstTmpParts / 2] : &DstTmpRegs[0], NumDstParts);
  MIRBuilder.buildMerge(DstReg, DstRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarExtract(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy) {
  if (TypeIdx != 1)
    return UnableToLegalize;

  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  int64_t SizeOp1 = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
  // FIXME: add support for when SizeOp1 isn't an exact multiple of
  // NarrowSize.
  if (SizeOp1 % NarrowSize != 0)
    return UnableToLegalize;
  int NumParts = SizeOp1 / NarrowSize;

  SmallVector<unsigned, 2> SrcRegs, DstRegs;
  SmallVector<uint64_t, 2> Indexes;
  extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

  unsigned OpReg = MI.getOperand(0).getReg();
  uint64_t OpStart = MI.getOperand(2).getImm();
  uint64_t OpSize = MRI.getType(OpReg).getSizeInBits();
  for (int i = 0; i < NumParts; ++i) {
    unsigned SrcStart = i * NarrowSize;

    if (SrcStart + NarrowSize <= OpStart || SrcStart >= OpStart + OpSize) {
      // No part of the extract uses this subregister, ignore it.
      continue;
    } else if (SrcStart == OpStart && NarrowTy == MRI.getType(OpReg)) {
      // The entire subregister is extracted, forward the value.
      DstRegs.push_back(SrcRegs[i]);
      continue;
    }

    // OpSegStart is where this destination segment would start in OpReg if it
    // extended infinitely in both directions.
    int64_t ExtractOffset;
    uint64_t SegSize;
    if (OpStart < SrcStart) {
      ExtractOffset = 0;
      SegSize = std::min(NarrowSize, OpStart + OpSize - SrcStart);
    } else {
      ExtractOffset = OpStart - SrcStart;
      SegSize = std::min(SrcStart + NarrowSize - OpStart, OpSize);
    }

    unsigned SegReg = SrcRegs[i];
    if (ExtractOffset != 0 || SegSize != NarrowSize) {
      // A genuine extract is needed.
      SegReg = MRI.createGenericVirtualRegister(LLT::scalar(SegSize));
      MIRBuilder.buildExtract(SegReg, SrcRegs[i], ExtractOffset);
    }

    DstRegs.push_back(SegReg);
  }

  unsigned DstReg = MI.getOperand(0).getReg();
  if(MRI.getType(DstReg).isVector())
    MIRBuilder.buildBuildVector(DstReg, DstRegs);
  else
    MIRBuilder.buildMerge(DstReg, DstRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarInsert(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;

  uint64_t SizeOp0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  uint64_t NarrowSize = NarrowTy.getSizeInBits();

  // FIXME: add support for when SizeOp0 isn't an exact multiple of
  // NarrowSize.
  if (SizeOp0 % NarrowSize != 0)
    return UnableToLegalize;

  int NumParts = SizeOp0 / NarrowSize;

  SmallVector<unsigned, 2> SrcRegs, DstRegs;
  SmallVector<uint64_t, 2> Indexes;
  extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

  unsigned OpReg = MI.getOperand(2).getReg();
  uint64_t OpStart = MI.getOperand(3).getImm();
  uint64_t OpSize = MRI.getType(OpReg).getSizeInBits();
  for (int i = 0; i < NumParts; ++i) {
    unsigned DstStart = i * NarrowSize;

    if (DstStart + NarrowSize <= OpStart || DstStart >= OpStart + OpSize) {
      // No part of the insert affects this subregister, forward the original.
      DstRegs.push_back(SrcRegs[i]);
      continue;
    } else if (DstStart == OpStart && NarrowTy == MRI.getType(OpReg)) {
      // The entire subregister is defined by this insert, forward the new
      // value.
      DstRegs.push_back(OpReg);
      continue;
    }

    // OpSegStart is where this destination segment would start in OpReg if it
    // extended infinitely in both directions.
    int64_t ExtractOffset, InsertOffset;
    uint64_t SegSize;
    if (OpStart < DstStart) {
      InsertOffset = 0;
      ExtractOffset = DstStart - OpStart;
      SegSize = std::min(NarrowSize, OpStart + OpSize - DstStart);
    } else {
      InsertOffset = OpStart - DstStart;
      ExtractOffset = 0;
      SegSize =
        std::min(NarrowSize - InsertOffset, OpStart + OpSize - DstStart);
    }

    unsigned SegReg = OpReg;
    if (ExtractOffset != 0 || SegSize != OpSize) {
      // A genuine extract is needed.
      SegReg = MRI.createGenericVirtualRegister(LLT::scalar(SegSize));
      MIRBuilder.buildExtract(SegReg, OpReg, ExtractOffset);
    }

    unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
    MIRBuilder.buildInsert(DstReg, SrcRegs[i], SegReg, InsertOffset);
    DstRegs.push_back(DstReg);
  }

  assert(DstRegs.size() == (unsigned)NumParts && "not all parts covered");
  unsigned DstReg = MI.getOperand(0).getReg();
  if(MRI.getType(DstReg).isVector())
    MIRBuilder.buildBuildVector(DstReg, DstRegs);
  else
    MIRBuilder.buildMerge(DstReg, DstRegs);
  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarBasic(MachineInstr &MI, unsigned TypeIdx,
                                   LLT NarrowTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  assert(MI.getNumOperands() == 3 && TypeIdx == 0);

  SmallVector<unsigned, 4> DstRegs, DstLeftoverRegs;
  SmallVector<unsigned, 4> Src0Regs, Src0LeftoverRegs;
  SmallVector<unsigned, 4> Src1Regs, Src1LeftoverRegs;
  LLT LeftoverTy;
  if (!extractParts(MI.getOperand(1).getReg(), DstTy, NarrowTy, LeftoverTy,
                    Src0Regs, Src0LeftoverRegs))
    return UnableToLegalize;

  LLT Unused;
  if (!extractParts(MI.getOperand(2).getReg(), DstTy, NarrowTy, Unused,
                    Src1Regs, Src1LeftoverRegs))
    llvm_unreachable("inconsistent extractParts result");

  for (unsigned I = 0, E = Src1Regs.size(); I != E; ++I) {
    auto Inst = MIRBuilder.buildInstr(MI.getOpcode(), {NarrowTy},
                                        {Src0Regs[I], Src1Regs[I]});
    DstRegs.push_back(Inst->getOperand(0).getReg());
  }

  for (unsigned I = 0, E = Src1LeftoverRegs.size(); I != E; ++I) {
    auto Inst = MIRBuilder.buildInstr(
      MI.getOpcode(),
      {LeftoverTy}, {Src0LeftoverRegs[I], Src1LeftoverRegs[I]});
    DstLeftoverRegs.push_back(Inst->getOperand(0).getReg());
  }

  insertParts(DstReg, DstTy, NarrowTy, DstRegs,
              LeftoverTy, DstLeftoverRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarSelect(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy) {
  if (TypeIdx != 0)
    return UnableToLegalize;

  unsigned CondReg = MI.getOperand(1).getReg();
  LLT CondTy = MRI.getType(CondReg);
  if (CondTy.isVector()) // TODO: Handle vselect
    return UnableToLegalize;

  unsigned DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  SmallVector<unsigned, 4> DstRegs, DstLeftoverRegs;
  SmallVector<unsigned, 4> Src1Regs, Src1LeftoverRegs;
  SmallVector<unsigned, 4> Src2Regs, Src2LeftoverRegs;
  LLT LeftoverTy;
  if (!extractParts(MI.getOperand(2).getReg(), DstTy, NarrowTy, LeftoverTy,
                    Src1Regs, Src1LeftoverRegs))
    return UnableToLegalize;

  LLT Unused;
  if (!extractParts(MI.getOperand(3).getReg(), DstTy, NarrowTy, Unused,
                    Src2Regs, Src2LeftoverRegs))
    llvm_unreachable("inconsistent extractParts result");

  for (unsigned I = 0, E = Src1Regs.size(); I != E; ++I) {
    auto Select = MIRBuilder.buildSelect(NarrowTy,
                                         CondReg, Src1Regs[I], Src2Regs[I]);
    DstRegs.push_back(Select->getOperand(0).getReg());
  }

  for (unsigned I = 0, E = Src1LeftoverRegs.size(); I != E; ++I) {
    auto Select = MIRBuilder.buildSelect(
      LeftoverTy, CondReg, Src1LeftoverRegs[I], Src2LeftoverRegs[I]);
    DstLeftoverRegs.push_back(Select->getOperand(0).getReg());
  }

  insertParts(DstReg, DstTy, NarrowTy, DstRegs,
              LeftoverTy, DstLeftoverRegs);

  MI.eraseFromParent();
  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerBitCount(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  unsigned Opc = MI.getOpcode();
  auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  auto isSupported = [this](const LegalityQuery &Q) {
    auto QAction = LI.getAction(Q).Action;
    return QAction == Legal || QAction == Libcall || QAction == Custom;
  };
  switch (Opc) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_CTLZ_ZERO_UNDEF: {
    // This trivially expands to CTLZ.
    Observer.changingInstr(MI);
    MI.setDesc(TII.get(TargetOpcode::G_CTLZ));
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CTLZ: {
    unsigned SrcReg = MI.getOperand(1).getReg();
    unsigned Len = Ty.getSizeInBits();
    if (isSupported({TargetOpcode::G_CTLZ_ZERO_UNDEF, {Ty, Ty}})) {
      // If CTLZ_ZERO_UNDEF is supported, emit that and a select for zero.
      auto MIBCtlzZU = MIRBuilder.buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF,
                                             {Ty}, {SrcReg});
      auto MIBZero = MIRBuilder.buildConstant(Ty, 0);
      auto MIBLen = MIRBuilder.buildConstant(Ty, Len);
      auto MIBICmp = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, LLT::scalar(1),
                                          SrcReg, MIBZero);
      MIRBuilder.buildSelect(MI.getOperand(0).getReg(), MIBICmp, MIBLen,
                             MIBCtlzZU);
      MI.eraseFromParent();
      return Legalized;
    }
    // for now, we do this:
    // NewLen = NextPowerOf2(Len);
    // x = x | (x >> 1);
    // x = x | (x >> 2);
    // ...
    // x = x | (x >>16);
    // x = x | (x >>32); // for 64-bit input
    // Upto NewLen/2
    // return Len - popcount(x);
    //
    // Ref: "Hacker's Delight" by Henry Warren
    unsigned Op = SrcReg;
    unsigned NewLen = PowerOf2Ceil(Len);
    for (unsigned i = 0; (1U << i) <= (NewLen / 2); ++i) {
      auto MIBShiftAmt = MIRBuilder.buildConstant(Ty, 1ULL << i);
      auto MIBOp = MIRBuilder.buildInstr(
          TargetOpcode::G_OR, {Ty},
          {Op, MIRBuilder.buildInstr(TargetOpcode::G_LSHR, {Ty},
                                     {Op, MIBShiftAmt})});
      Op = MIBOp->getOperand(0).getReg();
    }
    auto MIBPop = MIRBuilder.buildInstr(TargetOpcode::G_CTPOP, {Ty}, {Op});
    MIRBuilder.buildInstr(TargetOpcode::G_SUB, {MI.getOperand(0).getReg()},
                          {MIRBuilder.buildConstant(Ty, Len), MIBPop});
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CTTZ_ZERO_UNDEF: {
    // This trivially expands to CTTZ.
    Observer.changingInstr(MI);
    MI.setDesc(TII.get(TargetOpcode::G_CTTZ));
    Observer.changedInstr(MI);
    return Legalized;
  }
  case TargetOpcode::G_CTTZ: {
    unsigned SrcReg = MI.getOperand(1).getReg();
    unsigned Len = Ty.getSizeInBits();
    if (isSupported({TargetOpcode::G_CTTZ_ZERO_UNDEF, {Ty, Ty}})) {
      // If CTTZ_ZERO_UNDEF is legal or custom, emit that and a select with
      // zero.
      auto MIBCttzZU = MIRBuilder.buildInstr(TargetOpcode::G_CTTZ_ZERO_UNDEF,
                                             {Ty}, {SrcReg});
      auto MIBZero = MIRBuilder.buildConstant(Ty, 0);
      auto MIBLen = MIRBuilder.buildConstant(Ty, Len);
      auto MIBICmp = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, LLT::scalar(1),
                                          SrcReg, MIBZero);
      MIRBuilder.buildSelect(MI.getOperand(0).getReg(), MIBICmp, MIBLen,
                             MIBCttzZU);
      MI.eraseFromParent();
      return Legalized;
    }
    // for now, we use: { return popcount(~x & (x - 1)); }
    // unless the target has ctlz but not ctpop, in which case we use:
    // { return 32 - nlz(~x & (x-1)); }
    // Ref: "Hacker's Delight" by Henry Warren
    auto MIBCstNeg1 = MIRBuilder.buildConstant(Ty, -1);
    auto MIBNot =
        MIRBuilder.buildInstr(TargetOpcode::G_XOR, {Ty}, {SrcReg, MIBCstNeg1});
    auto MIBTmp = MIRBuilder.buildInstr(
        TargetOpcode::G_AND, {Ty},
        {MIBNot, MIRBuilder.buildInstr(TargetOpcode::G_ADD, {Ty},
                                       {SrcReg, MIBCstNeg1})});
    if (!isSupported({TargetOpcode::G_CTPOP, {Ty, Ty}}) &&
        isSupported({TargetOpcode::G_CTLZ, {Ty, Ty}})) {
      auto MIBCstLen = MIRBuilder.buildConstant(Ty, Len);
      MIRBuilder.buildInstr(
          TargetOpcode::G_SUB, {MI.getOperand(0).getReg()},
          {MIBCstLen,
           MIRBuilder.buildInstr(TargetOpcode::G_CTLZ, {Ty}, {MIBTmp})});
      MI.eraseFromParent();
      return Legalized;
    }
    MI.setDesc(TII.get(TargetOpcode::G_CTPOP));
    MI.getOperand(1).setReg(MIBTmp->getOperand(0).getReg());
    return Legalized;
  }
  }
}

// Expand s32 = G_UITOFP s64 using bit operations to an IEEE float
// representation.
LegalizerHelper::LegalizeResult
LegalizerHelper::lowerU64ToF32BitOps(MachineInstr &MI) {
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);
  const LLT S1 = LLT::scalar(1);

  assert(MRI.getType(Src) == S64 && MRI.getType(Dst) == S32);

  // unsigned cul2f(ulong u) {
  //   uint lz = clz(u);
  //   uint e = (u != 0) ? 127U + 63U - lz : 0;
  //   u = (u << lz) & 0x7fffffffffffffffUL;
  //   ulong t = u & 0xffffffffffUL;
  //   uint v = (e << 23) | (uint)(u >> 40);
  //   uint r = t > 0x8000000000UL ? 1U : (t == 0x8000000000UL ? v & 1U : 0U);
  //   return as_float(v + r);
  // }

  auto Zero32 = MIRBuilder.buildConstant(S32, 0);
  auto Zero64 = MIRBuilder.buildConstant(S64, 0);

  auto LZ = MIRBuilder.buildCTLZ_ZERO_UNDEF(S32, Src);

  auto K = MIRBuilder.buildConstant(S32, 127U + 63U);
  auto Sub = MIRBuilder.buildSub(S32, K, LZ);

  auto NotZero = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, Src, Zero64);
  auto E = MIRBuilder.buildSelect(S32, NotZero, Sub, Zero32);

  auto Mask0 = MIRBuilder.buildConstant(S64, (-1ULL) >> 1);
  auto ShlLZ = MIRBuilder.buildShl(S64, Src, LZ);

  auto U = MIRBuilder.buildAnd(S64, ShlLZ, Mask0);

  auto Mask1 = MIRBuilder.buildConstant(S64, 0xffffffffffULL);
  auto T = MIRBuilder.buildAnd(S64, U, Mask1);

  auto UShl = MIRBuilder.buildLShr(S64, U, MIRBuilder.buildConstant(S64, 40));
  auto ShlE = MIRBuilder.buildShl(S32, E, MIRBuilder.buildConstant(S32, 23));
  auto V = MIRBuilder.buildOr(S32, ShlE, MIRBuilder.buildTrunc(S32, UShl));

  auto C = MIRBuilder.buildConstant(S64, 0x8000000000ULL);
  auto RCmp = MIRBuilder.buildICmp(CmpInst::ICMP_UGT, S1, T, C);
  auto TCmp = MIRBuilder.buildICmp(CmpInst::ICMP_EQ, S1, T, C);
  auto One = MIRBuilder.buildConstant(S32, 1);

  auto VTrunc1 = MIRBuilder.buildAnd(S32, V, One);
  auto Select0 = MIRBuilder.buildSelect(S32, TCmp, VTrunc1, Zero32);
  auto R = MIRBuilder.buildSelect(S32, RCmp, One, Select0);
  MIRBuilder.buildAdd(Dst, V, R);

  return Legalized;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerUITOFP(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (SrcTy != LLT::scalar(64))
    return UnableToLegalize;

  if (DstTy == LLT::scalar(32)) {
    // TODO: SelectionDAG has several alternative expansions to port which may
    // be more reasonble depending on the available instructions. If a target
    // has sitofp, does not have CTLZ, or can efficiently use f64 as an
    // intermediate type, this is probably worse.
    return lowerU64ToF32BitOps(MI);
  }

  return UnableToLegalize;
}

LegalizerHelper::LegalizeResult
LegalizerHelper::lowerSITOFP(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  const LLT S64 = LLT::scalar(64);
  const LLT S32 = LLT::scalar(32);
  const LLT S1 = LLT::scalar(1);

  if (SrcTy != S64)
    return UnableToLegalize;

  if (DstTy == S32) {
    // signed cl2f(long l) {
    //   long s = l >> 63;
    //   float r = cul2f((l + s) ^ s);
    //   return s ? -r : r;
    // }
    unsigned L = Src;
    auto SignBit = MIRBuilder.buildConstant(S64, 63);
    auto S = MIRBuilder.buildAShr(S64, L, SignBit);

    auto LPlusS = MIRBuilder.buildAdd(S64, L, S);
    auto Xor = MIRBuilder.buildXor(S64, LPlusS, S);
    auto R = MIRBuilder.buildUITOFP(S32, Xor);

    auto RNeg = MIRBuilder.buildFNeg(S32, R);
    auto SignNotZero = MIRBuilder.buildICmp(CmpInst::ICMP_NE, S1, S,
                                            MIRBuilder.buildConstant(S64, 0));
    MIRBuilder.buildSelect(Dst, SignNotZero, RNeg, R);
    return Legalized;
  }

  return UnableToLegalize;
}
