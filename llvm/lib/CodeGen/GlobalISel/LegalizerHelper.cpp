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
  case TargetOpcode::G_FLOG2: {
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
    return narrowScalarMul(MI, TypeIdx, NarrowTy);
  case TargetOpcode::G_EXTRACT: {
    if (TypeIdx != 1)
      return UnableToLegalize;

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
  case TargetOpcode::G_INSERT: {
    // FIXME: Don't know how to handle secondary types yet.
    if (TypeIdx != 0)
      return UnableToLegalize;

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
  case TargetOpcode::G_LOAD: {
    const auto &MMO = **MI.memoperands_begin();
    unsigned DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    int NumParts = SizeOp0 / NarrowSize;
    unsigned HandledSize = NumParts * NarrowTy.getSizeInBits();
    unsigned LeftoverBits = DstTy.getSizeInBits() - HandledSize;

    if (DstTy.isVector() && LeftoverBits != 0)
      return UnableToLegalize;

    if (8 * MMO.getSize() != DstTy.getSizeInBits()) {
      unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildLoad(TmpReg, MI.getOperand(1).getReg(), MMO);
      MIRBuilder.buildAnyExt(DstReg, TmpReg);
      MI.eraseFromParent();
      return Legalized;
    }

    // This implementation doesn't work for atomics. Give up instead of doing
    // something invalid.
    if (MMO.getOrdering() != AtomicOrdering::NotAtomic ||
        MMO.getFailureOrdering() != AtomicOrdering::NotAtomic)
      return UnableToLegalize;

    LLT OffsetTy = LLT::scalar(
        MRI.getType(MI.getOperand(1).getReg()).getScalarSizeInBits());

    SmallVector<unsigned, 2> DstRegs;
    for (int i = 0; i < NumParts; ++i) {
      unsigned PartDstReg = MRI.createGenericVirtualRegister(NarrowTy);
      unsigned SrcReg = 0;
      unsigned Adjustment = i * NarrowSize / 8;
      unsigned Alignment = MinAlign(MMO.getAlignment(), Adjustment);

      MachineMemOperand *SplitMMO = MIRBuilder.getMF().getMachineMemOperand(
          MMO.getPointerInfo().getWithOffset(Adjustment), MMO.getFlags(),
          NarrowSize / 8, Alignment, MMO.getAAInfo(), MMO.getRanges(),
          MMO.getSyncScopeID(), MMO.getOrdering(), MMO.getFailureOrdering());

      MIRBuilder.materializeGEP(SrcReg, MI.getOperand(1).getReg(), OffsetTy,
                                Adjustment);

      MIRBuilder.buildLoad(PartDstReg, SrcReg, *SplitMMO);

      DstRegs.push_back(PartDstReg);
    }

    unsigned MergeResultReg = LeftoverBits == 0 ? DstReg :
      MRI.createGenericVirtualRegister(LLT::scalar(HandledSize));

    // For the leftover piece, still create the merge and insert it.
    // TODO: Would it be better to directly insert the intermediate pieces?
    if (DstTy.isVector())
      MIRBuilder.buildBuildVector(MergeResultReg, DstRegs);
    else
      MIRBuilder.buildMerge(MergeResultReg, DstRegs);

    if (LeftoverBits == 0) {
      MI.eraseFromParent();
      return Legalized;
    }

    unsigned ImpDefReg = MRI.createGenericVirtualRegister(DstTy);
    unsigned Insert0Reg = MRI.createGenericVirtualRegister(DstTy);
    MIRBuilder.buildUndef(ImpDefReg);
    MIRBuilder.buildInsert(Insert0Reg, ImpDefReg, MergeResultReg, 0);

    unsigned PartDstReg
      = MRI.createGenericVirtualRegister(LLT::scalar(LeftoverBits));
    unsigned Offset = HandledSize / 8;

    MachineMemOperand *SplitMMO = MIRBuilder.getMF().getMachineMemOperand(
      &MMO, Offset, LeftoverBits / 8);

    unsigned SrcReg = 0;
    MIRBuilder.materializeGEP(SrcReg, MI.getOperand(1).getReg(), OffsetTy,
                              Offset);
    MIRBuilder.buildLoad(PartDstReg, SrcReg, *SplitMMO);
    MIRBuilder.buildInsert(DstReg, Insert0Reg, PartDstReg, HandledSize);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    bool ZExt = MI.getOpcode() == TargetOpcode::G_ZEXTLOAD;
    unsigned DstReg = MI.getOperand(0).getReg();
    unsigned PtrReg = MI.getOperand(1).getReg();

    unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
    auto &MMO = **MI.memoperands_begin();
    if (MMO.getSize() * 8 == NarrowSize) {
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
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;

    const auto &MMO = **MI.memoperands_begin();

    unsigned SrcReg = MI.getOperand(0).getReg();
    LLT SrcTy = MRI.getType(SrcReg);

    if (8 * MMO.getSize() != SrcTy.getSizeInBits()) {
      unsigned TmpReg = MRI.createGenericVirtualRegister(NarrowTy);
      auto &MMO = **MI.memoperands_begin();
      MIRBuilder.buildTrunc(TmpReg, SrcReg);
      MIRBuilder.buildStore(TmpReg, MI.getOperand(1).getReg(), MMO);
      MI.eraseFromParent();
      return Legalized;
    }

    // This implementation doesn't work for atomics. Give up instead of doing
    // something invalid.
    if (MMO.getOrdering() != AtomicOrdering::NotAtomic ||
        MMO.getFailureOrdering() != AtomicOrdering::NotAtomic)
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowSize;
    LLT OffsetTy = LLT::scalar(
        MRI.getType(MI.getOperand(1).getReg()).getScalarSizeInBits());

    SmallVector<unsigned, 2> SrcRegs;
    extractParts(MI.getOperand(0).getReg(), NarrowTy, NumParts, SrcRegs);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = 0;
      unsigned Adjustment = i * NarrowSize / 8;
      unsigned Alignment = MinAlign(MMO.getAlignment(), Adjustment);

      MachineMemOperand *SplitMMO = MIRBuilder.getMF().getMachineMemOperand(
          MMO.getPointerInfo().getWithOffset(Adjustment), MMO.getFlags(),
          NarrowSize / 8, Alignment, MMO.getAAInfo(), MMO.getRanges(),
          MMO.getSyncScopeID(), MMO.getOrdering(), MMO.getFailureOrdering());

      MIRBuilder.materializeGEP(DstReg, MI.getOperand(1).getReg(), OffsetTy,
                                Adjustment);

      MIRBuilder.buildStore(SrcRegs[i], DstReg, *SplitMMO);
    }
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;
    const APInt &Cst = MI.getOperand(1).getCImm()->getValue();
    LLVMContext &Ctx = MIRBuilder.getMF().getFunction().getContext();

    SmallVector<unsigned, 2> DstRegs;
    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      ConstantInt *CI =
          ConstantInt::get(Ctx, Cst.lshr(NarrowSize * i).trunc(NarrowSize));
      MIRBuilder.buildConstant(DstReg, *CI);
      DstRegs.push_back(DstReg);
    }
    unsigned DstReg = MI.getOperand(0).getReg();
    if(MRI.getType(DstReg).isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
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

    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;

    // List the registers where the destination will be scattered.
    SmallVector<unsigned, 2> DstRegs;
    // List the registers where the first argument will be split.
    SmallVector<unsigned, 2> SrcsReg1;
    // List the registers where the second argument will be split.
    SmallVector<unsigned, 2> SrcsReg2;
    // Create all the temporary registers.
    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      unsigned SrcReg1 = MRI.createGenericVirtualRegister(NarrowTy);
      unsigned SrcReg2 = MRI.createGenericVirtualRegister(NarrowTy);

      DstRegs.push_back(DstReg);
      SrcsReg1.push_back(SrcReg1);
      SrcsReg2.push_back(SrcReg2);
    }
    // Explode the big arguments into smaller chunks.
    MIRBuilder.buildUnmerge(SrcsReg1, MI.getOperand(1).getReg());
    MIRBuilder.buildUnmerge(SrcsReg2, MI.getOperand(2).getReg());

    // Do the operation on each small part.
    for (int i = 0; i < NumParts; ++i)
      MIRBuilder.buildInstr(MI.getOpcode(), {DstRegs[i]},
                            {SrcsReg1[i], SrcsReg2[i]});

    // Gather the destination registers into the final destination.
    unsigned DstReg = MI.getOperand(0).getReg();
    if(MRI.getType(DstReg).isVector())
      MIRBuilder.buildBuildVector(DstReg, DstRegs);
    else
      MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {
    if (TypeIdx != 1)
      return UnableToLegalize; // TODO
    narrowScalarSrc(MI, NarrowTy, 2);
    return Legalized;
  }
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

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx, LLT WideTy) {
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_MERGE_VALUES: {
    if (TypeIdx != 1)
      return UnableToLegalize;

    unsigned DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (!DstTy.isScalar())
      return UnableToLegalize;

    unsigned NumSrc = MI.getNumOperands() - 1;
    unsigned EltSize = DstTy.getSizeInBits() / NumSrc;

    unsigned ResultReg = MRI.createGenericVirtualRegister(DstTy);
    unsigned Offset = 0;
    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I,
           Offset += EltSize) {
      assert(MRI.getType(MI.getOperand(I).getReg()) == LLT::scalar(EltSize));

      unsigned ShiftAmt = MRI.createGenericVirtualRegister(DstTy);
      unsigned Shl = MRI.createGenericVirtualRegister(DstTy);
      unsigned ZextInput = MRI.createGenericVirtualRegister(DstTy);
      MIRBuilder.buildZExt(ZextInput, MI.getOperand(I).getReg());

      if (Offset != 0) {
        unsigned NextResult = I + 1 == E ? DstReg :
          MRI.createGenericVirtualRegister(DstTy);

        MIRBuilder.buildConstant(ShiftAmt, Offset);
        MIRBuilder.buildShl(Shl, ZextInput, ShiftAmt);
        MIRBuilder.buildOr(NextResult, ResultReg, Shl);
        ResultReg = NextResult;
      } else {
        ResultReg = ZextInput;
      }
    }

    MI.eraseFromParent();
    return Legalized;
  }
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
    // First ZEXT the input.
    auto MIBSrc = MIRBuilder.buildZExt(WideTy, MI.getOperand(1).getReg());
    LLT CurTy = MRI.getType(MI.getOperand(0).getReg());
    if (MI.getOpcode() == TargetOpcode::G_CTTZ) {
      // The count is the same in the larger type except if the original
      // value was zero.  This can be handled by setting the bit just off
      // the top of the original type.
      auto TopBit =
          APInt::getOneBitSet(WideTy.getSizeInBits(), CurTy.getSizeInBits());
      MIBSrc = MIRBuilder.buildInstr(
          TargetOpcode::G_OR, {WideTy},
          {MIBSrc, MIRBuilder.buildConstant(WideTy, TopBit.getSExtValue())});
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
    auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
    // Make the original instruction a trunc now, and update its source.
    Observer.changingInstr(MI);
    MI.setDesc(TII.get(TargetOpcode::G_TRUNC));
    MI.getOperand(1).setReg(MIBNewOp->getOperand(0).getReg());
    Observer.changedInstr(MI);
    return Legalized;
  }

  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SUB:
    // Perform operation at larger width (any extension is fine here, high bits
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

  case TargetOpcode::G_INSERT:
    if (TypeIdx != 0)
      return UnableToLegalize;
    Observer.changingInstr(MI);
    widenScalarSrc(MI, WideTy, 1, TargetOpcode::G_ANYEXT);
    widenScalarDst(MI, WideTy);
    Observer.changedInstr(MI);
    return Legalized;

  case TargetOpcode::G_LOAD:
    // For some types like i24, we might try to widen to i32. To properly handle
    // this we should be using a dedicated extending load, until then avoid
    // trying to legalize.
    if (alignTo(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits(), 8) !=
        WideTy.getSizeInBits())
      return UnableToLegalize;
    LLVM_FALLTHROUGH;
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
      Val.convert(APFloat::IEEEsingle(), APFloat::rmTowardZero, &LosesInfo);
      break;
    case 64:
      Val.convert(APFloat::IEEEdouble(), APFloat::rmTowardZero, &LosesInfo);
      break;
    default:
      llvm_unreachable("Unhandled fp widen type");
    }
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
    widenScalarSrc(MI, WideTy, 0, TargetOpcode::G_ANYEXT);
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
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FSQRT:
    assert(TypeIdx == 0);
    Observer.changingInstr(MI);

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I)
      widenScalarSrc(MI, WideTy, I, TargetOpcode::G_FPEXT);

    widenScalarDst(MI, WideTy, 0, TargetOpcode::G_FPTRUNC);
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
    MIRBuilder.buildInstr(TargetOpcode::G_FSUB)
        .addDef(Res)
        .addUse(Zero->getOperand(0).getReg())
        .addUse(MI.getOperand(1).getReg());
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
    MIRBuilder.buildInstr(TargetOpcode::G_FADD)
        .addDef(Res)
        .addUse(LHS)
        .addUse(Neg);
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
      unsigned TmpReg = MRI.createGenericVirtualRegister(
          LLT::scalar(MMO.getSize() /* in bytes */ * 8));
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
      Offset += NarrowSize;
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

  if (NarrowTy.isScalar()) {
    NumParts = DstTy.getNumElements();
    NarrowTy1 = SrcTy.getElementType();
  } else {
    // Uneven breakdown not handled.
    NumParts = DstTy.getNumElements() / NarrowTy.getNumElements();
    if (NumParts * NarrowTy.getNumElements() != DstTy.getNumElements())
      return UnableToLegalize;

    NarrowTy1 = LLT::vector(NumParts, SrcTy.getElementType().getSizeInBits());
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
LegalizerHelper::fewerElementsVectorLoadStore(MachineInstr &MI, unsigned TypeIdx,
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
  unsigned NarrowSize = NarrowTy.getSizeInBits();
  unsigned Size = MRI.getType(ValReg).getSizeInBits();
  unsigned NumParts = Size / NarrowSize;

  SmallVector<unsigned, 8> NarrowRegs;
  if (!IsLoad)
    extractParts(ValReg, NarrowTy, NumParts, NarrowRegs);

  const LLT OffsetTy =
    LLT::scalar(MRI.getType(AddrReg).getScalarSizeInBits());
  MachineFunction &MF = *MI.getMF();

  for (unsigned Idx = 0; Idx < NumParts; ++Idx) {
    unsigned Adjustment = Idx * NarrowTy.getSizeInBits() / 8;
    unsigned Alignment = MinAlign(MMO->getAlignment(), Adjustment);
    unsigned NewAddrReg = 0;
    MIRBuilder.materializeGEP(NewAddrReg, AddrReg, OffsetTy, Adjustment);
    MachineMemOperand &NewMMO = *MF.getMachineMemOperand(
      MMO->getPointerInfo().getWithOffset(Adjustment), MMO->getFlags(),
      NarrowTy.getSizeInBits() / 8, Alignment);
    if (IsLoad) {
      unsigned Dst = MRI.createGenericVirtualRegister(NarrowTy);
      NarrowRegs.push_back(Dst);
      MIRBuilder.buildLoad(Dst, NewAddrReg, NewMMO);
    } else {
      MIRBuilder.buildStore(NarrowRegs[Idx], NewAddrReg, NewMMO);
    }
  }
  if (IsLoad) {
    if (NarrowTy.isVector())
      MIRBuilder.buildConcatVectors(ValReg, NarrowRegs);
    else
      MIRBuilder.buildBuildVector(ValReg, NarrowRegs);
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
  case G_FDIV:
  case G_FREM:
  case G_FMA:
  case G_FPOW:
  case G_FEXP:
  case G_FEXP2:
  case G_FLOG:
  case G_FLOG2:
  case G_FLOG10:
  case G_FCEIL:
  case G_INTRINSIC_ROUND:
  case G_INTRINSIC_TRUNC:
  case G_FCOS:
  case G_FSIN:
  case G_FSQRT:
    return fewerElementsVectorBasic(MI, TypeIdx, NarrowTy);
  case G_ZEXT:
  case G_SEXT:
  case G_ANYEXT:
  case G_FPEXT:
  case G_FPTRUNC:
  case G_SITOFP:
  case G_UITOFP:
  case G_FPTOSI:
  case G_FPTOUI:
    return fewerElementsVectorCasts(MI, TypeIdx, NarrowTy);
  case G_ICMP:
  case G_FCMP:
    return fewerElementsVectorCmp(MI, TypeIdx, NarrowTy);
  case G_SELECT:
    return fewerElementsVectorSelect(MI, TypeIdx, NarrowTy);
  case G_LOAD:
  case G_STORE:
    return fewerElementsVectorLoadStore(MI, TypeIdx, NarrowTy);
  default:
    return UnableToLegalize;
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::narrowScalarMul(MachineInstr &MI, unsigned TypeIdx, LLT NewTy) {
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned Src0 = MI.getOperand(1).getReg();
  unsigned Src1 = MI.getOperand(2).getReg();
  LLT Ty = MRI.getType(DstReg);
  if (Ty.isVector())
    return UnableToLegalize;

  unsigned Size = Ty.getSizeInBits();
  unsigned NewSize = Size / 2;
  if (Size != 2 * NewSize)
    return UnableToLegalize;

  LLT HalfTy = LLT::scalar(NewSize);
  // TODO: if HalfTy != NewTy, handle the breakdown all at once?

  unsigned ShiftAmt = MRI.createGenericVirtualRegister(Ty);
  unsigned Lo = MRI.createGenericVirtualRegister(HalfTy);
  unsigned Hi = MRI.createGenericVirtualRegister(HalfTy);
  unsigned ExtLo = MRI.createGenericVirtualRegister(Ty);
  unsigned ExtHi = MRI.createGenericVirtualRegister(Ty);
  unsigned ShiftedHi = MRI.createGenericVirtualRegister(Ty);

  SmallVector<unsigned, 2> Src0Parts;
  SmallVector<unsigned, 2> Src1Parts;

  extractParts(Src0, HalfTy, 2, Src0Parts);
  extractParts(Src1, HalfTy, 2, Src1Parts);

  MIRBuilder.buildMul(Lo, Src0Parts[0], Src1Parts[0]);

  // TODO: Use smulh or umulh depending on what the target has.
  MIRBuilder.buildUMulH(Hi, Src0Parts[1], Src1Parts[1]);

  MIRBuilder.buildConstant(ShiftAmt, NewSize);
  MIRBuilder.buildAnyExt(ExtHi, Hi);
  MIRBuilder.buildShl(ShiftedHi, ExtHi, ShiftAmt);

  MIRBuilder.buildZExt(ExtLo, Lo);
  MIRBuilder.buildOr(DstReg, ExtLo, ShiftedHi);
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
    if (isSupported({TargetOpcode::G_CTLZ_ZERO_UNDEF, {Ty}})) {
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
    if (isSupported({TargetOpcode::G_CTTZ_ZERO_UNDEF, {Ty}})) {
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
    if (!isSupported({TargetOpcode::G_CTPOP, {Ty}}) &&
        isSupported({TargetOpcode::G_CTLZ, {Ty}})) {
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
