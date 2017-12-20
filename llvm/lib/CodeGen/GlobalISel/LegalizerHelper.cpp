//===-- llvm/CodeGen/GlobalISel/LegalizerHelper.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG_TYPE "legalizer"

using namespace llvm;

LegalizerHelper::LegalizerHelper(MachineFunction &MF)
    : MRI(MF.getRegInfo()), LI(*MF.getSubtarget().getLegalizerInfo()) {
  MIRBuilder.setMF(MF);
}

LegalizerHelper::LegalizeResult
LegalizerHelper::legalizeInstrStep(MachineInstr &MI) {
  DEBUG(dbgs() << "Legalizing: "; MI.print(dbgs()));

  auto Action = LI.getAction(MI, MRI);
  switch (std::get<0>(Action)) {
  case LegalizerInfo::Legal:
    DEBUG(dbgs() << ".. Already legal\n");
    return AlreadyLegal;
  case LegalizerInfo::Libcall:
    DEBUG(dbgs() << ".. Convert to libcall\n");
    return libcall(MI);
  case LegalizerInfo::NarrowScalar:
    DEBUG(dbgs() << ".. Narrow scalar\n");
    return narrowScalar(MI, std::get<1>(Action), std::get<2>(Action));
  case LegalizerInfo::WidenScalar:
    DEBUG(dbgs() << ".. Widen scalar\n");
    return widenScalar(MI, std::get<1>(Action), std::get<2>(Action));
  case LegalizerInfo::Lower:
    DEBUG(dbgs() << ".. Lower\n");
    return lower(MI, std::get<1>(Action), std::get<2>(Action));
  case LegalizerInfo::FewerElements:
    DEBUG(dbgs() << ".. Reduce number of elements\n");
    return fewerElementsVector(MI, std::get<1>(Action), std::get<2>(Action));
  case LegalizerInfo::Custom:
    DEBUG(dbgs() << ".. Custom legalization\n");
    return LI.legalizeCustom(MI, MRI, MIRBuilder) ? Legalized
                                                  : UnableToLegalize;
  default:
    DEBUG(dbgs() << ".. Unable to legalize\n");
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
    assert(Size == 32 && "Unsupported size");
    return RTLIB::SDIV_I32;
  case TargetOpcode::G_UDIV:
    assert(Size == 32 && "Unsupported size");
    return RTLIB::UDIV_I32;
  case TargetOpcode::G_SREM:
    assert(Size == 32 && "Unsupported size");
    return RTLIB::SREM_I32;
  case TargetOpcode::G_UREM:
    assert(Size == 32 && "Unsupported size");
    return RTLIB::UREM_I32;
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

static LegalizerHelper::LegalizeResult
simpleLibcall(MachineInstr &MI, MachineIRBuilder &MIRBuilder, unsigned Size,
              Type *OpType) {
  auto Libcall = getRTLibDesc(MI.getOpcode(), Size);
  return createLibcall(MIRBuilder, Libcall, {MI.getOperand(0).getReg(), OpType},
                       {{MI.getOperand(1).getReg(), OpType},
                        {MI.getOperand(2).getReg(), OpType}});
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
  case TargetOpcode::G_UREM: {
    Type *HLTy = Type::getInt32Ty(Ctx);
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
    if (Status != Legalized)
      return Status;
    break;
  }
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_FREM: {
    Type *HLTy = Size == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx);
    auto Status = simpleLibcall(MI, MIRBuilder, Size, HLTy);
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
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0 && MI.getOpcode() != TargetOpcode::G_EXTRACT)
    return UnableToLegalize;

  MIRBuilder.setInstr(MI);

  int64_t SizeOp0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  int64_t NarrowSize = NarrowTy.getSizeInBits();

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
    for (int i = 0; i < NumParts; ++i) {
      unsigned Dst = MRI.createGenericVirtualRegister(NarrowTy);
      MIRBuilder.buildUndef(Dst);
      DstRegs.push_back(Dst);
    }
    MIRBuilder.buildMerge(MI.getOperand(0).getReg(), DstRegs);
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
    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
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
    int64_t OpStart = MI.getOperand(2).getImm();
    int64_t OpSize = MRI.getType(OpReg).getSizeInBits();
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
      int64_t ExtractOffset, SegSize;
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

    MIRBuilder.buildMerge(MI.getOperand(0).getReg(), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_INSERT: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;

    int NumParts = SizeOp0 / NarrowSize;

    SmallVector<unsigned, 2> SrcRegs, DstRegs;
    SmallVector<uint64_t, 2> Indexes;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, SrcRegs);

    unsigned OpReg = MI.getOperand(2).getReg();
    int64_t OpStart = MI.getOperand(3).getImm();
    int64_t OpSize = MRI.getType(OpReg).getSizeInBits();
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
      int64_t ExtractOffset, InsertOffset, SegSize;
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
    MIRBuilder.buildMerge(MI.getOperand(0).getReg(), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_LOAD: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;
    LLT OffsetTy = LLT::scalar(
        MRI.getType(MI.getOperand(1).getReg()).getScalarSizeInBits());

    SmallVector<unsigned, 2> DstRegs;
    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      unsigned SrcReg = 0;
      unsigned Adjustment = i * NarrowSize / 8;

      MIRBuilder.materializeGEP(SrcReg, MI.getOperand(1).getReg(), OffsetTy,
                                Adjustment);

      // TODO: This is conservatively correct, but we probably want to split the
      // memory operands in the future.
      MIRBuilder.buildLoad(DstReg, SrcReg, **MI.memoperands_begin());

      DstRegs.push_back(DstReg);
    }
    unsigned DstReg = MI.getOperand(0).getReg();
    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    // FIXME: add support for when SizeOp0 isn't an exact multiple of
    // NarrowSize.
    if (SizeOp0 % NarrowSize != 0)
      return UnableToLegalize;
    int NumParts = SizeOp0 / NarrowSize;
    LLT OffsetTy = LLT::scalar(
        MRI.getType(MI.getOperand(1).getReg()).getScalarSizeInBits());

    SmallVector<unsigned, 2> SrcRegs;
    extractParts(MI.getOperand(0).getReg(), NarrowTy, NumParts, SrcRegs);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = 0;
      unsigned Adjustment = i * NarrowSize / 8;

      MIRBuilder.materializeGEP(DstReg, MI.getOperand(1).getReg(), OffsetTy,
                                Adjustment);

      // TODO: This is conservatively correct, but we probably want to split the
      // memory operands in the future.
      MIRBuilder.buildStore(SrcRegs[i], DstReg, **MI.memoperands_begin());
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
    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_OR: {
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
      MIRBuilder.buildOr(DstRegs[i], SrcsReg1[i], SrcsReg2[i]);

    // Gather the destination registers into the final destination.
    unsigned DstReg = MI.getOperand(0).getReg();
    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx, LLT WideTy) {
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_SHL: {
    // Perform operation at larger width (any extension is fine here, high bits
    // don't affect the result) and then truncate the result back to the
    // original type.
    unsigned Src1Ext = MRI.createGenericVirtualRegister(WideTy);
    unsigned Src2Ext = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildAnyExt(Src1Ext, MI.getOperand(1).getReg());
    MIRBuilder.buildAnyExt(Src2Ext, MI.getOperand(2).getReg());

    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(MI.getOpcode())
        .addDef(DstExt)
        .addUse(Src1Ext)
        .addUse(Src2Ext);

    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR: {
    unsigned ExtOp = MI.getOpcode() == TargetOpcode::G_SDIV ||
                             MI.getOpcode() == TargetOpcode::G_SREM ||
                             MI.getOpcode() == TargetOpcode::G_ASHR
                         ? TargetOpcode::G_SEXT
                         : TargetOpcode::G_ZEXT;

    unsigned LHSExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(ExtOp).addDef(LHSExt).addUse(
        MI.getOperand(1).getReg());

    unsigned RHSExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(ExtOp).addDef(RHSExt).addUse(
        MI.getOperand(2).getReg());

    unsigned ResExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(MI.getOpcode())
        .addDef(ResExt)
        .addUse(LHSExt)
        .addUse(RHSExt);

    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), ResExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SELECT: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    // Perform operation at larger width (any extension is fine here, high bits
    // don't affect the result) and then truncate the result back to the
    // original type.
    unsigned Src1Ext = MRI.createGenericVirtualRegister(WideTy);
    unsigned Src2Ext = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildAnyExt(Src1Ext, MI.getOperand(2).getReg());
    MIRBuilder.buildAnyExt(Src2Ext, MI.getOperand(3).getReg());

    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(TargetOpcode::G_SELECT)
        .addDef(DstExt)
        .addReg(MI.getOperand(1).getReg())
        .addUse(Src1Ext)
        .addUse(Src2Ext);

    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(MI.getOpcode())
        .addDef(DstExt)
        .addUse(MI.getOperand(1).getReg());

    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP: {
    if (TypeIdx != 1)
      return UnableToLegalize;

    unsigned Src = MI.getOperand(1).getReg();
    unsigned SrcExt = MRI.createGenericVirtualRegister(WideTy);

    if (MI.getOpcode() == TargetOpcode::G_SITOFP) {
      MIRBuilder.buildSExt(SrcExt, Src);
    } else {
      assert(MI.getOpcode() == TargetOpcode::G_UITOFP && "Unexpected conv op");
      MIRBuilder.buildZExt(SrcExt, Src);
    }

    MIRBuilder.buildInstr(MI.getOpcode())
        .addDef(MI.getOperand(0).getReg())
        .addUse(SrcExt);

    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_INSERT: {
    if (TypeIdx != 0)
      return UnableToLegalize;

    unsigned Src = MI.getOperand(1).getReg();
    unsigned SrcExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildAnyExt(SrcExt, Src);

    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    auto MIB = MIRBuilder.buildInsert(DstExt, SrcExt, MI.getOperand(2).getReg(),
                                      MI.getOperand(3).getImm());
    for (unsigned OpNum = 4; OpNum < MI.getNumOperands(); OpNum += 2) {
      MIB.addReg(MI.getOperand(OpNum).getReg());
      MIB.addImm(MI.getOperand(OpNum + 1).getImm());
    }

    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_LOAD: {
    assert(alignTo(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits(), 8) ==
               WideTy.getSizeInBits() &&
           "illegal to increase number of bytes loaded");

    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildLoad(DstExt, MI.getOperand(1).getReg(),
                         **MI.memoperands_begin());
    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    if (MRI.getType(MI.getOperand(0).getReg()) != LLT::scalar(1) ||
        WideTy != LLT::scalar(8))
      return UnableToLegalize;

    auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
    auto Content = TLI.getBooleanContents(false, false);

    unsigned ExtOp = TargetOpcode::G_ANYEXT;
    if (Content == TargetLoweringBase::ZeroOrOneBooleanContent)
      ExtOp = TargetOpcode::G_ZEXT;
    else if (Content == TargetLoweringBase::ZeroOrNegativeOneBooleanContent)
      ExtOp = TargetOpcode::G_SEXT;
    else
      ExtOp = TargetOpcode::G_ANYEXT;

    unsigned SrcExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildInstr(ExtOp).addDef(SrcExt).addUse(
        MI.getOperand(0).getReg());
    MIRBuilder.buildStore(SrcExt, MI.getOperand(1).getReg(),
                          **MI.memoperands_begin());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildConstant(DstExt, *MI.getOperand(1).getCImm());
    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FCONSTANT: {
    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildFConstant(DstExt, *MI.getOperand(1).getFPImm());
    MIRBuilder.buildFPTrunc(MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_BRCOND: {
    unsigned TstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildAnyExt(TstExt, MI.getOperand(0).getReg());
    MIRBuilder.buildBrCond(TstExt, *MI.getOperand(1).getMBB());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FCMP: {
    unsigned Op0Ext, Op1Ext, DstReg;
    unsigned Cmp1 = MI.getOperand(2).getReg();
    unsigned Cmp2 = MI.getOperand(3).getReg();
    if (TypeIdx == 0) {
      Op0Ext = Cmp1;
      Op1Ext = Cmp2;
      DstReg = MRI.createGenericVirtualRegister(WideTy);
    } else {
      Op0Ext = MRI.createGenericVirtualRegister(WideTy);
      Op1Ext = MRI.createGenericVirtualRegister(WideTy);
      DstReg = MI.getOperand(0).getReg();
      MIRBuilder.buildInstr(TargetOpcode::G_FPEXT, Op0Ext, Cmp1);
      MIRBuilder.buildInstr(TargetOpcode::G_FPEXT, Op1Ext, Cmp2);
    }
    MIRBuilder.buildFCmp(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()),
        DstReg, Op0Ext, Op1Ext);
    if (TypeIdx == 0)
      MIRBuilder.buildInstr(TargetOpcode::G_TRUNC, MI.getOperand(0).getReg(),
                            DstReg);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_ICMP: {
    bool IsSigned = CmpInst::isSigned(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()));
    unsigned Cmp1 = MI.getOperand(2).getReg();
    unsigned Cmp2 = MI.getOperand(3).getReg();
    unsigned Op0Ext, Op1Ext, DstReg;
    if (TypeIdx == 0) {
      Op0Ext = Cmp1;
      Op1Ext = Cmp2;
      DstReg = MRI.createGenericVirtualRegister(WideTy);
    } else {
      Op0Ext = MRI.createGenericVirtualRegister(WideTy);
      Op1Ext = MRI.createGenericVirtualRegister(WideTy);
      DstReg = MI.getOperand(0).getReg();
      if (IsSigned) {
        MIRBuilder.buildSExt(Op0Ext, Cmp1);
        MIRBuilder.buildSExt(Op1Ext, Cmp2);
      } else {
        MIRBuilder.buildZExt(Op0Ext, Cmp1);
        MIRBuilder.buildZExt(Op1Ext, Cmp2);
      }
    }
    MIRBuilder.buildICmp(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()),
        DstReg, Op0Ext, Op1Ext);
    if (TypeIdx == 0)
      MIRBuilder.buildInstr(TargetOpcode::G_TRUNC, MI.getOperand(0).getReg(),
                            DstReg);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_GEP: {
    assert(TypeIdx == 1 && "unable to legalize pointer of GEP");
    unsigned OffsetExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildSExt(OffsetExt, MI.getOperand(2).getReg());
    MI.getOperand(2).setReg(OffsetExt);
    return Legalized;
  }
  case TargetOpcode::G_PHI: {
    assert(TypeIdx == 0 && "Expecting only Idx 0");
    auto getExtendedReg = [&](unsigned Reg, MachineBasicBlock &MBB) {
      auto FirstTermIt = MBB.getFirstTerminator();
      MIRBuilder.setInsertPt(MBB, FirstTermIt);
      MachineInstr *DefMI = MRI.getVRegDef(Reg);
      MachineInstrBuilder MIB;
      if (DefMI->getOpcode() == TargetOpcode::G_TRUNC)
        MIB = MIRBuilder.buildAnyExtOrTrunc(WideTy,
                                            DefMI->getOperand(1).getReg());
      else
        MIB = MIRBuilder.buildAnyExt(WideTy, Reg);
      return MIB->getOperand(0).getReg();
    };
    auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_PHI, WideTy);
    for (auto OpIt = MI.operands_begin() + 1, OpE = MI.operands_end();
         OpIt != OpE;) {
      unsigned Reg = OpIt++->getReg();
      MachineBasicBlock *OpMBB = OpIt++->getMBB();
      MIB.addReg(getExtendedReg(Reg, *OpMBB));
      MIB.addMBB(OpMBB);
    }
    auto *MBB = MI.getParent();
    MIRBuilder.setInsertPt(*MBB, MBB->getFirstNonPHI());
    MIRBuilder.buildTrunc(MI.getOperand(0).getReg(),
                          MIB->getOperand(0).getReg());
    MI.eraseFromParent();
    return Legalized;
  }
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
    MIRBuilder.buildICmp(CmpInst::ICMP_NE, Overflow, HiPart, Zero);
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
    unsigned Zero = MRI.createGenericVirtualRegister(Ty);
    MIRBuilder.buildFConstant(Zero, ZeroForNegation);
    MIRBuilder.buildInstr(TargetOpcode::G_FSUB)
        .addDef(Res)
        .addUse(Zero)
        .addUse(MI.getOperand(1).getReg());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FSUB: {
    // Lower (G_FSUB LHS, RHS) to (G_FADD LHS, (G_FNEG RHS)).
    // First, check if G_FNEG is marked as Lower. If so, we may
    // end up with an infinite loop as G_FSUB is used to legalize G_FNEG.
    if (LI.getAction({G_FNEG, Ty}).first == LegalizerInfo::Lower)
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
  }
}

LegalizerHelper::LegalizeResult
LegalizerHelper::fewerElementsVector(MachineInstr &MI, unsigned TypeIdx,
                                     LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;
  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD: {
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    unsigned DstReg = MI.getOperand(0).getReg();
    unsigned Size = MRI.getType(DstReg).getSizeInBits();
    int NumParts = Size / NarrowSize;
    // FIXME: Don't know how to handle the situation where the small vectors
    // aren't all the same size yet.
    if (Size % NarrowSize != 0)
      return UnableToLegalize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      MIRBuilder.buildAdd(DstReg, Src1Regs[i], Src2Regs[i]);
      DstRegs.push_back(DstReg);
    }

    MIRBuilder.buildMerge(DstReg, DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}
