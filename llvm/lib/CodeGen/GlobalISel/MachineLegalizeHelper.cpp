//===-- llvm/CodeGen/GlobalISel/MachineLegalizeHelper.cpp -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the MachineLegalizeHelper class to legalize
/// individual instructions and the LegalizeMachineIR wrapper pass for the
/// primary legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/MachineLegalizeHelper.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#include <sstream>

#define DEBUG_TYPE "legalize-mir"

using namespace llvm;

MachineLegalizeHelper::MachineLegalizeHelper(MachineFunction &MF)
  : MRI(MF.getRegInfo()) {
  MIRBuilder.setMF(MF);
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::legalizeInstrStep(MachineInstr &MI,
                                         const MachineLegalizer &Legalizer) {
  auto Action = Legalizer.getAction(MI, MRI);
  switch (std::get<0>(Action)) {
  case MachineLegalizer::Legal:
    return AlreadyLegal;
  case MachineLegalizer::Libcall:
    return libcall(MI);
  case MachineLegalizer::NarrowScalar:
    return narrowScalar(MI, std::get<1>(Action), std::get<2>(Action));
  case MachineLegalizer::WidenScalar:
    return widenScalar(MI, std::get<1>(Action), std::get<2>(Action));
  case MachineLegalizer::Lower:
    return lower(MI, std::get<1>(Action), std::get<2>(Action));
  case MachineLegalizer::FewerElements:
    return fewerElementsVector(MI, std::get<1>(Action), std::get<2>(Action));
  default:
    return UnableToLegalize;
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::legalizeInstr(MachineInstr &MI,
                                     const MachineLegalizer &Legalizer) {
  SmallVector<MachineInstr *, 4> WorkList;
  MIRBuilder.recordInsertions(
      [&](MachineInstr *MI) { WorkList.push_back(MI); });
  WorkList.push_back(&MI);

  bool Changed = false;
  LegalizeResult Res;
  unsigned Idx = 0;
  do {
    Res = legalizeInstrStep(*WorkList[Idx], Legalizer);
    if (Res == UnableToLegalize) {
      MIRBuilder.stopRecordingInsertions();
      return UnableToLegalize;
    }
    Changed |= Res == Legalized;
    ++Idx;
  } while (Idx < WorkList.size());

  MIRBuilder.stopRecordingInsertions();

  return Changed ? Legalized : AlreadyLegal;
}

void MachineLegalizeHelper::extractParts(unsigned Reg, LLT Ty, int NumParts,
                                         SmallVectorImpl<unsigned> &VRegs) {
  unsigned Size = Ty.getSizeInBits();
  SmallVector<uint64_t, 4> Indexes;
  for (int i = 0; i < NumParts; ++i) {
    VRegs.push_back(MRI.createGenericVirtualRegister(Ty));
    Indexes.push_back(i * Size);
  }
  MIRBuilder.buildExtract(VRegs, Indexes, Reg);
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::libcall(MachineInstr &MI) {
  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_FREM: {
    auto &Ctx = MIRBuilder.getMF().getFunction()->getContext();
    Type *Ty = Size == 64 ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx);
    auto &CLI = *MIRBuilder.getMF().getSubtarget().getCallLowering();
    auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
    const char *Name =
        TLI.getLibcallName(Size == 64 ? RTLIB::REM_F64 : RTLIB::REM_F32);

    CLI.lowerCall(MIRBuilder, MachineOperand::CreateES(Name), Ty,
                  MI.getOperand(0).getReg(), {Ty, Ty},
                  {MI.getOperand(1).getReg(), MI.getOperand(2).getReg()});
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::narrowScalar(MachineInstr &MI, unsigned TypeIdx,
                                    LLT NarrowTy) {
  // FIXME: Don't know how to handle secondary types yet.
  if (TypeIdx != 0)
    return UnableToLegalize;
  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD: {
    // Expand in terms of carry-setting/consuming G_ADDE instructions.
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    int NumParts = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() /
                   NarrowTy.getSizeInBits();

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs, Indexes;
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
      Indexes.push_back(i * NarrowSize);
      CarryIn = CarryOut;
    }
    unsigned DstReg = MI.getOperand(0).getReg();
    MIRBuilder.buildSequence(DstReg, DstRegs, Indexes);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx,
                                   LLT WideTy) {
  MIRBuilder.setInstr(MI);

  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SUB: {
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
  case TargetOpcode::G_UDIV: {
    unsigned ExtOp = MI.getOpcode() == TargetOpcode::G_SDIV
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
    assert(alignTo(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits(), 8) ==
               WideTy.getSizeInBits() &&
           "illegal to increase number of bytes modified by a store");

    unsigned SrcExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildAnyExt(SrcExt, MI.getOperand(0).getReg());
    MIRBuilder.buildStore(SrcExt, MI.getOperand(1).getReg(),
                          **MI.memoperands_begin());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    unsigned DstExt = MRI.createGenericVirtualRegister(WideTy);
    MIRBuilder.buildConstant(DstExt, MI.getOperand(1).getImm());
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
  case TargetOpcode::G_ICMP: {
    assert(TypeIdx == 1 && "unable to legalize predicate");
    bool IsSigned = CmpInst::isSigned(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()));
    unsigned Op0Ext = MRI.createGenericVirtualRegister(WideTy);
    unsigned Op1Ext = MRI.createGenericVirtualRegister(WideTy);
    if (IsSigned) {
      MIRBuilder.buildSExt(Op0Ext, MI.getOperand(2).getReg());
      MIRBuilder.buildSExt(Op1Ext, MI.getOperand(3).getReg());
    } else {
      MIRBuilder.buildZExt(Op0Ext, MI.getOperand(2).getReg());
      MIRBuilder.buildZExt(Op1Ext, MI.getOperand(3).getReg());
    }
    MIRBuilder.buildICmp(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()),
        MI.getOperand(0).getReg(), Op0Ext, Op1Ext);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::lower(MachineInstr &MI, unsigned TypeIdx, LLT Ty) {
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
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::fewerElementsVector(MachineInstr &MI, unsigned TypeIdx,
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
    int NumParts = MRI.getType(DstReg).getSizeInBits() / NarrowSize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs, Indexes;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowTy);
      MIRBuilder.buildAdd(DstReg, Src1Regs[i], Src2Regs[i]);
      DstRegs.push_back(DstReg);
      Indexes.push_back(i * NarrowSize);
    }

    MIRBuilder.buildSequence(DstReg, DstRegs, Indexes);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}
