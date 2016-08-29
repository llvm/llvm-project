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
#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
  auto Action = Legalizer.getAction(MI);
  switch (std::get<0>(Action)) {
  case MachineLegalizer::Legal:
    return AlreadyLegal;
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
  std::queue<MachineInstr *> WorkList;
  MIRBuilder.recordInsertions([&](MachineInstr *MI) { WorkList.push(MI); });
  WorkList.push(&MI);

  bool Changed = false;
  LegalizeResult Res;
  do {
    Res = legalizeInstrStep(*WorkList.front(), Legalizer);
    if (Res == UnableToLegalize) {
      MIRBuilder.stopRecordingInsertions();
      return UnableToLegalize;
    }
    Changed |= Res == Legalized;
    WorkList.pop();
  } while (!WorkList.empty());

  MIRBuilder.stopRecordingInsertions();

  return Changed ? Legalized : AlreadyLegal;
}

void MachineLegalizeHelper::extractParts(unsigned Reg, LLT Ty, int NumParts,
                                         SmallVectorImpl<unsigned> &VRegs) {
  unsigned Size = Ty.getSizeInBits();
  SmallVector<uint64_t, 4> Indexes;
  SmallVector<LLT, 4> ResTys;
  for (int i = 0; i < NumParts; ++i) {
    VRegs.push_back(MRI.createGenericVirtualRegister(Size));
    Indexes.push_back(i * Size);
    ResTys.push_back(Ty);
  }
  MIRBuilder.buildExtract(ResTys, VRegs, Indexes,
                          LLT::scalar(Ty.getSizeInBits() * NumParts), Reg);
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
    int NumParts = MI.getType().getSizeInBits() / NarrowSize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs, Indexes;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    unsigned CarryIn = MRI.createGenericVirtualRegister(1);
    MIRBuilder.buildConstant(LLT::scalar(1), CarryIn, 0);

    SmallVector<LLT, 2> DstTys;
    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowSize);
      unsigned CarryOut = MRI.createGenericVirtualRegister(1);

      MIRBuilder.buildUAdde(NarrowTy, DstReg, CarryOut, Src1Regs[i],
                            Src2Regs[i], CarryIn);

      DstTys.push_back(NarrowTy);
      DstRegs.push_back(DstReg);
      Indexes.push_back(i * NarrowSize);
      CarryIn = CarryOut;
    }
    MIRBuilder.buildSequence(MI.getType(), MI.getOperand(0).getReg(), DstTys,
                             DstRegs, Indexes);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::widenScalar(MachineInstr &MI, unsigned TypeIdx,
                                   LLT WideTy) {
  LLT Ty = MI.getType();
  unsigned WideSize = WideTy.getSizeInBits();
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
    unsigned Src1Ext = MRI.createGenericVirtualRegister(WideSize);
    unsigned Src2Ext = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildAnyExt({WideTy, Ty}, Src1Ext, MI.getOperand(1).getReg());
    MIRBuilder.buildAnyExt({WideTy, Ty}, Src2Ext, MI.getOperand(2).getReg());

    unsigned DstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildInstr(MI.getOpcode(), WideTy)
      .addDef(DstExt).addUse(Src1Ext).addUse(Src2Ext);

    MIRBuilder.buildTrunc({Ty, WideTy}, MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV: {
    unsigned ExtOp = MI.getOpcode() == TargetOpcode::G_SDIV
                          ? TargetOpcode::G_SEXT
                          : TargetOpcode::G_ZEXT;

    unsigned LHSExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildInstr(ExtOp, {WideTy, MI.getType()})
        .addDef(LHSExt)
        .addUse(MI.getOperand(1).getReg());

    unsigned RHSExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildInstr(ExtOp, {WideTy, MI.getType()})
        .addDef(RHSExt)
        .addUse(MI.getOperand(2).getReg());

    unsigned ResExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildInstr(MI.getOpcode(), WideTy)
        .addDef(ResExt)
        .addUse(LHSExt)
        .addUse(RHSExt);

    MIRBuilder.buildTrunc({MI.getType(), WideTy}, MI.getOperand(0).getReg(),
                          ResExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_LOAD: {
    assert(alignTo(Ty.getSizeInBits(), 8) == WideSize &&
           "illegal to increase number of bytes loaded");

    unsigned DstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildLoad(WideTy, MI.getType(1), DstExt,
                         MI.getOperand(1).getReg(), **MI.memoperands_begin());
    MIRBuilder.buildTrunc({Ty, WideTy}, MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_STORE: {
    assert(alignTo(Ty.getSizeInBits(), 8) == WideSize &&
           "illegal to increase number of bytes modified by a store");

    unsigned SrcExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildAnyExt({WideTy, Ty}, SrcExt, MI.getOperand(0).getReg());
    MIRBuilder.buildStore(WideTy, MI.getType(1), SrcExt,
                          MI.getOperand(1).getReg(), **MI.memoperands_begin());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_CONSTANT: {
    unsigned DstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildConstant(WideTy, DstExt, MI.getOperand(1).getImm());
    MIRBuilder.buildTrunc({Ty, WideTy}, MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_FCONSTANT: {
    unsigned DstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildFConstant(WideTy, DstExt, *MI.getOperand(1).getFPImm());
    MIRBuilder.buildFPTrunc({Ty, WideTy}, MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_BRCOND: {
    unsigned TstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildAnyExt({WideTy, Ty}, TstExt, MI.getOperand(0).getReg());
    MIRBuilder.buildBrCond(WideTy, TstExt, *MI.getOperand(1).getMBB());
    MI.eraseFromParent();
    return Legalized;
  }
  case TargetOpcode::G_ICMP: {
    assert(TypeIdx == 1 && "unable to legalize predicate");
    bool IsSigned = CmpInst::isSigned(
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate()));
    unsigned Op0Ext = MRI.createGenericVirtualRegister(WideSize);
    unsigned Op1Ext = MRI.createGenericVirtualRegister(WideSize);
    if (IsSigned) {
      MIRBuilder.buildSExt({WideTy, MI.getType(1)}, Op0Ext,
                           MI.getOperand(2).getReg());
      MIRBuilder.buildSExt({WideTy, MI.getType(1)}, Op1Ext,
                           MI.getOperand(3).getReg());
    } else {
      MIRBuilder.buildZExt({WideTy, MI.getType(1)}, Op0Ext,
                           MI.getOperand(2).getReg());
      MIRBuilder.buildZExt({WideTy, MI.getType(1)}, Op1Ext,
                           MI.getOperand(3).getReg());
    }
    MIRBuilder.buildICmp(
        {MI.getType(0), WideTy},
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
  unsigned Size = Ty.getSizeInBits();
  MIRBuilder.setInstr(MI);

  switch(MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM: {
    unsigned QuotReg = MRI.createGenericVirtualRegister(Size);
    MIRBuilder.buildInstr(MI.getOpcode() == G_SREM ? G_SDIV : G_UDIV, Ty)
        .addDef(QuotReg)
        .addUse(MI.getOperand(1).getReg())
        .addUse(MI.getOperand(2).getReg());

    unsigned ProdReg = MRI.createGenericVirtualRegister(Size);
    MIRBuilder.buildMul(Ty, ProdReg, QuotReg, MI.getOperand(2).getReg());
    MIRBuilder.buildSub(Ty, MI.getOperand(0).getReg(),
                        MI.getOperand(1).getReg(), ProdReg);
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
    int NumParts = MI.getType().getSizeInBits() / NarrowSize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs, Indexes;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    SmallVector<LLT, 2> DstTys;
    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowSize);
      MIRBuilder.buildAdd(NarrowTy, DstReg, Src1Regs[i], Src2Regs[i]);
      DstTys.push_back(NarrowTy);
      DstRegs.push_back(DstReg);
      Indexes.push_back(i * NarrowSize);
    }

    MIRBuilder.buildSequence(MI.getType(), MI.getOperand(0).getReg(), DstTys,
                             DstRegs, Indexes);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}
