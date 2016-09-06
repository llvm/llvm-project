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

MachineLegalizeHelper::LegalizeResult MachineLegalizeHelper::legalizeInstr(
    MachineInstr &MI, const MachineLegalizer &Legalizer) {
  auto Action = Legalizer.getAction(MI);
  switch (Action.first) {
  case MachineLegalizer::Legal:
    return AlreadyLegal;
  case MachineLegalizer::NarrowScalar:
    return narrowScalar(MI, Action.second);
  case MachineLegalizer::WidenScalar:
    return widenScalar(MI, Action.second);
  case MachineLegalizer::FewerElements:
    return fewerElementsVector(MI, Action.second);
  default:
    return UnableToLegalize;
  }
}

void MachineLegalizeHelper::extractParts(unsigned Reg, LLT Ty, int NumParts,
                                         SmallVectorImpl<unsigned> &VRegs) {
  unsigned Size = Ty.getSizeInBits();
  SmallVector<unsigned, 4> Indexes;
  for (int i = 0; i < NumParts; ++i) {
    VRegs.push_back(MRI.createGenericVirtualRegister(Size));
    Indexes.push_back(i * Size);
  }
  MIRBuilder.buildExtract(Ty, VRegs, Reg, Indexes);
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::narrowScalar(MachineInstr &MI, LLT NarrowTy) {
  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD: {
    // Expand in terms of carry-setting/consuming G_ADDE instructions.
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    int NumParts = MI.getType().getSizeInBits() / NarrowSize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    unsigned CarryIn = MRI.createGenericVirtualRegister(1);
    MIRBuilder.buildConstant(LLT::scalar(1), CarryIn, 0);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowSize);
      unsigned CarryOut = MRI.createGenericVirtualRegister(1);

      MIRBuilder.buildAdde(NarrowTy, DstReg, CarryOut, Src1Regs[i], Src2Regs[i],
                           CarryIn);

      DstRegs.push_back(DstReg);
      CarryIn = CarryOut;
    }
    MIRBuilder.buildSequence(MI.getType(), MI.getOperand(0).getReg(), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::widenScalar(MachineInstr &MI, LLT WideTy) {
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
    unsigned WideSize = WideTy.getSizeInBits();

    MIRBuilder.setInstr(MI);

    unsigned Src1Ext = MRI.createGenericVirtualRegister(WideSize);
    unsigned Src2Ext = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildAnyExtend(WideTy, Src1Ext, MI.getOperand(1).getReg());
    MIRBuilder.buildAnyExtend(WideTy, Src2Ext, MI.getOperand(2).getReg());

    unsigned DstExt = MRI.createGenericVirtualRegister(WideSize);
    MIRBuilder.buildInstr(MI.getOpcode(), WideTy)
      .addDef(DstExt).addUse(Src1Ext).addUse(Src2Ext);

    MIRBuilder.buildTrunc(MI.getType(), MI.getOperand(0).getReg(), DstExt);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}

MachineLegalizeHelper::LegalizeResult
MachineLegalizeHelper::fewerElementsVector(MachineInstr &MI, LLT NarrowTy) {
  switch (MI.getOpcode()) {
  default:
    return UnableToLegalize;
  case TargetOpcode::G_ADD: {
    unsigned NarrowSize = NarrowTy.getSizeInBits();
    int NumParts = MI.getType().getSizeInBits() / NarrowSize;

    MIRBuilder.setInstr(MI);

    SmallVector<unsigned, 2> Src1Regs, Src2Regs, DstRegs;
    extractParts(MI.getOperand(1).getReg(), NarrowTy, NumParts, Src1Regs);
    extractParts(MI.getOperand(2).getReg(), NarrowTy, NumParts, Src2Regs);

    for (int i = 0; i < NumParts; ++i) {
      unsigned DstReg = MRI.createGenericVirtualRegister(NarrowSize);
      MIRBuilder.buildAdd(NarrowTy, DstReg, Src1Regs[i], Src2Regs[i]);
      DstRegs.push_back(DstReg);
    }

    MIRBuilder.buildSequence(MI.getType(), MI.getOperand(0).getReg(), DstRegs);
    MI.eraseFromParent();
    return Legalized;
  }
  }
}
