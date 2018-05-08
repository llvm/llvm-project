//== ---lib/CodeGen/GlobalISel/GICombinerHelper.cpp --------------------- == //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "gi-combine"

using namespace llvm;

CombinerHelper::CombinerHelper(MachineIRBuilder &B) :
  Builder(B), MRI(Builder.getMF().getRegInfo()) {}

bool CombinerHelper::tryCombineCopy(MachineInstr &MI) {
  if (MI.getOpcode() != TargetOpcode::COPY)
    return false;
  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  // Simple Copy Propagation.
  // a(sx) = COPY b(sx) -> Replace all uses of a with b.
  if (DstTy.isValid() && SrcTy.isValid() && DstTy == SrcTy) {
    MI.eraseFromParent();
    MRI.replaceRegWith(DstReg, SrcReg);
    return true;
  }
  return false;
}

bool CombinerHelper::tryCombineExtendingLoads(MachineInstr &MI) {
  if (MI.getOpcode() != TargetOpcode::G_ANYEXT &&
      MI.getOpcode() != TargetOpcode::G_SEXT &&
      MI.getOpcode() != TargetOpcode::G_ZEXT)
    return false;

  unsigned DstReg = MI.getOperand(0).getReg();
  unsigned SrcReg = MI.getOperand(1).getReg();

  LLT DstTy = MRI.getType(DstReg);
  if (!DstTy.isScalar())
    return false;

  if (MachineInstr *DefMI = getOpcodeDef(TargetOpcode::G_LOAD, SrcReg, MRI)) {
    unsigned PtrReg = DefMI->getOperand(1).getReg();
    MachineMemOperand &MMO = **DefMI->memoperands_begin();
    DEBUG(dbgs() << ".. Combine MI: " << MI;);
    Builder.setInstr(MI);
    Builder.buildLoadInstr(MI.getOpcode() == TargetOpcode::G_SEXT
                               ? TargetOpcode::G_SEXTLOAD
                               : MI.getOpcode() == TargetOpcode::G_ZEXT
                                     ? TargetOpcode::G_ZEXTLOAD
                                     : TargetOpcode::G_LOAD,
                           DstReg, PtrReg, MMO);
    MI.eraseFromParent();
    return true;
  }
  return false;
}

bool CombinerHelper::tryCombine(MachineInstr &MI) {
  if (tryCombineCopy(MI))
    return true;
  return tryCombineExtendingLoads(MI);
}
