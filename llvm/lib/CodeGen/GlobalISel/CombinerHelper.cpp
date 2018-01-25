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

bool CombinerHelper::tryCombine(MachineInstr &MI) {
  return tryCombineCopy(MI);
}
