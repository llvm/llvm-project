//===- CombinerHelperSelect.cpp--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for G_SELECT.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

// select(slt(lhs,rhs),sub(rhs,lhs),sub(lhs,rhs) -> abds(lhs, rhs)
bool CombinerHelper::matchSelectAbds(const MachineInstr &MI) const {
  const GSelect *Select = cast<GSelect>(&MI);
  GSub *LHS = cast<GSub>(MRI.getVRegDef(Select->getTrueReg()));
  GSub *RHS = cast<GSub>(MRI.getVRegDef(Select->getFalseReg()));

  if (!MRI.hasOneNonDBGUse(Select->getCondReg()) ||
      !MRI.hasOneNonDBGUse(LHS->getReg(0)) ||
      !MRI.hasOneNonDBGUse(RHS->getReg(0)))
    return false;

  Register Dst = Select->getReg(0);
  LLT DstTy = MRI.getType(Dst);

  return isLegalOrBeforeLegalizer({TargetOpcode::G_ABDS, {DstTy}});
}

// select(ult(lhs,rhs),sub(rhs,lhs),sub(lhs,rhs)) -> abdu(lhs, rhs)
bool CombinerHelper::matchSelectAbdu(const MachineInstr &MI) const {
  const GSelect *Select = cast<GSelect>(&MI);
  GSub *LHS = cast<GSub>(MRI.getVRegDef(Select->getTrueReg()));
  GSub *RHS = cast<GSub>(MRI.getVRegDef(Select->getFalseReg()));

  if (!MRI.hasOneNonDBGUse(Select->getCondReg()) ||
      !MRI.hasOneNonDBGUse(LHS->getReg(0)) ||
      !MRI.hasOneNonDBGUse(RHS->getReg(0)))
    return false;

  Register Dst = Select->getReg(0);
  LLT DstTy = MRI.getType(Dst);

  return isLegalOrBeforeLegalizer({TargetOpcode::G_ABDU, {DstTy}});
}
