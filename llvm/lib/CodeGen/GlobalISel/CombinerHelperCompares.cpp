//===- CombinerHelperCompares.cpp------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for G_ICMP.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

bool CombinerHelper::constantFoldICmp(const GICmp &ICmp,
                                      const GIConstant &LHSCst,
                                      const GIConstant &RHSCst,
                                      BuildFnTy &MatchInfo) {
  if (LHSCst.getKind() != GIConstant::GIConstantKind::Scalar)
    return false;

  Register Dst = ICmp.getReg(0);
  LLT DstTy = MRI.getType(Dst);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return false;

  CmpInst::Predicate Pred = ICmp.getCond();
  APInt LHS = LHSCst.getScalarValue();
  APInt RHS = RHSCst.getScalarValue();

  bool Result = ICmpInst::compare(LHS, RHS, Pred);

  MatchInfo = [=](MachineIRBuilder &B) {
    if (Result)
      B.buildConstant(Dst, getICmpTrueVal(getTargetLowering(),
                                          /*IsVector=*/DstTy.isVector(),
                                          /*IsFP=*/false));
    else
      B.buildConstant(Dst, 0);
  };

  return true;
}

bool CombinerHelper::constantFoldFCmp(const GFCmp &FCmp,
                                      const GFConstant &LHSCst,
                                      const GFConstant &RHSCst,
                                      BuildFnTy &MatchInfo) {
  if (LHSCst.getKind() != GFConstant::GFConstantKind::Scalar)
    return false;

  Register Dst = FCmp.getReg(0);
  LLT DstTy = MRI.getType(Dst);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return false;

  CmpInst::Predicate Pred = FCmp.getCond();
  APFloat LHS = LHSCst.getScalarValue();
  APFloat RHS = RHSCst.getScalarValue();

  bool Result = FCmpInst::compare(LHS, RHS, Pred);

  MatchInfo = [=](MachineIRBuilder &B) {
    if (Result)
      B.buildConstant(Dst, getICmpTrueVal(getTargetLowering(),
                                          /*IsVector=*/DstTy.isVector(),
                                          /*IsFP=*/true));
    else
      B.buildConstant(Dst, 0);
  };

  return true;
}

bool CombinerHelper::matchCanonicalizeICmp(const MachineInstr &MI,
                                           BuildFnTy &MatchInfo) {
  const GICmp *Cmp = cast<GICmp>(&MI);

  Register Dst = Cmp->getReg(0);
  Register LHS = Cmp->getLHSReg();
  Register RHS = Cmp->getRHSReg();

  CmpInst::Predicate Pred = Cmp->getCond();
  assert(CmpInst::isIntPredicate(Pred) && "Not an integer compare!");
  if (auto CLHS = GIConstant::getConstant(LHS, MRI)) {
    if (auto CRHS = GIConstant::getConstant(RHS, MRI))
      return constantFoldICmp(*Cmp, *CLHS, *CRHS, MatchInfo);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);

    MatchInfo = [=](MachineIRBuilder &B) { B.buildICmp(Pred, Dst, LHS, RHS); };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCanonicalizeFCmp(const MachineInstr &MI,
                                           BuildFnTy &MatchInfo) {
  const GFCmp *Cmp = cast<GFCmp>(&MI);

  Register Dst = Cmp->getReg(0);
  Register LHS = Cmp->getLHSReg();
  Register RHS = Cmp->getRHSReg();

  CmpInst::Predicate Pred = Cmp->getCond();
  assert(CmpInst::isFPPredicate(Pred) && "Not an FP compare!");

  if (auto CLHS = GFConstant::getConstant(LHS, MRI)) {
    if (auto CRHS = GFConstant::getConstant(RHS, MRI))
      return constantFoldFCmp(*Cmp, *CLHS, *CRHS, MatchInfo);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);

    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildFCmp(Pred, Dst, LHS, RHS, Cmp->getFlags());
    };
    return true;
  }

  return false;
}
