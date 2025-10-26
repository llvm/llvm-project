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
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include <cstdlib>

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

bool CombinerHelper::constantFoldICmp(const GICmp &ICmp,
                                      const GIConstant &LHSCst,
                                      const GIConstant &RHSCst,
                                      BuildFnTy &MatchInfo) const {
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
                                      BuildFnTy &MatchInfo) const {
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
                                           BuildFnTy &MatchInfo) const {
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
                                           BuildFnTy &MatchInfo) const {
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

bool CombinerHelper::combineMergedBFXCompare(MachineInstr &MI) const {
  const GICmp *Cmp = cast<GICmp>(&MI);

  ICmpInst::Predicate CC = Cmp->getCond();
  if (CC != CmpInst::ICMP_EQ && CC != CmpInst::ICMP_NE)
    return false;

  Register CmpLHS = Cmp->getLHSReg();
  Register CmpRHS = Cmp->getRHSReg();

  LLT OpTy = MRI.getType(CmpLHS);
  if (!OpTy.isScalar() || OpTy.isPointer())
    return false;

  assert(isZeroOrZeroSplat(CmpRHS, /*AllowUndefs=*/false));

  Register Src;
  const auto IsSrc = [&](Register R) {
    if (!Src) {
      Src = R;
      return true;
    }

    return Src == R;
  };

  MachineInstr *CmpLHSDef = MRI.getVRegDef(CmpLHS);
  if (CmpLHSDef->getOpcode() != TargetOpcode::G_OR)
    return false;

  APInt PartsMask(OpTy.getSizeInBits(), 0);
  SmallVector<MachineInstr *> Worklist = {CmpLHSDef};
  while (!Worklist.empty()) {
    MachineInstr *Cur = Worklist.pop_back_val();

    Register Dst = Cur->getOperand(0).getReg();
    if (!MRI.hasOneUse(Dst) && Dst != Src)
      return false;

    if (Cur->getOpcode() == TargetOpcode::G_OR) {
      Worklist.push_back(MRI.getVRegDef(Cur->getOperand(1).getReg()));
      Worklist.push_back(MRI.getVRegDef(Cur->getOperand(2).getReg()));
      continue;
    }

    if (Cur->getOpcode() == TargetOpcode::G_UBFX) {
      Register Op = Cur->getOperand(1).getReg();
      Register Off = Cur->getOperand(2).getReg();
      Register Width = Cur->getOperand(3).getReg();

      auto WidthCst = getIConstantVRegVal(Width, MRI);
      auto OffCst = getIConstantVRegVal(Off, MRI);
      if (!WidthCst || !OffCst || !IsSrc(Op))
        return false;

      unsigned Start = OffCst->getZExtValue();
      unsigned End = Start + WidthCst->getZExtValue();
      if (End > OpTy.getScalarSizeInBits())
        return false;
      PartsMask.setBits(Start, End);
      continue;
    }

    if (Cur->getOpcode() == TargetOpcode::G_AND) {
      Register LHS = Cur->getOperand(1).getReg();
      Register RHS = Cur->getOperand(2).getReg();

      auto MaskCst = getIConstantVRegVal(RHS, MRI);
      if (!MaskCst || !IsSrc(LHS))
        return false;

      PartsMask |= *MaskCst;
      continue;
    }

    return false;
  }

  if (!Src)
    return false;

  assert(OpTy == MRI.getType(Src) && "Ignored a type casting operation?");
  auto MaskedSrc =
      Builder.buildAnd(OpTy, Src, Builder.buildConstant(OpTy, PartsMask));
  Builder.buildICmp(CC, Cmp->getReg(0), MaskedSrc, CmpRHS, Cmp->getFlags());
  MI.eraseFromParent();
  return true;
}
