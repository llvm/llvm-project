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

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
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

bool CombinerHelper::matchFCmpIntToFP(const MachineInstr &MI,
                                      BuildFnTy &MatchInfo) const {
  const auto &Cmp = cast<GFCmp>(MI);
  Register Dst = Cmp.getReg(0);
  Register CastReg = Cmp.getLHSReg();
  MachineInstr *Cast = getDefIgnoringCopies(CastReg, MRI);
  if (!Cast)
    return false;
  unsigned Opcode = Cast->getOpcode();
  if ((Opcode != TargetOpcode::G_SITOFP && Opcode != TargetOpcode::G_UITOFP) ||
      !MRI.hasOneNonDBGUse(CastReg))
    return false;

  Register Src = Cast->getOperand(1).getReg();
  LLT SrcTy = MRI.getType(Src);
  LLT DstTy = MRI.getType(Dst);
  auto C = MRI.getType(CastReg).isVector()
               ? getFConstantSplat(Cmp.getRHSReg(), MRI)
               : getFConstantVRegValWithLookThrough(Cmp.getRHSReg(), MRI);
  if (!C || !isConstantLegalOrBeforeLegalizer(SrcTy) ||
      !isLegalOrBeforeLegalizer({TargetOpcode::G_ICMP, {DstTy, SrcTy}}))
    return false;

  bool IsSigned = Opcode == TargetOpcode::G_SITOFP;
  APSInt IntC(SrcTy.getScalarSizeInBits(), !IsSigned);
  bool IsExact;
  if (C->Value.convertToInteger(IntC, APFloat::rmTowardZero, &IsExact) !=
          APFloat::opOK ||
      !IsExact)
    return false;

  CmpInst::Predicate Pred;
  switch (Cmp.getCond()) {
  default:
    return false;
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_UEQ:
    Pred = CmpInst::ICMP_EQ;
    break;
  case CmpInst::FCMP_ONE:
  case CmpInst::FCMP_UNE:
    Pred = CmpInst::ICMP_NE;
    break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT:
    Pred = IsSigned ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;
    break;
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE:
    Pred = IsSigned ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
    break;
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_ULT:
    Pred = IsSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
    break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULE:
    Pred = IsSigned ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE;
    break;
  }

  const APInt &Value = IntC;
  const fltSemantics &Sem = C->Value.getSemantics();
  auto IsExactInt = [&](const APInt &V) {
    return llvm::isKnownExactIntToFP(KnownBits::makeConstant(V), Sem, IsSigned);
  };
  // Exact neighbors prevent any rounded integer from colliding with C or
  // crossing the comparison boundary. Otherwise require an exact source.
  bool Min = IsSigned ? Value.isMinSignedValue() : Value.isMinValue();
  bool Max = IsSigned ? Value.isMaxSignedValue() : Value.isMaxValue();
  if (!((Min || IsExactInt(Value - 1)) && (Max || IsExactInt(Value + 1))) &&
      !(VT && VT->isKnownExactIntToFP(Src, Sem, IsSigned)))
    return false;

  APInt Const = IntC;
  MatchInfo = [=](MachineIRBuilder &B) {
    auto RHS = B.buildConstant(SrcTy, Const);
    B.buildICmp(Pred, Dst, Src, RHS);
  };
  return true;
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
