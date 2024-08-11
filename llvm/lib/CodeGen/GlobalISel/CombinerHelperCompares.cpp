//===- CombinerHelperCompares.cpp------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for G_ICMP
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

bool CombinerHelper::constantFoldICmp(const GICmp &ICmp,
                                      const GIConstant &LHSCst,
                                      const GIConstant &RHSCst,
                                      BuildFnTy &MatchInfo) {
  if (LHSCst.getKind() != GIConstantKind::Scalar)
    return false;

  Register Dst = ICmp.getReg(0);
  LLT DstTy = MRI.getType(Dst);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return false;

  CmpInst::Predicate Pred = ICmp.getCond();
  APInt LHS = LHSCst.getScalarValue();
  APInt RHS = RHSCst.getScalarValue();

  bool Result;

  switch (Pred) {
  case CmpInst::Predicate::ICMP_EQ:
    Result = LHS.eq(RHS);
    break;
  case CmpInst::Predicate::ICMP_NE:
    Result = LHS.ne(RHS);
    break;
  case CmpInst::Predicate::ICMP_UGT:
    Result = LHS.ugt(RHS);
    break;
  case CmpInst::Predicate::ICMP_UGE:
    Result = LHS.uge(RHS);
    break;
  case CmpInst::Predicate::ICMP_ULT:
    Result = LHS.ult(RHS);
    break;
  case CmpInst::Predicate::ICMP_ULE:
    Result = LHS.ule(RHS);
    break;
  case CmpInst::Predicate::ICMP_SGT:
    Result = LHS.sgt(RHS);
    break;
  case CmpInst::Predicate::ICMP_SGE:
    Result = LHS.sge(RHS);
    break;
  case CmpInst::Predicate::ICMP_SLT:
    Result = LHS.slt(RHS);
    break;
  case CmpInst::Predicate::ICMP_SLE:
    Result = LHS.sle(RHS);
    break;
  default:
    llvm_unreachable("Unexpected predicate");
  }

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

bool CombinerHelper::visitICmp(const MachineInstr &MI, BuildFnTy &MatchInfo) {
  const GICmp *Cmp = cast<GICmp>(&MI);

  Register Dst = Cmp->getReg(0);
  LLT DstTy = MRI.getType(Dst);
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

  [[maybe_unused]] MachineInstr *MILHS = MRI.getVRegDef(LHS);
  MachineInstr *MIRHS = MRI.getVRegDef(RHS);

  // For EQ and NE, we can always pick a value for the undef to make the
  // predicate pass or fail, so we can return undef.
  // Matches behavior in llvm::ConstantFoldCompareInstruction.
  if (isa<GImplicitDef>(MIRHS) && ICmpInst::isEquality(Pred) &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_IMPLICIT_DEF, {DstTy}})) {
    MatchInfo = [=](MachineIRBuilder &B) { B.buildUndef(Dst); };
    return true;
  }

  // icmp X, X -> true/false
  // icmp X, undef -> true/false because undef could be X.
  if ((LHS == RHS || isa<GImplicitDef>(MIRHS)) &&
      isConstantLegalOrBeforeLegalizer(DstTy)) {
    MatchInfo = [=](MachineIRBuilder &B) {
      if (CmpInst::isTrueWhenEqual(Pred))
        B.buildConstant(Dst, getICmpTrueVal(getTargetLowering(),
                                            /*IsVector=*/DstTy.isVector(),
                                            /*IsFP=*/false));
      else
        B.buildConstant(Dst, 0);
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchSextOfICmp(const MachineInstr &MI,
                                     BuildFnTy &MatchInfo) {
  const GICmp *Cmp = cast<GICmp>(&MI);

  Register Dst = Cmp->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  Register LHS = Cmp->getLHSReg();
  Register RHS = Cmp->getRHSReg();
  CmpInst::Predicate Pred = Cmp->getCond();

  GSext *SL = cast<GSext>(MRI.getVRegDef(LHS));
  GSext *SR = cast<GSext>(MRI.getVRegDef(RHS));

  LLT SLTy = MRI.getType(SL->getSrcReg());
  LLT SRTy = MRI.getType(SR->getSrcReg());

  // Turn icmp (sext X), (sext Y) into a compare of X and Y if they have the
  // same type.
  if (SLTy != SRTy)
    return false;

  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_ICMP, {DstTy, SLTy}}))
    return false;

  // Compare X and Y. Note that the predicate does not change.
  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildICmp(Pred, Dst, SL->getSrcReg(), SR->getSrcReg());
  };
  return true;
}

bool CombinerHelper::matchZextOfICmp(const MachineInstr &MI,
                                     BuildFnTy &MatchInfo) {
  const GICmp *Cmp = cast<GICmp>(&MI);

  Register Dst = Cmp->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  Register LHS = Cmp->getLHSReg();
  Register RHS = Cmp->getRHSReg();
  CmpInst::Predicate Pred = Cmp->getCond();

  /*
    %x:_(p0) = COPY $x0
    %y:_(p0) = COPY $x1
    %zero:_(p0) = G_CONSTANT i64 0
    %cmp1:_(s1) = G_ICMP intpred(eq), %x:_(p0), %zero:_
   */

  if (MRI.getType(LHS).isPointer() || MRI.getType(RHS).isPointer())
    return false;

  if (!MRI.getType(LHS).isScalar() || !MRI.getType(RHS).isScalar())
    return false;

  GZext *ZL = cast<GZext>(MRI.getVRegDef(LHS));
  GZext *ZR = cast<GZext>(MRI.getVRegDef(RHS));

  LLT ZLTy = MRI.getType(ZL->getSrcReg());
  LLT ZRTy = MRI.getType(ZR->getSrcReg());

  // Turn icmp (zext X), (zext Y) into a compare of X and Y if they have
  // the same type.
  if (ZLTy != ZRTy)
    return false;

  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_ICMP, {DstTy, ZLTy}}))
    return false;

  // Compare X and Y. Note that signed predicates become unsigned.
  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildICmp(ICmpInst::getUnsignedPredicate(Pred), Dst, ZL->getSrcReg(),
                ZR->getSrcReg());
  };
  return true;
}

bool CombinerHelper::matchCmpOfZero(const MachineInstr &MI,
                                    BuildFnTy &MatchInfo) {
  const GICmp *Cmp = cast<GICmp>(&MI);

  Register Dst = Cmp->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  Register LHS = Cmp->getLHSReg();
  CmpInst::Predicate Pred = Cmp->getCond();

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return false;

  std::optional<bool> Result;

  switch (Pred) {
  default:
    llvm_unreachable("Unkonwn ICmp predicate!");
  case ICmpInst::ICMP_ULT:
    Result = false;
    break;
  case ICmpInst::ICMP_UGE:
    Result = true;
    break;
  case ICmpInst::ICMP_EQ:
  case ICmpInst::ICMP_ULE:
    if (isKnownNonZero(LHS, MRI, KB))
      Result = false;
    break;
  case ICmpInst::ICMP_NE:
  case ICmpInst::ICMP_UGT:
    if (isKnownNonZero(LHS, MRI, KB))
      Result = true;
    break;
  case ICmpInst::ICMP_SLT: {
    KnownBits LHSKnown = KB->getKnownBits(LHS);
    if (LHSKnown.isNegative())
      Result = true;
    if (LHSKnown.isNonNegative())
      Result = false;
    break;
  }
  case ICmpInst::ICMP_SLE: {
    KnownBits LHSKnown = KB->getKnownBits(LHS);
    if (LHSKnown.isNegative())
      Result = true;
    if (LHSKnown.isNonNegative() && isKnownNonZero(LHS, MRI, KB))
      Result = false;
    break;
  }
  case ICmpInst::ICMP_SGE: {
    KnownBits LHSKnown = KB->getKnownBits(LHS);
    if (LHSKnown.isNegative())
      Result = false;
    if (LHSKnown.isNonNegative())
      Result = true;
    break;
  }
  case ICmpInst::ICMP_SGT: {
    KnownBits LHSKnown = KB->getKnownBits(LHS);
    if (LHSKnown.isNegative())
      Result = false;
    if (LHSKnown.isNonNegative() && isKnownNonZero(LHS, MRI, KB))
      Result = true;
    break;
  }
  }

  if (!Result)
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    if (*Result)
      B.buildConstant(Dst, getICmpTrueVal(getTargetLowering(),
                                          /*IsVector=*/DstTy.isVector(),
                                          /*IsFP=*/false));
    else
      B.buildConstant(Dst, 0);
  };

  return true;
}
