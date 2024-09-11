//===- CombinerHelperCasts.cpp---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for G_ANYEXT, G_SEXT, G_TRUNC, and
// G_ZEXT
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

bool CombinerHelper::matchSextOfTrunc(const MachineOperand &MO,
                                      BuildFnTy &MatchInfo) {
  GSext *Sext = cast<GSext>(getDefIgnoringCopies(MO.getReg(), MRI));
  GTrunc *Trunc = cast<GTrunc>(getDefIgnoringCopies(Sext->getSrcReg(), MRI));

  Register Dst = Sext->getReg(0);
  Register Src = Trunc->getSrcReg();

  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (DstTy == SrcTy) {
    MatchInfo = [=](MachineIRBuilder &B) { B.buildCopy(Dst, Src); };
    return true;
  }

  if (DstTy.getScalarSizeInBits() < SrcTy.getScalarSizeInBits() &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_TRUNC, {DstTy, SrcTy}})) {
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildTrunc(Dst, Src, MachineInstr::MIFlag::NoSWrap);
    };
    return true;
  }

  if (DstTy.getScalarSizeInBits() > SrcTy.getScalarSizeInBits() &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_SEXT, {DstTy, SrcTy}})) {
    MatchInfo = [=](MachineIRBuilder &B) { B.buildSExt(Dst, Src); };
    return true;
  }

  return false;
}

bool CombinerHelper::matchZextOfTrunc(const MachineOperand &MO,
                                      BuildFnTy &MatchInfo) {
  GZext *Zext = cast<GZext>(getDefIgnoringCopies(MO.getReg(), MRI));
  GTrunc *Trunc = cast<GTrunc>(getDefIgnoringCopies(Zext->getSrcReg(), MRI));

  Register Dst = Zext->getReg(0);
  Register Src = Trunc->getSrcReg();

  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (DstTy == SrcTy) {
    MatchInfo = [=](MachineIRBuilder &B) { B.buildCopy(Dst, Src); };
    return true;
  }

  if (DstTy.getScalarSizeInBits() < SrcTy.getScalarSizeInBits() &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_TRUNC, {DstTy, SrcTy}})) {
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildTrunc(Dst, Src, MachineInstr::MIFlag::NoUWrap);
    };
    return true;
  }

  if (DstTy.getScalarSizeInBits() > SrcTy.getScalarSizeInBits() &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_ZEXT, {DstTy, SrcTy}})) {
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildZExt(Dst, Src, MachineInstr::MIFlag::NonNeg);
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchNonNegZext(const MachineOperand &MO,
                                     BuildFnTy &MatchInfo) {
  GZext *Zext = cast<GZext>(MRI.getVRegDef(MO.getReg()));

  Register Dst = Zext->getReg(0);
  Register Src = Zext->getSrcReg();

  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);
  const auto &TLI = getTargetLowering();

  // Convert zext nneg to sext if sext is the preferred form for the target.
  if (isLegalOrBeforeLegalizer({TargetOpcode::G_SEXT, {DstTy, SrcTy}}) &&
      TLI.isSExtCheaperThanZExt(getMVTForLLT(SrcTy), getMVTForLLT(DstTy))) {
    MatchInfo = [=](MachineIRBuilder &B) { B.buildSExt(Dst, Src); };
    return true;
  }

  return false;
}

bool CombinerHelper::matchTruncateOfExt(const MachineInstr &Root,
                                        const MachineInstr &ExtMI,
                                        BuildFnTy &MatchInfo) {
  const GTrunc *Trunc = cast<GTrunc>(&Root);
  const GExtOp *Ext = cast<GExtOp>(&ExtMI);

  if (!MRI.hasOneNonDBGUse(Ext->getReg(0)))
    return false;

  Register Dst = Trunc->getReg(0);
  Register Src = Ext->getSrcReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (SrcTy == DstTy) {
    // The source and the destination are equally sized. We need to copy.
    MatchInfo = [=](MachineIRBuilder &B) { B.buildCopy(Dst, Src); };

    return true;
  }

  if (SrcTy.getScalarSizeInBits() < DstTy.getScalarSizeInBits()) {
    // If the source is smaller than the destination, we need to extend.

    if (!isLegalOrBeforeLegalizer({Ext->getOpcode(), {DstTy, SrcTy}}))
      return false;

    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildInstr(Ext->getOpcode(), {Dst}, {Src});
    };

    return true;
  }

  if (SrcTy.getScalarSizeInBits() > DstTy.getScalarSizeInBits()) {
    // If the source is larger than the destination, then we need to truncate.

    if (!isLegalOrBeforeLegalizer({TargetOpcode::G_TRUNC, {DstTy, SrcTy}}))
      return false;

    MatchInfo = [=](MachineIRBuilder &B) { B.buildTrunc(Dst, Src); };

    return true;
  }

  return false;
}

bool CombinerHelper::isCastFree(unsigned Opcode, LLT ToTy, LLT FromTy) const {
  const TargetLowering &TLI = getTargetLowering();
  const DataLayout &DL = getDataLayout();
  LLVMContext &Ctx = getContext();

  switch (Opcode) {
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_ZEXT:
    return TLI.isZExtFree(FromTy, ToTy, DL, Ctx);
  case TargetOpcode::G_TRUNC:
    return TLI.isTruncateFree(FromTy, ToTy, DL, Ctx);
  default:
    return false;
  }
}

bool CombinerHelper::matchCastOfSelect(const MachineInstr &CastMI,
                                       const MachineInstr &SelectMI,
                                       BuildFnTy &MatchInfo) {
  const GExtOrTruncOp *Cast = cast<GExtOrTruncOp>(&CastMI);
  const GSelect *Select = cast<GSelect>(&SelectMI);

  if (!MRI.hasOneNonDBGUse(Select->getReg(0)))
    return false;

  Register Dst = Cast->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  LLT CondTy = MRI.getType(Select->getCondReg());
  Register TrueReg = Select->getTrueReg();
  Register FalseReg = Select->getFalseReg();
  LLT SrcTy = MRI.getType(TrueReg);
  Register Cond = Select->getCondReg();

  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_SELECT, {DstTy, CondTy}}))
    return false;

  if (!isCastFree(Cast->getOpcode(), DstTy, SrcTy))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    auto True = B.buildInstr(Cast->getOpcode(), {DstTy}, {TrueReg});
    auto False = B.buildInstr(Cast->getOpcode(), {DstTy}, {FalseReg});
    B.buildSelect(Dst, Cond, True, False);
  };

  return true;
}

bool CombinerHelper::matchExtOfExt(const MachineInstr &FirstMI,
                                   const MachineInstr &SecondMI,
                                   BuildFnTy &MatchInfo) {
  const GExtOp *First = cast<GExtOp>(&FirstMI);
  const GExtOp *Second = cast<GExtOp>(&SecondMI);

  Register Dst = First->getReg(0);
  Register Src = Second->getSrcReg();
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Src);

  if (!MRI.hasOneNonDBGUse(Second->getReg(0)))
    return false;

  // ext of ext -> later ext
  if (First->getOpcode() == Second->getOpcode() &&
      isLegalOrBeforeLegalizer({Second->getOpcode(), {DstTy, SrcTy}})) {
    if (Second->getOpcode() == TargetOpcode::G_ZEXT) {
      MachineInstr::MIFlag Flag = MachineInstr::MIFlag::NoFlags;
      if (Second->getFlag(MachineInstr::MIFlag::NonNeg))
        Flag = MachineInstr::MIFlag::NonNeg;
      MatchInfo = [=](MachineIRBuilder &B) { B.buildZExt(Dst, Src, Flag); };
      return true;
    }
    // not zext -> no flags
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildInstr(Second->getOpcode(), {Dst}, {Src});
    };
    return true;
  }

  // anyext of sext/zext  -> sext/zext
  // -> pick anyext as second ext, then ext of ext
  if (First->getOpcode() == TargetOpcode::G_ANYEXT &&
      isLegalOrBeforeLegalizer({Second->getOpcode(), {DstTy, SrcTy}})) {
    if (Second->getOpcode() == TargetOpcode::G_ZEXT) {
      MachineInstr::MIFlag Flag = MachineInstr::MIFlag::NoFlags;
      if (Second->getFlag(MachineInstr::MIFlag::NonNeg))
        Flag = MachineInstr::MIFlag::NonNeg;
      MatchInfo = [=](MachineIRBuilder &B) { B.buildZExt(Dst, Src, Flag); };
      return true;
    }
    MatchInfo = [=](MachineIRBuilder &B) { B.buildSExt(Dst, Src); };
    return true;
  }

  // sext/zext of anyext -> sext/zext
  // -> pick anyext as first ext, then ext of ext
  if (Second->getOpcode() == TargetOpcode::G_ANYEXT &&
      isLegalOrBeforeLegalizer({First->getOpcode(), {DstTy, SrcTy}})) {
    if (First->getOpcode() == TargetOpcode::G_ZEXT) {
      MachineInstr::MIFlag Flag = MachineInstr::MIFlag::NoFlags;
      if (First->getFlag(MachineInstr::MIFlag::NonNeg))
        Flag = MachineInstr::MIFlag::NonNeg;
      MatchInfo = [=](MachineIRBuilder &B) { B.buildZExt(Dst, Src, Flag); };
      return true;
    }
    MatchInfo = [=](MachineIRBuilder &B) { B.buildSExt(Dst, Src); };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCastOfBuildVector(const MachineInstr &CastMI,
                                            const MachineInstr &BVMI,
                                            BuildFnTy &MatchInfo) {
  const GExtOrTruncOp *Cast = cast<GExtOrTruncOp>(&CastMI);
  const GBuildVector *BV = cast<GBuildVector>(&BVMI);

  if (!MRI.hasOneNonDBGUse(BV->getReg(0)))
    return false;

  Register Dst = Cast->getReg(0);
  // The type of the new build vector.
  LLT DstTy = MRI.getType(Dst);
  // The scalar or element type of the new build vector.
  LLT ElemTy = DstTy.getScalarType();
  // The scalar or element type of the old build vector.
  LLT InputElemTy = MRI.getType(BV->getReg(0)).getElementType();

  // Check legality of new build vector, the scalar casts, and profitability of
  // the many casts.
  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_BUILD_VECTOR, {DstTy, ElemTy}}) ||
      !isLegalOrBeforeLegalizer({Cast->getOpcode(), {ElemTy, InputElemTy}}) ||
      !isCastFree(Cast->getOpcode(), ElemTy, InputElemTy))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    SmallVector<Register> Casts;
    unsigned Elements = BV->getNumSources();
    for (unsigned I = 0; I < Elements; ++I) {
      auto CastI =
          B.buildInstr(Cast->getOpcode(), {ElemTy}, {BV->getSourceReg(I)});
      Casts.push_back(CastI.getReg(0));
    }

    B.buildBuildVector(Dst, Casts);
  };

  return true;
}
