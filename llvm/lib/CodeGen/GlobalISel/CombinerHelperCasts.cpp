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
