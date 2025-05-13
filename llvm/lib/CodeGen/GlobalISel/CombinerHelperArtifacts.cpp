//===- CombinerHelperArtifacts.cpp-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for legalization artifacts.
//
//===----------------------------------------------------------------------===//
//
// G_MERGE_VALUES
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

bool CombinerHelper::matchMergeXAndUndef(const MachineInstr &MI,
                                         BuildFnTy &MatchInfo) const {
  const GMerge *Merge = cast<GMerge>(&MI);

  Register Dst = Merge->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Merge->getSourceReg(0));

  // Otherwise, we would miscompile.
  assert(Merge->getNumSources() == 2 && "Unexpected number of operands");

  //
  //   %bits_8_15:_(s8) = G_IMPLICIT_DEF
  //   %0:_(s16) = G_MERGE_VALUES %bits_0_7:(s8), %bits_8_15:(s8)
  //
  // ->
  //
  //   %0:_(s16) = G_ANYEXT %bits_0_7:(s8)
  //

  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_ANYEXT, {DstTy, SrcTy}}))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildAnyExt(Dst, Merge->getSourceReg(0));
  };
  return true;
}

bool CombinerHelper::matchMergeXAndZero(const MachineInstr &MI,
                                        BuildFnTy &MatchInfo) const {
  const GMerge *Merge = cast<GMerge>(&MI);

  Register Dst = Merge->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Merge->getSourceReg(0));

  // No multi-use check. It is a constant.

  //
  //   %bits_8_15:_(s8) = G_CONSTANT i8 0
  //   %0:_(s16) = G_MERGE_VALUES %bits_0_7:(s8), %bits_8_15:(s8)
  //
  // ->
  //
  //   %0:_(s16) = G_ZEXT %bits_0_7:(s8)
  //

  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_ZEXT, {DstTy, SrcTy}}))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildZExt(Dst, Merge->getSourceReg(0));
  };
  return true;
}
