//===-- SPIRVCombinerHelper.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVCombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace MIPatternMatch;

SPIRVCombinerHelper::SPIRVCombinerHelper(
    GISelChangeObserver &Observer, MachineIRBuilder &B, bool IsPreLegalize,
    GISelValueTracking *VT, MachineDominatorTree *MDT, const LegalizerInfo *LI,
    const SPIRVSubtarget &STI)
    : CombinerHelper(Observer, B, IsPreLegalize, VT, MDT, LI), STI(STI) {}

/// This match is part of a combine that
/// rewrites length(X - Y) to distance(X, Y)
///   (f32 (g_intrinsic length
///           (g_fsub (vXf32 X) (vXf32 Y))))
/// ->
///   (f32 (g_intrinsic distance
///           (vXf32 X) (vXf32 Y)))
///
bool SPIRVCombinerHelper::matchLengthToDistance(MachineInstr &MI) const {
  if (MI.getOpcode() != TargetOpcode::G_INTRINSIC ||
      cast<GIntrinsic>(MI).getIntrinsicID() != Intrinsic::spv_length)
    return false;

  // First operand of MI is `G_INTRINSIC` so start at operand 2.
  Register SubReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubReg);
  if (SubInstr->getOpcode() != TargetOpcode::G_FSUB)
    return false;

  return true;
}

void SPIRVCombinerHelper::applySPIRVDistance(MachineInstr &MI) const {
  // Extract the operands for X and Y from the match criteria.
  Register SubDestReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubDestReg);
  Register SubOperand1 = SubInstr->getOperand(1).getReg();
  Register SubOperand2 = SubInstr->getOperand(2).getReg();
  Register ResultReg = MI.getOperand(0).getReg();

  Builder.setInstrAndDebugLoc(MI);
  Builder.buildIntrinsic(Intrinsic::spv_distance, ResultReg)
      .addUse(SubOperand1)
      .addUse(SubOperand2);

  MI.eraseFromParent();
}
