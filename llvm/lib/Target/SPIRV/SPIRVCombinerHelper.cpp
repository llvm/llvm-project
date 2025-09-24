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

static void removeAllUses(Register Reg, MachineRegisterInfo &MRI,
                          SPIRVGlobalRegistry *GR) {
  SmallVector<MachineInstr *, 4> UsesToErase(
      llvm::make_pointer_range(MRI.use_instructions(Reg)));

  // calling eraseFromParent too early invalidates the iterator.
  for (auto *MIToErase : UsesToErase) {
    GR->invalidateMachineInstr(MIToErase);
    MIToErase->eraseFromParent();
  }
}

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

/// This match is part of a combine that
/// rewrites select(fcmp(dot(I, Ng), 0), N, -N) to faceforward(N, I, Ng)
///   (vXf32 (g_select
///             (g_fcmp
///                (g_intrinsic dot(vXf32 I) (vXf32 Ng)
///                 0)
///             (vXf32 N)
///             (vXf32 g_fneg (vXf32 N))))
/// ->
///   (vXf32 (g_intrinsic faceforward
///             (vXf32 N) (vXf32 I) (vXf32 Ng)))
///
/// This only works for Vulkan targets.
///
bool SPIRVCombinerHelper::matchSelectToFaceForward(MachineInstr &MI) const {
  if (!MI.getMF()->getSubtarget<SPIRVSubtarget>().isShader())
    return false;

  // Match overall select pattern.
  Register CondReg, TrueReg, FalseReg;
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GISelect(m_Reg(CondReg), m_Reg(TrueReg), m_Reg(FalseReg))))
    return false;

  // Match the FCMP condition.
  Register DotReg, CondZeroReg;
  CmpInst::Predicate Pred;
  if (!mi_match(CondReg, MRI,
                m_GFCmp(m_Pred(Pred), m_Reg(DotReg), m_Reg(CondZeroReg))) ||
      Pred != CmpInst::FCMP_OLT)
    return false;

  // Check if FCMP is a comparison between a dot product and 0.
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  if (DotInstr->getOpcode() != TargetOpcode::G_INTRINSIC ||
      cast<GIntrinsic>(DotInstr)->getIntrinsicID() != Intrinsic::spv_fdot) {
    Register DotOperand1, DotOperand2;
    // Check for scalar dot product.
    if (!mi_match(DotReg, MRI,
                  m_GFMul(m_Reg(DotOperand1), m_Reg(DotOperand2))) ||
        !MRI.getType(DotOperand1).isScalar() ||
        !MRI.getType(DotOperand2).isScalar())
      return false;
  }

  const ConstantFP *ZeroVal;
  if (!mi_match(CondZeroReg, MRI, m_GFCst(ZeroVal)) || !ZeroVal->isZero())
    return false;

  // Check if select's false operand is the negation of the true operand.
  auto AreNegatedConstants = [&](Register TrueReg, Register FalseReg) {
    const ConstantFP *TrueVal, *FalseVal;
    if (!mi_match(TrueReg, MRI, m_GFCst(TrueVal)) ||
        !mi_match(FalseReg, MRI, m_GFCst(FalseVal)))
      return false;
    APFloat TrueValNegated = TrueVal->getValue();
    TrueValNegated.changeSign();
    return FalseVal->getValue().compare(TrueValNegated) == APFloat::cmpEqual;
  };

  if (!mi_match(FalseReg, MRI, m_GFNeg(m_SpecificReg(TrueReg))) &&
      !mi_match(TrueReg, MRI, m_GFNeg(m_SpecificReg(FalseReg)))) {
    // Check if they're constant opposites.
    MachineInstr *TrueInstr = MRI.getVRegDef(TrueReg);
    MachineInstr *FalseInstr = MRI.getVRegDef(FalseReg);
    if (TrueInstr->getOpcode() == TargetOpcode::G_BUILD_VECTOR &&
        FalseInstr->getOpcode() == TargetOpcode::G_BUILD_VECTOR &&
        TrueInstr->getNumOperands() == FalseInstr->getNumOperands()) {
      for (unsigned I = 1; I < TrueInstr->getNumOperands(); ++I)
        if (!AreNegatedConstants(TrueInstr->getOperand(I).getReg(),
                                 FalseInstr->getOperand(I).getReg()))
          return false;
    } else if (!AreNegatedConstants(TrueReg, FalseReg))
      return false;
  }

  return true;
}

void SPIRVCombinerHelper::applySPIRVFaceForward(MachineInstr &MI) const {
  // Extract the operands for N, I, and Ng from the match criteria.
  Register CondReg, TrueReg, DotReg, DotOperand1, DotOperand2;
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GISelect(m_Reg(CondReg), m_Reg(TrueReg), m_Reg())))
    return;
  if (!mi_match(CondReg, MRI, m_GFCmp(m_Pred(), m_Reg(DotReg), m_Reg())))
    return;
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  if (!mi_match(DotReg, MRI, m_GFMul(m_Reg(DotOperand1), m_Reg(DotOperand2)))) {
    DotOperand1 = DotInstr->getOperand(2).getReg();
    DotOperand2 = DotInstr->getOperand(3).getReg();
  }

  // Remove the original `select` instruction.
  Register ResultReg = MI.getOperand(0).getReg();
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::iterator InsertPt = MI.getIterator();

  // Build the `spv_faceforward` intrinsic.
  MachineInstrBuilder NewInstr = BuildMI(
      MBB, InsertPt, DL, Builder.getTII().get(TargetOpcode::G_INTRINSIC));
  NewInstr
      .addDef(ResultReg)                          // Result register
      .addIntrinsicID(Intrinsic::spv_faceforward) // Intrinsic ID
      .addUse(TrueReg)                            // Operand N
      .addUse(DotOperand1)                        // Operand I
      .addUse(DotOperand2);                       // Operand Ng

  SPIRVGlobalRegistry *GR =
      MI.getMF()->getSubtarget<SPIRVSubtarget>().getSPIRVGlobalRegistry();
  removeAllUses(CondReg, MRI, GR); // Remove all uses of FCMP result
  MachineInstr *CondInstr = MRI.getVRegDef(CondReg);
  GR->invalidateMachineInstr(CondInstr);
  CondInstr->eraseFromParent();   // Remove FCMP instruction
  removeAllUses(DotReg, MRI, GR); // Remove all uses of dot product result
  GR->invalidateMachineInstr(DotInstr);
  DotInstr->eraseFromParent(); // Remove dot product instruction
}
