
//===-- SPIRVPreLegalizerCombiner.cpp - combine legalization ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// before the legalizer.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#define GET_GICOMBINER_DEPS
#include "SPIRVGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "spirv-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

#define GET_GICOMBINER_TYPES
#include "SPIRVGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

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
bool matchLengthToDistance(MachineInstr &MI, MachineRegisterInfo &MRI) {
  if (MI.getOpcode() != TargetOpcode::G_INTRINSIC ||
      cast<GIntrinsic>(MI).getIntrinsicID() != Intrinsic::spv_length)
    return false;

  // First operand of MI is `G_INTRINSIC` so start at operand 2.
  Register SubReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubReg);
  if (!SubInstr || SubInstr->getOpcode() != TargetOpcode::G_FSUB)
    return false;

  return true;
}
void applySPIRVDistance(MachineInstr &MI, MachineRegisterInfo &MRI,
                        MachineIRBuilder &B) {

  // Extract the operands for X and Y from the match criteria.
  Register SubDestReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubDestReg);
  Register SubOperand1 = SubInstr->getOperand(1).getReg();
  Register SubOperand2 = SubInstr->getOperand(2).getReg();

  // Remove the original `spv_length` instruction.

  Register ResultReg = MI.getOperand(0).getReg();
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::iterator InsertPt = MI.getIterator();

  // Build the `spv_distance` intrinsic.
  MachineInstrBuilder NewInstr =
      BuildMI(MBB, InsertPt, DL, B.getTII().get(TargetOpcode::G_INTRINSIC));
  NewInstr
      .addDef(ResultReg)                       // Result register
      .addIntrinsicID(Intrinsic::spv_distance) // Intrinsic ID
      .addUse(SubOperand1)                     // Operand X
      .addUse(SubOperand2);                    // Operand Y

  SPIRVGlobalRegistry *GR =
      MI.getMF()->getSubtarget<SPIRVSubtarget>().getSPIRVGlobalRegistry();
  removeAllUses(SubDestReg, MRI, GR); // remove all uses of FSUB Result
  GR->invalidateMachineInstr(SubInstr);
  SubInstr->eraseFromParent(); // remove FSUB instruction
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
bool matchSelectToFaceForward(MachineInstr &MI, MachineRegisterInfo &MRI) {
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
void applySPIRVFaceForward(MachineInstr &MI, MachineRegisterInfo &MRI,
                           MachineIRBuilder &B) {

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
  MachineInstrBuilder NewInstr =
      BuildMI(MBB, InsertPt, DL, B.getTII().get(TargetOpcode::G_INTRINSIC));
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

class SPIRVPreLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const SPIRVPreLegalizerCombinerImplRuleConfig &RuleConfig;
  const SPIRVSubtarget &STI;

public:
  SPIRVPreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const SPIRVPreLegalizerCombinerImplRuleConfig &RuleConfig,
      const SPIRVSubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "SPIRVPreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

  bool tryCombineAllImpl(MachineInstr &I) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "SPIRVGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "SPIRVGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

SPIRVPreLegalizerCombinerImpl::SPIRVPreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const SPIRVPreLegalizerCombinerImplRuleConfig &RuleConfig,
    const SPIRVSubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "SPIRVGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool SPIRVPreLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  return tryCombineAllImpl(MI);
}

// Pass boilerplate
// ================

class SPIRVPreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  SPIRVPreLegalizerCombiner();

  StringRef getPassName() const override { return "SPIRVPreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  SPIRVPreLegalizerCombinerImplRuleConfig RuleConfig;
};

} // end anonymous namespace

void SPIRVPreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

SPIRVPreLegalizerCombiner::SPIRVPreLegalizerCombiner()
    : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool SPIRVPreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);
  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // This is the first Combiner, so the input IR might contain dead
  // instructions.
  CInfo.EnableFullDCE = false;
  SPIRVPreLegalizerCombinerImpl Impl(MF, CInfo, &TPC, *VT, /*CSEInfo*/ nullptr,
                                     RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char SPIRVPreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(SPIRVPreLegalizerCombiner, DEBUG_TYPE,
                      "Combine SPIRV machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(SPIRVPreLegalizerCombiner, DEBUG_TYPE,
                    "Combine SPIRV machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createSPIRVPreLegalizerCombiner() {
  return new SPIRVPreLegalizerCombiner();
}
} // end namespace llvm
