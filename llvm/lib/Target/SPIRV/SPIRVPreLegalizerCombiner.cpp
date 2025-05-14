
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
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
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Support/Debug.h"

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
/// rewrites select(fcmp(dot(I, Ng), 0), N, 0 - N) to faceforward(N, I, Ng)
///   (vXf32 (g_select
///             (g_fcmp
///                (g_intrinsic dot(vXf32 I) (vXf32 Ng)
///                 0)
///             (vXf32 N)
///             (vXf32 g_fsub (0) (vXf32 N))))
/// ->
///   (vXf32 (g_intrinsic faceforward
///             (vXf32 N) (vXf32 I) (vXf32 Ng)))
///
bool matchSelectToFaceForward(MachineInstr &MI, MachineRegisterInfo &MRI) {
  if (MI.getOpcode() != TargetOpcode::G_SELECT)
    return false;

  // Check if select's condition is a comparison between a dot product and 0.
  Register CondReg = MI.getOperand(1).getReg();
  MachineInstr *CondInstr = MRI.getVRegDef(CondReg);
  if (!CondInstr || CondInstr->getOpcode() != TargetOpcode::G_FCMP)
    return false;

  Register DotReg = CondInstr->getOperand(2).getReg();
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  if (DotInstr->getOpcode() != TargetOpcode::G_FMUL &&
      (DotInstr->getOpcode() != TargetOpcode::G_INTRINSIC ||
       cast<GIntrinsic>(DotInstr)->getIntrinsicID() != Intrinsic::spv_fdot))
    return false;

  Register CondZeroReg = CondInstr->getOperand(3).getReg();
  MachineInstr *CondZeroInstr = MRI.getVRegDef(CondZeroReg);
  if (CondZeroInstr->getOpcode() != TargetOpcode::G_FCONSTANT ||
      !CondZeroInstr->getOperand(1).getFPImm()->isZero())
    return false;

  // Check if select's false operand is the negation of the true operand.
  Register TrueReg = MI.getOperand(2).getReg();
  Register FalseReg = MI.getOperand(3).getReg();
  MachineInstr *FalseInstr = MRI.getVRegDef(FalseReg);
  if (FalseInstr->getOpcode() != TargetOpcode::G_FNEG)
    return false;
  if (TrueReg != FalseInstr->getOperand(1).getReg())
    return false;

  return true;
}
void applySPIRVFaceForward(MachineInstr &MI, MachineRegisterInfo &MRI,
                           MachineIRBuilder &B) {

  // Extract the operands for N, I, and Ng from the match criteria.
  Register CondReg = MI.getOperand(1).getReg();
  MachineInstr *CondInstr = MRI.getVRegDef(CondReg);
  Register DotReg = CondInstr->getOperand(2).getReg();
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  Register DotOperand1, DotOperand2;
  if (DotInstr->getOpcode() == TargetOpcode::G_FMUL) {
    DotOperand1 = DotInstr->getOperand(1).getReg();
    DotOperand2 = DotInstr->getOperand(2).getReg();
  } else {
    DotOperand1 = DotInstr->getOperand(2).getReg();
    DotOperand2 = DotInstr->getOperand(3).getReg();
  }
  Register TrueReg = MI.getOperand(2).getReg();

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
  removeAllUses(CondReg, MRI, GR); // remove all uses of FCMP Result
  GR->invalidateMachineInstr(CondInstr);
  CondInstr->eraseFromParent(); // remove FCMP instruction
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
  AU.addRequired<GISelValueTrackingAnalysis>();
  AU.addPreserved<GISelValueTrackingAnalysis>();
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
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);
  GISelValueTracking *VT = &getAnalysis<GISelValueTrackingAnalysis>().get(MF);
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
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysis)
INITIALIZE_PASS_END(SPIRVPreLegalizerCombiner, DEBUG_TYPE,
                    "Combine SPIRV machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createSPIRVPreLegalizerCombiner() {
  return new SPIRVPreLegalizerCombiner();
}
} // end namespace llvm
