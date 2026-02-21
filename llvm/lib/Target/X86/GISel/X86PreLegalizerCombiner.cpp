//===---------------- X86PreLegalizerCombiner.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass does combining of machine instructions at the generic MI level,
/// before the legalizer.
///
//===----------------------------------------------------------------------===//
#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"

#define GET_GICOMBINER_DEPS
#include "X86GenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "x86-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

#define GET_GICOMBINER_TYPES
#include "X86GenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class X86PreLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const X86PreLegalizerCombinerImplRuleConfig &RuleConfig;
  const X86Subtarget &STI;

public:
  X86PreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const X86PreLegalizerCombinerImplRuleConfig &RuleConfig,
      MachineDominatorTree *MDT);

  static const char *getName() { return "X86PreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

  bool tryCombineAllImpl(MachineInstr &I) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "X86GenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "X86GenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

X86PreLegalizerCombinerImpl::X86PreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const X86PreLegalizerCombinerImplRuleConfig &RuleConfig,
    MachineDominatorTree *MDT)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize=*/true, &VT, MDT,
             MF.getSubtarget<X86Subtarget>().getLegalizerInfo()),
      RuleConfig(RuleConfig), STI(MF.getSubtarget<X86Subtarget>()),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "X86GenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool X86PreLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  return tryCombineAllImpl(MI);
}

class X86PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  X86PreLegalizerCombiner();

  StringRef getPassName() const override { return "X86PreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  X86PreLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void X86PreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

X86PreLegalizerCombiner::X86PreLegalizerCombiner() : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool X86PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC.getCSEConfig());

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const LegalizerInfo *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);
  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  CombinerInfo CInfo(/*AllowIllegalOps=*/true, /*ShouldLegalizeIllegal=*/false,
                     /*LegalizerInfo=*/LI, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());

  // This is the first Combiner, so the input IR might contain dead
  // instructions.
  CInfo.EnableFullDCE = true;
  X86PreLegalizerCombinerImpl Impl(MF, CInfo, &TPC, *VT, CSEInfo, RuleConfig,
                                   MDT);
  return Impl.combineMachineInstrs();
}

char X86PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(X86PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine X86 machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(X86PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine X86 machine instrs before legalization", false,
                    false)

namespace llvm {

PreservedAnalyses
X86PreLegalizerCombinerPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (MF.getProperties().hasFailedISel())
    return PreservedAnalyses::all();

  X86PreLegalizerCombinerImplRuleConfig RuleConfig;
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");

  auto &CSEInfo = MFAM.getResult<GISelCSEAnalysis>(MF);

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const LegalizerInfo *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt = MF.getTarget().getOptLevel() != CodeGenOptLevel::None;
  GISelValueTracking &VT = MFAM.getResult<GISelValueTrackingAnalysis>(MF);
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  CombinerInfo CInfo(/*AllowIllegalOps=*/true, /*ShouldLegalizeIllegal=*/false,
                     /*LegalizerInfo=*/LI, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());

  // This is the first Combiner, so the input IR might contain dead
  // instructions.
  CInfo.EnableFullDCE = true;
  X86PreLegalizerCombinerImpl Impl(MF, CInfo, nullptr, VT, CSEInfo.get(),
                                   RuleConfig, &MDT);
  Impl.combineMachineInstrs();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GISelCSEAnalysis>();
  PA.preserve<GISelValueTrackingAnalysis>();
  return PA;
}

FunctionPass *createX86PreLegalizerCombiner() {
  return new X86PreLegalizerCombiner();
}
} // end namespace llvm
