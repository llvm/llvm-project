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
#include "SPIRVCombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Analysis.h"

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

class SPIRVPreLegalizerCombinerImpl : public Combiner {
protected:
  const SPIRVCombinerHelper Helper;
  const SPIRVPreLegalizerCombinerImplRuleConfig &RuleConfig;
  const SPIRVSubtarget &STI;

public:
  SPIRVPreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
      GISelCSEInfo *CSEInfo,
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
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
    GISelCSEInfo *CSEInfo,
    const SPIRVPreLegalizerCombinerImplRuleConfig &RuleConfig,
    const SPIRVSubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true, &VT, MDT, LI, STI),
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

class SPIRVPreLegalizerCombinerLegacy : public MachineFunctionPass {
public:
  static char ID;

  SPIRVPreLegalizerCombinerLegacy();

  StringRef getPassName() const override { return "SPIRVPreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // end anonymous namespace

void SPIRVPreLegalizerCombinerLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

SPIRVPreLegalizerCombinerLegacy::SPIRVPreLegalizerCombinerLegacy()
    : MachineFunctionPass(ID) {}

static bool
runPreLegalizerCombiner(MachineFunction &MF, bool ShouldSkip,
                        function_ref<GISelValueTracking *()> GetVT,
                        function_ref<MachineDominatorTree *()> GetMDT) {
  if (MF.getProperties().hasFailedISel())
    return false;

  SPIRVPreLegalizerCombinerImplRuleConfig RuleConfig;
  if (!RuleConfig.parseCommandLineOption())
    reportFatalUsageError("Invalid rule identifier");

  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !ShouldSkip;
  GISelValueTracking *VT = GetVT();
  MachineDominatorTree *MDT = GetMDT();
  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // This is the first Combiner, so the input IR might contain dead
  // instructions.
  CInfo.EnableFullDCE = false;
  SPIRVPreLegalizerCombinerImpl Impl(MF, CInfo, *VT, /*CSEInfo*/ nullptr,
                                     RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char SPIRVPreLegalizerCombinerLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(SPIRVPreLegalizerCombinerLegacy, DEBUG_TYPE,
                      "Combine SPIRV machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(SPIRVPreLegalizerCombinerLegacy, DEBUG_TYPE,
                    "Combine SPIRV machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createSPIRVPreLegalizerCombinerLegacyPass() {
  return new SPIRVPreLegalizerCombinerLegacy();
}
} // end namespace llvm

bool SPIRVPreLegalizerCombinerLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runPreLegalizerCombiner(
      MF, skipFunction(MF.getFunction()),
      [&]() {
        return &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
      },
      [&]() {
        return &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
      });
}

PreservedAnalyses
SPIRVPreLegalizerCombinerPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runPreLegalizerCombiner(
      MF, MF.getFunction().hasOptNone(),
      [&]() { return &MFAM.getResult<GISelValueTrackingAnalysis>(MF); },
      [&]() { return &MFAM.getResult<MachineDominatorTreeAnalysis>(MF); });
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<GISelValueTrackingAnalysis>();
}
