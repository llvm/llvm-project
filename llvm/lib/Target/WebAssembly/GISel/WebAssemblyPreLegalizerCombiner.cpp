//=== WebAssemblyPreLegalizerCombiner.cpp ---------------------------------===//
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

#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "WebAssemblyGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "wasm-prelegalizer-combiner"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "WebAssemblyGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class WebAssemblyPreLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const WebAssemblyPreLegalizerCombinerImplRuleConfig &RuleConfig;
  const WebAssemblySubtarget &STI;

public:
  WebAssemblyPreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
      GISelCSEInfo *CSEInfo,
      const WebAssemblyPreLegalizerCombinerImplRuleConfig &RuleConfig,
      const WebAssemblySubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "WebAssembly00PreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "WebAssemblyGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "WebAssemblyGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

WebAssemblyPreLegalizerCombinerImpl::WebAssemblyPreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
    GISelCSEInfo *CSEInfo,
    const WebAssemblyPreLegalizerCombinerImplRuleConfig &RuleConfig,
    const WebAssemblySubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "WebAssemblyGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

// Pass boilerplate
// ================

class WebAssemblyPreLegalizerCombinerLegacy : public MachineFunctionPass {
public:
  static char ID;

  WebAssemblyPreLegalizerCombinerLegacy();

  StringRef getPassName() const override {
    return "WebAssemblyPreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void WebAssemblyPreLegalizerCombinerLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

WebAssemblyPreLegalizerCombinerLegacy::WebAssemblyPreLegalizerCombinerLegacy()
    : MachineFunctionPass(ID) {}

static bool runCombinerOnMachineFunction(
    MachineFunction &MF, function_ref<GISelCSEInfo *()> GetCSEInfo,
    function_ref<bool()> ShouldSkip, function_ref<GISelValueTracking *()> GetVT,
    function_ref<MachineDominatorTree *()> GetMDT) {
  if (MF.getProperties().hasFailedISel())
    return false;

  WebAssemblyPreLegalizerCombinerImplRuleConfig RuleConfig;
  if (!RuleConfig.parseCommandLineOption())
    reportFatalUsageError("Invalid rule identifier");

  GISelCSEInfo *CSEInfo = GetCSEInfo();

  const WebAssemblySubtarget &ST = MF.getSubtarget<WebAssemblySubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !ShouldSkip();
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
  CInfo.EnableFullDCE = true;
  WebAssemblyPreLegalizerCombinerImpl Impl(MF, CInfo, *VT, CSEInfo, RuleConfig,
                                           ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char WebAssemblyPreLegalizerCombinerLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyPreLegalizerCombinerLegacy, DEBUG_TYPE,
                      "Combine WebAssembly machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(WebAssemblyPreLegalizerCombinerLegacy, DEBUG_TYPE,
                    "Combine WebAssembly machine instrs before legalization",
                    false, false)

FunctionPass *llvm::createWebAssemblyPreLegalizerCombinerLegacyPass() {
  return new WebAssemblyPreLegalizerCombinerLegacy();
}

bool WebAssemblyPreLegalizerCombinerLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runCombinerOnMachineFunction(
      MF,
      [&]() {
        TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
        GISelCSEAnalysisWrapper &Wrapper =
            getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
        return &Wrapper.get(TPC.getCSEConfig());
      },
      [&]() { return skipFunction(MF.getFunction()); },
      [&]() {
        return &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
      },
      [&]() {
        return &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
      });
}

PreservedAnalyses
WebAssemblyPreLegalizerCombinerPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runCombinerOnMachineFunction(
      MF, [&]() { return MFAM.getResult<GISelCSEAnalysis>(MF).get(); },
      [&]() { return MF.getFunction().hasOptNone(); },
      [&]() { return &MFAM.getResult<GISelValueTrackingAnalysis>(MF); },
      [&]() { return &MFAM.getResult<MachineDominatorTreeAnalysis>(MF); });
  return Changed ? getMachineFunctionPassPreservedAnalyses()
                       .preserveSet<CFGAnalyses>()
                 : PreservedAnalyses::all();
}
