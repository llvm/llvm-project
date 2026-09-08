//=== WebAssemblyPostLegalizerCombiner.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Post-legalization combines on generic MachineInstrs.
///
/// The combines here must preserve instruction legality.
///
/// Combines which don't rely on instruction legality should go in the
/// WebAssemblyPreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Analysis.h"

#define GET_GICOMBINER_DEPS
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "wasm-postlegalizer-combiner"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class WebAssemblyPostLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const WebAssemblyPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const WebAssemblySubtarget &STI;

public:
  WebAssemblyPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
      GISelCSEInfo *CSEInfo,
      const WebAssemblyPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const WebAssemblySubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "WebAssemblyPostLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

WebAssemblyPostLegalizerCombinerImpl::WebAssemblyPostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
    GISelCSEInfo *CSEInfo,
    const WebAssemblyPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const WebAssemblySubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ false, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class WebAssemblyPostLegalizerCombinerLegacy : public MachineFunctionPass {
public:
  static char ID;

  WebAssemblyPostLegalizerCombinerLegacy();

  StringRef getPassName() const override {
    return "WebAssemblyPostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void WebAssemblyPostLegalizerCombinerLegacy::getAnalysisUsage(
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

WebAssemblyPostLegalizerCombinerLegacy::WebAssemblyPostLegalizerCombinerLegacy()
    : MachineFunctionPass(ID) {}

static bool
runCombinerOnMachineFunction(MachineFunction &MF,
                             function_ref<bool()> ShouldSkip,
                             function_ref<GISelValueTracking *()> GetVT,
                             function_ref<MachineDominatorTree *()> GetMDT,
                             function_ref<GISelCSEInfo *()> GetCSEInfo) {
  if (MF.getProperties().hasFailedISel())
    return false;
  assert(MF.getProperties().hasLegalized() && "Expected a legalized function?");
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !ShouldSkip;

  WebAssemblyPostLegalizerCombinerImplRuleConfig RuleConfig;
  if (!RuleConfig.parseCommandLineOption())
    reportFatalUsageError("Invalid rule identifier");

  const WebAssemblySubtarget &ST = MF.getSubtarget<WebAssemblySubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  GISelValueTracking *VT = GetVT();
  MachineDominatorTree *MDT = GetMDT();
  GISelCSEInfo *CSEInfo = GetCSEInfo();

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // Legalizer performs DCE, so a full DCE pass is unnecessary.
  CInfo.EnableFullDCE = false;
  WebAssemblyPostLegalizerCombinerImpl Impl(MF, CInfo, *VT, CSEInfo, RuleConfig,
                                            ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char WebAssemblyPostLegalizerCombinerLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyPostLegalizerCombinerLegacy, DEBUG_TYPE,
                      "Combine WebAssembly MachineInstrs after legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(WebAssemblyPostLegalizerCombinerLegacy, DEBUG_TYPE,
                    "Combine WebAssembly MachineInstrs after legalization",
                    false, false)

FunctionPass *llvm::createWebAssemblyPostLegalizerCombinerLegacyPass() {
  return new WebAssemblyPostLegalizerCombinerLegacy();
}

bool WebAssemblyPostLegalizerCombinerLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runCombinerOnMachineFunction(
      MF, [&]() { return skipFunction(MF.getFunction()); },
      [&]() {
        return &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
      },
      [&]() {
        return &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
      },
      [&]() {
        TargetPassConfig *TPC = &getAnalysis<TargetPassConfig>();
        GISelCSEAnalysisWrapper &Wrapper =
            getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
        return &Wrapper.get(TPC->getCSEConfig());
      });
}

PreservedAnalyses WebAssemblyPostLegalizerCombinerPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runCombinerOnMachineFunction(
      MF, [&]() { return MF.getFunction().hasOptNone(); },
      [&]() { return &MFAM.getResult<GISelValueTrackingAnalysis>(MF); },
      [&]() { return &MFAM.getResult<MachineDominatorTreeAnalysis>(MF); },
      [&]() { return MFAM.getResult<GISelCSEAnalysis>(MF).get(); });
  return Changed ? getMachineFunctionPassPreservedAnalyses()
                       .preserveSet<CFGAnalyses>()
                 : PreservedAnalyses::all();
}
