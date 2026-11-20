//=== RISCVO0PreLegalizerCombiner.cpp -------------------------------------===//
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

#include "RISCV.h"
#include "RISCVSubtarget.h"
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
#include "llvm/IR/Analysis.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_GICOMBINER_DEPS
#include "RISCVGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "riscv-O0-prelegalizer-combiner"

using namespace llvm;

namespace {
#define GET_GICOMBINER_TYPES
#include "RISCVGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class RISCVO0PreLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const RISCVO0PreLegalizerCombinerImplRuleConfig &RuleConfig;
  const RISCVSubtarget &STI;

public:
  RISCVO0PreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
      GISelCSEInfo *CSEInfo,
      const RISCVO0PreLegalizerCombinerImplRuleConfig &RuleConfig,
      const RISCVSubtarget &STI);

  static const char *getName() { return "RISCVO0PreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "RISCVGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "RISCVGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

RISCVO0PreLegalizerCombinerImpl::RISCVO0PreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &VT,
    GISelCSEInfo *CSEInfo,
    const RISCVO0PreLegalizerCombinerImplRuleConfig &RuleConfig,
    const RISCVSubtarget &STI)
    : Combiner(MF, CInfo, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true, &VT), RuleConfig(RuleConfig),
      STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "RISCVGenO0PreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

// Pass boilerplate
// ================

class RISCVO0PreLegalizerCombinerLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVO0PreLegalizerCombinerLegacy();

  StringRef getPassName() const override {
    return "RISCVO0PreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void RISCVO0PreLegalizerCombinerLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

RISCVO0PreLegalizerCombinerLegacy::RISCVO0PreLegalizerCombinerLegacy()
    : MachineFunctionPass(ID) {}

static bool
runO0PreLegalizerCombiner(MachineFunction &MF,
                          function_ref<GISelValueTracking *()> GetVT) {
  if (MF.getProperties().hasFailedISel())
    return false;

  RISCVO0PreLegalizerCombinerImplRuleConfig RuleConfig;
  if (!RuleConfig.parseCommandLineOption())
    reportFatalUsageError("Invalid rule identifier");

  const Function &F = MF.getFunction();
  GISelValueTracking *VT = GetVT();

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*EnableOpt*/ false,
                     F.hasOptSize(), F.hasMinSize());
  // Disable fixed-point iteration in the Combiner. This improves compile-time
  // at the cost of possibly missing optimizations. See PR#94291 for details.
  CInfo.MaxIterations = 1;

  RISCVO0PreLegalizerCombinerImpl Impl(MF, CInfo, *VT,
                                       /*CSEInfo*/ nullptr, RuleConfig, ST);
  return Impl.combineMachineInstrs();
}

char RISCVO0PreLegalizerCombinerLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVO0PreLegalizerCombinerLegacy, DEBUG_TYPE,
                      "Combine RISC-V machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(RISCVO0PreLegalizerCombinerLegacy, DEBUG_TYPE,
                    "Combine RISC-V machine instrs before legalization", false,
                    false)

FunctionPass *llvm::createRISCVO0PreLegalizerCombinerLegacyPass() {
  return new RISCVO0PreLegalizerCombinerLegacy();
}

bool RISCVO0PreLegalizerCombinerLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return runO0PreLegalizerCombiner(MF, [&]() {
    return &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  });
}

PreservedAnalyses
RISCVO0PreLegalizerCombinerPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runO0PreLegalizerCombiner(
      MF, [&]() { return &MFAM.getResult<GISelValueTrackingAnalysis>(MF); });
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<GISelValueTrackingAnalysis>();
}
