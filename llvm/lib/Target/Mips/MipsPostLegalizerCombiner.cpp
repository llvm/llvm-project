//=== lib/CodeGen/GlobalISel/MipsPostLegalizerCombiner.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// after the legalizer.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "Mips.h"
#include "MipsLegalizerInfo.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "mips-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

namespace {
#define GET_GICOMBINER_TYPES
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class MipsPostLegalizerCombinerImpl : public Combiner {
protected:
  const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const MipsSubtarget &STI;
  const CombinerHelper Helper;

public:
  MipsPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const MipsSubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "MipsPostLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

MipsPostLegalizerCombinerImpl::MipsPostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const MipsSubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo), RuleConfig(RuleConfig), STI(STI),
      Helper(Observer, B, /*IsPreLegalize*/ false, &VT, MDT, LI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

// Pass boilerplate
// ================

class MipsPostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  MipsPostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override { return "MipsPostLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
  MipsPostLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void MipsPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
  }
  MachineFunctionPass::getAnalysisUsage(AU);
}

MipsPostLegalizerCombiner::MipsPostLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool MipsPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const MipsSubtarget &ST = MF.getSubtarget<MipsSubtarget>();
  const MipsLegalizerInfo *LI =
      static_cast<const MipsLegalizerInfo *>(ST.getLegalizerInfo());

  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr
                : &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  CombinerInfo CInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     LI, EnableOpt, F.hasOptSize(), F.hasMinSize());
  MipsPostLegalizerCombinerImpl Impl(MF, CInfo, TPC, *VT, /*CSEInfo*/ nullptr,
                                     RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char MipsPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(MipsPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine Mips machine instrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(MipsPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine Mips machine instrs after legalization", false,
                    false)

FunctionPass *llvm::createMipsPostLegalizeCombiner(bool IsOptNone) {
  return new MipsPostLegalizerCombiner(IsOptNone);
}
