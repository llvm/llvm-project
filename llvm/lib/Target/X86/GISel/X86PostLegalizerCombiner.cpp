//=== X86PostLegalizerCombiner.cpp --------------------------*- C++ -*-===//
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
/// Lowering combines (e.g. pseudo matching) should be handled by
/// X86PostLegalizerLowering.
///
/// Combines which don't rely on instruction legality should go in the
/// X86PreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//
#include "X86.h"
#include "X86TargetMachine.h"
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
#include "llvm/Support/Debug.h"

#define GET_GICOMBINER_DEPS
#include "X86GenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "X86-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

#define GET_GICOMBINER_TYPES
#include "X86GenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class X86PostLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const X86PostLegalizerCombinerImplRuleConfig &RuleConfig;
  const X86Subtarget &STI;

public:
  X86PostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const X86PostLegalizerCombinerImplRuleConfig &RuleConfig,
      const X86Subtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "X86PostLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;
  bool tryCombineAllImpl(MachineInstr &I) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "X86GenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "X86GenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

X86PostLegalizerCombinerImpl::X86PostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const X86PostLegalizerCombinerImplRuleConfig &RuleConfig,
    const X86Subtarget &STI, MachineDominatorTree *MDT, const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ false, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "X86GenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool X86PostLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  return tryCombineAllImpl(MI);
}

class X86PostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  X86PostLegalizerCombiner();

  StringRef getPassName() const override { return "X86PostLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  X86PostLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void X86PostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  // This is only added when processing level is not OptNone.
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();

  MachineFunctionPass::getAnalysisUsage(AU);
}

X86PostLegalizerCombiner::X86PostLegalizerCombiner() : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool X86PostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  assert(MF.getProperties().hasLegalized() && "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const auto *LI = ST.getLegalizerInfo();

  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC->getCSEConfig());

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // Legalizer performs DCE, so a full DCE pass is unnecessary.
  CInfo.EnableFullDCE = false;
  X86PostLegalizerCombinerImpl Impl(MF, CInfo, TPC, *VT, CSEInfo, RuleConfig,
                                    ST, MDT, LI);
  bool Changed = Impl.combineMachineInstrs();
  return Changed;
}

char X86PostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(X86PostLegalizerCombiner, DEBUG_TYPE,
                      "Combine X86 MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(X86PostLegalizerCombiner, DEBUG_TYPE,
                    "Combine X86 MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createX86PostLegalizerCombiner() {
  return new X86PostLegalizerCombiner();
}
} // end namespace llvm
