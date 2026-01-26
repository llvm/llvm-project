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
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"

#define GET_GICOMBINER_DEPS
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "wasm-postlegalizer-combiner"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

bool matchFoldGlobalOffset(MachineOperand &Dst, MachineOperand &Global,
                           MachineOperand &Offset, BuildFnTy &MatchInfo) {
  if (Dst.getParent()->getMF()->getTarget().isPositionIndependent())
    return false;

  // We assume at this point that we are looking at the Global as the operand of
  // a G_GLOBAL_VALUE, which is on the left-hand side of a NUW G_PTRADD, with
  // Offset being the immediate from a G_CONSTANT on the right-hand side.
  if (!Dst.isReg() || !Global.isGlobal() || !Offset.isCImm())
    return false;

  Register DstReg = Dst.getReg();
  auto *GV = Global.getGlobal();

  if (GV->isThreadLocal())
    return false;

  uint64_t NewOffset = Global.getOffset() + Offset.getCImm()->getSExtValue();

  if (int64_t(NewOffset) < 0) {
    return false;
  }

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildInstr(TargetOpcode::G_GLOBAL_VALUE)
        .addDef(DstReg)
        .addGlobalAddress(GV, NewOffset);
  };

  return true;
}

bool matchFormTruncstore(MachineInstr &MI, MachineRegisterInfo &MRI,
                         Register &SrcReg) {
  assert(MI.getOpcode() == TargetOpcode::G_STORE);
  Register DstReg = MI.getOperand(0).getReg();
  if (MRI.getType(DstReg).isVector())
    return false;
  // Match a store of a truncate.
  if (!mi_match(DstReg, MRI, m_GTrunc(MIPatternMatch::m_Reg(SrcReg))))
    return false;
  // Only form truncstores for value types of max 64b.
  return MRI.getType(SrcReg).getSizeInBits() <= 64;
}

void applyFormTruncstore(MachineInstr &MI, MachineRegisterInfo &MRI,
                         MachineIRBuilder &B, GISelChangeObserver &Observer,
                         Register &SrcReg) {
  assert(MI.getOpcode() == TargetOpcode::G_STORE);
  Observer.changingInstr(MI);
  MI.getOperand(0).setReg(SrcReg);
  Observer.changedInstr(MI);
}

class WebAssemblyPostLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const WebAssemblyPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const WebAssemblySubtarget &STI;

public:
  WebAssemblyPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
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
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const WebAssemblyPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const WebAssemblySubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ false, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "WebAssemblyGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class WebAssemblyPostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  WebAssemblyPostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "WebAssemblyPostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
  WebAssemblyPostLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void WebAssemblyPostLegalizerCombiner::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  }
  MachineFunctionPass::getAnalysisUsage(AU);
}

WebAssemblyPostLegalizerCombiner::WebAssemblyPostLegalizerCombiner(
    bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool WebAssemblyPostLegalizerCombiner::runOnMachineFunction(
    MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  assert(MF.getProperties().hasLegalized() && "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const WebAssemblySubtarget &ST = MF.getSubtarget<WebAssemblySubtarget>();
  const auto *LI = ST.getLegalizerInfo();

  GISelValueTracking *VT =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT = nullptr;
  GISelCSEInfo *CSEInfo = nullptr;

  if (!IsOptNone) {
    MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

    GISelCSEAnalysisWrapper &Wrapper =
        getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
    CSEInfo = &Wrapper.get(TPC->getCSEConfig());
  }

  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, F.hasOptSize(),
                     F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // Legalizer performs DCE, so a full DCE pass is unnecessary.
  CInfo.EnableFullDCE = false;
  WebAssemblyPostLegalizerCombinerImpl Impl(MF, CInfo, TPC, *VT, CSEInfo,
                                            RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char WebAssemblyPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine WebAssembly MachineInstrs after legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(WebAssemblyPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine WebAssembly MachineInstrs after legalization",
                    false, false)

FunctionPass *llvm::createWebAssemblyPostLegalizerCombiner(bool IsOptNone) {
  return new WebAssemblyPostLegalizerCombiner(IsOptNone);
}
