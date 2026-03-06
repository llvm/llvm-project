//=== RISCVPostLegalizerCombiner.cpp --------------------------*- C++ -*-===//
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
/// RISCVPreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/FormatVariadic.h"

#define GET_GICOMBINER_DEPS
#include "RISCVGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "riscv-postlegalizer-combiner"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "RISCVGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

/// Match: G_STORE (G_FCONSTANT +0.0), addr
/// Return the source vreg in MatchInfo if matched.
bool matchFoldFPZeroStore(MachineInstr &MI, MachineRegisterInfo &MRI,
                          const RISCVSubtarget &STI, Register &MatchInfo) {
  if (MI.getOpcode() != TargetOpcode::G_STORE)
    return false;

  Register SrcReg = MI.getOperand(0).getReg();
  if (!SrcReg.isVirtual())
    return false;

  MachineInstr *Def = MRI.getVRegDef(SrcReg);
  if (!Def || Def->getOpcode() != TargetOpcode::G_FCONSTANT)
    return false;

  auto *CFP = Def->getOperand(1).getFPImm();
  if (!CFP || !CFP->getValueAPF().isPosZero())
    return false;

  unsigned ValBits = MRI.getType(SrcReg).getSizeInBits();
  if ((ValBits == 16 && !STI.hasStdExtZfh()) ||
      (ValBits == 32 && !STI.hasStdExtF()) ||
      (ValBits == 64 && (!STI.hasStdExtD() || !STI.is64Bit())))
    return false;

  MatchInfo = SrcReg;
  return true;
}

/// Apply: rewrite to G_STORE (G_CONSTANT 0 [XLEN]), addr
void applyFoldFPZeroStore(MachineInstr &MI, MachineRegisterInfo &MRI,
                          MachineIRBuilder &B, const RISCVSubtarget &STI,
                          Register &MatchInfo) {
  const unsigned XLen = STI.getXLen();

  auto Zero = B.buildConstant(LLT::scalar(XLen), 0);
  MI.getOperand(0).setReg(Zero.getReg(0));

  MachineInstr *Def = MRI.getVRegDef(MatchInfo);
  if (Def && MRI.use_nodbg_empty(MatchInfo))
    Def->eraseFromParent();

#ifndef NDEBUG
  unsigned ValBits = MRI.getType(MatchInfo).getSizeInBits();
  LLVM_DEBUG(dbgs() << formatv("[{0}] Fold FP zero store -> int zero "
                               "(XLEN={1}, ValBits={2}):\n  {3}\n",
                               DEBUG_TYPE, XLen, ValBits, MI));
#endif
}

class RISCVPostLegalizerCombinerImpl : public Combiner {
protected:
  const CombinerHelper Helper;
  const RISCVPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const RISCVSubtarget &STI;

public:
  RISCVPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
      const RISCVPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const RISCVSubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "RISCVPostLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "RISCVGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "RISCVGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

RISCVPostLegalizerCombinerImpl::RISCVPostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
    const RISCVPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const RISCVSubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &VT, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ false, &VT, MDT, LI),
      RuleConfig(RuleConfig), STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "RISCVGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class RISCVPostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostLegalizerCombiner();

  StringRef getPassName() const override {
    return "RISCVPostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  RISCVPostLegalizerCombinerImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void RISCVPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

RISCVPostLegalizerCombiner::RISCVPostLegalizerCombiner()
    : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool RISCVPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;
  assert(MF.getProperties().hasLegalized() && "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
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
  RISCVPostLegalizerCombinerImpl Impl(MF, CInfo, TPC, *VT, CSEInfo, RuleConfig,
                                      ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char RISCVPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine RISC-V MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(RISCVPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine RISC-V MachineInstrs after legalization", false,
                    false)

FunctionPass *llvm::createRISCVPostLegalizerCombiner() {
  return new RISCVPostLegalizerCombiner();
}
