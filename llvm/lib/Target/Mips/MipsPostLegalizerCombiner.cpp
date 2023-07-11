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
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
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

class MipsPostLegalizerCombinerImpl : public GIMatchTableExecutor {
protected:
  CombinerHelper &Helper;
  const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig;

  const MipsSubtarget &STI;
  GISelChangeObserver &Observer;
  MachineIRBuilder &B;
  MachineFunction &MF;

  MachineRegisterInfo &MRI;

public:
  MipsPostLegalizerCombinerImpl(
      const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const MipsSubtarget &STI, GISelChangeObserver &Observer,
      MachineIRBuilder &B, CombinerHelper &Helper);

  static const char *getName() { return "MipsPostLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

MipsPostLegalizerCombinerImpl::MipsPostLegalizerCombinerImpl(
    const MipsPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const MipsSubtarget &STI, GISelChangeObserver &Observer,
    MachineIRBuilder &B, CombinerHelper &Helper)
    : Helper(Helper), RuleConfig(RuleConfig), STI(STI), Observer(Observer),
      B(B), MF(B.getMF()), MRI(*B.getMRI()),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class MipsPostLegalizerCombinerInfo final : public CombinerInfo {
  GISelKnownBits *KB;

public:
  MipsPostLegalizerCombinerImplRuleConfig RuleConfig;

  MipsPostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                GISelKnownBits *KB, const MipsLegalizerInfo *LI)
      : CombinerInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     /*LegalizerInfo*/ LI, EnableOpt, OptSize, MinSize),
        KB(KB) {
    if (!RuleConfig.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
               MachineIRBuilder &B) const override;
};

bool MipsPostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                            MachineInstr &MI,
                                            MachineIRBuilder &B) const {
  const auto &STI = MI.getMF()->getSubtarget<MipsSubtarget>();
  CombinerHelper Helper(Observer, B, /* IsPreLegalize*/ false, KB,
                        /*DominatorTree*/ nullptr, LInfo);
  MipsPostLegalizerCombinerImpl Impl(RuleConfig, STI, Observer, B, Helper);
  Impl.setupMF(*MI.getMF(), KB);
  return Impl.tryCombineAll(MI);
}

// Pass boilerplate
// ================

class MipsPostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  MipsPostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "MipsPostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
};
} // end anonymous namespace

void MipsPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
  }
  MachineFunctionPass::getAnalysisUsage(AU);
}

MipsPostLegalizerCombiner::MipsPostLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeMipsPostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool MipsPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);

  const MipsSubtarget &ST = MF.getSubtarget<MipsSubtarget>();
  const MipsLegalizerInfo *LI =
      static_cast<const MipsLegalizerInfo *>(ST.getLegalizerInfo());

  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MipsPostLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                       F.hasMinSize(), KB, LI);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char MipsPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(MipsPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine Mips machine instrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(MipsPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine Mips machine instrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createMipsPostLegalizeCombiner(bool IsOptNone) {
  return new MipsPostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
