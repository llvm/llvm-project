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
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mips-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

#define MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "MipsGenPostLegalizeGICombiner.inc"
#undef MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "MipsGenPostLegalizeGICombiner.inc"
#undef MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class MipsPostLegalizerCombinerInfo final : public CombinerInfo {
  GISelKnownBits *KB;

public:
  MipsGenPostLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

  MipsPostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                GISelKnownBits *KB, const MipsLegalizerInfo *LI)
      : CombinerInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     /*LegalizerInfo*/ LI, EnableOpt, OptSize, MinSize),
        KB(KB) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
               MachineIRBuilder &B) const override;
};

bool MipsPostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                            MachineInstr &MI,
                                            MachineIRBuilder &B) const {

  CombinerHelper Helper(Observer, B, KB,
                        /*DominatorTree*/ nullptr, LInfo);
  MipsGenPostLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "MipsGenPostLegalizeGICombiner.inc"
#undef MIPSPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

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
