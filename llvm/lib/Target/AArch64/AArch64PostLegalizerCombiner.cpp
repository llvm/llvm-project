//=== lib/CodeGen/GlobalISel/AArch64PostLegalizerCombiner.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This performs post-legalization combines on generic MachineInstrs.
//
// Any combine that this pass performs must preserve instruction legality.
// Combines unconcerned with legality should be handled by the
// PreLegalizerCombiner instead.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-postlegalizer-combiner"

using namespace llvm;

#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AArch64PostLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AArch64GenPostLegalizerCombinerHelper Generated;

  AArch64PostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                   GISelKnownBits *KB,
                                   MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!Generated.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                               MachineInstr &MI,
                                               MachineIRBuilder &B) const {
  const auto *LI =
      MI.getParent()->getParent()->getSubtarget().getLegalizerInfo();
  CombinerHelper Helper(Observer, B, KB, MDT, LI);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

class AArch64PostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AArch64PostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "AArch64PostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
};
} // end anonymous namespace

void AArch64PostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

AArch64PostLegalizerCombiner::AArch64PostLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAArch64PostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AArch64PostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  AArch64PostLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                          F.hasMinSize(), KB, MDT);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AArch64PostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PostLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AArch64PostLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createAArch64PostLegalizeCombiner(bool IsOptNone) {
  return new AArch64PostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
