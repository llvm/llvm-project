//=== lib/CodeGen/GlobalISel/AArch64O0PreLegalizerCombiner.cpp ------------===//
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

#include "AArch64GlobalISelUtils.h"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-O0-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

class AArch64O0PreLegalizerCombinerHelperState {
protected:
  CombinerHelper &Helper;

public:
  AArch64O0PreLegalizerCombinerHelperState(CombinerHelper &Helper)
      : Helper(Helper) {}
};

#define AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenO0PreLegalizeGICombiner.inc"
#undef AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AArch64GenO0PreLegalizeGICombiner.inc"
#undef AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AArch64O0PreLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;
  AArch64GenO0PreLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

public:
  AArch64O0PreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                    GISelKnownBits *KB,
                                    MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64O0PreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                                MachineInstr &MI,
                                                MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);
  AArch64GenO0PreLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper);

  if (Generated.tryCombineAll(Observer, MI, B))
    return true;

  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  case TargetOpcode::G_MEMCPY:
  case TargetOpcode::G_MEMMOVE:
  case TargetOpcode::G_MEMSET: {
    // At -O0 set a maxlen of 32 to inline;
    unsigned MaxLen = 32;
    // Try to inline memcpy type calls if optimizations are enabled.
    if (Helper.tryCombineMemCpyFamily(MI, MaxLen))
      return true;
    if (Opc == TargetOpcode::G_MEMSET)
      return llvm::AArch64GISelUtils::tryEmitBZero(MI, B, EnableMinSize);
    return false;
  }
  }

  return false;
}

#define AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenO0PreLegalizeGICombiner.inc"
#undef AARCH64O0PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AArch64O0PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AArch64O0PreLegalizerCombiner();

  StringRef getPassName() const override {
    return "AArch64O0PreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void AArch64O0PreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AArch64O0PreLegalizerCombiner::AArch64O0PreLegalizerCombiner()
    : MachineFunctionPass(ID) {
  initializeAArch64O0PreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AArch64O0PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto &TPC = getAnalysis<TargetPassConfig>();

  const Function &F = MF.getFunction();
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  AArch64O0PreLegalizerCombinerInfo PCInfo(
      false, F.hasOptSize(), F.hasMinSize(), KB, nullptr /* MDT */);
  Combiner C(PCInfo, &TPC);
  return C.combineMachineInstrs(MF, nullptr /* CSEInfo */);
}

char AArch64O0PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64O0PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(AArch64O0PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createAArch64O0PreLegalizerCombiner() {
  return new AArch64O0PreLegalizerCombiner();
}
} // end namespace llvm
