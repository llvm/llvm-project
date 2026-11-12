//=== lib/CodeGen/GlobalISel/MipsPreLegalizerCombiner.cpp --------------===//
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

#include "Mips.h"
#include "MipsLegalizerInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"

#define DEBUG_TYPE "mips-prelegalizer-combiner"

using namespace llvm;

namespace {
struct MipsPreLegalizerCombinerInfo : public CombinerInfo {
public:
  MipsPreLegalizerCombinerInfo()
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*EnableOpt*/ false,
                     /*EnableOptSize*/ false, /*EnableMinSize*/ false) {}
};

class MipsPreLegalizerCombinerImpl : public Combiner {
protected:
  const MipsSubtarget &STI;
  const CombinerHelper Helper;

public:
  MipsPreLegalizerCombinerImpl(MachineFunction &MF, CombinerInfo &CInfo,
                               GISelValueTracking &VT, GISelCSEInfo *CSEInfo,
                               const MipsSubtarget &STI,
                               MachineDominatorTree *MDT,
                               const LegalizerInfo *LI)
      : Combiner(MF, CInfo, &VT, CSEInfo), STI(STI),
        Helper(Observer, B, /*IsPreLegalize*/ true, &VT, MDT, LI) {}

  static const char *getName() { return "MipsPreLegalizerCombiner"; }

  void setupGeneratedPerFunctionState(MachineFunction &MF) override {
    // TODO: TableGen-erate this class' impl.
  }

  bool tryCombineAll(MachineInstr &MI) const override {
    switch (MI.getOpcode()) {
    default:
      return false;
    case TargetOpcode::G_MEMCPY_INLINE:
    case TargetOpcode::G_MEMSET_INLINE:
      return Helper.tryCombineMemCpyFamily(MI);
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_SEXTLOAD:
    case TargetOpcode::G_ZEXTLOAD: {
      // Don't attempt to combine non power of 2 loads or unaligned loads when
      // subtarget doesn't support them.
      auto MMO = *MI.memoperands_begin();
      const MipsSubtarget &STI = MI.getMF()->getSubtarget<MipsSubtarget>();
      if (!MMO->getSize().hasValue() ||
          !isPowerOf2_64(MMO->getSize().getValue()))
        return false;
      bool isUnaligned = MMO->getAlign() < MMO->getSize().getValue();
      if (!STI.systemSupportsUnalignedAccess() && isUnaligned)
        return false;

      return Helper.tryCombineExtendingLoads(MI);
    }
    }

    return false;
  }
};

// Pass boilerplate
// ================

class MipsPreLegalizerCombinerLegacy : public MachineFunctionPass {
public:
  static char ID;

  MipsPreLegalizerCombinerLegacy();

  StringRef getPassName() const override { return "MipsPreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void MipsPreLegalizerCombinerLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

MipsPreLegalizerCombinerLegacy::MipsPreLegalizerCombinerLegacy()
    : MachineFunctionPass(ID) {}

static bool
runPreLegalizerCombinerOnMF(MachineFunction &MF,
                            function_ref<GISelValueTracking *()> GetVT) {
  if (MF.getProperties().hasFailedISel())
    return false;

  const MipsSubtarget &ST = MF.getSubtarget<MipsSubtarget>();
  const MipsLegalizerInfo *LI =
      static_cast<const MipsLegalizerInfo *>(ST.getLegalizerInfo());

  GISelValueTracking *VT = GetVT();
  MipsPreLegalizerCombinerInfo PCInfo;
  MipsPreLegalizerCombinerImpl Impl(MF, PCInfo, *VT, /*CSEInfo*/ nullptr, ST,
                                    /*MDT*/ nullptr, LI);
  return Impl.combineMachineInstrs();
}

char MipsPreLegalizerCombinerLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(MipsPreLegalizerCombinerLegacy, DEBUG_TYPE,
                      "Combine Mips machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_END(MipsPreLegalizerCombinerLegacy, DEBUG_TYPE,
                    "Combine Mips machine instrs before legalization", false,
                    false)

FunctionPass *llvm::createMipsPreLegalizeCombinerLegacyPass() {
  return new MipsPreLegalizerCombinerLegacy();
}

bool MipsPreLegalizerCombinerLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runPreLegalizerCombinerOnMF(MF, [&]() {
    return &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  });
}

PreservedAnalyses
MipsPreLegalizerCombinerPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runPreLegalizerCombinerOnMF(
      MF, [&]() { return &MFAM.getResult<GISelValueTrackingAnalysis>(MF); });
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<GISelValueTrackingAnalysis>();
}
