//=== lib/CodeGen/GlobalISel/AArch64PreLegalizerCombiner.cpp --------------===//
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

#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

/// Return true if a G_FCONSTANT instruction is known to be better-represented
/// as a G_CONSTANT.
static bool matchFConstantToConstant(MachineInstr &MI,
                                     MachineRegisterInfo &MRI) {
  assert(MI.getOpcode() == TargetOpcode::G_FCONSTANT);
  Register DstReg = MI.getOperand(0).getReg();
  const unsigned DstSize = MRI.getType(DstReg).getSizeInBits();
  if (DstSize != 32 && DstSize != 64)
    return false;

  // When we're storing a value, it doesn't matter what register bank it's on.
  // Since not all floating point constants can be materialized using a fmov,
  // it makes more sense to just use a GPR.
  return all_of(MRI.use_nodbg_instructions(DstReg),
                [](const MachineInstr &Use) { return Use.mayStore(); });
}

/// Change a G_FCONSTANT into a G_CONSTANT.
static void applyFConstantToConstant(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_FCONSTANT);
  MachineIRBuilder MIB(MI);
  const APFloat &ImmValAPF = MI.getOperand(1).getFPImm()->getValueAPF();
  MIB.buildConstant(MI.getOperand(0).getReg(), ImmValAPF.bitcastToAPInt());
  MI.eraseFromParent();
}

#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AArch64PreLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AArch64GenPreLegalizerCombinerHelper Generated;

  AArch64PreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!Generated.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);

  switch (MI.getOpcode()) {
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
    switch (MI.getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset: {
      // If we're at -O0 set a maxlen of 32 to inline, otherwise let the other
      // heuristics decide.
      unsigned MaxLen = EnableOpt ? 0 : 32;
      // Try to inline memcpy type calls if optimizations are enabled.
      return (!EnableMinSize) ? Helper.tryCombineMemCpyFamily(MI, MaxLen)
                              : false;
    }
    default:
      break;
    }
  }

  if (Generated.tryCombineAll(Observer, MI, B, Helper))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  }

  return false;
}

#define AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenPreLegalizeGICombiner.inc"
#undef AARCH64PRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AArch64PreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AArch64PreLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override { return "AArch64PreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  bool IsOptNone;
};
} // end anonymous namespace

void AArch64PreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

AArch64PreLegalizerCombiner::AArch64PreLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAArch64PreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AArch64PreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  AArch64PreLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                         F.hasMinSize(), KB, MDT);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AArch64PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 machine instrs before legalization", false,
                    false)


namespace llvm {
FunctionPass *createAArch64PreLegalizeCombiner(bool IsOptNone) {
  return new AArch64PreLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
