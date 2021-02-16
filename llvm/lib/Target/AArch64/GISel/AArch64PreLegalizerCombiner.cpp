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
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"
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

/// Try to match a G_ICMP of a G_TRUNC with zero, in which the truncated bits
/// are sign bits. In this case, we can transform the G_ICMP to directly compare
/// the wide value with a zero.
static bool matchICmpRedundantTrunc(MachineInstr &MI, MachineRegisterInfo &MRI,
                                    GISelKnownBits *KB, Register &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP && KB);

  auto Pred = (CmpInst::Predicate)MI.getOperand(1).getPredicate();
  if (!ICmpInst::isEquality(Pred))
    return false;

  Register LHS = MI.getOperand(2).getReg();
  LLT LHSTy = MRI.getType(LHS);
  if (!LHSTy.isScalar())
    return false;

  Register RHS = MI.getOperand(3).getReg();
  Register WideReg;

  if (!mi_match(LHS, MRI, m_GTrunc(m_Reg(WideReg))) ||
      !mi_match(RHS, MRI, m_SpecificICst(0)))
    return false;

  LLT WideTy = MRI.getType(WideReg);
  if (KB->computeNumSignBits(WideReg) <=
      WideTy.getSizeInBits() - LHSTy.getSizeInBits())
    return false;

  MatchInfo = WideReg;
  return true;
}

static bool applyICmpRedundantTrunc(MachineInstr &MI, MachineRegisterInfo &MRI,
                                    MachineIRBuilder &Builder,
                                    GISelChangeObserver &Observer,
                                    Register &WideReg) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);

  LLT WideTy = MRI.getType(WideReg);
  // We're going to directly use the wide register as the LHS, and then use an
  // equivalent size zero for RHS.
  Builder.setInstrAndDebugLoc(MI);
  auto WideZero = Builder.buildConstant(WideTy, 0);
  Observer.changingInstr(MI);
  MI.getOperand(2).setReg(WideReg);
  MI.getOperand(3).setReg(WideZero.getReg(0));
  Observer.changedInstr(MI);
  return true;
}

class AArch64PreLegalizerCombinerHelperState {
protected:
  CombinerHelper &Helper;

public:
  AArch64PreLegalizerCombinerHelperState(CombinerHelper &Helper)
      : Helper(Helper) {}
};

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
  AArch64GenPreLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

public:
  AArch64PreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);
  AArch64GenPreLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper);

  if (Generated.tryCombineAll(Observer, MI, B))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  case TargetOpcode::G_MEMCPY:
  case TargetOpcode::G_MEMMOVE:
  case TargetOpcode::G_MEMSET: {
    // If we're at -O0 set a maxlen of 32 to inline, otherwise let the other
    // heuristics decide.
    unsigned MaxLen = EnableOpt ? 0 : 32;
    // Try to inline memcpy type calls if optimizations are enabled.
    return !EnableMinSize ? Helper.tryCombineMemCpyFamily(MI, MaxLen) : false;
  }
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
  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
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
  auto &TPC = getAnalysis<TargetPassConfig>();

  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC.getCSEConfig());

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  AArch64PreLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                         F.hasMinSize(), KB, MDT);
  Combiner C(PCInfo, &TPC);
  return C.combineMachineInstrs(MF, CSEInfo);
}

char AArch64PreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(AArch64PreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 machine instrs before legalization", false,
                    false)


namespace llvm {
FunctionPass *createAArch64PreLegalizerCombiner(bool IsOptNone) {
  return new AArch64PreLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
