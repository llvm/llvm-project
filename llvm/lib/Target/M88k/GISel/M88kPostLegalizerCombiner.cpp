//=== M88kPostLegalizer.cpp --------------------------------------*- C++ -*-===//
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
/// M88kPostLegalizerLowering.
///
/// Combines which don't rely on instruction legality should go in the
/// M88kPostLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "GISel/M88kGlobalISelUtils.h"
#include "M88kTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
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

#define DEBUG_TYPE "m88k-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

/*
Match:

    %2:gr(s32) = G_CONSTANT i32 0
    %3:gr(s1) = G_ICMP intpred(eq), %0(s32), %2
    %4:gr(s32) = G_ZEXT %3(s1)
    %5:gr(s32) = nsw G_ADD %4, %1
=>
    %2:gr(s32) = G_CONSTANT i32 0
               = G_USUBO %2, %0(s32)
               = G_UADDE
        subu.co  %r0,%r0,%r2
        addu.ci  %r4,%r3,%r0

    %2:gr(s32) = G_CONSTANT i32 0
    %3:gr(s1) = G_ICMP intpred(ne), %1(s32), %2
    %4:gr(s32) = G_SEXT %3(s1)
    %5:gr(s32) = G_ADD %4, %0
=>
       subu.co  %r0,%r0,%r3
        subu.ci  %r2,%r2,%r0
*/

// Match G_ADD ...
bool matchAddCmpToSubAdd(MachineInstr &MI, MachineRegisterInfo &MRI,
                         std::tuple<Register, Register, Register> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD);

  Register SrcRegA;
  Register SrcRegB;
  Optional<ValueAndVReg> CstValReg;
  CmpInst::Predicate Pred;
  if (!mi_match(
          MI, MRI,
          m_GAdd(m_Reg(SrcRegA), m_GZExt(m_GICmp(m_Pred(Pred), m_Reg(SrcRegB),
                                                 m_GCst(CstValReg))))))
    return false;

  if (Pred != CmpInst::ICMP_EQ || !CstValReg || CstValReg->Value != 0)
    return false;

  MatchInfo = std::make_tuple(SrcRegA, SrcRegB, CstValReg->VReg);
/*
  MatchInfo = [=](MachineIRBuilder &B) {
    Register CarryOut = MRI.createGenericVirtualRegister(LLT::scalar(32));

    B.buildInstr(TargetOpcode::G_USUBO, {}, {});
    B.buildInstr(TargetOpcode::G_UADDE, {}, {});
  };
*/
  return true;
}

// Lower to ...
bool applyAddCmpToSubAdd(MachineInstr &MI, MachineRegisterInfo &MRI,
                         std::tuple<Register, Register, Register> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD);

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcRegA;
  Register SrcRegB;
  Register ZeroReg;
  std::tie(SrcRegA, SrcRegB, ZeroReg) = MatchInfo;

  MachineIRBuilder B(MI);
  Register Carry = MRI.createGenericVirtualRegister(LLT::scalar(32));
  Register UnusedReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
  Register UnusedCarry = MRI.createGenericVirtualRegister(LLT::scalar(32));

  B.buildInstr(TargetOpcode::G_USUBO, {UnusedReg, Carry}, {ZeroReg, SrcRegB});
  B.buildInstr(TargetOpcode::G_UADDE, {DstReg, UnusedCarry},
               {SrcRegA, ZeroReg, Carry});
  MI.eraseFromParent();
  return true;
}

#define M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "M88kGenPostLegalizeGICombiner.inc"
#undef M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "M88kGenPostLegalizeGICombiner.inc"
#undef M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class M88kPostLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;
  M88kGenPostLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

public:
  M88kPostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
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

bool M88kPostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                            MachineInstr &MI,
                                            MachineIRBuilder &B) const {
  const auto *LI =
      MI.getParent()->getParent()->getSubtarget().getLegalizerInfo();
  CombinerHelper Helper(Observer, B, KB, MDT, LI);
  M88kGenPostLegalizerCombinerHelper Generated(GeneratedRuleCfg);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "M88kGenPostLegalizeGICombiner.inc"
#undef M88KPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class M88kPostLegalizerCombiner : public MachineFunctionPass {
  bool IsOptNone;
public:
  static char ID;

  M88kPostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override { return "M88kPostLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void M88kPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  }
  MachineFunctionPass::getAnalysisUsage(AU);
}

M88kPostLegalizerCombiner::M88kPostLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeM88kPostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool M88kPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
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
  M88kPostLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                       F.hasMinSize(), KB, MDT);
  GISelCSEInfo *CSEInfo = nullptr;
  if (!IsOptNone) {
    GISelCSEAnalysisWrapper &Wrapper =
        getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
    CSEInfo = &Wrapper.get(TPC->getCSEConfig());
  }
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, CSEInfo);
}

char M88kPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(M88kPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine M88k machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(M88kPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine M88k machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createM88kPostLegalizerCombiner(bool IsOptNone) {
  return new M88kPostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
