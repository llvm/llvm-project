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

// Match
//  Dst = G_ADD SrcA, (G_ZEXT (G_ICMP Pred, SrcB, SrcC)
// with:
//  - Pred = unsigned >=
//  - Pred = unsigned <=
//  - Pred = equal and either SrcB or SrcC being zero
// The returned MatchInfo transforms this sequence into
//  Unused, Carry = G_USUBO SrcB', SrcC'
//  Dst, UnusedCarry = G_UADDE SrcA, Zero, Carry
// with SrcB' and SrcC' derived from SrcB and SrcC, inserting a zero constant
// value if necessary.
bool matchAddSubFromAddICmp(MachineInstr &MI, MachineRegisterInfo &MRI,
                            BuildFnTy &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD);

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcRegA;
  Register SrcRegB;
  Register SrcRegC;
  Register ZeroReg;
  CmpInst::Predicate Pred;
  if (!mi_match(
          MI, MRI,
          m_GAdd(m_Reg(SrcRegA), m_GZExt(m_GICmp(m_Pred(Pred), m_Reg(SrcRegB),
                                                 m_Reg(SrcRegC))))))
    return false;

  switch (Pred) {
  case CmpInst::ICMP_UGE:
    ZeroReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    break;
  case CmpInst::ICMP_ULE:
    std::swap(SrcRegB, SrcRegC);
    ZeroReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    break;
  case CmpInst::ICMP_EQ:
    if (mi_match(SrcRegB, MRI, m_ZeroInt())) {
      ZeroReg = SrcRegB;
    } else if (mi_match(SrcRegC, MRI, m_ZeroInt())) {
      std::swap(SrcRegB, SrcRegC);
      ZeroReg = SrcRegB;
    } else
      return false;
    break;
  default:
    return false;
  }

  Register Carry = MRI.createGenericVirtualRegister(LLT::scalar(1));
  Register UnusedReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
  Register UnusedCarry = MRI.createGenericVirtualRegister(LLT::scalar(1));

  MatchInfo = [=](MachineIRBuilder &B) {
    if (Pred != CmpInst::ICMP_EQ)
      B.buildConstant(ZeroReg, 0);
    B.buildInstr(TargetOpcode::G_USUBO, {UnusedReg, Carry}, {SrcRegB, SrcRegC});
    B.buildInstr(TargetOpcode::G_UADDE, {DstReg, UnusedCarry},
                 {SrcRegA, ZeroReg, Carry});
  };

  return true;
}

// Match
//  Dst = G_SUB SrcA, (G_ZEXT (G_ICMP Pred, SrcB, SrcC)
// with:
//  - Pred = unsigned >
//  - Pred = unsigned <
//  - Pred = not equal and either SrcB or SrcC being zero
// The returned MatchInfo transforms this sequence into
//  Unused, Carry = G_USUBO SrcB', SrcC'
//  Dst, UnusedCarry = G_USUBE SrcA, Zero, Carry
// with SrcB' and SrcC' derived from SrcB and SrcC, inserting a zero constant
// value if necessary.
bool matchSubSubFromSubICmp(MachineInstr &SubMI, MachineInstr &ICmpMI, MachineRegisterInfo &MRI,
                            BuildFnTy &MatchInfo) {
  assert(SubMI.getOpcode() == TargetOpcode::G_SUB);
  assert(ICmpMI.getOpcode() == TargetOpcode::G_ICMP);

  Register DstReg = SubMI.getOperand(0).getReg();
  Register SrcRegA = SubMI.getOperand(1).getReg();
  CmpInst::Predicate Pred =
      static_cast<CmpInst::Predicate>(ICmpMI.getOperand(1).getPredicate());
  Register SrcRegB = ICmpMI.getOperand(2).getReg();
  Register SrcRegC = ICmpMI.getOperand(3).getReg();
  Register ZeroReg;

  switch (Pred) {
  case CmpInst::ICMP_UGT:
    std::swap(SrcRegB, SrcRegC);
    ZeroReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    break;
  case CmpInst::ICMP_ULT:
    ZeroReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    break;
  case CmpInst::ICMP_NE:
    if (mi_match(SrcRegB, MRI, m_ZeroInt())) {
      ZeroReg = SrcRegB;
    } else if (mi_match(SrcRegC, MRI, m_ZeroInt())) {
      std::swap(SrcRegB, SrcRegC);
      ZeroReg = SrcRegB;
    } else
      return false;
    break;
  default:
    return false;
  }

  Register Carry = MRI.createGenericVirtualRegister(LLT::scalar(1));
  Register UnusedReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
  Register UnusedCarry = MRI.createGenericVirtualRegister(LLT::scalar(1));

  MatchInfo = [=](MachineIRBuilder &B) {
    if (Pred != CmpInst::ICMP_NE)
      B.buildConstant(ZeroReg, 0);
    B.buildInstr(TargetOpcode::G_USUBO, {UnusedReg, Carry}, {SrcRegB, SrcRegC});
    B.buildInstr(TargetOpcode::G_USUBE, {DstReg, UnusedCarry},
                 {SrcRegA, ZeroReg, Carry});
  };

  return true;
}

// Given a match
//  Dst = G_SUB SrcA, (G_ZEXT (G_ICMP Pred, SrcB, SrcC)
// check:
//  - Pred = signed >= and SrcC equals zero
//  - Pred = signed <=  and SrcB equals zero
// The returned MatchInfo transforms this sequence into
//  Unused, Carry = G_UADDO SrcB', SrcC'
//  Dst, UnusedCarry = G_USUBE SrcA, Zero, Carry
// with SrcB' and SrcC' derived from SrcB and SrcC.
bool matchSubAddFromSubICmp(MachineInstr &SubMI, MachineInstr &ICmpMI,
                            MachineRegisterInfo &MRI, BuildFnTy &MatchInfo) {
  assert(SubMI.getOpcode() == TargetOpcode::G_SUB);
  assert(ICmpMI.getOpcode() == TargetOpcode::G_ICMP);

  Register DstReg = SubMI.getOperand(0).getReg();
  Register SrcRegA = SubMI.getOperand(1).getReg();
  CmpInst::Predicate Pred =
      static_cast<CmpInst::Predicate>(ICmpMI.getOperand(1).getPredicate());
  Register SrcRegB = ICmpMI.getOperand(2).getReg();
  Register SrcRegC = ICmpMI.getOperand(3).getReg();
  Register ZeroReg;

  switch (Pred) {
  case CmpInst::ICMP_SGE:
    if (mi_match(SrcRegC, MRI, m_ZeroInt())) {
      ZeroReg = SrcRegC;
    } else
      return false;
    break;
  case CmpInst::ICMP_SLE:
    if (mi_match(SrcRegB, MRI, m_ZeroInt())) {
      std::swap(SrcRegB, SrcRegC);
      ZeroReg = SrcRegC;
    } else
      return false;
    break;
  default:
    return false;
  }

  Register Carry = MRI.createGenericVirtualRegister(LLT::scalar(1));
  Register UnusedReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
  Register UnusedCarry = MRI.createGenericVirtualRegister(LLT::scalar(1));

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildInstr(TargetOpcode::G_UADDO, {UnusedReg, Carry}, {SrcRegB, SrcRegB});
    B.buildInstr(TargetOpcode::G_USUBE, {DstReg, UnusedCarry},
                 {SrcRegA, ZeroReg, Carry});
  };

  return true;
}

// Match
//  Unused, CarryBit = G_USUBU SrcB, SrcC
//  Carry = G_ZEXT CarryBit
//  Dst = G_ADD SrcA, Carry
// The returned MatchInfo transforms this sequence into
//  Unused, Carry = G_USUBU SrcB, SrcC
//  Dst, UnusedCarry = G_UADDE SrcA, Zero, Carry
// inserting a zero constant value if necessary.
bool matchAddSubFromAddSub(MachineInstr &MI, MachineRegisterInfo &MRI,
                            BuildFnTy &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD);

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcRegA;
  Register Carry;
  Register ZeroReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
  MachineInstr *SubMI;
  if (!mi_match(
          MI, MRI,
          m_GAdd(m_Reg(SrcRegA), m_GZExt(m_Reg(Carry)))))
    return false;
  SubMI = MRI.getVRegDef(Carry);
  if (SubMI->getOpcode() != TargetOpcode::G_USUBO)
    return false;
  if (SubMI->getOperand(1).getReg() != Carry)
    return false;

  Register UnusedCarry = MRI.createGenericVirtualRegister(LLT::scalar(1));

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildConstant(ZeroReg, 0);
    B.buildInstr(TargetOpcode::G_UADDE, {DstReg, UnusedCarry},
                 {SrcRegA, ZeroReg, Carry});
  };

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
  CombinerHelper Helper(Observer, B, /*IsPreLegalize=*/false, KB, MDT, LI);
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
