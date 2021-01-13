//=== lib/CodeGen/GlobalISel/AMDGPURegBankCombiner.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// after register banks are known.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Target/TargetMachine.h"
#define DEBUG_TYPE "amdgpu-regbank-combiner"

using namespace llvm;
using namespace MIPatternMatch;

class AMDGPURegBankCombinerHelper {
protected:
  MachineIRBuilder &B;
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const RegisterBankInfo &RBI;
  const TargetRegisterInfo &TRI;
  CombinerHelper &Helper;

public:
  AMDGPURegBankCombinerHelper(MachineIRBuilder &B, CombinerHelper &Helper)
      : B(B), MF(B.getMF()), MRI(*B.getMRI()),
        RBI(*MF.getSubtarget().getRegBankInfo()),
        TRI(*MF.getSubtarget().getRegisterInfo()), Helper(Helper){};

  bool isVgprRegBank(Register Reg);

  struct RmUniformWFMatchInfo {
    Register Dst, WFReplaceReg;
  };

  // Combine support to remove amdgcn_waterfall_readfirstlane or
  // amdgcn_waterfall_begin intrinsics if the index is determined to be
  // uniform. The SIInsertWaterfall pass can handle their removal and in some
  // cases remove the waterfall altogether
  bool matchRmUniformWF(MachineInstr &MI, RmUniformWFMatchInfo &MatchInfo);
  void applyRmUniformWF(MachineInstr &MI, RmUniformWFMatchInfo &MatchInfo);
};

bool AMDGPURegBankCombinerHelper::isVgprRegBank(Register Reg) {
  return RBI.getRegBank(Reg, MRI, TRI)->getID() == AMDGPU::VGPRRegBankID;
}

bool AMDGPURegBankCombinerHelper::matchRmUniformWF(
    MachineInstr &MI, RmUniformWFMatchInfo &MatchInfo) {
  auto IntrID = MI.getIntrinsicID();
  Register Dst, WFReplaceReg;

  switch (IntrID) {
  case Intrinsic::amdgcn_waterfall_readfirstlane: {
    Register IdxReg = MI.getOperand(3).getReg();
    Register IdxSrcReg = getSrcRegIgnoringCopies(IdxReg, MRI);
    if (!isVgprRegBank(IdxSrcReg)) {
      WFReplaceReg = IdxSrcReg;
      break;
    }
    return false;
  }
  case Intrinsic::amdgcn_waterfall_begin: {
    Register IdxReg = MI.getOperand(3).getReg();
    if (!isVgprRegBank(getSrcRegIgnoringCopies(IdxReg, MRI))) {
      WFReplaceReg = MI.getOperand(2).getReg();
      break;
    }
    return false;
  }
  default:
    return false;
  }

  Dst = MI.getOperand(0).getReg();
  MatchInfo = {Dst, WFReplaceReg};
  return true;
}

void AMDGPURegBankCombinerHelper::applyRmUniformWF(
    MachineInstr &MI, RmUniformWFMatchInfo &MatchInfo) {
  MI.eraseFromParent();
  Helper.replaceRegWith(MRI, MatchInfo.Dst, MatchInfo.WFReplaceReg);
}

class AMDGPURegBankCombinerHelperState {
protected:
  CombinerHelper &Helper;
  AMDGPURegBankCombinerHelper &RegBankHelper;

public:
  AMDGPURegBankCombinerHelperState(CombinerHelper &Helper,
                                   AMDGPURegBankCombinerHelper &RegBankHelper)
      : Helper(Helper), RegBankHelper(RegBankHelper) {}
};

#define AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AMDGPUGenRegBankGICombiner.inc"
#undef AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AMDGPUGenRegBankGICombiner.inc"
#undef AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_H

class AMDGPURegBankCombinerInfo final : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AMDGPUGenRegBankCombinerHelperRuleConfig GeneratedRuleCfg;

  AMDGPURegBankCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  const AMDGPULegalizerInfo *LI,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     /*LegalizerInfo*/ LI, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
               MachineIRBuilder &B) const override;
};

bool AMDGPURegBankCombinerInfo::combine(GISelChangeObserver &Observer,
                                        MachineInstr &MI,
                                        MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);
  AMDGPURegBankCombinerHelper RegBankHelper(B, Helper);
  AMDGPUGenRegBankCombinerHelper Generated(GeneratedRuleCfg, Helper,
                                           RegBankHelper);

  if (Generated.tryCombineAll(Observer, MI, B))
    return true;

  return false;
}

#define AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AMDGPUGenRegBankGICombiner.inc"
#undef AMDGPUREGBANKCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AMDGPURegBankCombiner : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURegBankCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "AMDGPURegBankCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  bool IsOptNone;
};
} // end anonymous namespace

void AMDGPURegBankCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

AMDGPURegBankCombiner::AMDGPURegBankCombiner(bool IsOptNone)
  : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAMDGPURegBankCombinerPass(*PassRegistry::getPassRegistry());
}

bool AMDGPURegBankCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const AMDGPULegalizerInfo *LI
    = static_cast<const AMDGPULegalizerInfo *>(ST.getLegalizerInfo());

  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  AMDGPURegBankCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                         F.hasMinSize(), LI, KB, MDT);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AMDGPURegBankCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AMDGPURegBankCombiner, DEBUG_TYPE,
                      "Combine AMDGPU machine instrs after regbankselect",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AMDGPURegBankCombiner, DEBUG_TYPE,
                    "Combine AMDGPU machine instrs after regbankselect", false,
                    false)

namespace llvm {
FunctionPass *createAMDGPURegBankCombiner(bool IsOptNone) {
  return new AMDGPURegBankCombiner(IsOptNone);
}
} // end namespace llvm
