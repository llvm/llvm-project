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
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
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

  struct MinMaxMedOpc {
    unsigned Min, Max, Med;
  };

  struct Med3MatchInfo {
    unsigned Opc;
    Register Val0, Val1, Val2;
  };

  MinMaxMedOpc getMinMaxPair(unsigned Opc);

  template <class m_Cst>
  bool matchMed(MachineInstr &MI, MachineRegisterInfo &MRI, MinMaxMedOpc MMMOpc,
                Register &Val, Register &K0, Register &K1);

  bool matchIntMinMaxToMed3(MachineInstr &MI, Med3MatchInfo &MatchInfo);
  void applyMed3(MachineInstr &MI, Med3MatchInfo &MatchInfo);
};

bool AMDGPURegBankCombinerHelper::isVgprRegBank(Register Reg) {
  return RBI.getRegBank(Reg, MRI, TRI)->getID() == AMDGPU::VGPRRegBankID;
}

AMDGPURegBankCombinerHelper::MinMaxMedOpc
AMDGPURegBankCombinerHelper::getMinMaxPair(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Unsupported opcode");
  case AMDGPU::G_SMAX:
  case AMDGPU::G_SMIN:
    return {AMDGPU::G_SMIN, AMDGPU::G_SMAX, AMDGPU::G_AMDGPU_SMED3};
  case AMDGPU::G_UMAX:
  case AMDGPU::G_UMIN:
    return {AMDGPU::G_UMIN, AMDGPU::G_UMAX, AMDGPU::G_AMDGPU_UMED3};
  }
}

template <class m_Cst>
bool AMDGPURegBankCombinerHelper::matchMed(MachineInstr &MI,
                                           MachineRegisterInfo &MRI,
                                           MinMaxMedOpc MMMOpc, Register &Val,
                                           Register &K0, Register &K1) {
  // 4 operand commutes of: min(max(Val, K0), K1).
  // Find K1 from outer instr: min(max(...), K1) or min(K1, max(...)).
  // Find K0 and Val from inner instr: max(K0, Val) or max(Val, K0).
  // 4 operand commutes of: max(min(Val, K1), K0).
  // Find K0 from outer instr: max(min(...), K0) or max(K0, min(...)).
  // Find K1 and Val from inner instr: min(K1, Val) or min(Val, K1).
  return mi_match(
      MI, MRI,
      m_any_of(
          m_CommutativeBinOp(
              MMMOpc.Min, m_CommutativeBinOp(MMMOpc.Max, m_Reg(Val), m_Cst(K0)),
              m_Cst(K1)),
          m_CommutativeBinOp(
              MMMOpc.Max, m_CommutativeBinOp(MMMOpc.Min, m_Reg(Val), m_Cst(K1)),
              m_Cst(K0))));
}

bool AMDGPURegBankCombinerHelper::matchIntMinMaxToMed3(
    MachineInstr &MI, Med3MatchInfo &MatchInfo) {
  Register Dst = MI.getOperand(0).getReg();
  if (!isVgprRegBank(Dst))
    return false;

  if (MRI.getType(Dst).isVector())
    return false;

  MinMaxMedOpc OpcodeTriple = getMinMaxPair(MI.getOpcode());
  Register Val, K0, K1;
  // Match min(max(Val, K0), K1) or max(min(Val, K1), K0). Then see if K0 <= K1.
  if (!matchMed<ICstRegMatch>(MI, MRI, OpcodeTriple, Val, K0, K1))
    return false;

  const APInt &K0_Imm = getConstantIntVRegVal(K0, MRI)->getValue();
  const APInt &K1_Imm = getConstantIntVRegVal(K1, MRI)->getValue();
  if (OpcodeTriple.Med == AMDGPU::G_AMDGPU_SMED3 && K0_Imm.sgt(K1_Imm))
    return false;
  if (OpcodeTriple.Med == AMDGPU::G_AMDGPU_UMED3 && K0_Imm.ugt(K1_Imm))
    return false;

  MatchInfo = {OpcodeTriple.Med, Val, K0, K1};
  return true;
}

void AMDGPURegBankCombinerHelper::applyMed3(MachineInstr &MI,
                                            Med3MatchInfo &MatchInfo) {
  B.setInstrAndDebugLoc(MI);
  B.buildInstr(MatchInfo.Opc, {MI.getOperand(0)},
               {MatchInfo.Val0, MatchInfo.Val1, MatchInfo.Val2}, MI.getFlags());
  MI.eraseFromParent();
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
