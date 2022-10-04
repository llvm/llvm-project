//=== lib/CodeGen/GlobalISel/AMDGPUPreLegalizerCombiner.cpp ---------------===//
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

#include "AMDGPU.h"
#include "AMDGPUCombinerHelper.h"
#include "AMDGPULegalizerInfo.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-prelegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

class AMDGPUPreLegalizerCombinerHelper {
protected:
  MachineIRBuilder &B;
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  AMDGPUCombinerHelper &Helper;

public:
  AMDGPUPreLegalizerCombinerHelper(MachineIRBuilder &B,
                                   AMDGPUCombinerHelper &Helper)
      : B(B), MF(B.getMF()), MRI(*B.getMRI()), Helper(Helper){};

  struct ClampI64ToI16MatchInfo {
    int64_t Cmp1 = 0;
    int64_t Cmp2 = 0;
    Register Origin;
  };

  bool matchClampI64ToI16(MachineInstr &MI, MachineRegisterInfo &MRI,
                          MachineFunction &MF,
                          ClampI64ToI16MatchInfo &MatchInfo);

  void applyClampI64ToI16(MachineInstr &MI,
                          const ClampI64ToI16MatchInfo &MatchInfo);
};

bool AMDGPUPreLegalizerCombinerHelper::matchClampI64ToI16(
    MachineInstr &MI, MachineRegisterInfo &MRI, MachineFunction &MF,
    ClampI64ToI16MatchInfo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Invalid instruction!");

  // Try to find a pattern where an i64 value should get clamped to short.
  const LLT SrcType = MRI.getType(MI.getOperand(1).getReg());
  if (SrcType != LLT::scalar(64))
    return false;

  const LLT DstType = MRI.getType(MI.getOperand(0).getReg());
  if (DstType != LLT::scalar(16))
    return false;

  Register Base;

  auto IsApplicableForCombine = [&MatchInfo]() -> bool {
    const auto Cmp1 = MatchInfo.Cmp1;
    const auto Cmp2 = MatchInfo.Cmp2;
    const auto Diff = std::abs(Cmp2 - Cmp1);

    // If the difference between both comparison values is 0 or 1, there is no
    // need to clamp.
    if (Diff == 0 || Diff == 1)
      return false;

    const int64_t Min = std::numeric_limits<int16_t>::min();
    const int64_t Max = std::numeric_limits<int16_t>::max();

    // Check if the comparison values are between SHORT_MIN and SHORT_MAX.
    return ((Cmp2 >= Cmp1 && Cmp1 >= Min && Cmp2 <= Max) ||
            (Cmp1 >= Cmp2 && Cmp1 <= Max && Cmp2 >= Min));
  };

  // Try to match a combination of min / max MIR opcodes.
  if (mi_match(MI.getOperand(1).getReg(), MRI,
               m_GSMin(m_Reg(Base), m_ICst(MatchInfo.Cmp1)))) {
    if (mi_match(Base, MRI,
                 m_GSMax(m_Reg(MatchInfo.Origin), m_ICst(MatchInfo.Cmp2)))) {
      return IsApplicableForCombine();
    }
  }

  if (mi_match(MI.getOperand(1).getReg(), MRI,
               m_GSMax(m_Reg(Base), m_ICst(MatchInfo.Cmp1)))) {
    if (mi_match(Base, MRI,
                 m_GSMin(m_Reg(MatchInfo.Origin), m_ICst(MatchInfo.Cmp2)))) {
      return IsApplicableForCombine();
    }
  }

  return false;
}

// We want to find a combination of instructions that
// gets generated when an i64 gets clamped to i16.
// The corresponding pattern is:
// G_MAX / G_MAX for i16 <= G_TRUNC i64.
// This can be efficiently written as following:
// v_cvt_pk_i16_i32 v0, v0, v1
// v_med3_i32 v0, Clamp_Min, v0, Clamp_Max
void AMDGPUPreLegalizerCombinerHelper::applyClampI64ToI16(
    MachineInstr &MI, const ClampI64ToI16MatchInfo &MatchInfo) {

  Register Src = MatchInfo.Origin;
  assert(MI.getParent()->getParent()->getRegInfo().getType(Src) ==
         LLT::scalar(64));
  const LLT S32 = LLT::scalar(32);

  B.setInstrAndDebugLoc(MI);

  auto Unmerge = B.buildUnmerge(S32, Src);

  assert(MI.getOpcode() != AMDGPU::G_AMDGPU_CVT_PK_I16_I32);

  const LLT V2S16 = LLT::fixed_vector(2, 16);
  auto CvtPk =
      B.buildInstr(AMDGPU::G_AMDGPU_CVT_PK_I16_I32, {V2S16},
                   {Unmerge.getReg(0), Unmerge.getReg(1)}, MI.getFlags());

  auto MinBoundary = std::min(MatchInfo.Cmp1, MatchInfo.Cmp2);
  auto MaxBoundary = std::max(MatchInfo.Cmp1, MatchInfo.Cmp2);
  auto MinBoundaryDst = B.buildConstant(S32, MinBoundary);
  auto MaxBoundaryDst = B.buildConstant(S32, MaxBoundary);

  auto Bitcast = B.buildBitcast({S32}, CvtPk);

  auto Med3 = B.buildInstr(
      AMDGPU::G_AMDGPU_SMED3, {S32},
      {MinBoundaryDst.getReg(0), Bitcast.getReg(0), MaxBoundaryDst.getReg(0)},
      MI.getFlags());

  B.buildTrunc(MI.getOperand(0).getReg(), Med3);

  MI.eraseFromParent();
}

class AMDGPUPreLegalizerCombinerHelperState {
protected:
  AMDGPUCombinerHelper &Helper;
  AMDGPUPreLegalizerCombinerHelper &PreLegalizerHelper;

public:
  AMDGPUPreLegalizerCombinerHelperState(
      AMDGPUCombinerHelper &Helper,
      AMDGPUPreLegalizerCombinerHelper &PreLegalizerHelper)
      : Helper(Helper), PreLegalizerHelper(PreLegalizerHelper) {}
};

#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AMDGPUGenPreLegalizeGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AMDGPUGenPreLegalizeGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AMDGPUPreLegalizerCombinerInfo final : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AMDGPUGenPreLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

  AMDGPUPreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
               MachineIRBuilder &B) const override;
};

bool AMDGPUPreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  AMDGPUCombinerHelper Helper(Observer, B, /*IsPreLegalize*/ true, KB, MDT);
  AMDGPUPreLegalizerCombinerHelper PreLegalizerHelper(B, Helper);
  AMDGPUGenPreLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper,
                                                PreLegalizerHelper);

  if (Generated.tryCombineAll(Observer, MI, B))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  }

  return false;
}

#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AMDGPUGenPreLegalizeGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AMDGPUPreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "AMDGPUPreLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  bool IsOptNone;
};
} // end anonymous namespace

void AMDGPUPreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

AMDGPUPreLegalizerCombiner::AMDGPUPreLegalizerCombiner(bool IsOptNone)
  : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAMDGPUPreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AMDGPUPreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
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
  AMDGPUPreLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                        F.hasMinSize(), KB, MDT);
  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  auto *CSEInfo = &Wrapper.get(TPC->getCSEConfig());

  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, CSEInfo);
}

char AMDGPUPreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AMDGPUPreLegalizerCombiner, DEBUG_TYPE,
                      "Combine AMDGPU machine instrs before legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AMDGPUPreLegalizerCombiner, DEBUG_TYPE,
                    "Combine AMDGPU machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createAMDGPUPreLegalizeCombiner(bool IsOptNone) {
  return new AMDGPUPreLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
