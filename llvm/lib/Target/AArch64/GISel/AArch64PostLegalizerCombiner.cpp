//=== AArch64PostLegalizerCombiner.cpp --------------------------*- C++ -*-===//
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
/// AArch64PostLegalizerLowering.
///
/// Combines which don't rely on instruction legality should go in the
/// AArch64PreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-postlegalizer-combiner"

using namespace llvm;

/// This combine tries do what performExtractVectorEltCombine does in SDAG.
/// Rewrite for pairwise fadd pattern
///   (s32 (g_extract_vector_elt
///           (g_fadd (vXs32 Other)
///                  (g_vector_shuffle (vXs32 Other) undef <1,X,...> )) 0))
/// ->
///   (s32 (g_fadd (g_extract_vector_elt (vXs32 Other) 0)
///              (g_extract_vector_elt (vXs32 Other) 1))
bool matchExtractVecEltPairwiseAdd(
    MachineInstr &MI, MachineRegisterInfo &MRI,
    std::tuple<unsigned, LLT, Register> &MatchInfo) {
  Register Src1 = MI.getOperand(1).getReg();
  Register Src2 = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  auto Cst = getConstantVRegValWithLookThrough(Src2, MRI);
  if (!Cst || Cst->Value != 0)
    return false;
  // SDAG also checks for FullFP16, but this looks to be beneficial anyway.

  // Now check for an fadd operation. TODO: expand this for integer add?
  auto *FAddMI = getOpcodeDef(TargetOpcode::G_FADD, Src1, MRI);
  if (!FAddMI)
    return false;

  // If we add support for integer add, must restrict these types to just s64.
  unsigned DstSize = DstTy.getSizeInBits();
  if (DstSize != 16 && DstSize != 32 && DstSize != 64)
    return false;

  Register Src1Op1 = FAddMI->getOperand(1).getReg();
  Register Src1Op2 = FAddMI->getOperand(2).getReg();
  MachineInstr *Shuffle =
      getOpcodeDef(TargetOpcode::G_SHUFFLE_VECTOR, Src1Op2, MRI);
  MachineInstr *Other = MRI.getVRegDef(Src1Op1);
  if (!Shuffle) {
    Shuffle = getOpcodeDef(TargetOpcode::G_SHUFFLE_VECTOR, Src1Op1, MRI);
    Other = MRI.getVRegDef(Src1Op2);
  }

  // We're looking for a shuffle that moves the second element to index 0.
  if (Shuffle && Shuffle->getOperand(3).getShuffleMask()[0] == 1 &&
      Other == MRI.getVRegDef(Shuffle->getOperand(1).getReg())) {
    std::get<0>(MatchInfo) = TargetOpcode::G_FADD;
    std::get<1>(MatchInfo) = DstTy;
    std::get<2>(MatchInfo) = Other->getOperand(0).getReg();
    return true;
  }
  return false;
}

bool applyExtractVecEltPairwiseAdd(
    MachineInstr &MI, MachineRegisterInfo &MRI, MachineIRBuilder &B,
    std::tuple<unsigned, LLT, Register> &MatchInfo) {
  unsigned Opc = std::get<0>(MatchInfo);
  assert(Opc == TargetOpcode::G_FADD && "Unexpected opcode!");
  // We want to generate two extracts of elements 0 and 1, and add them.
  LLT Ty = std::get<1>(MatchInfo);
  Register Src = std::get<2>(MatchInfo);
  LLT s64 = LLT::scalar(64);
  B.setInstrAndDebugLoc(MI);
  auto Elt0 = B.buildExtractVectorElement(Ty, Src, B.buildConstant(s64, 0));
  auto Elt1 = B.buildExtractVectorElement(Ty, Src, B.buildConstant(s64, 1));
  B.buildInstr(Opc, {MI.getOperand(0).getReg()}, {Elt0, Elt1});
  MI.eraseFromParent();
  return true;
}

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
  AArch64GenPostLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

  AArch64PostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
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

bool AArch64PostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                               MachineInstr &MI,
                                               MachineIRBuilder &B) const {
  const auto *LI =
      MI.getParent()->getParent()->getSubtarget().getLegalizerInfo();
  CombinerHelper Helper(Observer, B, KB, MDT, LI);
  AArch64GenPostLegalizerCombinerHelper Generated(GeneratedRuleCfg);
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
FunctionPass *createAArch64PostLegalizerCombiner(bool IsOptNone) {
  return new AArch64PostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
