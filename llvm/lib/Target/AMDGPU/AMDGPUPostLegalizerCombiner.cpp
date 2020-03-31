//=== lib/CodeGen/GlobalISel/AMDGPUPostLegalizerCombiner.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass does combining of machine instructions at the generic MI level,
// after the legalizer.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPULegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#define DEBUG_TYPE "amdgpu-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

struct FMinFMaxLegacyInfo {
  Register LHS;
  Register RHS;
  Register True;
  Register False;
  CmpInst::Predicate Pred;
};

// TODO: Make sure fmin_legacy/fmax_legacy don't canonicalize
static bool matchFMinFMaxLegacy(MachineInstr &MI, MachineRegisterInfo &MRI,
                                MachineFunction &MF, FMinFMaxLegacyInfo &Info) {
  // FIXME: Combines should have subtarget predicates, and we shouldn't need
  // this here.
  if (!MF.getSubtarget<GCNSubtarget>().hasFminFmaxLegacy())
    return false;

  // FIXME: Type predicate on pattern
  if (MRI.getType(MI.getOperand(0).getReg()) != LLT::scalar(32))
    return false;

  Register Cond = MI.getOperand(1).getReg();
  if (!MRI.hasOneNonDBGUse(Cond) ||
      !mi_match(Cond, MRI,
                m_GFCmp(m_Pred(Info.Pred), m_Reg(Info.LHS), m_Reg(Info.RHS))))
    return false;

  Info.True = MI.getOperand(2).getReg();
  Info.False = MI.getOperand(3).getReg();

  if (!(Info.LHS == Info.True && Info.RHS == Info.False) &&
      !(Info.LHS == Info.False && Info.RHS == Info.True))
    return false;

  switch (Info.Pred) {
  case CmpInst::FCMP_FALSE:
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_ONE:
  case CmpInst::FCMP_ORD:
  case CmpInst::FCMP_UNO:
  case CmpInst::FCMP_UEQ:
  case CmpInst::FCMP_UNE:
  case CmpInst::FCMP_TRUE:
    return false;
  default:
    return true;
  }
}

static void applySelectFCmpToFMinToFMaxLegacy(MachineInstr &MI,
                                              const FMinFMaxLegacyInfo &Info) {

  auto buildNewInst = [&MI](unsigned Opc, Register X, Register Y) {
    MachineIRBuilder MIB(MI);
    MIB.buildInstr(Opc, {MI.getOperand(0)}, {X, Y}, MI.getFlags());
  };

  switch (Info.Pred) {
  case CmpInst::FCMP_ULT:
  case CmpInst::FCMP_ULE:
    if (Info.LHS == Info.True)
      buildNewInst(AMDGPU::G_AMDGPU_FMIN_LEGACY, Info.RHS, Info.LHS);
    else
      buildNewInst(AMDGPU::G_AMDGPU_FMAX_LEGACY, Info.LHS, Info.RHS);
    break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_OLT: {
    // We need to permute the operands to get the correct NaN behavior. The
    // selected operand is the second one based on the failing compare with NaN,
    // so permute it based on the compare type the hardware uses.
    if (Info.LHS == Info.True)
      buildNewInst(AMDGPU::G_AMDGPU_FMIN_LEGACY, Info.LHS, Info.RHS);
    else
      buildNewInst(AMDGPU::G_AMDGPU_FMAX_LEGACY, Info.RHS, Info.LHS);
    break;
  }
  case CmpInst::FCMP_UGE:
  case CmpInst::FCMP_UGT: {
    if (Info.LHS == Info.True)
      buildNewInst(AMDGPU::G_AMDGPU_FMAX_LEGACY, Info.RHS, Info.LHS);
    else
      buildNewInst(AMDGPU::G_AMDGPU_FMIN_LEGACY, Info.LHS, Info.RHS);
    break;
  }
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE: {
    if (Info.LHS == Info.True)
      buildNewInst(AMDGPU::G_AMDGPU_FMAX_LEGACY, Info.LHS, Info.RHS);
    else
      buildNewInst(AMDGPU::G_AMDGPU_FMIN_LEGACY, Info.RHS, Info.LHS);
    break;
  }
  default:
    llvm_unreachable("predicate should not have matched");
  }

  MI.eraseFromParent();
}

static bool matchUCharToFloat(MachineInstr &MI, MachineRegisterInfo &MRI,
                              MachineFunction &MF, CombinerHelper &Helper) {
  Register DstReg = MI.getOperand(0).getReg();

  // TODO: We could try to match extracting the higher bytes, which would be
  // easier if i8 vectors weren't promoted to i32 vectors, particularly after
  // types are legalized. v4i8 -> v4f32 is probably the only case to worry
  // about in practice.
  LLT Ty = MRI.getType(DstReg);
  if (Ty == LLT::scalar(32) || Ty == LLT::scalar(16)) {
    const APInt Mask = APInt::getHighBitsSet(32, 24);
    return Helper.getKnownBits()->maskedValueIsZero(MI.getOperand(1).getReg(),
                                                    Mask);
  }

  return false;
}

static void applyUCharToFloat(MachineInstr &MI) {
  MachineIRBuilder B(MI);

  const LLT S32 = LLT::scalar(32);

  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = B.getMRI()->getType(DstReg);

  if (Ty == S32) {
    B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {DstReg},
                   {MI.getOperand(1)}, MI.getFlags());
  } else {
    auto Cvt0 = B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {S32},
                             {MI.getOperand(1)}, MI.getFlags());
    B.buildFPTrunc(DstReg, Cvt0, MI.getFlags());
  }

  MI.eraseFromParent();
}

#define AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AMDGPUPostLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AMDGPUGenPostLegalizerCombinerHelper Generated;

  AMDGPUPostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                  const AMDGPULegalizerInfo *LI,
                                  GISelKnownBits *KB, MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     /*LegalizerInfo*/ LI, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!Generated.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
               MachineIRBuilder &B) const override;
};

bool AMDGPUPostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);

  if (Generated.tryCombineAll(Observer, MI, B, Helper))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR:
    // On some subtargets, 64-bit shift is a quarter rate instruction. In the
    // common case, splitting this into a move and a 32-bit shift is faster and
    // the same code size.
    return Helper.tryCombineShiftToUnmerge(MI, 32);
  }

  return false;
}

#define AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AMDGPUPostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "AMDGPUPostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  bool IsOptNone;
};
} // end anonymous namespace

void AMDGPUPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
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

AMDGPUPostLegalizerCombiner::AMDGPUPostLegalizerCombiner(bool IsOptNone)
  : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAMDGPUPostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AMDGPUPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
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
  AMDGPUPostLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                         F.hasMinSize(), LI, KB, MDT);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AMDGPUPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AMDGPUPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine AMDGPU machine instrs after legalization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AMDGPUPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine AMDGPU machine instrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createAMDGPUPostLegalizeCombiner(bool IsOptNone) {
  return new AMDGPUPostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
