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

#include "AMDGPUTargetMachine.h"
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

#define DEBUG_TYPE "amdgpu-prelegalizer-combiner"

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


#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AMDGPUGenGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AMDGPUGenGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AMDGPUPreLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AMDGPUGenPreLegalizerCombinerHelper Generated;

  AMDGPUPreLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
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

bool AMDGPUPreLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B, KB, MDT);

  if (Generated.tryCombineAll(Observer, MI, B, Helper))
    return true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
    // On some subtargets, 64-bit shift is a quarter rate instruction. In the
    // common case, splitting this into a move and a 32-bit shift is faster and
    // the same code size.
    return Helper.tryCombineShiftToUnmerge(MI, 32);
  case TargetOpcode::G_CONCAT_VECTORS:
    return Helper.tryCombineConcatVectors(MI);
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return Helper.tryCombineShuffleVector(MI);
  }

  return false;
}

#define AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AMDGPUGenGICombiner.inc"
#undef AMDGPUPRELEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

// Pass boilerplate
// ================

class AMDGPUPreLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override { return "AMDGPUPreLegalizerCombiner"; }

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
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
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
