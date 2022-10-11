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

#include "AMDGPU.h"
#include "AMDGPUCombinerHelper.h"
#include "AMDGPULegalizerInfo.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

class AMDGPUPostLegalizerCombinerHelper {
protected:
  MachineIRBuilder &B;
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  AMDGPUCombinerHelper &Helper;

public:
  AMDGPUPostLegalizerCombinerHelper(MachineIRBuilder &B,
                                    AMDGPUCombinerHelper &Helper)
      : B(B), MF(B.getMF()), MRI(*B.getMRI()), Helper(Helper){};

  struct FMinFMaxLegacyInfo {
    Register LHS;
    Register RHS;
    Register True;
    Register False;
    CmpInst::Predicate Pred;
  };

  // TODO: Make sure fmin_legacy/fmax_legacy don't canonicalize
  bool matchFMinFMaxLegacy(MachineInstr &MI, FMinFMaxLegacyInfo &Info);
  void applySelectFCmpToFMinToFMaxLegacy(MachineInstr &MI,
                                         const FMinFMaxLegacyInfo &Info);

  bool matchUCharToFloat(MachineInstr &MI);
  void applyUCharToFloat(MachineInstr &MI);

  bool matchRcpSqrtToRsq(MachineInstr &MI,
                         std::function<void(MachineIRBuilder &)> &MatchInfo);

  // FIXME: Should be able to have 2 separate matchdatas rather than custom
  // struct boilerplate.
  struct CvtF32UByteMatchInfo {
    Register CvtVal;
    unsigned ShiftOffset;
  };

  bool matchCvtF32UByteN(MachineInstr &MI, CvtF32UByteMatchInfo &MatchInfo);
  void applyCvtF32UByteN(MachineInstr &MI,
                         const CvtF32UByteMatchInfo &MatchInfo);

  bool matchRemoveFcanonicalize(MachineInstr &MI, Register &Reg);
};

bool AMDGPUPostLegalizerCombinerHelper::matchFMinFMaxLegacy(
    MachineInstr &MI, FMinFMaxLegacyInfo &Info) {
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

void AMDGPUPostLegalizerCombinerHelper::applySelectFCmpToFMinToFMaxLegacy(
    MachineInstr &MI, const FMinFMaxLegacyInfo &Info) {
  B.setInstrAndDebugLoc(MI);
  auto buildNewInst = [&MI, this](unsigned Opc, Register X, Register Y) {
    B.buildInstr(Opc, {MI.getOperand(0)}, {X, Y}, MI.getFlags());
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

bool AMDGPUPostLegalizerCombinerHelper::matchUCharToFloat(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();

  // TODO: We could try to match extracting the higher bytes, which would be
  // easier if i8 vectors weren't promoted to i32 vectors, particularly after
  // types are legalized. v4i8 -> v4f32 is probably the only case to worry
  // about in practice.
  LLT Ty = MRI.getType(DstReg);
  if (Ty == LLT::scalar(32) || Ty == LLT::scalar(16)) {
    Register SrcReg = MI.getOperand(1).getReg();
    unsigned SrcSize = MRI.getType(SrcReg).getSizeInBits();
    assert(SrcSize == 16 || SrcSize == 32 || SrcSize == 64);
    const APInt Mask = APInt::getHighBitsSet(SrcSize, SrcSize - 8);
    return Helper.getKnownBits()->maskedValueIsZero(SrcReg, Mask);
  }

  return false;
}

void AMDGPUPostLegalizerCombinerHelper::applyUCharToFloat(MachineInstr &MI) {
  B.setInstrAndDebugLoc(MI);

  const LLT S32 = LLT::scalar(32);

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy != S32)
    SrcReg = B.buildAnyExtOrTrunc(S32, SrcReg).getReg(0);

  if (Ty == S32) {
    B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {DstReg},
                   {SrcReg}, MI.getFlags());
  } else {
    auto Cvt0 = B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {S32},
                             {SrcReg}, MI.getFlags());
    B.buildFPTrunc(DstReg, Cvt0, MI.getFlags());
  }

  MI.eraseFromParent();
}

bool AMDGPUPostLegalizerCombinerHelper::matchRcpSqrtToRsq(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {

  auto getRcpSrc = [=](const MachineInstr &MI) {
    MachineInstr *ResMI = nullptr;
    if (MI.getOpcode() == TargetOpcode::G_INTRINSIC &&
        MI.getIntrinsicID() == Intrinsic::amdgcn_rcp)
      ResMI = MRI.getVRegDef(MI.getOperand(2).getReg());

    return ResMI;
  };

  auto getSqrtSrc = [=](const MachineInstr &MI) {
    MachineInstr *SqrtSrcMI = nullptr;
    auto Match =
        mi_match(MI.getOperand(0).getReg(), MRI, m_GFSqrt(m_MInstr(SqrtSrcMI)));
    (void)Match;
    return SqrtSrcMI;
  };

  MachineInstr *RcpSrcMI = nullptr, *SqrtSrcMI = nullptr;
  // rcp(sqrt(x))
  if ((RcpSrcMI = getRcpSrc(MI)) && (SqrtSrcMI = getSqrtSrc(*RcpSrcMI))) {
    MatchInfo = [SqrtSrcMI, &MI](MachineIRBuilder &B) {
      B.buildIntrinsic(Intrinsic::amdgcn_rsq, {MI.getOperand(0)}, false)
          .addUse(SqrtSrcMI->getOperand(0).getReg())
          .setMIFlags(MI.getFlags());
    };
    return true;
  }

  // sqrt(rcp(x))
  if ((SqrtSrcMI = getSqrtSrc(MI)) && (RcpSrcMI = getRcpSrc(*SqrtSrcMI))) {
    MatchInfo = [RcpSrcMI, &MI](MachineIRBuilder &B) {
      B.buildIntrinsic(Intrinsic::amdgcn_rsq, {MI.getOperand(0)}, false)
          .addUse(RcpSrcMI->getOperand(0).getReg())
          .setMIFlags(MI.getFlags());
    };
    return true;
  }

  return false;
}

bool AMDGPUPostLegalizerCombinerHelper::matchCvtF32UByteN(
    MachineInstr &MI, CvtF32UByteMatchInfo &MatchInfo) {
  Register SrcReg = MI.getOperand(1).getReg();

  // Look through G_ZEXT.
  bool IsShr = mi_match(SrcReg, MRI, m_GZExt(m_Reg(SrcReg)));

  Register Src0;
  int64_t ShiftAmt;
  IsShr = mi_match(SrcReg, MRI, m_GLShr(m_Reg(Src0), m_ICst(ShiftAmt)));
  if (IsShr || mi_match(SrcReg, MRI, m_GShl(m_Reg(Src0), m_ICst(ShiftAmt)))) {
    const unsigned Offset = MI.getOpcode() - AMDGPU::G_AMDGPU_CVT_F32_UBYTE0;

    unsigned ShiftOffset = 8 * Offset;
    if (IsShr)
      ShiftOffset += ShiftAmt;
    else
      ShiftOffset -= ShiftAmt;

    MatchInfo.CvtVal = Src0;
    MatchInfo.ShiftOffset = ShiftOffset;
    return ShiftOffset < 32 && ShiftOffset >= 8 && (ShiftOffset % 8) == 0;
  }

  // TODO: Simplify demanded bits.
  return false;
}

void AMDGPUPostLegalizerCombinerHelper::applyCvtF32UByteN(
    MachineInstr &MI, const CvtF32UByteMatchInfo &MatchInfo) {
  B.setInstrAndDebugLoc(MI);
  unsigned NewOpc = AMDGPU::G_AMDGPU_CVT_F32_UBYTE0 + MatchInfo.ShiftOffset / 8;

  const LLT S32 = LLT::scalar(32);
  Register CvtSrc = MatchInfo.CvtVal;
  LLT SrcTy = MRI.getType(MatchInfo.CvtVal);
  if (SrcTy != S32) {
    assert(SrcTy.isScalar() && SrcTy.getSizeInBits() >= 8);
    CvtSrc = B.buildAnyExt(S32, CvtSrc).getReg(0);
  }

  assert(MI.getOpcode() != NewOpc);
  B.buildInstr(NewOpc, {MI.getOperand(0)}, {CvtSrc}, MI.getFlags());
  MI.eraseFromParent();
}

bool AMDGPUPostLegalizerCombinerHelper::matchRemoveFcanonicalize(
    MachineInstr &MI, Register &Reg) {
  const SITargetLowering *TLI = static_cast<const SITargetLowering *>(
      MF.getSubtarget().getTargetLowering());
  Reg = MI.getOperand(1).getReg();
  return TLI->isCanonicalized(Reg, MF);
}

class AMDGPUPostLegalizerCombinerHelperState {
protected:
  AMDGPUCombinerHelper &Helper;
  AMDGPUPostLegalizerCombinerHelper &PostLegalizerHelper;

public:
  AMDGPUPostLegalizerCombinerHelperState(
      AMDGPUCombinerHelper &Helper,
      AMDGPUPostLegalizerCombinerHelper &PostLegalizerHelper)
      : Helper(Helper), PostLegalizerHelper(PostLegalizerHelper) {}
};

#define AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUPOSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AMDGPUPostLegalizerCombinerInfo final : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AMDGPUGenPostLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

  AMDGPUPostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
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

bool AMDGPUPostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                              MachineInstr &MI,
                                              MachineIRBuilder &B) const {
  AMDGPUCombinerHelper Helper(Observer, B, /*IsPreLegalize*/ false, KB, MDT,
                              LInfo);
  AMDGPUPostLegalizerCombinerHelper PostLegalizerHelper(B, Helper);
  AMDGPUGenPostLegalizerCombinerHelper Generated(GeneratedRuleCfg, Helper,
                                                 PostLegalizerHelper);

  if (Generated.tryCombineAll(Observer, MI, B))
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
