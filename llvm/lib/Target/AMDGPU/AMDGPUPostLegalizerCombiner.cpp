//=== lib/CodeGen/GlobalISel/AMDGPUPostLegalizerCombiner.cpp --------------===//
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
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "AMDGPUGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "amdgpu-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

namespace {
#define GET_GICOMBINER_TYPES
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class AMDGPUPostLegalizerCombinerImpl : public Combiner {
protected:
  const AMDGPUPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const GCNSubtarget &STI;
  const SIInstrInfo &TII;
  // TODO: Make CombinerHelper methods const.
  mutable AMDGPUCombinerHelper Helper;

public:
  AMDGPUPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
      const AMDGPUPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const GCNSubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "AMDGPUPostLegalizerCombinerImpl"; }

  bool tryCombineAllImpl(MachineInstr &I) const;
  bool tryCombineAll(MachineInstr &I) const override;

  struct FMinFMaxLegacyInfo {
    Register LHS;
    Register RHS;
    Register True;
    Register False;
    CmpInst::Predicate Pred;
  };

  // TODO: Make sure fmin_legacy/fmax_legacy don't canonicalize
  bool matchFMinFMaxLegacy(MachineInstr &MI, FMinFMaxLegacyInfo &Info) const;
  void applySelectFCmpToFMinToFMaxLegacy(MachineInstr &MI,
                                         const FMinFMaxLegacyInfo &Info) const;

  bool matchUCharToFloat(MachineInstr &MI) const;
  void applyUCharToFloat(MachineInstr &MI) const;

  bool
  matchRcpSqrtToRsq(MachineInstr &MI,
                    std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  // FIXME: Should be able to have 2 separate matchdatas rather than custom
  // struct boilerplate.
  struct CvtF32UByteMatchInfo {
    Register CvtVal;
    unsigned ShiftOffset;
  };

  bool matchCvtF32UByteN(MachineInstr &MI,
                         CvtF32UByteMatchInfo &MatchInfo) const;
  void applyCvtF32UByteN(MachineInstr &MI,
                         const CvtF32UByteMatchInfo &MatchInfo) const;

  bool matchRemoveFcanonicalize(MachineInstr &MI, Register &Reg) const;

  // Combine unsigned buffer load and signed extension instructions to generate
  // signed buffer laod instructions.
  bool matchCombineSignExtendInReg(MachineInstr &MI,
                                   MachineInstr *&MatchInfo) const;
  void applyCombineSignExtendInReg(MachineInstr &MI,
                                   MachineInstr *&MatchInfo) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#define AMDGPUSubtarget GCNSubtarget
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
#undef AMDGPUSubtarget
};

#define GET_GICOMBINER_IMPL
#define AMDGPUSubtarget GCNSubtarget
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef AMDGPUSubtarget
#undef GET_GICOMBINER_IMPL

AMDGPUPostLegalizerCombinerImpl::AMDGPUPostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelKnownBits &KB, GISelCSEInfo *CSEInfo,
    const AMDGPUPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const GCNSubtarget &STI, MachineDominatorTree *MDT, const LegalizerInfo *LI)
    : Combiner(MF, CInfo, TPC, &KB, CSEInfo), RuleConfig(RuleConfig), STI(STI),
      TII(*STI.getInstrInfo()),
      Helper(Observer, B, /*IsPreLegalize*/ false, &KB, MDT, LI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "AMDGPUGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool AMDGPUPostLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  if (tryCombineAllImpl(MI))
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

bool AMDGPUPostLegalizerCombinerImpl::matchFMinFMaxLegacy(
    MachineInstr &MI, FMinFMaxLegacyInfo &Info) const {
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

  // TODO: Handle case where the the selected value is an fneg and the compared
  // constant is the negation of the selected value.
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

void AMDGPUPostLegalizerCombinerImpl::applySelectFCmpToFMinToFMaxLegacy(
    MachineInstr &MI, const FMinFMaxLegacyInfo &Info) const {
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

bool AMDGPUPostLegalizerCombinerImpl::matchUCharToFloat(
    MachineInstr &MI) const {
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

void AMDGPUPostLegalizerCombinerImpl::applyUCharToFloat(
    MachineInstr &MI) const {
  B.setInstrAndDebugLoc(MI);

  const LLT S32 = LLT::scalar(32);

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy != S32)
    SrcReg = B.buildAnyExtOrTrunc(S32, SrcReg).getReg(0);

  if (Ty == S32) {
    B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {DstReg}, {SrcReg},
                 MI.getFlags());
  } else {
    auto Cvt0 = B.buildInstr(AMDGPU::G_AMDGPU_CVT_F32_UBYTE0, {S32}, {SrcReg},
                             MI.getFlags());
    B.buildFPTrunc(DstReg, Cvt0, MI.getFlags());
  }

  MI.eraseFromParent();
}

bool AMDGPUPostLegalizerCombinerImpl::matchRcpSqrtToRsq(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  auto getRcpSrc = [=](const MachineInstr &MI) -> MachineInstr * {
    if (!MI.getFlag(MachineInstr::FmContract))
      return nullptr;

    if (auto *GI = dyn_cast<GIntrinsic>(&MI)) {
      if (GI->is(Intrinsic::amdgcn_rcp))
        return MRI.getVRegDef(MI.getOperand(2).getReg());
    }
    return nullptr;
  };

  auto getSqrtSrc = [=](const MachineInstr &MI) -> MachineInstr * {
    if (!MI.getFlag(MachineInstr::FmContract))
      return nullptr;
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
      B.buildIntrinsic(Intrinsic::amdgcn_rsq, {MI.getOperand(0)})
          .addUse(SqrtSrcMI->getOperand(0).getReg())
          .setMIFlags(MI.getFlags());
    };
    return true;
  }

  // sqrt(rcp(x))
  if ((SqrtSrcMI = getSqrtSrc(MI)) && (RcpSrcMI = getRcpSrc(*SqrtSrcMI))) {
    MatchInfo = [RcpSrcMI, &MI](MachineIRBuilder &B) {
      B.buildIntrinsic(Intrinsic::amdgcn_rsq, {MI.getOperand(0)})
          .addUse(RcpSrcMI->getOperand(0).getReg())
          .setMIFlags(MI.getFlags());
    };
    return true;
  }
  return false;
}

bool AMDGPUPostLegalizerCombinerImpl::matchCvtF32UByteN(
    MachineInstr &MI, CvtF32UByteMatchInfo &MatchInfo) const {
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

void AMDGPUPostLegalizerCombinerImpl::applyCvtF32UByteN(
    MachineInstr &MI, const CvtF32UByteMatchInfo &MatchInfo) const {
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

bool AMDGPUPostLegalizerCombinerImpl::matchRemoveFcanonicalize(
    MachineInstr &MI, Register &Reg) const {
  const SITargetLowering *TLI = static_cast<const SITargetLowering *>(
      MF.getSubtarget().getTargetLowering());
  Reg = MI.getOperand(1).getReg();
  return TLI->isCanonicalized(Reg, MF);
}

// The buffer_load_{i8, i16} intrinsics are intially lowered as buffer_load_{u8,
// u16} instructions. Here, the buffer_load_{u8, u16} instructions are combined
// with sign extension instrucions in order to generate buffer_load_{i8, i16}
// instructions.

// Identify buffer_load_{u8, u16}.
bool AMDGPUPostLegalizerCombinerImpl::matchCombineSignExtendInReg(
    MachineInstr &MI, MachineInstr *&SubwordBufferLoad) const {
  Register Op0Reg = MI.getOperand(1).getReg();
  SubwordBufferLoad = MRI.getVRegDef(Op0Reg);

  if (!MRI.hasOneNonDBGUse(Op0Reg))
    return false;

  // Check if the first operand of the sign extension is a subword buffer load
  // instruction.
  return SubwordBufferLoad->getOpcode() == AMDGPU::G_AMDGPU_BUFFER_LOAD_UBYTE ||
         SubwordBufferLoad->getOpcode() == AMDGPU::G_AMDGPU_BUFFER_LOAD_USHORT;
}

// Combine buffer_load_{u8, u16} and the sign extension instruction to generate
// buffer_load_{i8, i16}.
void AMDGPUPostLegalizerCombinerImpl::applyCombineSignExtendInReg(
    MachineInstr &MI, MachineInstr *&SubwordBufferLoad) const {
  // Modify the opcode and the destination of buffer_load_{u8, u16}:
  // Replace the opcode.
  unsigned Opc =
      SubwordBufferLoad->getOpcode() == AMDGPU::G_AMDGPU_BUFFER_LOAD_UBYTE
          ? AMDGPU::G_AMDGPU_BUFFER_LOAD_SBYTE
          : AMDGPU::G_AMDGPU_BUFFER_LOAD_SSHORT;
  SubwordBufferLoad->setDesc(TII.get(Opc));
  // Update the destination register of SubwordBufferLoad with the destination
  // register of the sign extension.
  Register SignExtendInsnDst = MI.getOperand(0).getReg();
  SubwordBufferLoad->getOperand(0).setReg(SignExtendInsnDst);
  // Remove the sign extension.
  MI.eraseFromParent();
}

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
  AMDGPUPostLegalizerCombinerImplRuleConfig RuleConfig;
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

  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool AMDGPUPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const AMDGPULegalizerInfo *LI =
      static_cast<const AMDGPULegalizerInfo *>(ST.getLegalizerInfo());

  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();

  CombinerInfo CInfo(/*AllowIllegalOps*/ false, /*ShouldLegalizeIllegal*/ true,
                     LI, EnableOpt, F.hasOptSize(), F.hasMinSize());

  AMDGPUPostLegalizerCombinerImpl Impl(MF, CInfo, TPC, *KB, /*CSEInfo*/ nullptr,
                                       RuleConfig, ST, MDT, LI);
  return Impl.combineMachineInstrs();
}

char AMDGPUPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AMDGPUPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine AMDGPU machine instrs after legalization", false,
                      false)
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
