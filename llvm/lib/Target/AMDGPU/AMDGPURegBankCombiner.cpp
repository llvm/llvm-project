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
#include "SIMachineFunctionInfo.h"
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
  const SIInstrInfo &TII;
  CombinerHelper &Helper;

public:
  AMDGPURegBankCombinerHelper(MachineIRBuilder &B, CombinerHelper &Helper)
      : B(B), MF(B.getMF()), MRI(*B.getMRI()),
        RBI(*MF.getSubtarget().getRegBankInfo()),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget<GCNSubtarget>().getInstrInfo()), Helper(Helper){};

  bool isVgprRegBank(Register Reg);
  Register getAsVgpr(Register Reg);

  struct MinMaxMedOpc {
    unsigned Min, Max, Med;
  };

  struct Med3MatchInfo {
    unsigned Opc;
    Register Val0, Val1, Val2;
  };

  MinMaxMedOpc getMinMaxPair(unsigned Opc);

  template <class m_Cst, typename CstTy>
  bool matchMed(MachineInstr &MI, MachineRegisterInfo &MRI, MinMaxMedOpc MMMOpc,
                Register &Val, CstTy &K0, CstTy &K1);

  bool matchIntMinMaxToMed3(MachineInstr &MI, Med3MatchInfo &MatchInfo);
  bool matchFPMinMaxToMed3(MachineInstr &MI, Med3MatchInfo &MatchInfo);
  bool matchFPMinMaxToClamp(MachineInstr &MI, Register &Reg);
  bool matchFPMed3ToClamp(MachineInstr &MI, Register &Reg);
  void applyMed3(MachineInstr &MI, Med3MatchInfo &MatchInfo);
  void applyClamp(MachineInstr &MI, Register &Reg);

private:
  AMDGPU::SIModeRegisterDefaults getMode();
  bool getIEEE();
  bool getDX10Clamp();
  bool isFminnumIeee(const MachineInstr &MI);
  bool isFCst(MachineInstr *MI);
  bool isClampZeroToOne(MachineInstr *K0, MachineInstr *K1);
};

bool AMDGPURegBankCombinerHelper::isVgprRegBank(Register Reg) {
  return RBI.getRegBank(Reg, MRI, TRI)->getID() == AMDGPU::VGPRRegBankID;
}

Register AMDGPURegBankCombinerHelper::getAsVgpr(Register Reg) {
  if (isVgprRegBank(Reg))
    return Reg;

  // Search for existing copy of Reg to vgpr.
  for (MachineInstr &Use : MRI.use_instructions(Reg)) {
    Register Def = Use.getOperand(0).getReg();
    if (Use.getOpcode() == AMDGPU::COPY && isVgprRegBank(Def))
      return Def;
  }

  // Copy Reg to vgpr.
  Register VgprReg = B.buildCopy(MRI.getType(Reg), Reg).getReg(0);
  MRI.setRegBank(VgprReg, RBI.getRegBank(AMDGPU::VGPRRegBankID));
  return VgprReg;
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
  case AMDGPU::G_FMAXNUM:
  case AMDGPU::G_FMINNUM:
    return {AMDGPU::G_FMINNUM, AMDGPU::G_FMAXNUM, AMDGPU::G_AMDGPU_FMED3};
  case AMDGPU::G_FMAXNUM_IEEE:
  case AMDGPU::G_FMINNUM_IEEE:
    return {AMDGPU::G_FMINNUM_IEEE, AMDGPU::G_FMAXNUM_IEEE,
            AMDGPU::G_AMDGPU_FMED3};
  }
}

template <class m_Cst, typename CstTy>
bool AMDGPURegBankCombinerHelper::matchMed(MachineInstr &MI,
                                           MachineRegisterInfo &MRI,
                                           MinMaxMedOpc MMMOpc, Register &Val,
                                           CstTy &K0, CstTy &K1) {
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
  Register Val;
  Optional<ValueAndVReg> K0, K1;
  // Match min(max(Val, K0), K1) or max(min(Val, K1), K0). Then see if K0 <= K1.
  if (!matchMed<GCstAndRegMatch>(MI, MRI, OpcodeTriple, Val, K0, K1))
    return false;

  if (OpcodeTriple.Med == AMDGPU::G_AMDGPU_SMED3 && K0->Value.sgt(K1->Value))
    return false;
  if (OpcodeTriple.Med == AMDGPU::G_AMDGPU_UMED3 && K0->Value.ugt(K1->Value))
    return false;

  MatchInfo = {OpcodeTriple.Med, Val, K0->VReg, K1->VReg};
  return true;
}

// fmed3(NaN, K0, K1) = min(min(NaN, K0), K1)
// ieee = true  : min/max(SNaN, K) = QNaN, min/max(QNaN, K) = K
// ieee = false : min/max(NaN, K) = K
// clamp(NaN) = dx10_clamp ? 0.0 : NaN
// Consider values of min(max(Val, K0), K1) and max(min(Val, K1), K0) as input.
// Other operand commutes (see matchMed) give same result since min and max are
// commutative.

// Try to replace fp min(max(Val, K0), K1) or max(min(Val, K1), K0), KO<=K1
// with fmed3(Val, K0, K1) or clamp(Val). Clamp requires K0 = 0.0 and K1 = 1.0.
// Val = SNaN only for ieee = true
// fmed3(SNaN, K0, K1) = min(min(SNaN, K0), K1) = min(QNaN, K1) = K1
// min(max(SNaN, K0), K1) = min(QNaN, K1) = K1
// max(min(SNaN, K1), K0) = max(K1, K0) = K1
// Val = NaN,ieee = false or Val = QNaN,ieee = true
// fmed3(NaN, K0, K1) = min(min(NaN, K0), K1) = min(K0, K1) = K0
// min(max(NaN, K0), K1) = min(K0, K1) = K0 (can clamp when dx10_clamp = true)
// max(min(NaN, K1), K0) = max(K1, K0) = K1 != K0
bool AMDGPURegBankCombinerHelper::matchFPMinMaxToMed3(
    MachineInstr &MI, Med3MatchInfo &MatchInfo) {
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  if (Ty != LLT::scalar(16) && Ty != LLT::scalar(32))
    return false;

  auto OpcodeTriple = getMinMaxPair(MI.getOpcode());

  Register Val;
  Optional<FPValueAndVReg> K0, K1;
  // Match min(max(Val, K0), K1) or max(min(Val, K1), K0). Then see if K0 <= K1.
  if (!matchMed<GFCstAndRegMatch>(MI, MRI, OpcodeTriple, Val, K0, K1))
    return false;

  if (K0->Value > K1->Value)
    return false;

  // For IEEE=false perform combine only when it's safe to assume that there are
  // no NaN inputs. Most often MI is marked with nnan fast math flag.
  // For IEEE=true consider NaN inputs. fmed3(NaN, K0, K1) is equivalent to
  // min(min(NaN, K0), K1). Safe to fold for min(max(Val, K0), K1) since inner
  // nodes(max/min) have same behavior when one input is NaN and other isn't.
  // Don't consider max(min(SNaN, K1), K0) since there is no isKnownNeverQNaN,
  // also post-legalizer inputs to min/max are fcanonicalized (never SNaN).
  if ((getIEEE() && isFminnumIeee(MI)) || isKnownNeverNaN(Dst, MRI)) {
    // Don't fold single use constant that can't be inlined.
    if ((!MRI.hasOneNonDBGUse(K0->VReg) || TII.isInlineConstant(K0->Value)) &&
        (!MRI.hasOneNonDBGUse(K1->VReg) || TII.isInlineConstant(K1->Value))) {
      MatchInfo = {OpcodeTriple.Med, Val, K0->VReg, K1->VReg};
      return true;
    }
  }

  return false;
}

bool AMDGPURegBankCombinerHelper::matchFPMinMaxToClamp(MachineInstr &MI,
                                                       Register &Reg) {
  // Clamp is available on all types after regbankselect (f16, f32, f64, v2f16).
  auto OpcodeTriple = getMinMaxPair(MI.getOpcode());
  Register Val;
  Optional<FPValueAndVReg> K0, K1;
  // Match min(max(Val, K0), K1) or max(min(Val, K1), K0).
  if (!matchMed<GFCstOrSplatGFCstMatch>(MI, MRI, OpcodeTriple, Val, K0, K1))
    return false;

  if (!K0->Value.isExactlyValue(0.0) || !K1->Value.isExactlyValue(1.0))
    return false;

  // For IEEE=false perform combine only when it's safe to assume that there are
  // no NaN inputs. Most often MI is marked with nnan fast math flag.
  // For IEEE=true consider NaN inputs. Only min(max(QNaN, 0.0), 1.0) evaluates
  // to 0.0 requires dx10_clamp = true.
  if ((getIEEE() && getDX10Clamp() && isFminnumIeee(MI) &&
       isKnownNeverSNaN(Val, MRI)) ||
      isKnownNeverNaN(MI.getOperand(0).getReg(), MRI)) {
    Reg = Val;
    return true;
  }

  return false;
}

// Replacing fmed3(NaN, 0.0, 1.0) with clamp. Requires dx10_clamp = true.
// Val = SNaN only for ieee = true. It is important which operand is NaN.
// min(min(SNaN, 0.0), 1.0) = min(QNaN, 1.0) = 1.0
// min(min(SNaN, 1.0), 0.0) = min(QNaN, 0.0) = 0.0
// min(min(0.0, 1.0), SNaN) = min(0.0, SNaN) = QNaN
// Val = NaN,ieee = false or Val = QNaN,ieee = true
// min(min(NaN, 0.0), 1.0) = min(0.0, 1.0) = 0.0
// min(min(NaN, 1.0), 0.0) = min(1.0, 0.0) = 0.0
// min(min(0.0, 1.0), NaN) = min(0.0, NaN) = 0.0
bool AMDGPURegBankCombinerHelper::matchFPMed3ToClamp(MachineInstr &MI,
                                                     Register &Reg) {
  if (MI.getIntrinsicID() != Intrinsic::amdgcn_fmed3)
    return false;

  // In llvm-ir, clamp is often represented as an intrinsic call to
  // @llvm.amdgcn.fmed3.f32(%Val, 0.0, 1.0). Check for other operand orders.
  MachineInstr *Src0 = getDefIgnoringCopies(MI.getOperand(2).getReg(), MRI);
  MachineInstr *Src1 = getDefIgnoringCopies(MI.getOperand(3).getReg(), MRI);
  MachineInstr *Src2 = getDefIgnoringCopies(MI.getOperand(4).getReg(), MRI);

  if (isFCst(Src0) && !isFCst(Src1))
    std::swap(Src0, Src1);
  if (isFCst(Src1) && !isFCst(Src2))
    std::swap(Src1, Src2);
  if (isFCst(Src0) && !isFCst(Src1))
    std::swap(Src0, Src1);
  if (!isClampZeroToOne(Src1, Src2))
    return false;

  Register Val = Src0->getOperand(0).getReg();

  auto isOp3Zero = [&]() {
    MachineInstr *Op3 = getDefIgnoringCopies(MI.getOperand(4).getReg(), MRI);
    if (Op3->getOpcode() == TargetOpcode::G_FCONSTANT)
      return Op3->getOperand(1).getFPImm()->isExactlyValue(0.0);
    return false;
  };
  // For IEEE=false perform combine only when it's safe to assume that there are
  // no NaN inputs. Most often MI is marked with nnan fast math flag.
  // For IEEE=true consider NaN inputs. Requires dx10_clamp = true. Safe to fold
  // when Val could be QNaN. If Val can also be SNaN third input should be 0.0.
  if (isKnownNeverNaN(MI.getOperand(0).getReg(), MRI) ||
      (getIEEE() && getDX10Clamp() &&
       (isKnownNeverSNaN(Val, MRI) || isOp3Zero()))) {
    Reg = Val;
    return true;
  }

  return false;
}

void AMDGPURegBankCombinerHelper::applyClamp(MachineInstr &MI, Register &Reg) {
  B.setInstrAndDebugLoc(MI);
  B.buildInstr(AMDGPU::G_AMDGPU_CLAMP, {MI.getOperand(0)}, {Reg},
               MI.getFlags());
  MI.eraseFromParent();
}

void AMDGPURegBankCombinerHelper::applyMed3(MachineInstr &MI,
                                            Med3MatchInfo &MatchInfo) {
  B.setInstrAndDebugLoc(MI);
  B.buildInstr(MatchInfo.Opc, {MI.getOperand(0)},
               {getAsVgpr(MatchInfo.Val0), getAsVgpr(MatchInfo.Val1),
                getAsVgpr(MatchInfo.Val2)},
               MI.getFlags());
  MI.eraseFromParent();
}

AMDGPU::SIModeRegisterDefaults AMDGPURegBankCombinerHelper::getMode() {
  return MF.getInfo<SIMachineFunctionInfo>()->getMode();
}

bool AMDGPURegBankCombinerHelper::getIEEE() { return getMode().IEEE; }

bool AMDGPURegBankCombinerHelper::getDX10Clamp() { return getMode().DX10Clamp; }

bool AMDGPURegBankCombinerHelper::isFminnumIeee(const MachineInstr &MI) {
  return MI.getOpcode() == AMDGPU::G_FMINNUM_IEEE;
}

bool AMDGPURegBankCombinerHelper::isFCst(MachineInstr *MI) {
  return MI->getOpcode() == AMDGPU::G_FCONSTANT;
}

bool AMDGPURegBankCombinerHelper::isClampZeroToOne(MachineInstr *K0,
                                                   MachineInstr *K1) {
  if (isFCst(K0) && isFCst(K1)) {
    const ConstantFP *KO_FPImm = K0->getOperand(1).getFPImm();
    const ConstantFP *K1_FPImm = K1->getOperand(1).getFPImm();
    return (KO_FPImm->isExactlyValue(0.0) && K1_FPImm->isExactlyValue(1.0)) ||
           (KO_FPImm->isExactlyValue(1.0) && K1_FPImm->isExactlyValue(0.0));
  }
  return false;
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
