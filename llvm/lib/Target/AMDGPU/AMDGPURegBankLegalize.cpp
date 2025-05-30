//===-- AMDGPURegBankLegalize.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Lower G_ instructions that can't be inst-selected with register bank
/// assignment from AMDGPURegBankSelect based on machine uniformity info.
/// Given types on all operands, some register bank assignments require lowering
/// while others do not.
/// Note: cases where all register bank assignments would require lowering are
/// lowered in legalizer.
/// For example vgpr S64 G_AND requires lowering to S32 while sgpr S64 does not.
/// Eliminate sgpr S1 by lowering to sgpr S32.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURegBankLegalizeHelper.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-regbanklegalize"

using namespace llvm;
using namespace AMDGPU;

namespace {

class AMDGPURegBankLegalize : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPURegBankLegalize() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Register Bank Legalize";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // If there were no phis and we do waterfall expansion machine verifier would
  // fail.
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURegBankLegalize, DEBUG_TYPE,
                      "AMDGPU Register Bank Legalize", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPURegBankLegalize, DEBUG_TYPE,
                    "AMDGPU Register Bank Legalize", false, false)

char AMDGPURegBankLegalize::ID = 0;

char &llvm::AMDGPURegBankLegalizeID = AMDGPURegBankLegalize::ID;

FunctionPass *llvm::createAMDGPURegBankLegalizePass() {
  return new AMDGPURegBankLegalize();
}

const RegBankLegalizeRules &getRules(const GCNSubtarget &ST,
                                     MachineRegisterInfo &MRI) {
  static std::mutex GlobalMutex;
  static SmallDenseMap<unsigned, std::unique_ptr<RegBankLegalizeRules>>
      CacheForRuleSet;
  std::lock_guard<std::mutex> Lock(GlobalMutex);
  auto [It, Inserted] = CacheForRuleSet.try_emplace(ST.getGeneration());
  if (Inserted)
    It->second = std::make_unique<RegBankLegalizeRules>(ST, MRI);
  else
    It->second->refreshRefs(ST, MRI);
  return *It->second;
}

class AMDGPURegBankLegalizeCombiner {
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  const SIRegisterInfo &TRI;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

  static constexpr LLT S1 = LLT::scalar(1);
  static constexpr LLT S16 = LLT::scalar(16);
  static constexpr LLT S32 = LLT::scalar(32);
  static constexpr LLT S64 = LLT::scalar(64);

public:
  AMDGPURegBankLegalizeCombiner(MachineIRBuilder &B, const SIRegisterInfo &TRI,
                                const RegisterBankInfo &RBI)
      : B(B), MRI(*B.getMRI()), TRI(TRI),
        SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
        VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
        VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {};

  bool isLaneMask(Register Reg) {
    const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
    if (RB && RB->getID() == AMDGPU::VCCRegBankID)
      return true;

    const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
    return RC && TRI.isSGPRClass(RC) && MRI.getType(Reg) == LLT::scalar(1);
  }

  void cleanUpAfterCombine(MachineInstr &MI, MachineInstr *Optional0) {
    MI.eraseFromParent();
    if (Optional0 && isTriviallyDead(*Optional0, MRI))
      Optional0->eraseFromParent();
  }

  std::pair<MachineInstr *, Register> tryMatch(Register Src, unsigned Opcode) {
    MachineInstr *MatchMI = MRI.getVRegDef(Src);
    if (MatchMI->getOpcode() != Opcode)
      return {nullptr, Register()};
    return {MatchMI, MatchMI->getOperand(1).getReg()};
  }

  void tryCombineCopy(MachineInstr &MI) {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    // Skip copies of physical registers.
    if (!Dst.isVirtual() || !Src.isVirtual())
      return;

    // This is a cross bank copy, sgpr S1 to lane mask.
    //
    // %Src:sgpr(s1) = G_TRUNC %TruncS32Src:sgpr(s32)
    // %Dst:lane-mask(s1) = COPY %Src:sgpr(s1)
    // ->
    // %Dst:lane-mask(s1) = G_AMDGPU_COPY_VCC_SCC %TruncS32Src:sgpr(s32)
    if (isLaneMask(Dst) && MRI.getRegBankOrNull(Src) == SgprRB) {
      auto [Trunc, TruncS32Src] = tryMatch(Src, AMDGPU::G_TRUNC);
      assert(Trunc && MRI.getType(TruncS32Src) == S32 &&
             "sgpr S1 must be result of G_TRUNC of sgpr S32");

      B.setInstr(MI);
      // Ensure that truncated bits in BoolSrc are 0.
      auto One = B.buildConstant({SgprRB, S32}, 1);
      auto BoolSrc = B.buildAnd({SgprRB, S32}, TruncS32Src, One);
      B.buildInstr(AMDGPU::G_AMDGPU_COPY_VCC_SCC, {Dst}, {BoolSrc});
      cleanUpAfterCombine(MI, Trunc);
      return;
    }

    // Src = G_AMDGPU_READANYLANE RALSrc
    // Dst = COPY Src
    // ->
    // Dst = RALSrc
    if (MRI.getRegBankOrNull(Dst) == VgprRB &&
        MRI.getRegBankOrNull(Src) == SgprRB) {
      auto [RAL, RALSrc] = tryMatch(Src, AMDGPU::G_AMDGPU_READANYLANE);
      if (!RAL)
        return;

      assert(MRI.getRegBank(RALSrc) == VgprRB);
      MRI.replaceRegWith(Dst, RALSrc);
      cleanUpAfterCombine(MI, RAL);
      return;
    }
  }

  void tryCombineS1AnyExt(MachineInstr &MI) {
    // %Src:sgpr(S1) = G_TRUNC %TruncSrc
    // %Dst = G_ANYEXT %Src:sgpr(S1)
    // ->
    // %Dst = G_... %TruncSrc
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    if (MRI.getType(Src) != S1)
      return;

    auto [Trunc, TruncSrc] = tryMatch(Src, AMDGPU::G_TRUNC);
    if (!Trunc)
      return;

    LLT DstTy = MRI.getType(Dst);
    LLT TruncSrcTy = MRI.getType(TruncSrc);

    if (DstTy == TruncSrcTy) {
      MRI.replaceRegWith(Dst, TruncSrc);
      cleanUpAfterCombine(MI, Trunc);
      return;
    }

    B.setInstr(MI);

    if (DstTy == S32 && TruncSrcTy == S64) {
      auto Unmerge = B.buildUnmerge({SgprRB, S32}, TruncSrc);
      MRI.replaceRegWith(Dst, Unmerge.getReg(0));
      cleanUpAfterCombine(MI, Trunc);
      return;
    }

    if (DstTy == S32 && TruncSrcTy == S16) {
      B.buildAnyExt(Dst, TruncSrc);
      cleanUpAfterCombine(MI, Trunc);
      return;
    }

    if (DstTy == S16 && TruncSrcTy == S32) {
      B.buildTrunc(Dst, TruncSrc);
      cleanUpAfterCombine(MI, Trunc);
      return;
    }

    llvm_unreachable("missing anyext + trunc combine");
  }
};

// Search through MRI for virtual registers with sgpr register bank and S1 LLT.
[[maybe_unused]] static Register getAnySgprS1(const MachineRegisterInfo &MRI) {
  const LLT S1 = LLT::scalar(1);
  for (unsigned i = 0; i < MRI.getNumVirtRegs(); ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI.def_empty(Reg) || MRI.getType(Reg) != S1)
      continue;

    const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
    if (RB && RB->getID() == AMDGPU::SGPRRegBankID) {
      LLVM_DEBUG(dbgs() << "Warning: detected sgpr S1 register in: ";
                 MRI.getVRegDef(Reg)->dump(););
      return Reg;
    }
  }

  return {};
}

bool AMDGPURegBankLegalize::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasFailedISel())
    return false;

  // Setup the instruction builder with CSE.
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  GISelCSEInfo &CSEInfo = Wrapper.get(TPC.getCSEConfig());
  GISelObserverWrapper Observer;
  Observer.addObserver(&CSEInfo);

  CSEMIRBuilder B(MF);
  B.setCSEInfo(&CSEInfo);
  B.setChangeObserver(Observer);

  RAIIDelegateInstaller DelegateInstaller(MF, &Observer);
  RAIIMFObserverInstaller MFObserverInstaller(MF, Observer);

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RegisterBankInfo &RBI = *ST.getRegBankInfo();
  const MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();

  // RegBankLegalizeRules is initialized with assigning sets of IDs to opcodes.
  const RegBankLegalizeRules &RBLRules = getRules(ST, MRI);

  // Logic that does legalization based on IDs assigned to Opcode.
  RegBankLegalizeHelper RBLHelper(B, MUI, RBI, RBLRules);

  SmallVector<MachineInstr *> AllInst;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      AllInst.push_back(&MI);
    }
  }

  for (MachineInstr *MI : AllInst) {
    if (!MI->isPreISelOpcode())
      continue;

    unsigned Opc = MI->getOpcode();
    // Insert point for use operands needs some calculation.
    if (Opc == AMDGPU::G_PHI) {
      RBLHelper.applyMappingPHI(*MI);
      continue;
    }

    // Opcodes that support pretty much all combinations of reg banks and LLTs
    // (except S1). There is no point in writing rules for them.
    if (Opc == AMDGPU::G_BUILD_VECTOR || Opc == AMDGPU::G_UNMERGE_VALUES ||
        Opc == AMDGPU::G_MERGE_VALUES) {
      RBLHelper.applyMappingTrivial(*MI);
      continue;
    }

    // Opcodes that also support S1.
    if (Opc == G_FREEZE &&
        MRI.getType(MI->getOperand(0).getReg()) != LLT::scalar(1)) {
      RBLHelper.applyMappingTrivial(*MI);
      continue;
    }

    if ((Opc == AMDGPU::G_CONSTANT || Opc == AMDGPU::G_FCONSTANT ||
         Opc == AMDGPU::G_IMPLICIT_DEF)) {
      Register Dst = MI->getOperand(0).getReg();
      // Non S1 types are trivially accepted.
      if (MRI.getType(Dst) != LLT::scalar(1)) {
        assert(MRI.getRegBank(Dst)->getID() == AMDGPU::SGPRRegBankID);
        continue;
      }

      // S1 rules are in RegBankLegalizeRules.
    }

    RBLHelper.findRuleAndApplyMapping(*MI);
  }

  // Sgpr S1 clean up combines:
  // - Sgpr S1(S32) to sgpr S1(S32) Copy: anyext + trunc combine.
  //   In RegBankLegalize 'S1 Dst' are legalized into S32 as
  //   'S1Dst = Trunc S32Dst' and 'S1 Src' into 'S32Src = Anyext S1Src'.
  //   S1 Truncs and Anyexts that come from legalizer, that can have non-S32
  //   types e.g. S16 = Anyext S1 or S1 = Trunc S64, will also be cleaned up.
  // - Sgpr S1(S32) to vcc Copy: G_AMDGPU_COPY_VCC_SCC combine.
  //   Divergent instruction uses sgpr S1 as input that should be lane mask(vcc)
  //   Legalizing this use creates sgpr S1(S32) to vcc Copy.

  // Note: Remaining S1 copies, S1s are either sgpr S1(S32) or vcc S1:
  // - Vcc to vcc Copy: nothing to do here, just a regular copy.
  // - Vcc to sgpr S1 Copy: Should not exist in a form of COPY instruction(*).
  //   Note: For 'uniform-in-vcc to sgpr-S1 copy' G_AMDGPU_COPY_SCC_VCC is used
  //   instead. When only available instruction creates vcc result, use of
  //   UniformInVcc results in creating G_AMDGPU_COPY_SCC_VCC.

  // (*)Explanation for 'sgpr S1(uniform) = COPY vcc(divergent)':
  // Copy from divergent to uniform register indicates an error in either:
  // - Uniformity analysis: Uniform instruction has divergent input. If one of
  //   the inputs is divergent, instruction should be divergent!
  // - RegBankLegalizer not executing in waterfall loop (missing implementation)

  AMDGPURegBankLegalizeCombiner Combiner(B, *ST.getRegisterInfo(), RBI);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() == AMDGPU::COPY) {
        Combiner.tryCombineCopy(MI);
        continue;
      }
      if (MI.getOpcode() == AMDGPU::G_ANYEXT) {
        Combiner.tryCombineS1AnyExt(MI);
        continue;
      }
    }
  }

  assert(!getAnySgprS1(MRI).isValid() &&
         "Registers with sgpr reg bank and S1 LLT are not legal after "
         "AMDGPURegBankLegalize. Should lower to sgpr S32");

  return true;
}
