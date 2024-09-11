//===-- AMDGPURBLegalize.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Lower G_ instructions that can't be inst-selected with register bank
/// assignment given by RB-select based on machine uniformity info.
/// Given types on all operands, some register bank assignments require lowering
/// while other do not.
/// Note: cases where all register bank assignments would require lowering are
/// lowered in legalizer.
/// For example vgpr S64 G_AND requires lowering to S32 while SGPR S64 does not.
/// Eliminate sgpr S1 by lowering to sgpr S32.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURBLegalizeHelper.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "rb-legalize"

using namespace llvm;

namespace {

class AMDGPURBLegalize : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPURBLegalize() : MachineFunctionPass(ID) {
    initializeAMDGPURBLegalizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU RB Legalize"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // If there were no phis and we do waterfall expansion machine verifier would
  // fail.
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURBLegalize, DEBUG_TYPE, "AMDGPU RB Legalize", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPURBLegalize, DEBUG_TYPE, "AMDGPU RB Legalize", false,
                    false)

char AMDGPURBLegalize::ID = 0;

char &llvm::AMDGPURBLegalizeID = AMDGPURBLegalize::ID;

FunctionPass *llvm::createAMDGPURBLegalizePass() {
  return new AMDGPURBLegalize();
}

using namespace AMDGPU;

const RegBankLegalizeRules &getRules(const GCNSubtarget &ST,
                                     MachineRegisterInfo &MRI) {
  static std::mutex GlobalMutex;
  static SmallDenseMap<unsigned, std::unique_ptr<RegBankLegalizeRules>>
      CacheForRuleSet;
  std::lock_guard<std::mutex> Lock(GlobalMutex);
  if (!CacheForRuleSet.contains(ST.getGeneration())) {
    auto Rules = std::make_unique<RegBankLegalizeRules>(ST, MRI);
    CacheForRuleSet[ST.getGeneration()] = std::move(Rules);
  } else {
    CacheForRuleSet[ST.getGeneration()]->refreshRefs(ST, MRI);
  }
  return *CacheForRuleSet[ST.getGeneration()];
}

bool AMDGPURBLegalize::runOnMachineFunction(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Setup the instruction builder with CSE.
  std::unique_ptr<MachineIRBuilder> MIRBuilder;
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  GISelCSEInfo *CSEInfo = nullptr;
  GISelObserverWrapper Observer;

  if (TPC.isGISelCSEEnabled()) {
    MIRBuilder = std::make_unique<CSEMIRBuilder>();
    CSEInfo = &Wrapper.get(TPC.getCSEConfig());
    MIRBuilder->setCSEInfo(CSEInfo);
    Observer.addObserver(CSEInfo);
    MIRBuilder->setChangeObserver(Observer);
  } else {
    MIRBuilder = std::make_unique<MachineIRBuilder>();
  }
  MIRBuilder->setMF(MF);

  RAIIDelegateInstaller DelegateInstaller(MF, &Observer);
  RAIIMFObserverInstaller MFObserverInstaller(MF, Observer);

  const MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  const RegisterBankInfo &RBI = *MF.getSubtarget().getRegBankInfo();

  // RegBankLegalizeRules is initialized with assigning sets of IDs to opcodes.
  const RegBankLegalizeRules &RBLRules = getRules(ST, MRI);

  // Logic that does legalization based on IDs assigned to Opcode.
  RegBankLegalizeHelper RBLegalizeHelper(*MIRBuilder, MRI, MUI, RBI, RBLRules);

  SmallVector<MachineInstr *> AllInst;

  for (auto &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      AllInst.push_back(&MI);
    }
  }

  for (auto &MI : AllInst) {
    if (!MI->isPreISelOpcode())
      continue;

    unsigned Opc = MI->getOpcode();

    // Insert point for use operands needs some calculation.
    if (Opc == G_PHI) {
      RBLegalizeHelper.applyMappingPHI(*MI);
      continue;
    }

    // Opcodes that support pretty much all combinations of reg banks and LLTs
    // (except S1). There is no point in writing rules for them.
    if (Opc == G_BUILD_VECTOR || Opc == G_UNMERGE_VALUES ||
        Opc == G_MERGE_VALUES) {
      RBLegalizeHelper.applyMappingTrivial(*MI);
      continue;
    }

    // Opcodes that also support S1. S1 rules are in RegBankLegalizeRules.
    // Remaining reg bank and LLT combinations are trivially accepted.
    if ((Opc == G_CONSTANT || Opc == G_FCONSTANT || Opc == G_IMPLICIT_DEF) &&
        !isS1(MI->getOperand(0).getReg(), MRI)) {
      assert(isSgprRB(MI->getOperand(0).getReg(), MRI));
      continue;
    }

    if (!RBLegalizeHelper.findRuleAndApplyMapping(*MI)) {
      MI->dump();
      llvm_unreachable("failed to match any of the rules");
    }
  }

  LLT S1 = LLT::scalar(1);
  LLT S16 = LLT::scalar(16);
  LLT S32 = LLT::scalar(32);
  LLT S64 = LLT::scalar(64);

  // SGPR S1 clean up combines:
  // - SGPR S1(S32) to SGPR S1(S32) Copy: anyext + trunc combine.
  //   In RBLegalize 'S1 Dst' are legalized into S32 as'S1Dst = Trunc S32Dst'
  //   and 'S1 Src' into 'S32Src = Anyext S1Src'.
  //   S1 Truncs and Anyexts that come from legalizer will also be cleaned up.
  //   Note: they can have non-S32 types e.g. S16 = Anyext S1 or S1 = Trunc S64.
  // - Sgpr S1(S32) to VCC Copy: G_COPY_VCC_SCC combine.
  //   Divergent instruction uses Sgpr S1 as input that should be lane mask(VCC)
  //   Legalizing this use creates Sgpr S1(S32) to VCC Copy.

  // Note: Remaining S1 copies, S1s are either SGPR S1(S32) or VCC S1:
  // - VCC to VCC Copy: nothing to do here, just a regular copy.
  // - VCC to SGPR S1 Copy: Should not exist in a form of COPY instruction(*).
  //   Note: For 'uniform-in-VCC to SGPR-S1 copy' G_COPY_SCC_VCC is used
  //   instead. When only available instruction creates VCC result, use of
  //   UniformInVcc results in creating G_COPY_SCC_VCC.

  // (*)Explanation for 'SGPR S1(uniform) = COPY VCC(divergent)':
  // Copy from divergent to uniform register indicates an error in either:
  // - Uniformity analysis: Uniform instruction has divergent input. If one of
  //   the inputs is divergent, instruction should be divergent!
  // - RBLegalizer not executing in waterfall loop (missing implementation)

  using namespace MIPatternMatch;
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  for (auto &MBB : MF) {
    for (auto &MI : make_early_inc_range(MBB)) {

      if (MI.getOpcode() == G_TRUNC && isTriviallyDead(MI, MRI)) {
        MI.eraseFromParent();
        continue;
      }

      if (MI.getOpcode() == COPY) {
        Register Dst = MI.getOperand(0).getReg();
        Register Src = MI.getOperand(1).getReg();
        if (!Dst.isVirtual() || !Src.isVirtual())
          continue;

        // This is cross bank copy, sgpr S1 to lane mask.
        // sgpr S1 must be result of G_TRUNC of SGPR S32.
        if (isLaneMask(Dst, MRI, TRI) && isSgprRB(Src, MRI)) {
          MachineInstr *Trunc = MRI.getVRegDef(Src);
          Register SrcSgpr32 = Trunc->getOperand(1).getReg();

          assert(Trunc->getOpcode() == G_TRUNC);
          assert(isSgprRB(SrcSgpr32, MRI) && MRI.getType(SrcSgpr32) == S32);

          MIRBuilder->setInstr(MI);
          const RegisterBank *SgprRB = &RBI.getRegBank(SGPRRegBankID);
          Register BoolSrc = MRI.createVirtualRegister({SgprRB, S32});
          Register One = MRI.createVirtualRegister({SgprRB, S32});
          // Ensure that truncated bits in BoolSrc are 0.
          MIRBuilder->buildConstant(One, 1);
          MIRBuilder->buildAnd(BoolSrc, SrcSgpr32, One);
          MIRBuilder->buildInstr(G_COPY_VCC_SCC, {Dst}, {BoolSrc});
          cleanUpAfterCombine(MI, MRI, Trunc);
          continue;
        }

        // Src = G_READANYLANE VgprRBSrc
        // Dst = COPY Src
        // ->
        // Dst = VgprRBSrc
        if (isVgprRB(Dst, MRI) && isSgprRB(Src, MRI)) {
          MachineInstr *RFL = MRI.getVRegDef(Src);
          Register VgprRBSrc;
          if (mi_match(RFL, MRI, m_GReadAnyLane(m_Reg(VgprRBSrc)))) {
            assert(isVgprRB(VgprRBSrc, MRI));
            MRI.replaceRegWith(Dst, VgprRBSrc);
            cleanUpAfterCombine(MI, MRI, RFL);
            continue;
          }
        }
      }

      // Sgpr(S1) = G_TRUNC TruncSrc
      // Dst = G_ANYEXT Sgpr(S1)
      // ->
      // Dst = G_... TruncSrc
      if (MI.getOpcode() == G_ANYEXT) {
        Register Dst = MI.getOperand(0).getReg();
        Register Src = MI.getOperand(1).getReg();
        if (!Dst.isVirtual() || !Src.isVirtual() || MRI.getType(Src) != S1)
          continue;

        // Note: Sgpr S16 is anyextened to S32 for some opcodes and could use
        // same combine but it is not required for correctness so we skip it.
        // S16 is legal because there is instruction with VGPR S16.

        MachineInstr *Trunc = MRI.getVRegDef(Src);
        if (Trunc->getOpcode() != G_TRUNC)
          continue;

        Register TruncSrc = Trunc->getOperand(1).getReg();
        LLT DstTy = MRI.getType(Dst);
        LLT TruncSrcTy = MRI.getType(TruncSrc);

        if (DstTy == TruncSrcTy) {
          MRI.replaceRegWith(Dst, TruncSrc);
          cleanUpAfterCombine(MI, MRI, Trunc);
          continue;
        }

        MIRBuilder->setInstr(MI);

        if (DstTy == S32 && TruncSrcTy == S64) {
          const RegisterBank *SgprRB = &RBI.getRegBank(SGPRRegBankID);
          Register Lo = MRI.createVirtualRegister({SgprRB, S32});
          Register Hi = MRI.createVirtualRegister({SgprRB, S32});
          MIRBuilder->buildUnmerge({Lo, Hi}, TruncSrc);

          MRI.replaceRegWith(Dst, Lo);
          cleanUpAfterCombine(MI, MRI, Trunc);
          continue;
        }

        if (DstTy == S32 && TruncSrcTy == S16) {
          MIRBuilder->buildAnyExt(Dst, TruncSrc);
          cleanUpAfterCombine(MI, MRI, Trunc);
          continue;
        }

        if (DstTy == S16 && TruncSrcTy == S32) {
          MIRBuilder->buildTrunc(Dst, TruncSrc);
          cleanUpAfterCombine(MI, MRI, Trunc);
          continue;
        }

        llvm_unreachable("missing anyext + trunc combine\n");
      }
    }
  }

  assert(!hasSGPRS1(MF, MRI) && "detetected SGPR S1 register");

  return true;
}
