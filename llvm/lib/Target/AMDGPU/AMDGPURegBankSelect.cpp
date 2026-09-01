//===-- AMDGPURegBankSelect.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Assign register banks to all register operands of G_ instructions using
/// machine uniformity analysis.
/// Sgpr - uniform values and some lane masks
/// Vgpr - divergent, non S1, values
/// Vcc  - divergent S1 values(lane masks)
/// However in some cases G_ instructions with this register bank assignment
/// can't be inst-selected. This is solved in AMDGPURegBankLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-reg-bank-select"

using namespace llvm;
using namespace AMDGPU;

namespace {

class AMDGPURegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURegBankSelect() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Register Bank Select";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<GISelCSEAnalysisWrapperPass>();
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // This pass assigns register banks to all virtual registers, and we maintain
  // this property in subsequent passes
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setRegBankSelected();
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURegBankSelect, DEBUG_TYPE,
                      "AMDGPU Register Bank Select", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPURegBankSelect, DEBUG_TYPE,
                    "AMDGPU Register Bank Select", false, false)

char AMDGPURegBankSelect::ID = 0;

char &llvm::AMDGPURegBankSelectID = AMDGPURegBankSelect::ID;

FunctionPass *llvm::createAMDGPURegBankSelectPass() {
  return new AMDGPURegBankSelect();
}

class RegBankSelectHelper {
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA;
  const MachineUniformityInfo &MUI;
  const SIRegisterInfo &TRI;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

public:
  RegBankSelectHelper(MachineIRBuilder &B,
                      AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA,
                      const MachineUniformityInfo &MUI,
                      const SIRegisterInfo &TRI, const RegisterBankInfo &RBI)
      : B(B), MRI(*B.getMRI()), ILMA(ILMA), MUI(MUI), TRI(TRI),
        SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
        VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
        VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {}

  // Temporal divergence copy: COPY to vgpr with implicit use of $exec inside of
  // the cycle
  // Note: uniformity analysis does not consider that registers with vgpr def
  // are divergent (you can have uniform value in vgpr).
  // - TODO: implicit use of $exec could be implemented as indicator that
  //   instruction is divergent
  bool isTemporalDivergenceCopy(Register Reg) {
    MachineInstr *MI = MRI.getVRegDef(Reg);
    if (!MI->isCopy() || MI->getNumImplicitOperands() != 1)
      return false;

    return MI->implicit_operands().begin()->getReg() == TRI.getExec();
  }

  const RegisterBank *getRegBankToAssign(Register Reg) {
    if (!isTemporalDivergenceCopy(Reg) &&
        (MUI.isUniformAtDef(Reg) || ILMA.isS32S64LaneMask(Reg)))
      return SgprRB;
    if (MRI.getType(Reg) == LLT::scalar(1))
      return VccRB;
    return VgprRB;
  }

  // %rc:RegClass(s32) = G_ ...
  // ...
  // %a = G_ ..., %rc
  // ->
  // %rb:RegBank(s32) = G_ ...
  // %rc:RegClass(s32) = COPY %rb
  // ...
  // %a = G_ ..., %rb
  void reAssignRegBankOnDef(MachineInstr &MI, MachineOperand &DefOP,
                            const RegisterBank *RB) {
    // Register that already has Register class got it during pre-inst selection
    // of another instruction. Maybe cross bank copy was required so we insert a
    // copy that can be removed later. This simplifies post regbanklegalize
    // combiner and avoids need to special case some patterns.
    Register Reg = DefOP.getReg();
    LLT Ty = MRI.getType(Reg);
    Register NewReg = MRI.createVirtualRegister({RB, Ty});
    DefOP.setReg(NewReg);

    auto &MBB = *MI.getParent();
    B.setInsertPt(MBB, MBB.SkipPHIsAndLabels(std::next(MI.getIterator())));
    B.buildCopy(Reg, NewReg);

    // The problem was discovered for uniform S1 that was used as both
    // lane mask(vcc) and regular sgpr S1.
    // - lane-mask(vcc) use was by si_if, this use is divergent and requires
    //   non-trivial sgpr-S1-to-vcc copy. But pre-inst-selection of si_if sets
    //   sreg_64_xexec(S1) on def of uniform S1 making it lane-mask.
    // - the regular sgpr S1(uniform) instruction is now broken since
    //   it uses sreg_64_xexec(S1) which is divergent.

    // Replace virtual registers with register class on generic instructions
    // uses with virtual registers with register bank.
    for (auto &UseMI : make_early_inc_range(MRI.use_instructions(Reg))) {
      if (UseMI.isPreISelOpcode()) {
        for (MachineOperand &Op : UseMI.operands()) {
          if (Op.isReg() && Op.getReg() == Reg)
            Op.setReg(NewReg);
        }
      }
    }
  }

  Register buildHandoffReadFirstLane(Register Src) {
    LLT Ty = MRI.getType(Src);
    Register VgprSrc = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    MRI.setType(VgprSrc, Ty);
    B.buildCopy(VgprSrc, Src);

    Register Sgpr = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
    MRI.setType(Sgpr, Ty);
    B.buildInstr(AMDGPU::V_READFIRSTLANE_B32, {Sgpr}, {VgprSrc});
    return Sgpr;
  }

  bool isSGPR(Register Reg) {
    if (Reg.isPhysical())
      return TRI.isSGPRPhysReg(Reg);
    if (const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg))
      return SIRegisterInfo::isSGPRClass(RC);
    if (const RegisterBank *RB = MRI.getRegBankOrNull(Reg))
      return RB == SgprRB;
    return getRegBankToAssign(Reg) == SgprRB;
  }

  bool isVectorOnly(Register Reg) {
    if (const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg))
      return SIRegisterInfo::hasVectorRegisters(RC) &&
             !SIRegisterInfo::hasSGPRs(RC);
    if (const RegisterBank *RB = MRI.getRegBankOrNull(Reg))
      return RB == VgprRB;
    return getRegBankToAssign(Reg) == VgprRB;
  }

  // %a = G_ ..., %rc
  // ->
  // %rb:RegBank(s32) = COPY %rc
  // %a = G_ ..., %rb
  void constrainRegBankUse(MachineInstr &MI, MachineOperand &UseOP,
                           const RegisterBank *RB) {
    Register Reg = UseOP.getReg();

    if (MI.isPHI()) {
      auto DefMI = MRI.getVRegDef(Reg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();
      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));
    } else {
      B.setInstr(MI);
    }

    MachineInstr *DefMI = MRI.getVRegDef(Reg);
    // The chosen insertion point may be behind this pass's cursor, so repair
    // a direct scalar handoff use immediately.
    if (RB == SgprRB && SIInstrInfo::isRegAllocHandoff(DefMI->getOpcode()))
      Reg = buildHandoffReadFirstLane(Reg);

    Register NewReg = MRI.createVirtualRegister({RB, MRI.getType(Reg)});
    B.buildCopy(NewReg, Reg);
    UseOP.setReg(NewReg);
  }

  bool lowerRegAllocHandoff(MachineInstr &MI, bool HasMAIInsts) {
    if (MI.getOpcode() != AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS ||
        MI.getNumExplicitOperands() != 4 || !MI.getOperand(1).isIntrinsicID() ||
        MI.getOperand(1).getIntrinsicID() !=
            Intrinsic::experimental_regalloc_handoff)
      return false;

    MachineOperand &DstOp = MI.getOperand(0);
    MachineOperand &SrcOp = MI.getOperand(2);
    MachineOperand &ConstraintOp = MI.getOperand(3);
    if (!DstOp.isReg() || !DstOp.isDef() || !DstOp.getReg().isVirtual() ||
        !SrcOp.isReg() || !SrcOp.isUse() || !SrcOp.getReg().isVirtual() ||
        !ConstraintOp.isMetadata())
      return false;

    const auto *ConstraintNode = dyn_cast<MDNode>(ConstraintOp.getMetadata());
    const MDString *Constraint =
        ConstraintNode && ConstraintNode->getNumOperands() == 1
            ? dyn_cast<MDString>(ConstraintNode->getOperand(0))
            : nullptr;
    StringRef ConstraintName = Constraint ? Constraint->getString() : "";
    const TargetRegisterClass *DstRC = nullptr;
    unsigned HandoffOpcode = AMDGPU::INSTRUCTION_LIST_END;
    if (ConstraintName == "amdgpu.vgpr") {
      DstRC = &AMDGPU::VGPR_32RegClass;
      HandoffOpcode = AMDGPU::REGALLOC_HANDOFF_VGPR;
    } else if (ConstraintName == "amdgpu.agpr" && HasMAIInsts) {
      DstRC = &AMDGPU::AGPR_32RegClass;
      HandoffOpcode = AMDGPU::REGALLOC_HANDOFF_AGPR;
    }

    Register Dst = DstOp.getReg();
    Register Src = SrcOp.getReg();
    if (MRI.getType(Dst) != LLT::scalar(32) ||
        MRI.getType(Src) != LLT::scalar(32))
      return false;

    bool DstConstrained =
        MRI.getRegClassOrNull(Dst) || MRI.getRegBankOrNull(Dst);
    if (!DstRC) {
      if (Dst == Src) {
        MI.eraseFromParent();
        return true;
      }

      if (!DstConstrained) {
        MRI.replaceRegWith(Dst, Src);
      } else {
        B.setInstr(MI);
        Register CopySrc = Src;
        if (isSGPR(Dst) && isVectorOnly(Src))
          CopySrc = buildHandoffReadFirstLane(Src);
        B.buildCopy(Dst, CopySrc);
      }
      MI.eraseFromParent();
      return true;
    }

    Register HandoffDst = Dst;
    if (DstConstrained) {
      HandoffDst = MRI.createVirtualRegister(DstRC);
      MRI.setType(HandoffDst, MRI.getType(Dst));
    } else {
      MRI.setRegClass(Dst, DstRC);
    }

    Register AVSrc = MRI.createVirtualRegister(&AMDGPU::AV_32RegClass);
    MRI.setType(AVSrc, MRI.getType(Src));
    B.setInstr(MI);
    B.buildCopy(AVSrc, Src);
    B.buildInstr(HandoffOpcode, {HandoffDst}, {AVSrc})
        .setMIFlag(MachineInstr::NoMerge);
    if (HandoffDst != Dst)
      B.buildCopy(Dst, HandoffDst);
    MI.eraseFromParent();
    return true;
  }

  bool repairHandoffScalarCopy(MachineInstr &MI) {
    if (!MI.isCopy() || !MI.getOperand(0).isReg() || !MI.getOperand(1).isReg())
      return false;

    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    if (!Src.isVirtual())
      return false;
    MachineInstr *SrcDef = MRI.getVRegDef(Src);
    if (!SrcDef || !SIInstrInfo::isRegAllocHandoff(SrcDef->getOpcode()))
      return false;

    if (!isSGPR(Dst))
      return false;

    B.setInstr(MI);
    MI.getOperand(1).setReg(buildHandoffReadFirstLane(Src));
    return true;
  }
};

static Register getVReg(MachineOperand &Op) {
  if (!Op.isReg())
    return {};

  // Operands of COPY and G_SI_CALL can be physical registers.
  Register Reg = Op.getReg();
  if (!Reg.isVirtual())
    return {};

  return Reg;
}

bool AMDGPURegBankSelect::runOnMachineFunction(MachineFunction &MF) {
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

  IntrinsicLaneMaskAnalyzer ILMA(MF);
  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  MachineRegisterInfo &MRI = *B.getMRI();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  RegBankSelectHelper RBSHelper(B, ILMA, MUI, *ST.getRegisterInfo(),
                                *ST.getRegBankInfo());

  // Lower all handoff markers before ordinary bank assignment. This makes the
  // exact requested class visible even when a PHI or another use appears
  // before its defining block in machine layout.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS ||
          MI.getNumExplicitOperands() < 2 ||
          !MI.getOperand(1).isIntrinsicID() ||
          MI.getOperand(1).getIntrinsicID() !=
              Intrinsic::experimental_regalloc_handoff)
        continue;
      if (!RBSHelper.lowerRegAllocHandoff(MI, ST.hasMAIInsts())) {
        MF.getProperties().setFailedISel();
        return true;
      }
    }
  }

  // Virtual registers at this point don't have register banks.
  // Virtual registers in def and use operands of already inst-selected
  // instruction have register class.

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Vregs in def and use operands of COPY can have either register class
      // or bank. If there is neither on vreg in def operand, assign bank.
      if (MI.isCopy()) {
        RBSHelper.repairHandoffScalarCopy(MI);
        Register DefReg = getVReg(MI.getOperand(0));
        if (!DefReg.isValid() || MRI.getRegClassOrNull(DefReg))
          continue;

        // Scalar handoff repair can insert a banked copy in a predecessor that
        // has not been visited yet.
        if (MRI.getRegBankOrNull(DefReg))
          continue;
        MRI.setRegBank(DefReg, *RBSHelper.getRegBankToAssign(DefReg));
        continue;
      }

      if (!MI.isPreISelOpcode())
        continue;

      // Vregs in def and use operands of G_ instructions need to have register
      // banks assigned. Before this loop possible case are
      // - (1) vreg without register class or bank in def or use operand
      // - (2) vreg with register class in def operand
      // - (3) vreg, defined by G_ instruction, in use operand
      // - (4) vreg, defined by pre-inst-selected instruction, in use operand

      // First three cases are handled in loop through all def operands of G_
      // instructions. For case (1) simply setRegBank. Cases (2) and (3) are
      // handled by reAssignRegBankOnDef.
      for (MachineOperand &DefOP : MI.defs()) {
        Register DefReg = getVReg(DefOP);
        if (!DefReg.isValid())
          continue;

        const RegisterBank *RB = RBSHelper.getRegBankToAssign(DefReg);
        if (MRI.getRegClassOrNull(DefReg))
          RBSHelper.reAssignRegBankOnDef(MI, DefOP, RB);
        else {
          assert(!MRI.getRegBankOrNull(DefReg));
          MRI.setRegBank(DefReg, *RB);
        }
      }

      // Register bank select doesn't modify pre-inst-selected instructions.
      // For case (4) need to insert a copy, handled by constrainRegBankUse.
      for (MachineOperand &UseOP : MI.uses()) {
        Register UseReg = getVReg(UseOP);
        if (!UseReg.isValid())
          continue;

        // Skip case (3).
        if (!MRI.getRegClassOrNull(UseReg) ||
            MRI.getVRegDef(UseReg)->isPreISelOpcode())
          continue;

        // Use with register class defined by pre-inst-selected instruction.
        const RegisterBank *RB = RBSHelper.getRegBankToAssign(UseReg);
        RBSHelper.constrainRegBankUse(MI, UseOP, RB);
      }
    }
  }

  return true;
}
