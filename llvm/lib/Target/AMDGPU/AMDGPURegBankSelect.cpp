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
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-regbankselect"

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
        (MUI.isUniform(Reg) || ILMA.isS32S64LaneMask(Reg)))
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

  // %a = G_ ..., %rc
  // ->
  // %rb:RegBank(s32) = COPY %rc
  // %a = G_ ..., %rb
  void constrainRegBankUse(MachineInstr &MI, MachineOperand &UseOP,
                           const RegisterBank *RB) {
    Register Reg = UseOP.getReg();

    LLT Ty = MRI.getType(Reg);
    Register NewReg = MRI.createVirtualRegister({RB, Ty});
    UseOP.setReg(NewReg);

    if (MI.isPHI()) {
      auto DefMI = MRI.getVRegDef(Reg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();
      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));
    } else {
      B.setInstr(MI);
    }

    B.buildCopy(NewReg, Reg);
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
  // Virtual registers at this point don't have register banks.
  // Virtual registers in def and use operands of already inst-selected
  // instruction have register class.

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Vregs in def and use operands of COPY can have either register class
      // or bank. If there is neither on vreg in def operand, assign bank.
      if (MI.isCopy()) {
        Register DefReg = getVReg(MI.getOperand(0));
        if (!DefReg.isValid() || MRI.getRegClassOrNull(DefReg))
          continue;

        assert(!MRI.getRegBankOrNull(DefReg));
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
