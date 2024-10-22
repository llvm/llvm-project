//===-- AMDGPUStandaloneRegBankSelect.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Assign register banks to all register operands of G_ instructions using
/// machine uniformity analysis.
/// SGPR - uniform values and some lane masks
/// VGPR - divergent, non S1, values
/// VCC  - divergent S1 values(lane masks)
/// However in some cases G_ instructions with this register bank assignment
/// can't be inst-selected. This is solved in RegBankLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURegisterBankInfo.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-standalone-regbankselect"

using namespace llvm;

namespace {

class AMDGPUStandaloneRegBankSelect : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUStandaloneRegBankSelect() : MachineFunctionPass(ID) {
    initializeAMDGPUStandaloneRegBankSelectPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Standalone Register Bank Select";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineUniformityAnalysisPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // This pass assigns register banks to all virtual registers, and we maintain
  // this property in subsequent passes
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::RegBankSelected);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUStandaloneRegBankSelect, DEBUG_TYPE,
                      "AMDGPU Standalone Register Bank Select", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPUStandaloneRegBankSelect, DEBUG_TYPE,
                    "AMDGPU Standalone Register Bank Select", false, false)

char AMDGPUStandaloneRegBankSelect::ID = 0;

char &llvm::AMDGPUStandaloneRegBankSelectID = AMDGPUStandaloneRegBankSelect::ID;

FunctionPass *llvm::createAMDGPUStandaloneRegBankSelectPass() {
  return new AMDGPUStandaloneRegBankSelect();
}

class RegBankSelectHelper {
  MachineFunction &MF;
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA;
  const MachineUniformityInfo &MUI;
  const SIRegisterInfo &TRI;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

public:
  RegBankSelectHelper(MachineFunction &MF, MachineIRBuilder &B,
                      MachineRegisterInfo &MRI,
                      AMDGPU::IntrinsicLaneMaskAnalyzer &ILMA,
                      const MachineUniformityInfo &MUI,
                      const SIRegisterInfo &TRI, const RegisterBankInfo &RBI)
      : MF(MF), B(B), MRI(MRI), ILMA(ILMA), MUI(MUI), TRI(TRI),
        SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
        VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
        VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {}

  bool shouldRegBankSelect(MachineInstr &MI) {
    return MI.isPreISelOpcode() || MI.isCopy();
  }

  void setRBDef(MachineInstr &MI, MachineOperand &DefOP,
                const RegisterBank *RB) {
    Register Reg = DefOP.getReg();
    // Register that already has Register class got it during pre-inst selection
    // of another instruction. Maybe cross bank copy was required so we insert a
    // copy that can be removed later. This simplifies post-rb-legalize artifact
    // combiner and avoids need to special case some patterns.
    if (MRI.getRegClassOrNull(Reg)) {
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

      // "Clear" reg classes from uses on generic instructions and put register
      // banks instead.
      for (auto &UseMI : MRI.use_instructions(Reg)) {
        if (shouldRegBankSelect(UseMI)) {
          for (MachineOperand &Op : UseMI.operands()) {
            if (Op.isReg() && Op.getReg() == Reg)
              Op.setReg(NewReg);
          }
        }
      }

    } else {
      MRI.setRegBank(Reg, *RB);
    }
  }

  void constrainRBUse(MachineInstr &MI, MachineOperand &UseOP,
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

  std::optional<Register> tryGetVReg(MachineOperand &Op) {
    if (!Op.isReg())
      return std::nullopt;

    Register Reg = Op.getReg();
    if (!Reg.isVirtual())
      return std::nullopt;

    return Reg;
  }

  void assignBanksOnDefs() {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!shouldRegBankSelect(MI))
          continue;

        for (MachineOperand &DefOP : MI.defs()) {
          auto MaybeDefReg = tryGetVReg(DefOP);
          if (!MaybeDefReg)
            continue;
          Register DefReg = *MaybeDefReg;

          // Copies can have register class on def registers.
          if (MI.isCopy() && MRI.getRegClassOrNull(DefReg)) {
            continue;
          }

          if (MUI.isUniform(DefReg) || ILMA.isS32S64LaneMask(DefReg)) {
            setRBDef(MI, DefOP, SgprRB);
          } else {
            if (MRI.getType(DefReg) == LLT::scalar(1))
              setRBDef(MI, DefOP, VccRB);
            else
              setRBDef(MI, DefOP, VgprRB);
          }
        }
      }
    }
  }

  // Temporal divergence copy: COPY to vgpr with implicit use of $exec inside of
  // the cycle
  // Note: uniformity analysis does not consider that registers with vgpr def
  // are divergent (you can have uniform value in vgpr).
  // - TODO: implicit use of $exec could be implemented as indicator that
  //   instruction is divergent
  bool isTemporalDivergenceCopy(Register Reg) {
    MachineInstr *MI = MRI.getVRegDef(Reg);
    if (!MI->isCopy())
      return false;

    for (auto Op : MI->implicit_operands()) {
      if (!Op.isReg())
        continue;

      if (Op.getReg() == TRI.getExec()) {
        return true;
      }
    }

    return false;
  }

  void constrainBanksOnUses() {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!shouldRegBankSelect(MI))
          continue;

        // Copies can have register class on use registers.
        if (MI.isCopy())
          continue;

        for (MachineOperand &UseOP : MI.uses()) {
          auto MaybeUseReg = tryGetVReg(UseOP);
          if (!MaybeUseReg)
            continue;
          Register UseReg = *MaybeUseReg;

          // UseReg already has register bank.
          if (MRI.getRegBankOrNull(UseReg))
            continue;

          if (!isTemporalDivergenceCopy(UseReg) &&
              (MUI.isUniform(UseReg) || ILMA.isS32S64LaneMask(UseReg))) {
            constrainRBUse(MI, UseOP, SgprRB);
          } else {
            if (MRI.getType(UseReg) == LLT::scalar(1))
              constrainRBUse(MI, UseOP, VccRB);
            else
              constrainRBUse(MI, UseOP, VgprRB);
          }
        }
      }
    }
  }
};

bool AMDGPUStandaloneRegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  AMDGPU::IntrinsicLaneMaskAnalyzer ILMA(MF);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI =
      *MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  const RegisterBankInfo &RBI = *MF.getSubtarget().getRegBankInfo();

  MachineIRBuilder B(MF);
  RegBankSelectHelper RBSHelper(MF, B, MRI, ILMA, MUI, TRI, RBI);

  // Assign register banks to ALL def registers on G_ instructions.
  // Same for copies if they have no register bank or class on def.
  RBSHelper.assignBanksOnDefs();

  // At this point all virtual registers have register class or bank
  // - Defs of G_ instructions have register banks.
  // - Defs and uses of inst-selected instructions have register class.
  // - Defs and uses of copies can have either register class or bank
  // and most notably
  // - Uses of G_ instructions can have either register class or bank

  // Reassign uses of G_ instructions to only have register banks.
  RBSHelper.constrainBanksOnUses();

  // Defs and uses of G_ instructions have register banks exclusively.
  return true;
}
