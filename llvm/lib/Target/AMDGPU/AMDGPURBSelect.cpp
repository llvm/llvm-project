//===-- AMDGPURBSelect.cpp ------------------------------------------------===//
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
/// can't be inst-selected. This is solved in RBLegalize.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURegisterBankInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "rb-select"

using namespace llvm;

namespace {

class AMDGPURBSelect : public MachineFunctionPass {
public:
  static char ID;

public:
  AMDGPURBSelect() : MachineFunctionPass(ID) {
    initializeAMDGPURBSelectPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU RB select"; }

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

INITIALIZE_PASS_BEGIN(AMDGPURBSelect, DEBUG_TYPE, "AMDGPU RB select", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(AMDGPURBSelect, DEBUG_TYPE, "AMDGPU RB select", false,
                    false)

char AMDGPURBSelect::ID = 0;

char &llvm::AMDGPURBSelectID = AMDGPURBSelect::ID;

FunctionPass *llvm::createAMDGPURBSelectPass() { return new AMDGPURBSelect(); }

bool shouldRBSelect(MachineInstr &MI) {
  if (isTargetSpecificOpcode(MI.getOpcode()) && !MI.isPreISelOpcode())
    return false;

  if (MI.getOpcode() == AMDGPU::PHI || MI.getOpcode() == AMDGPU::IMPLICIT_DEF)
    return false;

  if (MI.isInlineAsm())
    return false;

  return true;
}

void setRB(MachineInstr &MI, MachineOperand &DefOP, MachineIRBuilder B,
           MachineRegisterInfo &MRI, const RegisterBank &RB) {
  Register Reg = DefOP.getReg();
  // Register that already has Register class got it during pre-inst selection
  // of another instruction. Maybe cross bank copy was required so we insert a
  // copy trat can be removed later. This simplifies post-rb-legalize artifact
  // combiner and avoids need to special case some patterns.
  if (MRI.getRegClassOrNull(Reg)) {
    LLT Ty = MRI.getType(Reg);
    Register NewReg = MRI.createVirtualRegister({&RB, Ty});
    DefOP.setReg(NewReg);

    auto &MBB = *MI.getParent();
    B.setInsertPt(MBB, MI.isPHI() ? MBB.getFirstNonPHI()
                                  : std::next(MI.getIterator()));
    B.buildCopy(Reg, NewReg);

    // The problem was discoverd for uniform S1 that was used as both
    // lane mask(vcc) and regular sgpr S1.
    // - lane-mask(vcc) use was by si_if, this use is divergent and requires
    //   non-trivial sgpr-S1-to-vcc copy. But pre-inst-selection of si_if sets
    //   sreg_64_xexec(S1) on def of uniform S1 making it lane-mask.
    // - the regular regular sgpr S1(uniform) instruction is now broken since
    //   it uses sreg_64_xexec(S1) which is divergent.

    // "Clear" reg classes from uses on generic instructions and but register
    // banks instead.
    for (auto &UseMI : MRI.use_instructions(Reg)) {
      if (shouldRBSelect(UseMI)) {
        for (MachineOperand &Op : UseMI.operands()) {
          if (Op.isReg() && Op.isUse() && Op.getReg() == Reg)
            Op.setReg(NewReg);
        }
      }
    }

  } else {
    MRI.setRegBank(Reg, RB);
  }
}

void setRBUse(MachineInstr &MI, MachineOperand &UseOP, MachineIRBuilder B,
              MachineRegisterInfo &MRI, const RegisterBank &RB) {
  Register Reg = UseOP.getReg();

  LLT Ty = MRI.getType(Reg);
  Register NewReg = MRI.createVirtualRegister({&RB, Ty});
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

// Temporal divergence copy: COPY to vgpr with implicit use of $exec inside of
// the cycle
// Note: uniformity analysis does not consider that registers with vgpr def are
// divergent (you can have uniform value in vgpr).
// - TODO: implicit use of $exec could be implemented as indicator that
//   instruction is divergent
bool isTemporalDivergenceCopy(Register Reg, MachineRegisterInfo &MRI) {
  MachineInstr *MI = MRI.getVRegDef(Reg);
  if (MI->getOpcode() == AMDGPU::COPY) {
    for (auto Op : MI->implicit_operands()) {
      if (!Op.isReg())
        continue;
      Register Reg = Op.getReg();
      if (Reg == AMDGPU::EXEC) {
        return true;
      }
    }
  }

  return false;
}

Register getVReg(MachineOperand &Op) {
  if (!Op.isReg())
    return 0;

  Register Reg = Op.getReg();
  if (!Reg.isVirtual())
    return 0;

  return Reg;
}

bool AMDGPURBSelect::runOnMachineFunction(MachineFunction &MF) {
  MachineUniformityInfo &MUI =
      getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  AMDGPU::IntrinsicLaneMaskAnalyzer ILMA(MF);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RegisterBankInfo &RBI = *MF.getSubtarget().getRegBankInfo();

  MachineIRBuilder B(MF);

  // Assign register banks to ALL def registers on G_ instructions.
  // Same for copies if they have no register bank or class on def.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!shouldRBSelect(MI))
        continue;

      for (MachineOperand &DefOP : MI.defs()) {
        Register DefReg = getVReg(DefOP);
        if (!DefReg)
          continue;

        // Copies can have register class on def registers.
        if (MI.isCopy() && MRI.getRegClassOrNull(DefReg)) {
          continue;
        }

        if (MUI.isUniform(DefReg) || ILMA.isS32S64LaneMask(DefReg)) {
          setRB(MI, DefOP, B, MRI, RBI.getRegBank(AMDGPU::SGPRRegBankID));
        } else {
          if (MRI.getType(DefReg) == LLT::scalar(1))
            setRB(MI, DefOP, B, MRI, RBI.getRegBank(AMDGPU::VCCRegBankID));
          else
            setRB(MI, DefOP, B, MRI, RBI.getRegBank(AMDGPU::VGPRRegBankID));
        }
      }
    }
  }

  // At this point all virtual registers have register class or bank
  // - Defs of G_ instructions have register banks.
  // - Defs and uses of inst-selected instructions have register class.
  // - Defs and uses of copies can have either register class or bank
  // and most notably
  // - Uses of G_ instructions can have either register class or bank

  // Reassign uses of G_ instructions to only have register banks.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!shouldRBSelect(MI))
        continue;

      // Copies can have register class on use registers.
      if (MI.isCopy())
        continue;

      for (MachineOperand &UseOP : MI.uses()) {
        Register UseReg = getVReg(UseOP);
        if (!UseReg)
          continue;

        if (!MRI.getRegClassOrNull(UseReg))
          continue;

        if (!isTemporalDivergenceCopy(UseReg, MRI) &&
            (MUI.isUniform(UseReg) || ILMA.isS32S64LaneMask(UseReg))) {
          setRBUse(MI, UseOP, B, MRI, RBI.getRegBank(AMDGPU::SGPRRegBankID));
        } else {
          if (MRI.getType(UseReg) == LLT::scalar(1))
            setRBUse(MI, UseOP, B, MRI, RBI.getRegBank(AMDGPU::VCCRegBankID));
          else
            setRBUse(MI, UseOP, B, MRI, RBI.getRegBank(AMDGPU::VGPRRegBankID));
        }
      }
    }
  }

  // Defs and uses of G_ instructions have register banks exclusively.

  return true;
}
