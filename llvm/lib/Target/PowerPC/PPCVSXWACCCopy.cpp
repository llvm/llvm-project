//===--------- PPCVSXWACCCopy.cpp - VSX and WACC Copy Legalization --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass which deals with the complexity of generating legal VSX register
// copies to/from register classes which partially overlap with the VSX
// register file and combines the wacc/wacc_hi copies when needed.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCInstrInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-vsx-copy"

namespace {
// PPCVSXWACCCopy pass - For copies between VSX registers and non-VSX registers
// (Altivec and scalar floating-point registers), we need to transform the
// copies into subregister copies with other restrictions.
struct PPCVSXWACCCopy : public MachineFunctionPass {
  static char ID;
  PPCVSXWACCCopy() : MachineFunctionPass(ID) {}

  const TargetInstrInfo *TII;

  bool IsRegInClass(unsigned Reg, const TargetRegisterClass *RC,
                    MachineRegisterInfo &MRI) {
    if (Register::isVirtualRegister(Reg)) {
      return RC->hasSubClassEq(MRI.getRegClass(Reg));
    } else if (RC->contains(Reg)) {
      return true;
    }

    return false;
  }

  bool IsVSReg(unsigned Reg, MachineRegisterInfo &MRI) {
    return IsRegInClass(Reg, &PPC::VSRCRegClass, MRI);
  }

  bool IsVRReg(unsigned Reg, MachineRegisterInfo &MRI) {
    return IsRegInClass(Reg, &PPC::VRRCRegClass, MRI);
  }

  bool IsF8Reg(unsigned Reg, MachineRegisterInfo &MRI) {
    return IsRegInClass(Reg, &PPC::F8RCRegClass, MRI);
  }

  bool IsVSFReg(unsigned Reg, MachineRegisterInfo &MRI) {
    return IsRegInClass(Reg, &PPC::VSFRCRegClass, MRI);
  }

  bool IsVSSReg(unsigned Reg, MachineRegisterInfo &MRI) {
    return IsRegInClass(Reg, &PPC::VSSRCRegClass, MRI);
  }

protected:
  bool processBlock(MachineBasicBlock &MBB) {
    bool Changed = false;

    MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    for (MachineInstr &MI : MBB) {
      if (!MI.isFullCopy())
        continue;

      MachineOperand &DstMO = MI.getOperand(0);
      MachineOperand &SrcMO = MI.getOperand(1);

      if (IsVSReg(DstMO.getReg(), MRI) && !IsVSReg(SrcMO.getReg(), MRI)) {
        // This is a copy *to* a VSX register from a non-VSX register.
        Changed = true;

        const TargetRegisterClass *SrcRC = &PPC::VSLRCRegClass;
        assert((IsF8Reg(SrcMO.getReg(), MRI) || IsVSSReg(SrcMO.getReg(), MRI) ||
                IsVSFReg(SrcMO.getReg(), MRI)) &&
               "Unknown source for a VSX copy");

        Register NewVReg = MRI.createVirtualRegister(SrcRC);
        BuildMI(MBB, MI, MI.getDebugLoc(),
                TII->get(TargetOpcode::SUBREG_TO_REG), NewVReg)
            .addImm(1) // add 1, not 0, because there is no implicit clearing
                       // of the high bits.
            .add(SrcMO)
            .addImm(PPC::sub_64);

        // The source of the original copy is now the new virtual register.
        SrcMO.setReg(NewVReg);
      } else if (!IsVSReg(DstMO.getReg(), MRI) &&
                 IsVSReg(SrcMO.getReg(), MRI)) {
        // This is a copy *from* a VSX register to a non-VSX register.
        Changed = true;

        const TargetRegisterClass *DstRC = &PPC::VSLRCRegClass;
        assert((IsF8Reg(DstMO.getReg(), MRI) || IsVSFReg(DstMO.getReg(), MRI) ||
                IsVSSReg(DstMO.getReg(), MRI)) &&
               "Unknown destination for a VSX copy");

        // Copy the VSX value into a new VSX register of the correct subclass.
        Register NewVReg = MRI.createVirtualRegister(DstRC);
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY),
                NewVReg)
            .add(SrcMO);

        // Transform the original copy into a subregister extraction copy.
        SrcMO.setReg(NewVReg);
        SrcMO.setSubReg(PPC::sub_64);
      } else if (IsRegInClass(DstMO.getReg(), &PPC::WACC_HIRCRegClass, MRI) &&
                 IsRegInClass(SrcMO.getReg(), &PPC::WACCRCRegClass, MRI)) {
        // Matches the pattern:
        //   %a:waccrc = COPY %b.sub_wacc_hi:dmrrc
        //   %c:wacc_hirc = COPY %a:waccrc
        // And replaces it with:
        //   %c:wacc_hirc = COPY %b.sub_wacc_hi:dmrrc
        MachineInstr *DefMI = MRI.getUniqueVRegDef(SrcMO.getReg());
        if (!DefMI || !DefMI->isCopy())
          continue;

        MachineOperand &OrigSrc = DefMI->getOperand(1);

        if (!IsRegInClass(OrigSrc.getReg(), &PPC::DMRRCRegClass, MRI))
          continue;

        if (OrigSrc.getSubReg() != PPC::sub_wacc_hi)
          continue;

        // Rewrite the second copy to use the original register's subreg
        SrcMO.setReg(OrigSrc.getReg());
        SrcMO.setSubReg(PPC::sub_wacc_hi);
        Changed = true;

        // Remove the intermediate copy if safe
        if (MRI.use_nodbg_empty(DefMI->getOperand(0).getReg()))
          DefMI->eraseFromParent();
      }
    }

    return Changed;
  }

public:
  bool runOnMachineFunction(MachineFunction &MF) override {
    // If we don't have VSX on the subtarget, don't do anything.
    const PPCSubtarget &STI = MF.getSubtarget<PPCSubtarget>();
    if (!STI.hasVSX())
      return false;
    TII = STI.getInstrInfo();

    bool Changed = false;

    for (MachineBasicBlock &B : llvm::make_early_inc_range(MF))
      if (processBlock(B))
        Changed = true;

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

INITIALIZE_PASS(PPCVSXWACCCopy, DEBUG_TYPE, "PowerPC VSX Copy Legalization",
                false, false)

char PPCVSXWACCCopy::ID = 0;
FunctionPass *llvm::createPPCVSXWACCCopyPass() { return new PPCVSXWACCCopy(); }
