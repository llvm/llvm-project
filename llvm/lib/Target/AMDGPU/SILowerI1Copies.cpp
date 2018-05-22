//===-- SILowerI1Copies.cpp - Lower I1 Copies -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// i1 values are usually inserted by the CFG Structurize pass and they are
/// unique in that they can be copied from VALU to SALU registers.
/// This is not possible for any other value type.  Since there are no
/// MOV instructions for i1, we to use V_CMP_* and V_CNDMASK to move the i1.
///
//===----------------------------------------------------------------------===//
//

#define DEBUG_TYPE "si-i1-copies"
#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPULaneDominator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

class SILowerI1Copies : public MachineFunctionPass {
public:
  static char ID;

public:
  SILowerI1Copies() : MachineFunctionPass(ID) {
    initializeSILowerI1CopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Lower i1 Copies"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SILowerI1Copies, DEBUG_TYPE,
                "SI Lower i1 Copies", false, false)

char SILowerI1Copies::ID = 0;

char &llvm::SILowerI1CopiesID = SILowerI1Copies::ID;

FunctionPass *llvm::createSILowerI1CopiesPass() {
  return new SILowerI1Copies();
}

bool SILowerI1Copies::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = &TII->getRegisterInfo();

  std::vector<unsigned> I1Defs;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      if (MI.getOpcode() == AMDGPU::IMPLICIT_DEF) {
        unsigned Reg = MI.getOperand(0).getReg();
        const TargetRegisterClass *RC = MRI.getRegClass(Reg);
        if (RC == &AMDGPU::VReg_1RegClass)
          MRI.setRegClass(Reg, &AMDGPU::SReg_64RegClass);
        continue;
      }

      if (MI.getOpcode() != AMDGPU::COPY)
        continue;

      const MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &Src = MI.getOperand(1);

      if (!TargetRegisterInfo::isVirtualRegister(Src.getReg()) ||
          !TargetRegisterInfo::isVirtualRegister(Dst.getReg()))
        continue;

      const TargetRegisterClass *DstRC = MRI.getRegClass(Dst.getReg());
      const TargetRegisterClass *SrcRC = MRI.getRegClass(Src.getReg());

      DebugLoc DL = MI.getDebugLoc();
      MachineInstr *DefInst = MRI.getUniqueVRegDef(Src.getReg());
      if (DstRC == &AMDGPU::VReg_1RegClass &&
          TRI->getCommonSubClass(SrcRC, &AMDGPU::SGPR_64RegClass)) {
        I1Defs.push_back(Dst.getReg());

        if (DefInst->getOpcode() == AMDGPU::S_MOV_B64) {
          if (DefInst->getOperand(1).isImm()) {
            I1Defs.push_back(Dst.getReg());

            int64_t Val = DefInst->getOperand(1).getImm();
            assert(Val == 0 || Val == -1);

            BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_MOV_B32_e32))
                .add(Dst)
                .addImm(Val);
            MI.eraseFromParent();
            continue;
          }
        }

        unsigned int TmpSrc = MRI.createVirtualRegister(&AMDGPU::SReg_64_XEXECRegClass);
        BuildMI(MBB, &MI, DL, TII->get(AMDGPU::COPY), TmpSrc)
            .add(Src);
        BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64))
            .add(Dst)
            .addImm(0)
            .addImm(-1)
            .addReg(TmpSrc);
        MI.eraseFromParent();
      } else if (TRI->getCommonSubClass(DstRC, &AMDGPU::SGPR_64RegClass) &&
                 SrcRC == &AMDGPU::VReg_1RegClass) {
        if (DefInst->getOpcode() == AMDGPU::V_CNDMASK_B32_e64 &&
            DefInst->getOperand(1).isImm() && DefInst->getOperand(2).isImm() &&
            DefInst->getOperand(1).getImm() == 0 &&
            DefInst->getOperand(2).getImm() != 0 &&
            DefInst->getOperand(3).isReg() &&
            TargetRegisterInfo::isVirtualRegister(
              DefInst->getOperand(3).getReg()) &&
            TRI->getCommonSubClass(
              MRI.getRegClass(DefInst->getOperand(3).getReg()),
              &AMDGPU::SGPR_64RegClass) &&
            AMDGPU::laneDominates(DefInst->getParent(), &MBB)) {
          BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_B64))
              .add(Dst)
              .addReg(AMDGPU::EXEC)
              .add(DefInst->getOperand(3));
        } else {
          BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CMP_NE_U32_e64))
              .add(Dst)
              .add(Src)
              .addImm(0);
        }
        MI.eraseFromParent();
      }
    }
  }

  for (unsigned Reg : I1Defs)
    MRI.setRegClass(Reg, &AMDGPU::VGPR_32RegClass);

  return false;
}
