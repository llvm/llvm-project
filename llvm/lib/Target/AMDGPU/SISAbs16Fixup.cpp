//===-- SISAbs16Fixup.cpp - Lower I1 Copies -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass matches the pattern for 16-bit ABS instructions after they have
// been lowered to for execution on the Scalar Unit.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "si-abs16-pattern"

using namespace llvm;

static Register pierceCopies(Register R, MachineRegisterInfo& MRI) {
  MachineInstr *CopyMI = MRI.getVRegDef(R);
  while (CopyMI && CopyMI->getOpcode() == AMDGPU::COPY) {
    Register T = CopyMI->getOperand(1).getReg();
    if (!T.isVirtual())
      break;

    R = T;
    CopyMI = MRI.getVRegDef(R);
  }

  return R;
}

static MachineInstr *matchExpandAbsPattern(MachineInstr &MI,
                                           MachineRegisterInfo &MRI) {
  std::array<MachineInstr *, 2> SextInstructions;
  for (unsigned I = 0; I < SextInstructions.size(); I++)
  {
    SextInstructions[I] = MRI.getVRegDef(MI.getOperand(I + 1).getReg());
    if (SextInstructions[I]->getOpcode() != AMDGPU::S_SEXT_I32_I16)
      return nullptr;
  }

  Register AbsSource;
  MachineInstr* SubIns = nullptr;
  for (MachineInstr *SextMI : SextInstructions) {
    Register SextReg = SextMI->getOperand(1).getReg();
    MachineInstr* OperandMI = MRI.getVRegDef(SextReg);
    if (OperandMI->getOpcode() == AMDGPU::S_SUB_I32)
      if(!SubIns)
        SubIns = OperandMI;
      else
        return nullptr;
    else
      AbsSource = pierceCopies(SextReg,MRI);
  }

  if (!SubIns)
    return nullptr;

  if (MRI.getRegClass(AbsSource) != &AMDGPU::SGPR_32RegClass)
    return nullptr;

  MachineInstr &MustBeZero =
      *MRI.getVRegDef(pierceCopies(SubIns->getOperand(1).getReg(), MRI));
  if (MustBeZero.getOpcode() != AMDGPU::S_MOV_B32 ||
      MustBeZero.getOperand(1).getImm())
    return nullptr;

  if (pierceCopies(SubIns->getOperand(2).getReg(), MRI) != AbsSource)
    return nullptr;

  return MRI.getVRegDef(AbsSource);
}

static bool runSAbs16Fixup(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  bool Changed = false;
  
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      bool IsPositive = MI.getOpcode() == AMDGPU::S_MAX_I32;
      bool IsNegative = MI.getOpcode() == AMDGPU::S_MIN_I32;
      MachineInstr* AbsSourceMI;
      if ((!IsPositive && !IsNegative) ||
          !(AbsSourceMI = matchExpandAbsPattern(MI, MRI)))
        continue;

      Register SextDestReg =
          MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
      Register AbsDestReg =
          IsNegative ? MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass)
                     : MI.getOperand(0).getReg();

      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(AMDGPU::S_SEXT_I32_I16),
              SextDestReg)
          .addReg(AbsSourceMI->getOperand(0).getReg());
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(AMDGPU::S_ABS_I32), AbsDestReg)
          .addReg(SextDestReg);

      if(IsNegative)
        BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(AMDGPU::S_SUB_I32),
                MI.getOperand(0).getReg())
            .addImm(0)
            .addReg(AbsDestReg);

      MI.eraseFromParent();
      Changed = true;
    }

  return Changed;
}

PreservedAnalyses SISAbs16FixupPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runSAbs16Fixup(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: Probably preserves most.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

class SISAbs16FixupLegacy : public MachineFunctionPass {
public:
  static char ID;

  SISAbs16FixupLegacy() : MachineFunctionPass(ID) {
    initializeSISAbs16FixupLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI SAbs16 Fixup"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

bool SISAbs16FixupLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runSAbs16Fixup(MF);
}

INITIALIZE_PASS_BEGIN(SISAbs16FixupLegacy, DEBUG_TYPE, "SI SAbs16 Fixup",
                      false, false)
INITIALIZE_PASS_END(SISAbs16FixupLegacy, DEBUG_TYPE, "SI SAbs16 Fixup",
                    false, false)

char SISAbs16FixupLegacy::ID = 0;

char &llvm::SISAbs16FixupLegacyID = SISAbs16FixupLegacy::ID;

FunctionPass *llvm::createSISAbs16FixupLegacyPass() {
  return new SISAbs16FixupLegacy();
}
