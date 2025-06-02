//===-- AMDGPUReserveWWMRegs.cpp - Add WWM Regs to reserved regs list -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass should be invoked at the end of wwm-regalloc pipeline.
/// It identifies the WWM regs allocated during this pipeline and add
/// them to the list of reserved registers so that they won't be available for
/// per-thread VGPR allocation in the subsequent regalloc pipeline.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUReserveWWMRegs.h"
#include "AMDGPU.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-reserve-wwm-regs"

namespace {

class AMDGPUReserveWWMRegsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUReserveWWMRegsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Reserve WWM Registers";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class AMDGPUReserveWWMRegs {
public:
  bool run(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUReserveWWMRegsLegacy, DEBUG_TYPE,
                "AMDGPU Reserve WWM Registers", false, false)

char AMDGPUReserveWWMRegsLegacy::ID = 0;

char &llvm::AMDGPUReserveWWMRegsLegacyID = AMDGPUReserveWWMRegsLegacy::ID;

bool AMDGPUReserveWWMRegsLegacy::runOnMachineFunction(MachineFunction &MF) {
  return AMDGPUReserveWWMRegs().run(MF);
}

PreservedAnalyses
AMDGPUReserveWWMRegsPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &) {
  AMDGPUReserveWWMRegs().run(MF);
  // TODO: This should abandon RegisterClassInfo once it is turned into an
  // analysis.
  return PreservedAnalyses::all();
}

bool AMDGPUReserveWWMRegs::run(MachineFunction &MF) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != AMDGPU::SI_SPILL_S32_TO_VGPR &&
          Opc != AMDGPU::SI_RESTORE_S32_FROM_VGPR)
        continue;

      Register Reg = Opc == AMDGPU::SI_SPILL_S32_TO_VGPR
                         ? MI.getOperand(0).getReg()
                         : MI.getOperand(1).getReg();

      assert(Reg.isPhysical() &&
             "All WWM registers should have been allocated by now.");

      MFI->reserveWWMRegister(Reg);
      Changed |= true;
    }
  }

  // The renamable flag can't be set for reserved registers. Reset the flag for
  // MOs involving wwm-regs as they will be reserved during vgpr-regalloc
  // pipeline.
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (Register Reg : MFI->getWWMReservedRegs()) {
    for (MachineOperand &MO : MRI.reg_operands(Reg))
      MO.setIsRenamable(false);
  }

  // Now clear the NonWWMRegMask earlier set during wwm-regalloc.
  MFI->clearNonWWMRegAllocMask();

  return Changed;
}
