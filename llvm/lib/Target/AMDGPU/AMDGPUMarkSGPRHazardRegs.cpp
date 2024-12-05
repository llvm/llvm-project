//===- AMDGPUMarkSGPRHazardRegs.cpp - Annotate SGPRs used by VALU ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to mark SGPRs used by VALU.
///       Marks can be used during register allocation to reduce hazards.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMarkSGPRHazardRegs.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-sgpr-hazard-regs"

namespace {

class AMDGPUMarkSGPRHazardRegs {
public:
  AMDGPUMarkSGPRHazardRegs() {}
  bool run(MachineFunction &MF);
};

class AMDGPUMarkSGPRHazardRegsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUMarkSGPRHazardRegsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return AMDGPUMarkSGPRHazardRegs().run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

bool AMDGPUMarkSGPRHazardRegs::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasVALUReadSGPRHazard())
    return false;

  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  if (!TRI->getSGPRHazardAvoidanceStrategy(MF))
    return false;

  LLVM_DEBUG(dbgs() << "AMDGPUMarkSGPRHazardRegs: function " << MF.getName()
                    << "\n");

  const MachineRegisterInfo *MRI = &MF.getRegInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    const auto *RC = MRI->getRegClass(Reg);
    if (!RC || !TRI->isSGPRClass(RC))
      continue;
    for (const auto &MO : MRI->reg_nodbg_operands(Reg)) {
      const MachineInstr &MI = *MO.getParent();
      if (SIInstrInfo::isVALU(MI) && MO.isUse()) {
        FuncInfo->setFlag(Reg, AMDGPU::VirtRegFlag::SGPR_HAZARD_REG);
        break;
      }
    }
  }

  return true;
}

INITIALIZE_PASS(AMDGPUMarkSGPRHazardRegsLegacy, DEBUG_TYPE,
                "AMDGPU Mark Hazard SGPRs", false, false)

char AMDGPUMarkSGPRHazardRegsLegacy::ID = 0;

char &llvm::AMDGPUMarkSGPRHazardRegsLegacyID =
    AMDGPUMarkSGPRHazardRegsLegacy::ID;

PreservedAnalyses
AMDGPUMarkSGPRHazardRegsPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  AMDGPUMarkSGPRHazardRegs().run(MF);
  return PreservedAnalyses::all();
}
