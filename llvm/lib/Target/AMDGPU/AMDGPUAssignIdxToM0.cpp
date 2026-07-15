//===- AMDGPUAssignIdxToM0.cpp - Copy VGPR-memory indices to M0 ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copy the register index of a VGPR "as memory" (address space 13)
/// V_LOAD_IDX / V_STORE_IDX pseudo into M0, which V_MOVREL[SD] reads when the
/// pseudo is lowered (see AMDGPULowerVGPREncoding). This runs before register
/// allocation so the copy to M0 is inserted while the index is still virtual.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-assign-idx-to-m0"

static bool assignIdxToM0(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMovrel())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      auto *LdSt = dyn_cast<AMDGPUMI::VLoadStoreIdxInst>(&MI);
      if (!LdSt)
        continue;

      MachineOperand &IdxOp = LdSt->getIdxOp();
      if (!IdxOp.isReg())
        continue;

      assert(!MI.isBundled());

      // Remove the implicit-def $m0 that instruction selection added (to pin a
      // divergent access inside its waterfall loop); M0 is written for real
      // below.
      int DefIdx = MI.findRegisterDefOperandIdx(AMDGPU::M0, /*TRI=*/nullptr);
      assert(DefIdx >= 0);
      MI.removeOperand(DefIdx);

      // Add a copy from the index register to M0 and rewrite MI to read M0.
      // No kill flag is set on the M0 use: kill flags are deprecated and are a
      // no-op on the reserved M0 register.
      BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY), AMDGPU::M0)
          .add(IdxOp);
      IdxOp.setReg(AMDGPU::M0);
      Changed = true;
    }
  }

  return Changed;
}

namespace {

class AMDGPUAssignIdxToM0Legacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUAssignIdxToM0Legacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return assignIdxToM0(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "AMDGPU Assign Idx To M0"; }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPUAssignIdxToM0Pass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  if (!assignIdxToM0(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUAssignIdxToM0Legacy::ID = 0;

char &llvm::AMDGPUAssignIdxToM0ID = AMDGPUAssignIdxToM0Legacy::ID;

INITIALIZE_PASS(AMDGPUAssignIdxToM0Legacy, DEBUG_TYPE,
                "AMDGPU Assign Idx To M0", false, false)
