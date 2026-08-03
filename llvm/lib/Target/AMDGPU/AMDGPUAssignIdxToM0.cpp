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

  // Only movrel takes its index from M0. Subtargets without it index with the
  // VGPR indexing mode instead, which AMDGPULowerVGPREncoding enables around
  // the move with s_set_gpr_idx_on, reading the index straight out of its SGPR.
  if (!ST.hasMovrel())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      auto *LdSt = dyn_cast<AMDGPUMI::VLoadStoreIdxInst>(&MI);
      if (!LdSt)
        continue;

      // The operand class of the index is a register class, so it never holds
      // an immediate that would have to be moved into M0 separately.
      MachineOperand &IdxOp = LdSt->getIdxOp();
      assert(IdxOp.isReg() && "VGPR-memory index must be a register");

      assert(!MI.isBundled());

      // Add a copy from the index register to M0 and rewrite MI to read M0. The
      // pseudo goes on declaring that it writes M0: it stands for the whole
      // sequence, and this copy is the write it describes.
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
    // This is required lowering, not an optimization: without the copy to M0
    // the movrel that AMDGPULowerVGPREncoding emits later reads a stale index.
    // It therefore must not be skipped for optnone functions.
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
