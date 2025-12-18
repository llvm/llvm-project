//===-- AMDGPUExpandCondBarrier.cpp - Expand conditional barriers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands SI_COND_BARRIER pseudo instructions into conditional
// control flow with actual barrier instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-expand-cond-barrier"

class AMDGPUExpandCondBarrier : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUExpandCondBarrier() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Expand Conditional Barriers";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We modify the CFG, so don't call setPreservesCFG().
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool expandCondBarrier(MachineBasicBlock &MBB, MachineInstr &MI);
};

char AMDGPUExpandCondBarrier::ID = 0;

char &llvm::AMDGPUExpandCondBarrierID = AMDGPUExpandCondBarrier::ID;

INITIALIZE_PASS(AMDGPUExpandCondBarrier, DEBUG_TYPE,
                "Expand conditional barrier pseudo instructions", false, false)

bool AMDGPUExpandCondBarrier::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  // Collect all SI_COND_BARRIER instructions first to avoid iterator
  // invalidation.
  SmallVector<MachineInstr *, 4> CondBarriers;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::SI_COND_BARRIER) {
        CondBarriers.push_back(&MI);
      }
    }
  }

  // Process collected instructions.
  for (MachineInstr *MI : CondBarriers) {
    MachineBasicBlock *MBB = MI->getParent();
    Changed |= expandCondBarrier(*MBB, *MI);
  }

  return Changed;
}

bool AMDGPUExpandCondBarrier::expandCondBarrier(MachineBasicBlock &MBB,
                                                MachineInstr &MI) {
  MachineFunction *MF = MBB.getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  // Get operands from SI_COND_BARRIER instruction.
  unsigned Variant = MI.getOperand(0).getImm();
  unsigned Src1 = MI.getOperand(1).getReg(); // First register operand (unused for now)

  // Split current block only if there are instructions after MI.
  MachineBasicBlock *ContinueMBB = nullptr;
  if (!MBB.succ_empty() || std::next(MI.getIterator()) != MBB.end()) {
    ContinueMBB = MBB.splitAt(MI, false /*UpdateLiveIns*/);
  }

  // Build simple linear expansion with proper basic block structure:
  // Split current block if needed to create continuation block.
  if (!ContinueMBB) {
    ContinueMBB = MF->CreateMachineBasicBlock();
    MF->push_back(ContinueMBB);
  }

  // Create barrier basic block - insert it immediately after current block.
  // to ensure proper layout for fallthrough.
  MachineBasicBlock *BarrierMBB = MF->CreateMachineBasicBlock();

  // Insert BarrierMBB right after MBB for proper fallthrough layout.
  MachineFunction::iterator MBBI = MBB.getIterator();
  ++MBBI;
  MF->insert(MBBI, BarrierMBB);

  // 1. Conditional branch to skip barrier based on variant:
  //    Variant 0: Execute barrier when SCC=1, skip when SCC=0 (use S_CBRANCH_SCC0)
  //    Variant 1: Execute barrier when SCC=0, skip when SCC=1 (use S_CBRANCH_SCC1)
  unsigned BranchOpcode =
      (Variant == 0) ? AMDGPU::S_CBRANCH_SCC0 : AMDGPU::S_CBRANCH_SCC1;
  BuildMI(MBB, &MI, DL, TII->get(BranchOpcode)).addMBB(ContinueMBB);
  LLVM_DEBUG(dbgs() << "ExpandCondBarrier: Variant " << Variant
                    << " expansion\n");

  // 2. Insert barrier in fallthrough block.
  BuildMI(*BarrierMBB, BarrierMBB->end(), DL, TII->get(AMDGPU::S_BARRIER));

  // 3. Add explicit unconditional branch from barrier block to continuation.
  BuildMI(*BarrierMBB, BarrierMBB->end(), DL, TII->get(AMDGPU::S_BRANCH))
      .addMBB(ContinueMBB);

  // 4. Set up CFG with both paths.
  // For S_CBRANCH_SCC0: SCC=0 -> branch to ContinueMBB, SCC=1 -> fallthrough to
  // BarrierMBB
  MBB.addSuccessor(
      BarrierMBB); // Barrier path (implicit fallthrough when SCC=1)
  MBB.addSuccessor(
      ContinueMBB); // Skip barrier path (explicit branch target when SCC=0)
  BarrierMBB->addSuccessor(ContinueMBB); // Barrier to continue

  // Remove the pseudo-instruction.
  MI.eraseFromParent();

  return true;
}

FunctionPass *llvm::createAMDGPUExpandCondBarrierPass() {
  return new AMDGPUExpandCondBarrier();
}
