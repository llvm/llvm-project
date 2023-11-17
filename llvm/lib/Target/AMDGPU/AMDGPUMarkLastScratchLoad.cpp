//===-- AMDGPUMarkLastScratchLoad.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark scratch load/spill instructions which are guaranteed to be the last time
// this scratch slot is used so it can be evicted from caches.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineOperand.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-last-scratch-load"

namespace {

class AMDGPUMarkLastScratchLoad : public MachineFunctionPass {
private:
  LiveStacks *LS = nullptr;
  SlotIndexes *SI = nullptr;
  const SIInstrInfo *SII = nullptr;

public:
  static char ID;

  AMDGPUMarkLastScratchLoad() : MachineFunctionPass(ID) {
    initializeAMDGPUMarkLastScratchLoadPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<SlotIndexes>();
    AU.addRequired<LiveStacks>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "AMDGPU Mark Last Scratch Load";
  }
};

} // end anonymous namespace

bool AMDGPUMarkLastScratchLoad::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG({
    dbgs() << "********** Mark Last Scratch Load **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() < AMDGPUSubtarget::GFX12)
    return false;

  LS = &getAnalysis<LiveStacks>();
  SI = &getAnalysis<SlotIndexes>();
  SII = ST.getInstrInfo();

  const unsigned NumSlots = LS->getNumIntervals();
  if (NumSlots == 0) {
    LLVM_DEBUG(dbgs() << "No live slots, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << LS->getNumIntervals() << " intervals\n");

  bool Changed = false;

  for (auto &[SS, LI] : *LS) {
    LLVM_DEBUG(dbgs() << "Checking interval: " << LI << "\n");

    for (const LiveRange::Segment &Segment : LI.segments) {
      LLVM_DEBUG(dbgs() << "  Checking segment: " << Segment << "\n");

      // Ignore segments that run to the end of basic block because in this case
      // slot is still live at the end of it.
      if (Segment.end.isBlock())
        continue;

      const int FrameIndex = Register::stackSlot2Index(LI.reg());
      MachineInstr *LastLoad = nullptr;

      MachineInstr *MISegmentStart = SI->getInstructionFromIndex(Segment.start);
      MachineInstr *MISegmentEnd = SI->getInstructionFromIndex(Segment.end);
      MachineBasicBlock *BB = MISegmentEnd->getParent();

      // Start iteration backwards from segment end until the start of basic
      // block or start of segment if it is in the same basic block.
      auto End = BB->instr_rend();
      if (MISegmentStart && MISegmentStart->getParent() == BB)
        End = MISegmentStart->getReverseIterator();

      for (auto MI = MISegmentEnd->getReverseIterator(); MI != End; ++MI) {
        int LoadFI = 0;

        if (SII->isLoadFromStackSlot(*MI, LoadFI) && LoadFI == FrameIndex) {
          LastLoad = &*MI;
          break;
        }
      }

      if (LastLoad) {
        MachineOperand *LastUse =
            SII->getNamedOperand(*LastLoad, AMDGPU::OpName::last_use);
        assert(LastUse && "This instruction must have a last_use operand");
        LastUse->setImm(1);
        Changed = true;
        LLVM_DEBUG(dbgs() << "  Found last load: " << *LastLoad;);
      }
    }
  }

  return Changed;
}

char AMDGPUMarkLastScratchLoad::ID = 0;

char &llvm::AMDGPUMarkLastScratchLoadID = AMDGPUMarkLastScratchLoad::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMarkLastScratchLoad, DEBUG_TYPE,
                      "AMDGPU Mark last scratch load", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_END(AMDGPUMarkLastScratchLoad, DEBUG_TYPE,
                    "AMDGPU Mark last scratch load", false, false)
