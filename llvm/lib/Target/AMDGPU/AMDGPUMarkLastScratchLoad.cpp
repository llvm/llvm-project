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
// TODO: Handle general stack accesses not just spilling.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineOperand.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-last-scratch-load"

namespace {

class AMDGPUMarkLastScratchLoad {
private:
  LiveStacks *LS = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *SI = nullptr;
  const SIInstrInfo *SII = nullptr;

public:
  AMDGPUMarkLastScratchLoad(LiveStacks *LS, LiveIntervals *LIS, SlotIndexes *SI)
      : LS(LS), LIS(LIS), SI(SI) {}
  bool run(MachineFunction &MF);
};

class AMDGPUMarkLastScratchLoadLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUMarkLastScratchLoadLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUMarkLastScratchLoadLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<LiveStacksWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "AMDGPU Mark Last Scratch Load";
  }
};

} // end anonymous namespace

bool AMDGPUMarkLastScratchLoadLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &LS = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto &SI = getAnalysis<SlotIndexesWrapperPass>().getSI();

  return AMDGPUMarkLastScratchLoad(&LS, &LIS, &SI).run(MF);
}

PreservedAnalyses
AMDGPUMarkLastScratchLoadPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  auto &LS = MFAM.getResult<LiveStacksAnalysis>(MF);
  auto &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  auto &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  AMDGPUMarkLastScratchLoad(&LS, &LIS, &SI).run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUMarkLastScratchLoad::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() < AMDGPUSubtarget::GFX12)
    return false;

  SII = ST.getInstrInfo();
  SlotIndexes &Slots = *LIS->getSlotIndexes();

  const unsigned NumSlots = LS->getNumIntervals();
  if (NumSlots == 0) {
    LLVM_DEBUG(dbgs() << "No live slots, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << LS->getNumIntervals() << " intervals\n");

  bool Changed = false;

  for (auto &[SS, LI] : *LS) {
    for (const LiveRange::Segment &Segment : LI.segments) {

      // Ignore segments that run to the end of basic block because in this case
      // slot is still live at the end of it.
      if (Segment.end.isBlock())
        continue;

      const int FrameIndex = LI.reg().stackSlotIndex();
      MachineInstr *LastLoad = nullptr;

      MachineInstr *MISegmentEnd = SI->getInstructionFromIndex(Segment.end);

      // If there is no instruction at this slot because it was deleted take the
      // instruction from the next slot.
      if (!MISegmentEnd) {
        SlotIndex NextSlot = Slots.getNextNonNullIndex(Segment.end);
        MISegmentEnd = SI->getInstructionFromIndex(NextSlot);
      }

      MachineInstr *MISegmentStart = SI->getInstructionFromIndex(Segment.start);
      MachineBasicBlock *BB = MISegmentEnd->getParent();

      // Start iteration backwards from segment end until the start of basic
      // block or start of segment if it is in the same basic block.
      auto End = BB->rend();
      if (MISegmentStart && MISegmentStart->getParent() == BB)
        End = MISegmentStart->getReverseIterator();

      for (auto MI = MISegmentEnd->getReverseIterator(); MI != End; ++MI) {
        int LoadFI = 0;

        if (SII->isLoadFromStackSlot(*MI, LoadFI) && LoadFI == FrameIndex) {
          LastLoad = &*MI;
          break;
        }
      }

      if (LastLoad && !LastLoad->memoperands_empty()) {
        MachineMemOperand *MMO = *LastLoad->memoperands_begin();
        MMO->setFlags(MOLastUse);
        Changed = true;
        LLVM_DEBUG(dbgs() << "  Found last load: " << *LastLoad);
      }
    }
  }

  return Changed;
}

char AMDGPUMarkLastScratchLoadLegacy::ID = 0;

char &llvm::AMDGPUMarkLastScratchLoadID = AMDGPUMarkLastScratchLoadLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMarkLastScratchLoadLegacy, DEBUG_TYPE,
                      "AMDGPU Mark last scratch load", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_END(AMDGPUMarkLastScratchLoadLegacy, DEBUG_TYPE,
                    "AMDGPU Mark last scratch load", false, false)
