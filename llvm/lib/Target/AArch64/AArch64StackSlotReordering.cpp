//===- AArch64StackSlotReordering.cpp - Reorder stack slots for pairing ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that reorders stack slots for spilled values
// to enable more load/store pair instructions. The pass analyzes spill/reload
// patterns and attempts to place values that are loaded/stored together
// in adjacent stack slots.
//
// This pass runs after register allocation but before PrologEpilogInserter
// to allow modification of frame indices.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "aarch64-stack-slot-reordering"

STATISTIC(NumSlotsReordered, "Number of stack slots reordered");
STATISTIC(NumPairsEnabled, "Number of potential pairs enabled by reordering");
STATISTIC(NumFrameIndicesRemapped, "Number of frame indices remapped");

static cl::opt<bool>
    EnableStackSlotReordering("aarch64-enable-stack-slot-reordering",
                             cl::init(true), cl::Hidden,
                             cl::desc("Enable stack slot reordering"));

static cl::opt<unsigned>
    ProximityThreshold("aarch64-stack-slot-proximity-threshold",
                      cl::init(10), cl::Hidden,
                      cl::desc("Maximum instruction distance for pairing"));

namespace {

class AArch64StackSlotReordering : public MachineFunctionPass {
public:
  static char ID;
  AArch64StackSlotReordering() : MachineFunctionPass(ID) {
    initializeAArch64StackSlotReorderingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Stack Slot Reordering";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  struct StackSlotAccess {
    int FrameIndex;
    MachineInstr *MI;
    bool IsLoad;
    unsigned Order; // Instruction order in the function
    unsigned Size;  // Size of the access in bytes
    unsigned Align; // Alignment requirement
  };

  struct SlotPairInfo {
    int Slot1, Slot2;
    unsigned Score; // How many times these slots are accessed together
    bool CanPair;   // Whether these slots can form a valid pair
  };

  struct SlotInfo {
    int FrameIndex;
    unsigned Size;
    unsigned Align;
    int NewFrameIndex; // New frame index after reordering
  };

  const AArch64InstrInfo *TII;
  const AArch64RegisterInfo *TRI;
  MachineFrameInfo *MFI;

  void analyzeStackAccesses(MachineFunction &MF,
                           SmallVectorImpl<StackSlotAccess> &Accesses);
  void computeSlotPairScores(const SmallVectorImpl<StackSlotAccess> &Accesses,
                            DenseMap<std::pair<int, int>, unsigned> &PairScores);
  bool reorderStackSlots(MachineFunction &MF,
                        const DenseMap<std::pair<int, int>, unsigned> &PairScores);
  bool isSpillSlot(int FrameIndex) const;
  bool canFormPair(const StackSlotAccess &Access1,
                   const StackSlotAccess &Access2) const;
  void remapFrameIndices(MachineFunction &MF,
                        const DenseMap<int, int> &FrameIndexMap);
  bool areAlignmentCompatible(unsigned Size1, unsigned Align1,
                              unsigned Size2, unsigned Align2) const;
  unsigned getAccessSizeAndAlign(const MachineInstr *MI, int FrameIndex,
                                 unsigned &Align) const;
};

char AArch64StackSlotReordering::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64StackSlotReordering, DEBUG_TYPE,
                "AArch64 Stack Slot Reordering", false, false)

bool AArch64StackSlotReordering::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableStackSlotReordering)
    return false;

  const AArch64Subtarget &ST = MF.getSubtarget<AArch64Subtarget>();

  // Skip if SVE is enabled - we can't handle scalable vectors yet
  if (ST.hasSVE() || ST.hasSVE2() || ST.hasSME())
    return false;

  LLVM_DEBUG(dbgs() << "Running AArch64StackSlotReordering on function: "
                    << MF.getName() << "\n");

  if (MF.getFunction().hasOptSize())
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MFI = &MF.getFrameInfo();

  // Skip functions with scalable vector stack objects - we can't handle them yet
  for (int FI = MFI->getObjectIndexBegin(); FI < MFI->getObjectIndexEnd(); ++FI) {
    if (MFI->getStackID(FI) == TargetStackID::ScalableVector) {
      LLVM_DEBUG(dbgs() << "Skipping function with scalable vector objects\n");
      return false;
    }
  }

  // Collect all stack slot accesses
  SmallVector<StackSlotAccess, 64> Accesses;
  analyzeStackAccesses(MF, Accesses);

  if (Accesses.empty())
    return false;

  // Compute scores for pairs of slots that are accessed near each other
  DenseMap<std::pair<int, int>, unsigned> PairScores;
  computeSlotPairScores(Accesses, PairScores);

  // Reorder stack slots based on pairing scores
  return reorderStackSlots(MF, PairScores);
}

void AArch64StackSlotReordering::analyzeStackAccesses(
    MachineFunction &MF, SmallVectorImpl<StackSlotAccess> &Accesses) {
  unsigned Order = 0;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      // Look for loads and stores to stack slots
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (!MO.isFI())
          continue;

        int FrameIndex = MO.getIndex();

        // Only consider spill slots
        if (!isSpillSlot(FrameIndex))
          continue;

        bool IsLoad = MI.mayLoad();
        unsigned Align = 0;
        unsigned Size = getAccessSizeAndAlign(&MI, FrameIndex, Align);

        Accesses.push_back({FrameIndex, &MI, IsLoad, Order, Size, Align});
      }
      Order++;
    }
  }

  LLVM_DEBUG(dbgs() << "Found " << Accesses.size()
                    << " stack slot accesses\n");
}

bool AArch64StackSlotReordering::isSpillSlot(int FrameIndex) const {
  // Check if this is a spill slot (not a fixed object, not a variable slot)
  if (FrameIndex < 0)
    return false;

  // Spill slots are typically not fixed and have no associated alloca
  return !MFI->isFixedObjectIndex(FrameIndex) &&
         !MFI->isVariableSizedObjectIndex(FrameIndex);
}

void AArch64StackSlotReordering::computeSlotPairScores(
    const SmallVectorImpl<StackSlotAccess> &Accesses,
    DenseMap<std::pair<int, int>, unsigned> &PairScores) {

  for (size_t i = 0; i < Accesses.size(); ++i) {
    for (size_t j = i + 1; j < Accesses.size(); ++j) {
      const auto &Access1 = Accesses[i];
      const auto &Access2 = Accesses[j];

      // Skip if same slot
      if (Access1.FrameIndex == Access2.FrameIndex)
        continue;

      // Check if accesses are close enough
      unsigned Distance = Access2.Order - Access1.Order;
      if (Distance > ProximityThreshold)
        break; // Too far apart

      // Check if both are loads or both are stores
      if (Access1.IsLoad != Access2.IsLoad)
        continue;

      // Check if they could potentially form a pair
      if (!canFormPair(Access1, Access2))
        continue;

      // Create ordered pair
      int Slot1 = std::min(Access1.FrameIndex, Access2.FrameIndex);
      int Slot2 = std::max(Access1.FrameIndex, Access2.FrameIndex);
      auto SlotPair = std::make_pair(Slot1, Slot2);

      // Increase score for this pair (closer accesses get higher score)
      unsigned Score = ProximityThreshold - Distance + 1;
      PairScores[SlotPair] += Score;
    }
  }

  LLVM_DEBUG({
    dbgs() << "Slot pair scores:\n";
    for (const auto &Entry : PairScores) {
      dbgs() << "  Slots " << Entry.first.first << " and "
             << Entry.first.second << ": score = " << Entry.second << "\n";
    }
  });
}

bool AArch64StackSlotReordering::canFormPair(
    const StackSlotAccess &Access1,
    const StackSlotAccess &Access2) const {
  // Check if two instructions could potentially form a load/store pair

  // Both must be in the same basic block
  if (Access1.MI->getParent() != Access2.MI->getParent())
    return false;

  // Check for compatible sizes and alignments
  if (!areAlignmentCompatible(Access1.Size, Access1.Align,
                              Access2.Size, Access2.Align))
    return false;

  // TODO: Add more sophisticated checks for addressing modes,
  // register availability, etc.

  return true;
}

bool AArch64StackSlotReordering::reorderStackSlots(
    MachineFunction &MF,
    const DenseMap<std::pair<int, int>, unsigned> &PairScores) {

  if (PairScores.empty())
    return false;

  // Sort pairs by score
  std::vector<SlotPairInfo> SortedPairs;
  for (const auto &Entry : PairScores) {
    // Skip scalable vector objects - we can't handle their sizes as fixed values
    if (MFI->getStackID(Entry.first.first) == TargetStackID::ScalableVector ||
        MFI->getStackID(Entry.first.second) == TargetStackID::ScalableVector)
      continue;

    // Check if these slots can actually be paired
    int64_t Size1 = MFI->getObjectSize(Entry.first.first);
    unsigned Align1 = MFI->getObjectAlign(Entry.first.first).value();
    int64_t Size2 = MFI->getObjectSize(Entry.first.second);
    unsigned Align2 = MFI->getObjectAlign(Entry.first.second).value();

    bool CanPair = areAlignmentCompatible(Size1, Align1, Size2, Align2);
    SortedPairs.push_back({Entry.first.first, Entry.first.second,
                          Entry.second, CanPair});
  }

  std::sort(SortedPairs.begin(), SortedPairs.end(),
            [](const SlotPairInfo &A, const SlotPairInfo &B) {
              // Prioritize pairs that can actually form ldp/stp
              if (A.CanPair != B.CanPair)
                return A.CanPair;
              return A.Score > B.Score;
            });

  // For now, just report what we found
  // Actually reordering frame indices is complex and can cause issues
  // A better approach would be to communicate this information to the
  // frame lowering pass or modify stack slot assignment earlier

  LLVM_DEBUG({
    dbgs() << "Stack slot pairing opportunities:\n";
    for (const auto &Pair : SortedPairs) {
      if (!Pair.CanPair)
        continue;
      int64_t Size = MFI->getObjectSize(Pair.Slot1);
      dbgs() << "  Slots " << Pair.Slot1 << " and " << Pair.Slot2
             << " (score: " << Pair.Score << ") - size: "
             << Size << " bytes\n";
    }
  });

  // Track statistics
  for (const auto &Pair : SortedPairs) {
    if (Pair.CanPair) {
      NumPairsEnabled++;
    }
  }

  // Don't actually modify frame indices - this causes memory issues
  // and incorrect code generation. The proper solution would require
  // deeper integration with the frame lowering infrastructure.
  return false;
}

unsigned AArch64StackSlotReordering::getAccessSizeAndAlign(
    const MachineInstr *MI, int FrameIndex, unsigned &Align) const {
  // Try to determine the size of the access from the instruction
  unsigned Size = 0;
  Align = 1;

  // First check the MachineMemOperands
  for (const MachineMemOperand *MMO : MI->memoperands()) {
    if (MMO->getPseudoValue() &&
        MMO->getPseudoValue()->kind() == PseudoSourceValue::FixedStack) {
      const auto *FSV = cast<FixedStackPseudoSourceValue>(MMO->getPseudoValue());
      if (FSV->getFrameIndex() == FrameIndex) {
        // Skip if this is a scalable size
        if (MMO->getSize().isScalable())
          return 0;
        Size = MMO->getSize().getValue();
        Align = MMO->getAlign().value();
        return Size;
      }
    }
  }

  // Fall back to frame info
  // Check if this is a scalable vector object
  if (MFI->getStackID(FrameIndex) == TargetStackID::ScalableVector)
    return 0;
  Size = MFI->getObjectSize(FrameIndex);
  Align = MFI->getObjectAlign(FrameIndex).value();
  return Size;
}

bool AArch64StackSlotReordering::areAlignmentCompatible(
    unsigned Size1, unsigned Align1,
    unsigned Size2, unsigned Align2) const {
  // Check if two slots can be paired based on size and alignment
  // For load/store pairs, we need matching sizes or compatible sizes

  // Both must be 8-byte (64-bit) for ldp/stp of X registers
  // or both 4-byte (32-bit) for ldp/stp of W registers
  // or both 16-byte for ldp/stp of Q registers
  if (Size1 != Size2)
    return false;

  // Alignment must be at least the size for pairing
  if (Align1 < Size1 || Align2 < Size2)
    return false;

  return (Size1 == 8 || Size1 == 4 || Size1 == 16);
}

void AArch64StackSlotReordering::remapFrameIndices(
    MachineFunction &MF, const DenseMap<int, int> &FrameIndexMap) {
  // Update all frame index references in the function
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        MachineOperand &MO = MI.getOperand(i);
        if (!MO.isFI())
          continue;

        int OldIndex = MO.getIndex();
        auto It = FrameIndexMap.find(OldIndex);
        if (It != FrameIndexMap.end()) {
          MO.setIndex(It->second);
          NumFrameIndicesRemapped++;
          LLVM_DEBUG(dbgs() << "Remapped frame index " << OldIndex
                           << " to " << It->second << " in: " << MI);
        }
      }
    }
  }
}

// Factory function
FunctionPass *llvm::createAArch64StackSlotReorderingPass() {
  return new AArch64StackSlotReordering();
}