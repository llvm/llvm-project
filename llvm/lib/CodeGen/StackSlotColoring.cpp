//===-- StackSlotColoring.cpp - Stack slot coloring pass. -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the stack slot coloring pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stackcoloring"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <vector>
using namespace llvm;

static cl::opt<bool>
DisableSharing("no-stack-slot-sharing",
             cl::init(false), cl::Hidden,
             cl::desc("Surpress slot sharing during stack coloring"));

STATISTIC(NumEliminated,   "Number of stack slots eliminated due to coloring");

namespace {
  class VISIBILITY_HIDDEN StackSlotColoring : public MachineFunctionPass {
    LiveStacks* LS;
    MachineFrameInfo *MFI;

    // SSIntervals - Spill slot intervals.
    std::vector<LiveInterval*> SSIntervals;

    // OrigAlignments - Alignments of stack objects before coloring.
    SmallVector<unsigned, 16> OrigAlignments;

    // OrigSizes - Sizess of stack objects before coloring.
    SmallVector<unsigned, 16> OrigSizes;

    // AllColors - If index is set, it's a spill slot, i.e. color.
    // FIXME: This assumes PEI locate spill slot with smaller indices
    // closest to stack pointer / frame pointer. Therefore, smaller
    // index == better color.
    BitVector AllColors;

    // NextColor - Next "color" that's not yet used.
    int NextColor;

    // UsedColors - "Colors" that have been assigned.
    BitVector UsedColors;

    // Assignments - Color to intervals mapping.
    SmallVector<SmallVector<LiveInterval*,4>,16> Assignments;

  public:
    static char ID; // Pass identification
    StackSlotColoring() : MachineFunctionPass(&ID), NextColor(-1) {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveStacks>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char* getPassName() const {
      return "Stack Slot Coloring";
    }

  private:
    bool InitializeSlots();
    bool OverlapWithAssignments(LiveInterval *li, int Color) const;
    int ColorSlot(LiveInterval *li);
    bool ColorSlots(MachineFunction &MF);
  };
} // end anonymous namespace

char StackSlotColoring::ID = 0;

static RegisterPass<StackSlotColoring>
X("stack-slot-coloring", "Stack Slot Coloring");

FunctionPass *llvm::createStackSlotColoringPass() {
  return new StackSlotColoring();
}

namespace {
  // IntervalSorter - Comparison predicate that sort live intervals by
  // their weight.
  struct IntervalSorter {
    bool operator()(LiveInterval* LHS, LiveInterval* RHS) const {
      return LHS->weight > RHS->weight;
    }
  };
}

/// InitializeSlots - Process all spill stack slot liveintervals and add them
/// to a sorted (by weight) list.
bool StackSlotColoring::InitializeSlots() {
  if (LS->getNumIntervals() < 2)
    return false;

  int LastFI = MFI->getObjectIndexEnd();
  OrigAlignments.resize(LastFI);
  OrigSizes.resize(LastFI);
  AllColors.resize(LastFI);
  UsedColors.resize(LastFI);
  Assignments.resize(LastFI);

  // Gather all spill slots into a list.
  for (LiveStacks::iterator i = LS->begin(), e = LS->end(); i != e; ++i) {
    LiveInterval &li = i->second;
    int FI = li.getStackSlotIndex();
    if (MFI->isDeadObjectIndex(FI))
      continue;
    SSIntervals.push_back(&li);
    OrigAlignments[FI] = MFI->getObjectAlignment(FI);
    OrigSizes[FI]      = MFI->getObjectSize(FI);
    AllColors.set(FI);
  }

  // Sort them by weight.
  std::stable_sort(SSIntervals.begin(), SSIntervals.end(), IntervalSorter());

  // Get first "color".
  NextColor = AllColors.find_first();
  return true;
}

/// OverlapWithAssignments - Return true if LiveInterval overlaps with any
/// LiveIntervals that have already been assigned to the specified color.
bool
StackSlotColoring::OverlapWithAssignments(LiveInterval *li, int Color) const {
  const SmallVector<LiveInterval*,4> &OtherLIs = Assignments[Color];
  for (unsigned i = 0, e = OtherLIs.size(); i != e; ++i) {
    LiveInterval *OtherLI = OtherLIs[i];
    if (OtherLI->overlaps(*li))
      return true;
  }
  return false;
}

/// ColorSlot - Assign a "color" (stack slot) to the specified stack slot.
///
int StackSlotColoring::ColorSlot(LiveInterval *li) {
  int Color = -1;
  bool Share = false;
  if (!DisableSharing) {
    // Check if it's possible to reuse any of the used colors.
    Color = UsedColors.find_first();
    while (Color != -1) {
      if (!OverlapWithAssignments(li, Color)) {
        Share = true;
        ++NumEliminated;
        break;
      }
      Color = UsedColors.find_next(Color);
    }
  }

  // Assign it to the first available color (assumed to be the best) if it's
  // not possible to share a used color with other objects.
  if (!Share) {
    assert(NextColor != -1 && "No more spill slots?");
    Color = NextColor;
    UsedColors.set(Color);
    NextColor = AllColors.find_next(NextColor);
  }

  // Record the assignment.
  Assignments[Color].push_back(li);
  int FI = li->getStackSlotIndex();
  DOUT << "Assigning fi#" << FI << " to fi#" << Color << "\n";

  // Change size and alignment of the allocated slot. If there are multiple
  // objects sharing the same slot, then make sure the size and alignment
  // are large enough for all.
  unsigned Align = OrigAlignments[FI];
  if (!Share || Align > MFI->getObjectAlignment(Color))
    MFI->setObjectAlignment(Color, Align);
  int64_t Size = OrigSizes[FI];
  if (!Share || Size > MFI->getObjectSize(Color))
    MFI->setObjectSize(Color, Size);
  return Color;
}

/// Colorslots - Color all spill stack slots and rewrite all frameindex machine
/// operands in the function.
bool StackSlotColoring::ColorSlots(MachineFunction &MF) {
  unsigned NumObjs = MFI->getObjectIndexEnd();
  std::vector<int> SlotMapping(NumObjs, -1);

  bool Changed = false;
  for (unsigned i = 0, e = SSIntervals.size(); i != e; ++i) {
    LiveInterval *li = SSIntervals[i];
    int SS = li->getStackSlotIndex();
    int NewSS = ColorSlot(li);
    SlotMapping[SS] = NewSS;
    Changed |= (SS != NewSS);
  }

  if (!Changed)
    return false;

  // Rewrite all MO_FrameIndex operands.
  // FIXME: Need the equivalent of MachineRegisterInfo for frameindex operands.
  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
       MBB != E; ++MBB) {
    for (MachineBasicBlock::iterator MII = MBB->begin(), EE = MBB->end();
         MII != EE; ++MII) {
      MachineInstr &MI = *MII;
      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI.getOperand(i);
        if (!MO.isFI())
          continue;
        int FI = MO.getIndex();
        if (FI < 0)
          continue;
        int NewFI = SlotMapping[FI];
        if (NewFI == -1)
          continue;
        MO.setIndex(NewFI);

        // Update the MachineMemOperand for the new memory location.
        // FIXME: We need a better method of managing these too.
        SmallVector<MachineMemOperand, 2> MMOs(MI.memoperands_begin(),
                                               MI.memoperands_end());
        MI.clearMemOperands(MF);
        const Value *OldSV = PseudoSourceValue::getFixedStack(FI);
        for (unsigned i = 0, e = MMOs.size(); i != e; ++i) {
          if (MMOs[i].getValue() == OldSV) {
            MachineMemOperand MMO(PseudoSourceValue::getFixedStack(NewFI),
                                  MMOs[i].getFlags(), MMOs[i].getOffset(),
                                  MMOs[i].getSize(), MMOs[i].getAlignment());
            MI.addMemOperand(MF, MMO);
          } else
            MI.addMemOperand(MF, MMOs[i]);
        }
      }
    }
  }

  // Delete unused stack slots.
  while (NextColor != -1) {
    DOUT << "Removing unused stack object fi#" << NextColor << "\n";
    MFI->RemoveStackObject(NextColor);
    NextColor = AllColors.find_next(NextColor);
  }

  return true;
}

bool StackSlotColoring::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "********** Stack Slot Coloring **********\n";

  MFI = MF.getFrameInfo();
  LS = &getAnalysis<LiveStacks>();

  bool Changed = false;
  if (InitializeSlots())
    Changed = ColorSlots(MF);

  NextColor = -1;
  SSIntervals.clear();
  OrigAlignments.clear();
  OrigSizes.clear();
  AllColors.clear();
  UsedColors.clear();
  for (unsigned i = 0, e = Assignments.size(); i != e; ++i)
    Assignments[i].clear();
  Assignments.clear();

  return Changed;
}
