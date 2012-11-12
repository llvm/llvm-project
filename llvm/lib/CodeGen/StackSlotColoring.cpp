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

#define DEBUG_TYPE "stackslotcoloring"
#include "llvm/Module.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <vector>
using namespace llvm;

static cl::opt<bool>
DisableSharing("no-stack-slot-sharing",
             cl::init(false), cl::Hidden,
             cl::desc("Suppress slot sharing during stack coloring"));

static cl::opt<int> DCELimit("ssc-dce-limit", cl::init(-1), cl::Hidden);

STATISTIC(NumEliminated, "Number of stack slots eliminated due to coloring");
STATISTIC(NumDead,       "Number of trivially dead stack accesses eliminated");

namespace {
  class StackSlotColoring : public MachineFunctionPass {
    LiveStacks* LS;
    MachineFrameInfo *MFI;
    const TargetInstrInfo  *TII;
    const MachineLoopInfo *loopInfo;

    // SSIntervals - Spill slot intervals.
    std::vector<LiveInterval*> SSIntervals;

    // SSRefs - Keep a list of frame index references for each spill slot.
    SmallVector<SmallVector<MachineInstr*, 8>, 16> SSRefs;

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
    SmallVector<SmallVector<LiveInterval*,4>, 16> Assignments;

  public:
    static char ID; // Pass identification
    StackSlotColoring() :
      MachineFunctionPass(ID), NextColor(-1) {
        initializeStackSlotColoringPass(*PassRegistry::getPassRegistry());
      }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<SlotIndexes>();
      AU.addPreserved<SlotIndexes>();
      AU.addRequired<LiveStacks>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    void InitializeSlots();
    void ScanForSpillSlotRefs(MachineFunction &MF);
    bool OverlapWithAssignments(LiveInterval *li, int Color) const;
    int ColorSlot(LiveInterval *li);
    bool ColorSlots(MachineFunction &MF);
    void RewriteInstruction(MachineInstr *MI, int OldFI, int NewFI,
                            MachineFunction &MF);
    bool RemoveDeadStores(MachineBasicBlock* MBB);
  };
} // end anonymous namespace

char StackSlotColoring::ID = 0;
char &llvm::StackSlotColoringID = StackSlotColoring::ID;

INITIALIZE_PASS_BEGIN(StackSlotColoring, "stack-slot-coloring",
                "Stack Slot Coloring", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(StackSlotColoring, "stack-slot-coloring",
                "Stack Slot Coloring", false, false)

namespace {
  // IntervalSorter - Comparison predicate that sort live intervals by
  // their weight.
  struct IntervalSorter {
    bool operator()(LiveInterval* LHS, LiveInterval* RHS) const {
      return LHS->weight > RHS->weight;
    }
  };
}

/// ScanForSpillSlotRefs - Scan all the machine instructions for spill slot
/// references and update spill slot weights.
void StackSlotColoring::ScanForSpillSlotRefs(MachineFunction &MF) {
  SSRefs.resize(MFI->getObjectIndexEnd());

  // FIXME: Need the equivalent of MachineRegisterInfo for frameindex operands.
  for (MachineFunction::iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = &*MBBI;
    unsigned loopDepth = loopInfo->getLoopDepth(MBB);
    for (MachineBasicBlock::iterator MII = MBB->begin(), EE = MBB->end();
         MII != EE; ++MII) {
      MachineInstr *MI = &*MII;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isFI())
          continue;
        int FI = MO.getIndex();
        if (FI < 0)
          continue;
        if (!LS->hasInterval(FI))
          continue;
        LiveInterval &li = LS->getInterval(FI);
        if (!MI->isDebugValue())
          li.weight += LiveIntervals::getSpillWeight(false, true, loopDepth);
        SSRefs[FI].push_back(MI);
      }
    }
  }
}

/// InitializeSlots - Process all spill stack slot liveintervals and add them
/// to a sorted (by weight) list.
void StackSlotColoring::InitializeSlots() {
  int LastFI = MFI->getObjectIndexEnd();
  OrigAlignments.resize(LastFI);
  OrigSizes.resize(LastFI);
  AllColors.resize(LastFI);
  UsedColors.resize(LastFI);
  Assignments.resize(LastFI);

  // Gather all spill slots into a list.
  DEBUG(dbgs() << "Spill slot intervals:\n");
  for (LiveStacks::iterator i = LS->begin(), e = LS->end(); i != e; ++i) {
    LiveInterval &li = i->second;
    DEBUG(li.dump());
    int FI = TargetRegisterInfo::stackSlot2Index(li.reg);
    if (MFI->isDeadObjectIndex(FI))
      continue;
    SSIntervals.push_back(&li);
    OrigAlignments[FI] = MFI->getObjectAlignment(FI);
    OrigSizes[FI]      = MFI->getObjectSize(FI);
    AllColors.set(FI);
  }
  DEBUG(dbgs() << '\n');

  // Sort them by weight.
  std::stable_sort(SSIntervals.begin(), SSIntervals.end(), IntervalSorter());

  // Get first "color".
  NextColor = AllColors.find_first();
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
  int FI = TargetRegisterInfo::stackSlot2Index(li->reg);
  DEBUG(dbgs() << "Assigning fi#" << FI << " to fi#" << Color << "\n");

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
  SmallVector<int, 16> SlotMapping(NumObjs, -1);
  SmallVector<float, 16> SlotWeights(NumObjs, 0.0);
  SmallVector<SmallVector<int, 4>, 16> RevMap(NumObjs);
  BitVector UsedColors(NumObjs);

  DEBUG(dbgs() << "Color spill slot intervals:\n");
  bool Changed = false;
  for (unsigned i = 0, e = SSIntervals.size(); i != e; ++i) {
    LiveInterval *li = SSIntervals[i];
    int SS = TargetRegisterInfo::stackSlot2Index(li->reg);
    int NewSS = ColorSlot(li);
    assert(NewSS >= 0 && "Stack coloring failed?");
    SlotMapping[SS] = NewSS;
    RevMap[NewSS].push_back(SS);
    SlotWeights[NewSS] += li->weight;
    UsedColors.set(NewSS);
    Changed |= (SS != NewSS);
  }

  DEBUG(dbgs() << "\nSpill slots after coloring:\n");
  for (unsigned i = 0, e = SSIntervals.size(); i != e; ++i) {
    LiveInterval *li = SSIntervals[i];
    int SS = TargetRegisterInfo::stackSlot2Index(li->reg);
    li->weight = SlotWeights[SS];
  }
  // Sort them by new weight.
  std::stable_sort(SSIntervals.begin(), SSIntervals.end(), IntervalSorter());

#ifndef NDEBUG
  for (unsigned i = 0, e = SSIntervals.size(); i != e; ++i)
    DEBUG(SSIntervals[i]->dump());
  DEBUG(dbgs() << '\n');
#endif

  if (!Changed)
    return false;

  // Rewrite all MO_FrameIndex operands.
  SmallVector<SmallSet<unsigned, 4>, 4> NewDefs(MF.getNumBlockIDs());
  for (unsigned SS = 0, SE = SSRefs.size(); SS != SE; ++SS) {
    int NewFI = SlotMapping[SS];
    if (NewFI == -1 || (NewFI == (int)SS))
      continue;

    SmallVector<MachineInstr*, 8> &RefMIs = SSRefs[SS];
    for (unsigned i = 0, e = RefMIs.size(); i != e; ++i)
      RewriteInstruction(RefMIs[i], SS, NewFI, MF);
  }

  // Delete unused stack slots.
  while (NextColor != -1) {
    DEBUG(dbgs() << "Removing unused stack object fi#" << NextColor << "\n");
    MFI->RemoveStackObject(NextColor);
    NextColor = AllColors.find_next(NextColor);
  }

  return true;
}

/// RewriteInstruction - Rewrite specified instruction by replacing references
/// to old frame index with new one.
void StackSlotColoring::RewriteInstruction(MachineInstr *MI, int OldFI,
                                           int NewFI, MachineFunction &MF) {
  // Update the operands.
  for (unsigned i = 0, ee = MI->getNumOperands(); i != ee; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isFI())
      continue;
    int FI = MO.getIndex();
    if (FI != OldFI)
      continue;
    MO.setIndex(NewFI);
  }

  // Update the memory references. This changes the MachineMemOperands
  // directly. They may be in use by multiple instructions, however all
  // instructions using OldFI are being rewritten to use NewFI.
  const Value *OldSV = PseudoSourceValue::getFixedStack(OldFI);
  const Value *NewSV = PseudoSourceValue::getFixedStack(NewFI);
  for (MachineInstr::mmo_iterator I = MI->memoperands_begin(),
       E = MI->memoperands_end(); I != E; ++I)
    if ((*I)->getValue() == OldSV)
      (*I)->setValue(NewSV);
}


/// RemoveDeadStores - Scan through a basic block and look for loads followed
/// by stores.  If they're both using the same stack slot, then the store is
/// definitely dead.  This could obviously be much more aggressive (consider
/// pairs with instructions between them), but such extensions might have a
/// considerable compile time impact.
bool StackSlotColoring::RemoveDeadStores(MachineBasicBlock* MBB) {
  // FIXME: This could be much more aggressive, but we need to investigate
  // the compile time impact of doing so.
  bool changed = false;

  SmallVector<MachineInstr*, 4> toErase;

  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
       I != E; ++I) {
    if (DCELimit != -1 && (int)NumDead >= DCELimit)
      break;

    MachineBasicBlock::iterator NextMI = llvm::next(I);
    if (NextMI == MBB->end()) continue;

    int FirstSS, SecondSS;
    unsigned LoadReg = 0;
    unsigned StoreReg = 0;
    if (!(LoadReg = TII->isLoadFromStackSlot(I, FirstSS))) continue;
    if (!(StoreReg = TII->isStoreToStackSlot(NextMI, SecondSS))) continue;
    if (FirstSS != SecondSS || LoadReg != StoreReg || FirstSS == -1) continue;

    ++NumDead;
    changed = true;

    if (NextMI->findRegisterUseOperandIdx(LoadReg, true, 0) != -1) {
      ++NumDead;
      toErase.push_back(I);
    }

    toErase.push_back(NextMI);
    ++I;
  }

  for (SmallVector<MachineInstr*, 4>::iterator I = toErase.begin(),
       E = toErase.end(); I != E; ++I)
    (*I)->eraseFromParent();

  return changed;
}


bool StackSlotColoring::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
      dbgs() << "********** Stack Slot Coloring **********\n"
             << "********** Function: " << MF.getName() << '\n';
    });

  MFI = MF.getFrameInfo();
  TII = MF.getTarget().getInstrInfo();
  LS = &getAnalysis<LiveStacks>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  bool Changed = false;

  unsigned NumSlots = LS->getNumIntervals();
  if (NumSlots == 0)
    // Nothing to do!
    return false;

  // If there are calls to setjmp or sigsetjmp, don't perform stack slot
  // coloring. The stack could be modified before the longjmp is executed,
  // resulting in the wrong value being used afterwards. (See
  // <rdar://problem/8007500>.)
  if (MF.exposesReturnsTwice())
    return false;

  // Gather spill slot references
  ScanForSpillSlotRefs(MF);
  InitializeSlots();
  Changed = ColorSlots(MF);

  NextColor = -1;
  SSIntervals.clear();
  for (unsigned i = 0, e = SSRefs.size(); i != e; ++i)
    SSRefs[i].clear();
  SSRefs.clear();
  OrigAlignments.clear();
  OrigSizes.clear();
  AllColors.clear();
  UsedColors.clear();
  for (unsigned i = 0, e = Assignments.size(); i != e; ++i)
    Assignments[i].clear();
  Assignments.clear();

  if (Changed) {
    for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
      Changed |= RemoveDeadStores(I);
  }

  return Changed;
}
