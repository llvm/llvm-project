//===-- StackColoring.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements the stack-coloring optimization that looks for
// lifetime markers machine instructions (LIFESTART_BEGIN and LIFESTART_END),
// which represent the possible lifetime of stack slots. It attempts to
// merge disjoint stack slots and reduce the used stack space.
// NOTE: This pass is not StackSlotColoring, which optimizes spill slots.
//
// TODO: In the future we plan to improve stack coloring in the following ways:
// 1. Allow merging multiple small slots into a single larger slot at different
//    offsets.
// 2. Merge this pass with StackSlotColoring and allow merging of allocas with
//    spill slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stackcoloring"
#include "MachineTraceMetrics.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/DebugInfo.h"
#include "llvm/Instructions.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
DisableColoring("no-stack-coloring",
        cl::init(false), cl::Hidden,
        cl::desc("Disable stack coloring"));

/// The user may write code that uses allocas outside of the declared lifetime
/// zone. This can happen when the user returns a reference to a local
/// data-structure. We can detect these cases and decide not to optimize the
/// code. If this flag is enabled, we try to save the user.
static cl::opt<bool>
ProtectFromEscapedAllocas("protect-from-escaped-allocas",
        cl::init(false), cl::Hidden,
        cl::desc("Do not optimize lifetime zones that are broken"));

STATISTIC(NumMarkerSeen,  "Number of lifetime markers found.");
STATISTIC(StackSpaceSaved, "Number of bytes saved due to merging slots.");
STATISTIC(StackSlotMerged, "Number of stack slot merged.");
STATISTIC(EscapedAllocas,
          "Number of allocas that escaped the lifetime region");

//===----------------------------------------------------------------------===//
//                           StackColoring Pass
//===----------------------------------------------------------------------===//

namespace {
/// StackColoring - A machine pass for merging disjoint stack allocations,
/// marked by the LIFETIME_START and LIFETIME_END pseudo instructions.
class StackColoring : public MachineFunctionPass {
  MachineFrameInfo *MFI;
  MachineFunction *MF;

  /// A class representing liveness information for a single basic block.
  /// Each bit in the BitVector represents the liveness property
  /// for a different stack slot.
  struct BlockLifetimeInfo {
    /// Which slots BEGINs in each basic block.
    BitVector Begin;
    /// Which slots ENDs in each basic block.
    BitVector End;
    /// Which slots are marked as LIVE_IN, coming into each basic block.
    BitVector LiveIn;
    /// Which slots are marked as LIVE_OUT, coming out of each basic block.
    BitVector LiveOut;
  };

  /// Maps active slots (per bit) for each basic block.
  DenseMap<MachineBasicBlock*, BlockLifetimeInfo> BlockLiveness;

  /// Maps serial numbers to basic blocks.
  DenseMap<MachineBasicBlock*, int> BasicBlocks;
  /// Maps basic blocks to a serial number.
  SmallVector<MachineBasicBlock*, 8> BasicBlockNumbering;

  /// Maps liveness intervals for each slot.
  SmallVector<LiveInterval*, 16> Intervals;
  /// VNInfo is used for the construction of LiveIntervals.
  VNInfo::Allocator VNInfoAllocator;
  /// SlotIndex analysis object.
  SlotIndexes *Indexes;

  /// The list of lifetime markers found. These markers are to be removed
  /// once the coloring is done.
  SmallVector<MachineInstr*, 8> Markers;

  /// SlotSizeSorter - A Sort utility for arranging stack slots according
  /// to their size.
  struct SlotSizeSorter {
    MachineFrameInfo *MFI;
    SlotSizeSorter(MachineFrameInfo *mfi) : MFI(mfi) { }
    bool operator()(int LHS, int RHS) {
      // We use -1 to denote a uninteresting slot. Place these slots at the end.
      if (LHS == -1) return false;
      if (RHS == -1) return true;
      // Sort according to size.
      return MFI->getObjectSize(LHS) > MFI->getObjectSize(RHS);
  }
};

public:
  static char ID;
  StackColoring() : MachineFunctionPass(ID) {
    initializeStackColoringPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const;
  bool runOnMachineFunction(MachineFunction &MF);

private:
  /// Debug.
  void dump();

  /// Removes all of the lifetime marker instructions from the function.
  /// \returns true if any markers were removed.
  bool removeAllMarkers();

  /// Scan the machine function and find all of the lifetime markers.
  /// Record the findings in the BEGIN and END vectors.
  /// \returns the number of markers found.
  unsigned collectMarkers(unsigned NumSlot);

  /// Perform the dataflow calculation and calculate the lifetime for each of
  /// the slots, based on the BEGIN/END vectors. Set the LifetimeLIVE_IN and
  /// LifetimeLIVE_OUT maps that represent which stack slots are live coming
  /// in and out blocks.
  void calculateLocalLiveness();

  /// Construct the LiveIntervals for the slots.
  void calculateLiveIntervals(unsigned NumSlots);

  /// Go over the machine function and change instructions which use stack
  /// slots to use the joint slots.
  void remapInstructions(DenseMap<int, int> &SlotRemap);

  /// The input program may contain intructions which are not inside lifetime
  /// markers. This can happen due to a bug in the compiler or due to a bug in
  /// user code (for example, returning a reference to a local variable).
  /// This procedure checks all of the instructions in the function and
  /// invalidates lifetime ranges which do not contain all of the instructions
  /// which access that frame slot.
  void removeInvalidSlotRanges();

  /// Map entries which point to other entries to their destination.
  ///   A->B->C becomes A->C.
   void expungeSlotMap(DenseMap<int, int> &SlotRemap, unsigned NumSlots);
};
} // end anonymous namespace

char StackColoring::ID = 0;
char &llvm::StackColoringID = StackColoring::ID;

INITIALIZE_PASS_BEGIN(StackColoring,
                   "stack-coloring", "Merge disjoint stack slots", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_END(StackColoring,
                   "stack-coloring", "Merge disjoint stack slots", false, false)

void StackColoring::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<SlotIndexes>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void StackColoring::dump() {
  for (df_iterator<MachineFunction*> FI = df_begin(MF), FE = df_end(MF);
       FI != FE; ++FI) {
    unsigned Num = BasicBlocks[*FI];
    DEBUG(dbgs()<<"Inspecting block #"<<Num<<" ["<<FI->getName()<<"]\n");
    Num = 0;
    DEBUG(dbgs()<<"BEGIN  : {");
    for (unsigned i=0; i < BlockLiveness[*FI].Begin.size(); ++i)
      DEBUG(dbgs()<<BlockLiveness[*FI].Begin.test(i)<<" ");
    DEBUG(dbgs()<<"}\n");

    DEBUG(dbgs()<<"END    : {");
    for (unsigned i=0; i < BlockLiveness[*FI].End.size(); ++i)
      DEBUG(dbgs()<<BlockLiveness[*FI].End.test(i)<<" ");

    DEBUG(dbgs()<<"}\n");

    DEBUG(dbgs()<<"LIVE_IN: {");
    for (unsigned i=0; i < BlockLiveness[*FI].LiveIn.size(); ++i)
      DEBUG(dbgs()<<BlockLiveness[*FI].LiveIn.test(i)<<" ");

    DEBUG(dbgs()<<"}\n");
    DEBUG(dbgs()<<"LIVEOUT: {");
    for (unsigned i=0; i < BlockLiveness[*FI].LiveOut.size(); ++i)
      DEBUG(dbgs()<<BlockLiveness[*FI].LiveOut.test(i)<<" ");
    DEBUG(dbgs()<<"}\n");
  }
}

unsigned StackColoring::collectMarkers(unsigned NumSlot) {
  unsigned MarkersFound = 0;
  // Scan the function to find all lifetime markers.
  // NOTE: We use the a reverse-post-order iteration to ensure that we obtain a
  // deterministic numbering, and because we'll need a post-order iteration
  // later for solving the liveness dataflow problem.
  for (df_iterator<MachineFunction*> FI = df_begin(MF), FE = df_end(MF);
       FI != FE; ++FI) {

    // Assign a serial number to this basic block.
    BasicBlocks[*FI] = BasicBlockNumbering.size();
    BasicBlockNumbering.push_back(*FI);

    BlockLiveness[*FI].Begin.resize(NumSlot);
    BlockLiveness[*FI].End.resize(NumSlot);

    for (MachineBasicBlock::iterator BI = (*FI)->begin(), BE = (*FI)->end();
         BI != BE; ++BI) {

      if (BI->getOpcode() != TargetOpcode::LIFETIME_START &&
          BI->getOpcode() != TargetOpcode::LIFETIME_END)
        continue;

      Markers.push_back(BI);

      bool IsStart = BI->getOpcode() == TargetOpcode::LIFETIME_START;
      MachineOperand &MI = BI->getOperand(0);
      unsigned Slot = MI.getIndex();

      MarkersFound++;

      const AllocaInst *Allocation = MFI->getObjectAllocation(Slot);
      if (Allocation) {
        DEBUG(dbgs()<<"Found a lifetime marker for slot #"<<Slot<<
              " with allocation: "<< Allocation->getName()<<"\n");
      }

      if (IsStart) {
        BlockLiveness[*FI].Begin.set(Slot);
      } else {
        if (BlockLiveness[*FI].Begin.test(Slot)) {
          // Allocas that start and end within a single block are handled
          // specially when computing the LiveIntervals to avoid pessimizing
          // the liveness propagation.
          BlockLiveness[*FI].Begin.reset(Slot);
        } else {
          BlockLiveness[*FI].End.set(Slot);
        }
      }
    }
  }

  // Update statistics.
  NumMarkerSeen += MarkersFound;
  return MarkersFound;
}

void StackColoring::calculateLocalLiveness() {
  // Perform a standard reverse dataflow computation to solve for
  // global liveness.  The BEGIN set here is equivalent to KILL in the standard
  // formulation, and END is equivalent to GEN.  The result of this computation
  // is a map from blocks to bitvectors where the bitvectors represent which
  // allocas are live in/out of that block.
  SmallPtrSet<MachineBasicBlock*, 8> BBSet(BasicBlockNumbering.begin(),
                                           BasicBlockNumbering.end());
  unsigned NumSSMIters = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    ++NumSSMIters;

    SmallPtrSet<MachineBasicBlock*, 8> NextBBSet;

    for (SmallVector<MachineBasicBlock*, 8>::iterator
         PI = BasicBlockNumbering.begin(), PE = BasicBlockNumbering.end();
         PI != PE; ++PI) {

      MachineBasicBlock *BB = *PI;
      if (!BBSet.count(BB)) continue;

      BitVector LocalLiveIn;
      BitVector LocalLiveOut;

      // Forward propagation from begins to ends.
      for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
           PE = BB->pred_end(); PI != PE; ++PI)
        LocalLiveIn |= BlockLiveness[*PI].LiveOut;
      LocalLiveIn |= BlockLiveness[BB].End;
      LocalLiveIn.reset(BlockLiveness[BB].Begin);

      // Reverse propagation from ends to begins.
      for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI)
        LocalLiveOut |= BlockLiveness[*SI].LiveIn;
      LocalLiveOut |= BlockLiveness[BB].Begin;
      LocalLiveOut.reset(BlockLiveness[BB].End);

      LocalLiveIn |= LocalLiveOut;
      LocalLiveOut |= LocalLiveIn;

      // After adopting the live bits, we need to turn-off the bits which
      // are de-activated in this block.
      LocalLiveOut.reset(BlockLiveness[BB].End);
      LocalLiveIn.reset(BlockLiveness[BB].Begin);

      // If we have both BEGIN and END markers in the same basic block then
      // we know that the BEGIN marker comes after the END, because we already
      // handle the case where the BEGIN comes before the END when collecting
      // the markers (and building the BEGIN/END vectore).
      // Want to enable the LIVE_IN and LIVE_OUT of slots that have both
      // BEGIN and END because it means that the value lives before and after
      // this basic block.
      BitVector LocalEndBegin = BlockLiveness[BB].End;
      LocalEndBegin &= BlockLiveness[BB].Begin;
      LocalLiveIn |= LocalEndBegin;
      LocalLiveOut |= LocalEndBegin;

      if (LocalLiveIn.test(BlockLiveness[BB].LiveIn)) {
        changed = true;
        BlockLiveness[BB].LiveIn |= LocalLiveIn;

        for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
             PE = BB->pred_end(); PI != PE; ++PI)
          NextBBSet.insert(*PI);
      }

      if (LocalLiveOut.test(BlockLiveness[BB].LiveOut)) {
        changed = true;
        BlockLiveness[BB].LiveOut |= LocalLiveOut;

        for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
             SE = BB->succ_end(); SI != SE; ++SI)
          NextBBSet.insert(*SI);
      }
    }

    BBSet = NextBBSet;
  }// while changed.
}

void StackColoring::calculateLiveIntervals(unsigned NumSlots) {
  SmallVector<SlotIndex, 16> Starts;
  SmallVector<SlotIndex, 16> Finishes;

  // For each block, find which slots are active within this block
  // and update the live intervals.
  for (MachineFunction::iterator MBB = MF->begin(), MBBe = MF->end();
       MBB != MBBe; ++MBB) {
    Starts.clear();
    Starts.resize(NumSlots);
    Finishes.clear();
    Finishes.resize(NumSlots);

    // Create the interval for the basic blocks with lifetime markers in them.
    for (SmallVector<MachineInstr*, 8>::iterator it = Markers.begin(),
         e = Markers.end(); it != e; ++it) {
      MachineInstr *MI = *it;
      if (MI->getParent() != MBB)
        continue;

      assert((MI->getOpcode() == TargetOpcode::LIFETIME_START ||
              MI->getOpcode() == TargetOpcode::LIFETIME_END) &&
             "Invalid Lifetime marker");

      bool IsStart = MI->getOpcode() == TargetOpcode::LIFETIME_START;
      MachineOperand &Mo = MI->getOperand(0);
      int Slot = Mo.getIndex();
      assert(Slot >= 0 && "Invalid slot");

      SlotIndex ThisIndex = Indexes->getInstructionIndex(MI);

      if (IsStart) {
        if (!Starts[Slot].isValid() || Starts[Slot] > ThisIndex)
          Starts[Slot] = ThisIndex;
      } else {
        if (!Finishes[Slot].isValid() || Finishes[Slot] < ThisIndex)
          Finishes[Slot] = ThisIndex;
      }
    }

    // Create the interval of the blocks that we previously found to be 'alive'.
    BitVector Alive = BlockLiveness[MBB].LiveIn;
    Alive |= BlockLiveness[MBB].LiveOut;

    if (Alive.any()) {
      for (int pos = Alive.find_first(); pos != -1;
           pos = Alive.find_next(pos)) {
        if (!Starts[pos].isValid())
          Starts[pos] = Indexes->getMBBStartIdx(MBB);
        if (!Finishes[pos].isValid())
          Finishes[pos] = Indexes->getMBBEndIdx(MBB);
      }
    }

    for (unsigned i = 0; i < NumSlots; ++i) {
      assert(Starts[i].isValid() == Finishes[i].isValid() && "Unmatched range");
      if (!Starts[i].isValid())
        continue;

      assert(Starts[i] && Finishes[i] && "Invalid interval");
      VNInfo *ValNum = Intervals[i]->getValNumInfo(0);
      SlotIndex S = Starts[i];
      SlotIndex F = Finishes[i];
      if (S < F) {
        // We have a single consecutive region.
        Intervals[i]->addRange(LiveRange(S, F, ValNum));
      } else {
        // We have two non consecutive regions. This happens when
        // LIFETIME_START appears after the LIFETIME_END marker.
        SlotIndex NewStart = Indexes->getMBBStartIdx(MBB);
        SlotIndex NewFin = Indexes->getMBBEndIdx(MBB);
        Intervals[i]->addRange(LiveRange(NewStart, F, ValNum));
        Intervals[i]->addRange(LiveRange(S, NewFin, ValNum));
      }
    }
  }
}

bool StackColoring::removeAllMarkers() {
  unsigned Count = 0;
  for (unsigned i = 0; i < Markers.size(); ++i) {
    Markers[i]->eraseFromParent();
    Count++;
  }
  Markers.clear();

  DEBUG(dbgs()<<"Removed "<<Count<<" markers.\n");
  return Count;
}

void StackColoring::remapInstructions(DenseMap<int, int> &SlotRemap) {
  unsigned FixedInstr = 0;
  unsigned FixedMemOp = 0;
  unsigned FixedDbg = 0;
  MachineModuleInfo *MMI = &MF->getMMI();

  // Remap debug information that refers to stack slots.
  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
       VE = VMap.end(); VI != VE; ++VI) {
    const MDNode *Var = VI->first;
    if (!Var) continue;
    std::pair<unsigned, DebugLoc> &VP = VI->second;
    if (SlotRemap.count(VP.first)) {
      DEBUG(dbgs()<<"Remapping debug info for ["<<Var->getName()<<"].\n");
      VP.first = SlotRemap[VP.first];
      FixedDbg++;
    }
  }

  // Keep a list of *allocas* which need to be remapped.
  DenseMap<const AllocaInst*, const AllocaInst*> Allocas;
  for (DenseMap<int, int>::iterator it = SlotRemap.begin(),
       e = SlotRemap.end(); it != e; ++it) {
    const AllocaInst *From = MFI->getObjectAllocation(it->first);
    const AllocaInst *To = MFI->getObjectAllocation(it->second);
    assert(To && From && "Invalid allocation object");
    Allocas[From] = To;
  }

  // Remap all instructions to the new stack slots.
  MachineFunction::iterator BB, BBE;
  MachineBasicBlock::iterator I, IE;
  for (BB = MF->begin(), BBE = MF->end(); BB != BBE; ++BB)
    for (I = BB->begin(), IE = BB->end(); I != IE; ++I) {

      // Skip lifetime markers. We'll remove them soon.
      if (I->getOpcode() == TargetOpcode::LIFETIME_START ||
          I->getOpcode() == TargetOpcode::LIFETIME_END)
        continue;

      // Update the MachineMemOperand to use the new alloca.
      for (MachineInstr::mmo_iterator MM = I->memoperands_begin(),
           E = I->memoperands_end(); MM != E; ++MM) {
        MachineMemOperand *MMO = *MM;

        const Value *V = MMO->getValue();

        if (!V)
          continue;

        // Climb up and find the original alloca.
        V = GetUnderlyingObject(V);
        // If we did not find one, or if the one that we found is not in our
        // map, then move on.
        if (!V || !isa<AllocaInst>(V)) {
          // Clear mem operand since we don't know for sure that it doesn't
          // alias a merged alloca.
          MMO->setValue(0);
          continue;
        }
        const AllocaInst *AI= cast<AllocaInst>(V);
        if (!Allocas.count(AI))
          continue;

        MMO->setValue(Allocas[AI]);
        FixedMemOp++;
      }

      // Update all of the machine instruction operands.
      for (unsigned i = 0 ; i <  I->getNumOperands(); ++i) {
        MachineOperand &MO = I->getOperand(i);

        if (!MO.isFI())
          continue;
        int FromSlot = MO.getIndex();

        // Don't touch arguments.
        if (FromSlot<0)
          continue;

        // Only look at mapped slots.
        if (!SlotRemap.count(FromSlot))
          continue;

        // In a debug build, check that the instruction that we are modifying is
        // inside the expected live range. If the instruction is not inside
        // the calculated range then it means that the alloca usage moved
        // outside of the lifetime markers, or that the user has a bug.
        // NOTE: Alloca address calculations which happen outside the lifetime
        // zone are are okay, despite the fact that we don't have a good way
        // for validating all of the usages of the calculation.
#ifndef NDEBUG
        bool TouchesMemory = I->mayLoad() || I->mayStore();
        // If we *don't* protect the user from escaped allocas, don't bother
        // validating the instructions.
        if (!I->isDebugValue() && TouchesMemory && ProtectFromEscapedAllocas) {
          SlotIndex Index = Indexes->getInstructionIndex(I);
          LiveInterval *Interval = Intervals[FromSlot];
          assert(Interval->find(Index) != Interval->end() &&
               "Found instruction usage outside of live range.");
        }
#endif

        // Fix the machine instructions.
        int ToSlot = SlotRemap[FromSlot];
        MO.setIndex(ToSlot);
        FixedInstr++;
      }
    }

  DEBUG(dbgs()<<"Fixed "<<FixedMemOp<<" machine memory operands.\n");
  DEBUG(dbgs()<<"Fixed "<<FixedDbg<<" debug locations.\n");
  DEBUG(dbgs()<<"Fixed "<<FixedInstr<<" machine instructions.\n");
}

void StackColoring::removeInvalidSlotRanges() {
  MachineFunction::iterator BB, BBE;
  MachineBasicBlock::iterator I, IE;
  for (BB = MF->begin(), BBE = MF->end(); BB != BBE; ++BB)
    for (I = BB->begin(), IE = BB->end(); I != IE; ++I) {

      if (I->getOpcode() == TargetOpcode::LIFETIME_START ||
          I->getOpcode() == TargetOpcode::LIFETIME_END || I->isDebugValue())
        continue;

      // Some intervals are suspicious! In some cases we find address
      // calculations outside of the lifetime zone, but not actual memory
      // read or write. Memory accesses outside of the lifetime zone are a clear
      // violation, but address calculations are okay. This can happen when
      // GEPs are hoisted outside of the lifetime zone.
      // So, in here we only check instructions which can read or write memory.
      if (!I->mayLoad() && !I->mayStore())
        continue;

      // Check all of the machine operands.
      for (unsigned i = 0 ; i <  I->getNumOperands(); ++i) {
        MachineOperand &MO = I->getOperand(i);

        if (!MO.isFI())
          continue;

        int Slot = MO.getIndex();

        if (Slot<0)
          continue;

        if (Intervals[Slot]->empty())
          continue;

        // Check that the used slot is inside the calculated lifetime range.
        // If it is not, warn about it and invalidate the range.
        LiveInterval *Interval = Intervals[Slot];
        SlotIndex Index = Indexes->getInstructionIndex(I);
        if (Interval->find(Index) == Interval->end()) {
          Intervals[Slot]->clear();
          DEBUG(dbgs()<<"Invalidating range #"<<Slot<<"\n");
          EscapedAllocas++;
        }
      }
    }
}

void StackColoring::expungeSlotMap(DenseMap<int, int> &SlotRemap,
                                   unsigned NumSlots) {
  // Expunge slot remap map.
  for (unsigned i=0; i < NumSlots; ++i) {
    // If we are remapping i
    if (SlotRemap.count(i)) {
      int Target = SlotRemap[i];
      // As long as our target is mapped to something else, follow it.
      while (SlotRemap.count(Target)) {
        Target = SlotRemap[Target];
        SlotRemap[i] = Target;
      }
    }
  }
}

bool StackColoring::runOnMachineFunction(MachineFunction &Func) {
  DEBUG(dbgs() << "********** Stack Coloring **********\n"
               << "********** Function: "
               << ((const Value*)Func.getFunction())->getName() << '\n');
  MF = &Func;
  MFI = MF->getFrameInfo();
  Indexes = &getAnalysis<SlotIndexes>();
  BlockLiveness.clear();
  BasicBlocks.clear();
  BasicBlockNumbering.clear();
  Markers.clear();
  Intervals.clear();
  VNInfoAllocator.Reset();

  unsigned NumSlots = MFI->getObjectIndexEnd();

  // If there are no stack slots then there are no markers to remove.
  if (!NumSlots)
    return false;

  SmallVector<int, 8> SortedSlots;

  SortedSlots.reserve(NumSlots);
  Intervals.reserve(NumSlots);

  unsigned NumMarkers = collectMarkers(NumSlots);

  unsigned TotalSize = 0;
  DEBUG(dbgs()<<"Found "<<NumMarkers<<" markers and "<<NumSlots<<" slots\n");
  DEBUG(dbgs()<<"Slot structure:\n");

  for (int i=0; i < MFI->getObjectIndexEnd(); ++i) {
    DEBUG(dbgs()<<"Slot #"<<i<<" - "<<MFI->getObjectSize(i)<<" bytes.\n");
    TotalSize += MFI->getObjectSize(i);
  }

  DEBUG(dbgs()<<"Total Stack size: "<<TotalSize<<" bytes\n\n");

  // Don't continue because there are not enough lifetime markers, or the
  // stack is too small, or we are told not to optimize the slots.
  if (NumMarkers < 2 || TotalSize < 16 || DisableColoring) {
    DEBUG(dbgs()<<"Will not try to merge slots.\n");
    return removeAllMarkers();
  }

  for (unsigned i=0; i < NumSlots; ++i) {
    LiveInterval *LI = new LiveInterval(i, 0);
    Intervals.push_back(LI);
    LI->getNextValue(Indexes->getZeroIndex(), VNInfoAllocator);
    SortedSlots.push_back(i);
  }

  // Calculate the liveness of each block.
  calculateLocalLiveness();

  // Propagate the liveness information.
  calculateLiveIntervals(NumSlots);

  // Search for allocas which are used outside of the declared lifetime
  // markers.
  if (ProtectFromEscapedAllocas)
    removeInvalidSlotRanges();

  // Maps old slots to new slots.
  DenseMap<int, int> SlotRemap;
  unsigned RemovedSlots = 0;
  unsigned ReducedSize = 0;

  // Do not bother looking at empty intervals.
  for (unsigned I = 0; I < NumSlots; ++I) {
    if (Intervals[SortedSlots[I]]->empty())
      SortedSlots[I] = -1;
  }

  // This is a simple greedy algorithm for merging allocas. First, sort the
  // slots, placing the largest slots first. Next, perform an n^2 scan and look
  // for disjoint slots. When you find disjoint slots, merge the samller one
  // into the bigger one and update the live interval. Remove the small alloca
  // and continue.

  // Sort the slots according to their size. Place unused slots at the end.
  std::sort(SortedSlots.begin(), SortedSlots.end(), SlotSizeSorter(MFI));

  bool Chanded = true;
  while (Chanded) {
    Chanded = false;
    for (unsigned I = 0; I < NumSlots; ++I) {
      if (SortedSlots[I] == -1)
        continue;

      for (unsigned J=I+1; J < NumSlots; ++J) {
        if (SortedSlots[J] == -1)
          continue;

        int FirstSlot = SortedSlots[I];
        int SecondSlot = SortedSlots[J];
        LiveInterval *First = Intervals[FirstSlot];
        LiveInterval *Second = Intervals[SecondSlot];
        assert (!First->empty() && !Second->empty() && "Found an empty range");

        // Merge disjoint slots.
        if (!First->overlaps(*Second)) {
          Chanded = true;
          First->MergeRangesInAsValue(*Second, First->getValNumInfo(0));
          SlotRemap[SecondSlot] = FirstSlot;
          SortedSlots[J] = -1;
          DEBUG(dbgs()<<"Merging #"<<FirstSlot<<" and slots #"<<
                SecondSlot<<" together.\n");
          unsigned MaxAlignment = std::max(MFI->getObjectAlignment(FirstSlot),
                                           MFI->getObjectAlignment(SecondSlot));

          assert(MFI->getObjectSize(FirstSlot) >=
                 MFI->getObjectSize(SecondSlot) &&
                 "Merging a small object into a larger one");

          RemovedSlots+=1;
          ReducedSize += MFI->getObjectSize(SecondSlot);
          MFI->setObjectAlignment(FirstSlot, MaxAlignment);
          MFI->RemoveStackObject(SecondSlot);
        }
      }
    }
  }// While changed.

  // Record statistics.
  StackSpaceSaved += ReducedSize;
  StackSlotMerged += RemovedSlots;
  DEBUG(dbgs()<<"Merge "<<RemovedSlots<<" slots. Saved "<<
        ReducedSize<<" bytes\n");

  // Scan the entire function and update all machine operands that use frame
  // indices to use the remapped frame index.
  expungeSlotMap(SlotRemap, NumSlots);
  remapInstructions(SlotRemap);

  // Release the intervals.
  for (unsigned I = 0; I < NumSlots; ++I) {
    delete Intervals[I];
  }

  return removeAllMarkers();
}
