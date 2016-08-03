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

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "stackcoloring"

static cl::opt<bool>
DisableColoring("no-stack-coloring",
        cl::init(false), cl::Hidden,
        cl::desc("Disable stack coloring"));

/// The user may write code that uses allocas outside of the declared lifetime
/// zone. This can happen when the user returns a reference to a local
/// data-structure. We can detect these cases and decide not to optimize the
/// code. If this flag is enabled, we try to save the user. This option
/// is treated as overriding LifetimeStartOnFirstUse below.
static cl::opt<bool>
ProtectFromEscapedAllocas("protect-from-escaped-allocas",
                          cl::init(false), cl::Hidden,
                          cl::desc("Do not optimize lifetime zones that "
                                   "are broken"));

/// Enable enhanced dataflow scheme for lifetime analysis (treat first
/// use of stack slot as start of slot lifetime, as opposed to looking
/// for LIFETIME_START marker). See "Implementation notes" below for
/// more info.
static cl::opt<bool>
LifetimeStartOnFirstUse("stackcoloring-lifetime-start-on-first-use",
        cl::init(true), cl::Hidden,
        cl::desc("Treat stack lifetimes as starting on first use, not on START marker."));


STATISTIC(NumMarkerSeen,  "Number of lifetime markers found.");
STATISTIC(StackSpaceSaved, "Number of bytes saved due to merging slots.");
STATISTIC(StackSlotMerged, "Number of stack slot merged.");
STATISTIC(EscapedAllocas, "Number of allocas that escaped the lifetime region");

//
// Implementation Notes:
// ---------------------
//
// Consider the following motivating example:
//
//     int foo() {
//       char b1[1024], b2[1024];
//       if (...) {
//         char b3[1024];
//         <uses of b1, b3>;
//         return x;
//       } else {
//         char b4[1024], b5[1024];
//         <uses of b2, b4, b5>;
//         return y;
//       }
//     }
//
// In the code above, "b3" and "b4" are declared in distinct lexical
// scopes, meaning that it is easy to prove that they can share the
// same stack slot. Variables "b1" and "b2" are declared in the same
// scope, meaning that from a lexical point of view, their lifetimes
// overlap. From a control flow pointer of view, however, the two
// variables are accessed in disjoint regions of the CFG, thus it
// should be possible for them to share the same stack slot. An ideal
// stack allocation for the function above would look like:
//
//     slot 0: b1, b2
//     slot 1: b3, b4
//     slot 2: b5
//
// Achieving this allocation is tricky, however, due to the way
// lifetime markers are inserted. Here is a simplified view of the
// control flow graph for the code above:
//
//                +------  block 0 -------+
//               0| LIFETIME_START b1, b2 |
//               1| <test 'if' condition> |
//                +-----------------------+
//                   ./              \.
//   +------  block 1 -------+   +------  block 2 -------+
//  2| LIFETIME_START b3     |  5| LIFETIME_START b4, b5 |
//  3| <uses of b1, b3>      |  6| <uses of b2, b4, b5>  |
//  4| LIFETIME_END b3       |  7| LIFETIME_END b4, b5   |
//   +-----------------------+   +-----------------------+
//                   \.              /.
//                +------  block 3 -------+
//               8| <cleanupcode>         |
//               9| LIFETIME_END b1, b2   |
//              10| return                |
//                +-----------------------+
//
// If we create live intervals for the variables above strictly based
// on the lifetime markers, we'll get the set of intervals on the
// left. If we ignore the lifetime start markers and instead treat a
// variable's lifetime as beginning with the first reference to the
// var, then we get the intervals on the right.
//
//            LIFETIME_START      First Use
//     b1:    [0,9]               [3,4] [8,9]
//     b2:    [0,9]               [6,9]
//     b3:    [2,4]               [3,4]
//     b4:    [5,7]               [6,7]
//     b5:    [5,7]               [6,7]
//
// For the intervals on the left, the best we can do is overlap two
// variables (b3 and b4, for example); this gives us a stack size of
// 4*1024 bytes, not ideal. When treating first-use as the start of a
// lifetime, we can additionally overlap b1 and b5, giving us a 3*1024
// byte stack (better).
//
// Relying entirely on first-use of stack slots is problematic,
// however, due to the fact that optimizations can sometimes migrate
// uses of a variable outside of its lifetime start/end region. Here
// is an example:
//
//     int bar() {
//       char b1[1024], b2[1024];
//       if (...) {
//         <uses of b2>
//         return y;
//       } else {
//         <uses of b1>
//         while (...) {
//           char b3[1024];
//           <uses of b3>
//         }
//       }
//     }
//
// Before optimization, the control flow graph for the code above
// might look like the following:
//
//                +------  block 0 -------+
//               0| LIFETIME_START b1, b2 |
//               1| <test 'if' condition> |
//                +-----------------------+
//                   ./              \.
//   +------  block 1 -------+    +------- block 2 -------+
//  2| <uses of b2>          |   3| <uses of b1>          |
//   +-----------------------+    +-----------------------+
//              |                            |
//              |                 +------- block 3 -------+ <-\.
//              |                4| <while condition>     |    |
//              |                 +-----------------------+    |
//              |               /          |                   |
//              |              /  +------- block 4 -------+
//              \             /  5| LIFETIME_START b3     |    |
//               \           /   6| <uses of b3>          |    |
//                \         /    7| LIFETIME_END b3       |    |
//                 \        |    +------------------------+    |
//                  \       |                 \                /
//                +------  block 5 -----+      \---------------
//               8| <cleanupcode>       |
//               9| LIFETIME_END b1, b2 |
//              10| return              |
//                +---------------------+
//
// During optimization, however, it can happen that an instruction
// computing an address in "b3" (for example, a loop-invariant GEP) is
// hoisted up out of the loop from block 4 to block 2.  [Note that
// this is not an actual load from the stack, only an instruction that
// computes the address to be loaded]. If this happens, there is now a
// path leading from the first use of b3 to the return instruction
// that does not encounter the b3 LIFETIME_END, hence b3's lifetime is
// now larger than if we were computing live intervals strictly based
// on lifetime markers. In the example above, this lengthened lifetime
// would mean that it would appear illegal to overlap b3 with b2.
//
// To deal with this such cases, the code in ::collectMarkers() below
// tries to identify "degenerate" slots -- those slots where on a single
// forward pass through the CFG we encounter a first reference to slot
// K before we hit the slot K lifetime start marker. For such slots,
// we fall back on using the lifetime start marker as the beginning of
// the variable's lifetime.  NB: with this implementation, slots can
// appear degenerate in cases where there is unstructured control flow:
//
//    if (q) goto mid;
//    if (x > 9) {
//         int b[100];
//         memcpy(&b[0], ...);
//    mid: b[k] = ...;
//         abc(&b);
//    }
//
// If in RPO ordering chosen to walk the CFG  we happen to visit the b[k]
// before visiting the memcpy block (which will contain the lifetime start
// for "b" then it will appear that 'b' has a degenerate lifetime.
//

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
  typedef DenseMap<const MachineBasicBlock*, BlockLifetimeInfo> LivenessMap;
  LivenessMap BlockLiveness;

  /// Maps serial numbers to basic blocks.
  DenseMap<const MachineBasicBlock*, int> BasicBlocks;
  /// Maps basic blocks to a serial number.
  SmallVector<const MachineBasicBlock*, 8> BasicBlockNumbering;

  /// Maps liveness intervals for each slot.
  SmallVector<std::unique_ptr<LiveInterval>, 16> Intervals;
  /// VNInfo is used for the construction of LiveIntervals.
  VNInfo::Allocator VNInfoAllocator;
  /// SlotIndex analysis object.
  SlotIndexes *Indexes;
  /// The stack protector object.
  StackProtector *SP;

  /// The list of lifetime markers found. These markers are to be removed
  /// once the coloring is done.
  SmallVector<MachineInstr*, 8> Markers;

  /// Record the FI slots for which we have seen some sort of
  /// lifetime marker (either start or end).
  BitVector InterestingSlots;

  /// FI slots that need to be handled conservatively (for these
  /// slots lifetime-start-on-first-use is disabled).
  BitVector ConservativeSlots;

  /// Number of iterations taken during data flow analysis.
  unsigned NumIterations;

public:
  static char ID;
  StackColoring() : MachineFunctionPass(ID) {
    initializeStackColoringPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// Debug.
  void dump() const;
  void dumpIntervals() const;
  void dumpBB(MachineBasicBlock *MBB) const;
  void dumpBV(const char *tag, const BitVector &BV) const;

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

  /// Returns TRUE if we're using the first-use-begins-lifetime method for
  /// this slot (if FALSE, then the start marker is treated as start of lifetime).
  bool applyFirstUse(int Slot) {
    if (!LifetimeStartOnFirstUse || ProtectFromEscapedAllocas)
      return false;
    if (ConservativeSlots.test(Slot))
      return false;
    return true;
  }

  /// Examines the specified instruction and returns TRUE if the instruction
  /// represents the start or end of an interesting lifetime. The slot or slots
  /// starting or ending are added to the vector "slots" and "isStart" is set
  /// accordingly.
  /// \returns True if inst contains a lifetime start or end
  bool isLifetimeStartOrEnd(const MachineInstr &MI,
                            SmallVector<int, 4> &slots,
                            bool &isStart);

  /// Construct the LiveIntervals for the slots.
  void calculateLiveIntervals(unsigned NumSlots);

  /// Go over the machine function and change instructions which use stack
  /// slots to use the joint slots.
  void remapInstructions(DenseMap<int, int> &SlotRemap);

  /// The input program may contain instructions which are not inside lifetime
  /// markers. This can happen due to a bug in the compiler or due to a bug in
  /// user code (for example, returning a reference to a local variable).
  /// This procedure checks all of the instructions in the function and
  /// invalidates lifetime ranges which do not contain all of the instructions
  /// which access that frame slot.
  void removeInvalidSlotRanges();

  /// Map entries which point to other entries to their destination.
  ///   A->B->C becomes A->C.
  void expungeSlotMap(DenseMap<int, int> &SlotRemap, unsigned NumSlots);

  /// Used in collectMarkers
  typedef DenseMap<const MachineBasicBlock*, BitVector> BlockBitVecMap;
};
} // end anonymous namespace

char StackColoring::ID = 0;
char &llvm::StackColoringID = StackColoring::ID;

INITIALIZE_PASS_BEGIN(StackColoring,
                   "stack-coloring", "Merge disjoint stack slots", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(StackProtector)
INITIALIZE_PASS_END(StackColoring,
                   "stack-coloring", "Merge disjoint stack slots", false, false)

void StackColoring::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<SlotIndexes>();
  AU.addRequired<StackProtector>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

#ifndef NDEBUG

LLVM_DUMP_METHOD void StackColoring::dumpBV(const char *tag,
                                            const BitVector &BV) const {
  DEBUG(dbgs() << tag << " : { ");
  for (unsigned I = 0, E = BV.size(); I != E; ++I)
    DEBUG(dbgs() << BV.test(I) << " ");
  DEBUG(dbgs() << "}\n");
}

LLVM_DUMP_METHOD void StackColoring::dumpBB(MachineBasicBlock *MBB) const {
  LivenessMap::const_iterator BI = BlockLiveness.find(MBB);
  assert(BI != BlockLiveness.end() && "Block not found");
  const BlockLifetimeInfo &BlockInfo = BI->second;

  dumpBV("BEGIN", BlockInfo.Begin);
  dumpBV("END", BlockInfo.End);
  dumpBV("LIVE_IN", BlockInfo.LiveIn);
  dumpBV("LIVE_OUT", BlockInfo.LiveOut);
}

LLVM_DUMP_METHOD void StackColoring::dump() const {
  for (MachineBasicBlock *MBB : depth_first(MF)) {
    DEBUG(dbgs() << "Inspecting block #" << MBB->getNumber() << " ["
                 << MBB->getName() << "]\n");
    DEBUG(dumpBB(MBB));
  }
}

LLVM_DUMP_METHOD void StackColoring::dumpIntervals() const {
  for (unsigned I = 0, E = Intervals.size(); I != E; ++I) {
    DEBUG(dbgs() << "Interval[" << I << "]:\n");
    DEBUG(Intervals[I]->dump());
  }
}

#endif // not NDEBUG

static inline int getStartOrEndSlot(const MachineInstr &MI)
{
  assert((MI.getOpcode() == TargetOpcode::LIFETIME_START ||
          MI.getOpcode() == TargetOpcode::LIFETIME_END) &&
         "Expected LIFETIME_START or LIFETIME_END op");
  const MachineOperand &MO = MI.getOperand(0);
  int Slot = MO.getIndex();
  if (Slot >= 0)
    return Slot;
  return -1;
}

//
// At the moment the only way to end a variable lifetime is with
// a VARIABLE_LIFETIME op (which can't contain a start). If things
// change and the IR allows for a single inst that both begins
// and ends lifetime(s), this interface will need to be reworked.
//
bool StackColoring::isLifetimeStartOrEnd(const MachineInstr &MI,
                                         SmallVector<int, 4> &slots,
                                         bool &isStart)
{
  if (MI.getOpcode() == TargetOpcode::LIFETIME_START ||
      MI.getOpcode() == TargetOpcode::LIFETIME_END) {
    int Slot = getStartOrEndSlot(MI);
    if (Slot < 0)
      return false;
    if (!InterestingSlots.test(Slot))
      return false;
    slots.push_back(Slot);
    if (MI.getOpcode() == TargetOpcode::LIFETIME_END) {
      isStart = false;
      return true;
    }
    if (! applyFirstUse(Slot)) {
      isStart = true;
      return true;
    }
  } else if (LifetimeStartOnFirstUse && !ProtectFromEscapedAllocas) {
    if (! MI.isDebugValue()) {
      bool found = false;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isFI())
          continue;
        int Slot = MO.getIndex();
        if (Slot<0)
          continue;
        if (InterestingSlots.test(Slot) && applyFirstUse(Slot)) {
          slots.push_back(Slot);
          found = true;
        }
      }
      if (found) {
        isStart = true;
        return true;
      }
    }
  }
  return false;
}

unsigned StackColoring::collectMarkers(unsigned NumSlot)
{
  unsigned MarkersFound = 0;
  BlockBitVecMap SeenStartMap;
  InterestingSlots.clear();
  InterestingSlots.resize(NumSlot);
  ConservativeSlots.clear();
  ConservativeSlots.resize(NumSlot);

  // number of start and end lifetime ops for each slot
  SmallVector<int, 8> NumStartLifetimes(NumSlot, 0);
  SmallVector<int, 8> NumEndLifetimes(NumSlot, 0);

  // Step 1: collect markers and populate the "InterestingSlots"
  // and "ConservativeSlots" sets.
  for (MachineBasicBlock *MBB : depth_first(MF)) {

    // Compute the set of slots for which we've seen a START marker but have
    // not yet seen an END marker at this point in the walk (e.g. on entry
    // to this bb).
    BitVector BetweenStartEnd;
    BetweenStartEnd.resize(NumSlot);
    for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
      BlockBitVecMap::const_iterator I = SeenStartMap.find(*PI);
      if (I != SeenStartMap.end()) {
        BetweenStartEnd |= I->second;
      }
    }

    // Walk the instructions in the block to look for start/end ops.
    for (MachineInstr &MI : *MBB) {
      if (MI.getOpcode() == TargetOpcode::LIFETIME_START ||
          MI.getOpcode() == TargetOpcode::LIFETIME_END) {
        int Slot = getStartOrEndSlot(MI);
        if (Slot < 0)
          continue;
        InterestingSlots.set(Slot);
        if (MI.getOpcode() == TargetOpcode::LIFETIME_START) {
          BetweenStartEnd.set(Slot);
          NumStartLifetimes[Slot] += 1;
        } else {
          BetweenStartEnd.reset(Slot);
          NumEndLifetimes[Slot] += 1;
        }
        const AllocaInst *Allocation = MFI->getObjectAllocation(Slot);
        if (Allocation) {
          DEBUG(dbgs() << "Found a lifetime ");
          DEBUG(dbgs() << (MI.getOpcode() == TargetOpcode::LIFETIME_START
                               ? "start"
                               : "end"));
          DEBUG(dbgs() << " marker for slot #" << Slot);
          DEBUG(dbgs() << " with allocation: " << Allocation->getName()
                       << "\n");
        }
        Markers.push_back(&MI);
        MarkersFound += 1;
      } else {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isFI())
            continue;
          int Slot = MO.getIndex();
          if (Slot < 0)
            continue;
          if (! BetweenStartEnd.test(Slot)) {
            ConservativeSlots.set(Slot);
          }
        }
      }
    }
    BitVector &SeenStart = SeenStartMap[MBB];
    SeenStart |= BetweenStartEnd;
  }
  if (!MarkersFound) {
    return 0;
  }

  // PR27903: slots with multiple start or end lifetime ops are not
  // safe to enable for "lifetime-start-on-first-use".
  for (unsigned slot = 0; slot < NumSlot; ++slot)
    if (NumStartLifetimes[slot] > 1 || NumEndLifetimes[slot] > 1)
      ConservativeSlots.set(slot);
  DEBUG(dumpBV("Conservative slots", ConservativeSlots));

  // Step 2: compute begin/end sets for each block

  // NOTE: We use a reverse-post-order iteration to ensure that we obtain a
  // deterministic numbering, and because we'll need a post-order iteration
  // later for solving the liveness dataflow problem.
  for (MachineBasicBlock *MBB : depth_first(MF)) {

    // Assign a serial number to this basic block.
    BasicBlocks[MBB] = BasicBlockNumbering.size();
    BasicBlockNumbering.push_back(MBB);

    // Keep a reference to avoid repeated lookups.
    BlockLifetimeInfo &BlockInfo = BlockLiveness[MBB];

    BlockInfo.Begin.resize(NumSlot);
    BlockInfo.End.resize(NumSlot);

    SmallVector<int, 4> slots;
    for (MachineInstr &MI : *MBB) {
      bool isStart = false;
      slots.clear();
      if (isLifetimeStartOrEnd(MI, slots, isStart)) {
        if (!isStart) {
          assert(slots.size() == 1 && "unexpected: MI ends multiple slots");
          int Slot = slots[0];
          if (BlockInfo.Begin.test(Slot)) {
            BlockInfo.Begin.reset(Slot);
          }
          BlockInfo.End.set(Slot);
        } else {
          for (auto Slot : slots) {
            DEBUG(dbgs() << "Found a use of slot #" << Slot);
            DEBUG(dbgs() << " at BB#" << MBB->getNumber() << " index ");
            DEBUG(Indexes->getInstructionIndex(MI).print(dbgs()));
            const AllocaInst *Allocation = MFI->getObjectAllocation(Slot);
            if (Allocation) {
              DEBUG(dbgs() << " with allocation: "<< Allocation->getName());
            }
            DEBUG(dbgs() << "\n");
            if (BlockInfo.End.test(Slot)) {
              BlockInfo.End.reset(Slot);
            }
            BlockInfo.Begin.set(Slot);
          }
        }
      }
    }
  }

  // Update statistics.
  NumMarkerSeen += MarkersFound;
  return MarkersFound;
}

void StackColoring::calculateLocalLiveness()
{
  unsigned NumIters = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    ++NumIters;

    for (const MachineBasicBlock *BB : BasicBlockNumbering) {

      // Use an iterator to avoid repeated lookups.
      LivenessMap::iterator BI = BlockLiveness.find(BB);
      assert(BI != BlockLiveness.end() && "Block not found");
      BlockLifetimeInfo &BlockInfo = BI->second;

      // Compute LiveIn by unioning together the LiveOut sets of all preds.
      BitVector LocalLiveIn;
      for (MachineBasicBlock::const_pred_iterator PI = BB->pred_begin(),
           PE = BB->pred_end(); PI != PE; ++PI) {
        LivenessMap::const_iterator I = BlockLiveness.find(*PI);
        assert(I != BlockLiveness.end() && "Predecessor not found");
        LocalLiveIn |= I->second.LiveOut;
      }

      // Compute LiveOut by subtracting out lifetimes that end in this
      // block, then adding in lifetimes that begin in this block.  If
      // we have both BEGIN and END markers in the same basic block
      // then we know that the BEGIN marker comes after the END,
      // because we already handle the case where the BEGIN comes
      // before the END when collecting the markers (and building the
      // BEGIN/END vectors).
      BitVector LocalLiveOut = LocalLiveIn;
      LocalLiveOut.reset(BlockInfo.End);
      LocalLiveOut |= BlockInfo.Begin;

      // Update block LiveIn set, noting whether it has changed.
      if (LocalLiveIn.test(BlockInfo.LiveIn)) {
        changed = true;
        BlockInfo.LiveIn |= LocalLiveIn;
      }

      // Update block LiveOut set, noting whether it has changed.
      if (LocalLiveOut.test(BlockInfo.LiveOut)) {
        changed = true;
        BlockInfo.LiveOut |= LocalLiveOut;
      }
    }
  }// while changed.

  NumIterations = NumIters;
}

void StackColoring::calculateLiveIntervals(unsigned NumSlots) {
  SmallVector<SlotIndex, 16> Starts;
  SmallVector<SlotIndex, 16> Finishes;

  // For each block, find which slots are active within this block
  // and update the live intervals.
  for (const MachineBasicBlock &MBB : *MF) {
    Starts.clear();
    Starts.resize(NumSlots);
    Finishes.clear();
    Finishes.resize(NumSlots);

    // Create the interval for the basic blocks containing lifetime begin/end.
    for (const MachineInstr &MI : MBB) {

      SmallVector<int, 4> slots;
      bool IsStart = false;
      if (!isLifetimeStartOrEnd(MI, slots, IsStart))
        continue;
      SlotIndex ThisIndex = Indexes->getInstructionIndex(MI);
      for (auto Slot : slots) {
        if (IsStart) {
          if (!Starts[Slot].isValid() || Starts[Slot] > ThisIndex)
            Starts[Slot] = ThisIndex;
        } else {
          if (!Finishes[Slot].isValid() || Finishes[Slot] < ThisIndex)
            Finishes[Slot] = ThisIndex;
        }
      }
    }

    // Create the interval of the blocks that we previously found to be 'alive'.
    BlockLifetimeInfo &MBBLiveness = BlockLiveness[&MBB];
    for (int pos = MBBLiveness.LiveIn.find_first(); pos != -1;
         pos = MBBLiveness.LiveIn.find_next(pos)) {
      Starts[pos] = Indexes->getMBBStartIdx(&MBB);
    }
    for (int pos = MBBLiveness.LiveOut.find_first(); pos != -1;
         pos = MBBLiveness.LiveOut.find_next(pos)) {
      Finishes[pos] = Indexes->getMBBEndIdx(&MBB);
    }

    for (unsigned i = 0; i < NumSlots; ++i) {
      //
      // When LifetimeStartOnFirstUse is turned on, data flow analysis
      // is forward (from starts to ends), not bidirectional. A
      // consequence of this is that we can wind up in situations
      // where Starts[i] is invalid but Finishes[i] is valid and vice
      // versa. Example:
      //
      //     LIFETIME_START x
      //     if (...) {
      //       <use of x>
      //       throw ...;
      //     }
      //     LIFETIME_END x
      //     return 2;
      //
      //
      // Here the slot for "x" will not be live into the block
      // containing the "return 2" (since lifetimes start with first
      // use, not at the dominating LIFETIME_START marker).
      //
      if (Starts[i].isValid() && !Finishes[i].isValid()) {
        Finishes[i] = Indexes->getMBBEndIdx(&MBB);
      }
      if (!Starts[i].isValid())
        continue;

      assert(Starts[i] && Finishes[i] && "Invalid interval");
      VNInfo *ValNum = Intervals[i]->getValNumInfo(0);
      SlotIndex S = Starts[i];
      SlotIndex F = Finishes[i];
      if (S < F) {
        // We have a single consecutive region.
        Intervals[i]->addSegment(LiveInterval::Segment(S, F, ValNum));
      } else {
        // We have two non-consecutive regions. This happens when
        // LIFETIME_START appears after the LIFETIME_END marker.
        SlotIndex NewStart = Indexes->getMBBStartIdx(&MBB);
        SlotIndex NewFin = Indexes->getMBBEndIdx(&MBB);
        Intervals[i]->addSegment(LiveInterval::Segment(NewStart, F, ValNum));
        Intervals[i]->addSegment(LiveInterval::Segment(S, NewFin, ValNum));
      }
    }
  }
}

bool StackColoring::removeAllMarkers() {
  unsigned Count = 0;
  for (MachineInstr *MI : Markers) {
    MI->eraseFromParent();
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
  for (auto &VI : MMI->getVariableDbgInfo()) {
    if (!VI.Var)
      continue;
    if (SlotRemap.count(VI.Slot)) {
      DEBUG(dbgs() << "Remapping debug info for ["
                   << cast<DILocalVariable>(VI.Var)->getName() << "].\n");
      VI.Slot = SlotRemap[VI.Slot];
      FixedDbg++;
    }
  }

  // Keep a list of *allocas* which need to be remapped.
  DenseMap<const AllocaInst*, const AllocaInst*> Allocas;
  for (const std::pair<int, int> &SI : SlotRemap) {
    const AllocaInst *From = MFI->getObjectAllocation(SI.first);
    const AllocaInst *To = MFI->getObjectAllocation(SI.second);
    assert(To && From && "Invalid allocation object");
    Allocas[From] = To;

    // AA might be used later for instruction scheduling, and we need it to be
    // able to deduce the correct aliasing releationships between pointers
    // derived from the alloca being remapped and the target of that remapping.
    // The only safe way, without directly informing AA about the remapping
    // somehow, is to directly update the IR to reflect the change being made
    // here.
    Instruction *Inst = const_cast<AllocaInst *>(To);
    if (From->getType() != To->getType()) {
      BitCastInst *Cast = new BitCastInst(Inst, From->getType());
      Cast->insertAfter(Inst);
      Inst = Cast;
    }

    // Allow the stack protector to adjust its value map to account for the
    // upcoming replacement.
    SP->adjustForColoring(From, To);

    // The new alloca might not be valid in a llvm.dbg.declare for this
    // variable, so undef out the use to make the verifier happy.
    AllocaInst *FromAI = const_cast<AllocaInst *>(From);
    if (FromAI->isUsedByMetadata())
      ValueAsMetadata::handleRAUW(FromAI, UndefValue::get(FromAI->getType()));
    for (auto &Use : FromAI->uses()) {
      if (BitCastInst *BCI = dyn_cast<BitCastInst>(Use.get()))
        if (BCI->isUsedByMetadata())
          ValueAsMetadata::handleRAUW(BCI, UndefValue::get(BCI->getType()));
    }

    // Note that this will not replace uses in MMOs (which we'll update below),
    // or anywhere else (which is why we won't delete the original
    // instruction).
    FromAI->replaceAllUsesWith(Inst);
  }

  // Remap all instructions to the new stack slots.
  for (MachineBasicBlock &BB : *MF)
    for (MachineInstr &I : BB) {
      // Skip lifetime markers. We'll remove them soon.
      if (I.getOpcode() == TargetOpcode::LIFETIME_START ||
          I.getOpcode() == TargetOpcode::LIFETIME_END)
        continue;

      // Update the MachineMemOperand to use the new alloca.
      for (MachineMemOperand *MMO : I.memoperands()) {
        // FIXME: In order to enable the use of TBAA when using AA in CodeGen,
        // we'll also need to update the TBAA nodes in MMOs with values
        // derived from the merged allocas. When doing this, we'll need to use
        // the same variant of GetUnderlyingObjects that is used by the
        // instruction scheduler (that can look through ptrtoint/inttoptr
        // pairs).

        // We've replaced IR-level uses of the remapped allocas, so we only
        // need to replace direct uses here.
        const AllocaInst *AI = dyn_cast_or_null<AllocaInst>(MMO->getValue());
        if (!AI)
          continue;

        if (!Allocas.count(AI))
          continue;

        MMO->setValue(Allocas[AI]);
        FixedMemOp++;
      }

      // Update all of the machine instruction operands.
      for (MachineOperand &MO : I.operands()) {
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
        bool TouchesMemory = I.mayLoad() || I.mayStore();
        // If we *don't* protect the user from escaped allocas, don't bother
        // validating the instructions.
        if (!I.isDebugValue() && TouchesMemory && ProtectFromEscapedAllocas) {
          SlotIndex Index = Indexes->getInstructionIndex(I);
          const LiveInterval *Interval = &*Intervals[FromSlot];
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

  // Update the location of C++ catch objects for the MSVC personality routine.
  if (WinEHFuncInfo *EHInfo = MF->getWinEHFuncInfo())
    for (WinEHTryBlockMapEntry &TBME : EHInfo->TryBlockMap)
      for (WinEHHandlerType &H : TBME.HandlerArray)
        if (H.CatchObj.FrameIndex != INT_MAX &&
            SlotRemap.count(H.CatchObj.FrameIndex))
          H.CatchObj.FrameIndex = SlotRemap[H.CatchObj.FrameIndex];

  DEBUG(dbgs()<<"Fixed "<<FixedMemOp<<" machine memory operands.\n");
  DEBUG(dbgs()<<"Fixed "<<FixedDbg<<" debug locations.\n");
  DEBUG(dbgs()<<"Fixed "<<FixedInstr<<" machine instructions.\n");
}

void StackColoring::removeInvalidSlotRanges() {
  for (MachineBasicBlock &BB : *MF)
    for (MachineInstr &I : BB) {
      if (I.getOpcode() == TargetOpcode::LIFETIME_START ||
          I.getOpcode() == TargetOpcode::LIFETIME_END || I.isDebugValue())
        continue;

      // Some intervals are suspicious! In some cases we find address
      // calculations outside of the lifetime zone, but not actual memory
      // read or write. Memory accesses outside of the lifetime zone are a clear
      // violation, but address calculations are okay. This can happen when
      // GEPs are hoisted outside of the lifetime zone.
      // So, in here we only check instructions which can read or write memory.
      if (!I.mayLoad() && !I.mayStore())
        continue;

      // Check all of the machine operands.
      for (const MachineOperand &MO : I.operands()) {
        if (!MO.isFI())
          continue;

        int Slot = MO.getIndex();

        if (Slot<0)
          continue;

        if (Intervals[Slot]->empty())
          continue;

        // Check that the used slot is inside the calculated lifetime range.
        // If it is not, warn about it and invalidate the range.
        LiveInterval *Interval = &*Intervals[Slot];
        SlotIndex Index = Indexes->getInstructionIndex(I);
        if (Interval->find(Index) == Interval->end()) {
          Interval->clear();
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
  MFI = &MF->getFrameInfo();
  Indexes = &getAnalysis<SlotIndexes>();
  SP = &getAnalysis<StackProtector>();
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
  if (NumMarkers < 2 || TotalSize < 16 || DisableColoring ||
      skipFunction(*Func.getFunction())) {
    DEBUG(dbgs()<<"Will not try to merge slots.\n");
    return removeAllMarkers();
  }

  for (unsigned i=0; i < NumSlots; ++i) {
    std::unique_ptr<LiveInterval> LI(new LiveInterval(i, 0));
    LI->getNextValue(Indexes->getZeroIndex(), VNInfoAllocator);
    Intervals.push_back(std::move(LI));
    SortedSlots.push_back(i);
  }

  // Calculate the liveness of each block.
  calculateLocalLiveness();
  DEBUG(dbgs() << "Dataflow iterations: " << NumIterations << "\n");
  DEBUG(dump());

  // Propagate the liveness information.
  calculateLiveIntervals(NumSlots);
  DEBUG(dumpIntervals());

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
  // Use stable sort to guarantee deterministic code generation.
  std::stable_sort(SortedSlots.begin(), SortedSlots.end(),
                   [this](int LHS, int RHS) {
    // We use -1 to denote a uninteresting slot. Place these slots at the end.
    if (LHS == -1) return false;
    if (RHS == -1) return true;
    // Sort according to size.
    return MFI->getObjectSize(LHS) > MFI->getObjectSize(RHS);
  });

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (unsigned I = 0; I < NumSlots; ++I) {
      if (SortedSlots[I] == -1)
        continue;

      for (unsigned J=I+1; J < NumSlots; ++J) {
        if (SortedSlots[J] == -1)
          continue;

        int FirstSlot = SortedSlots[I];
        int SecondSlot = SortedSlots[J];
        LiveInterval *First = &*Intervals[FirstSlot];
        LiveInterval *Second = &*Intervals[SecondSlot];
        assert (!First->empty() && !Second->empty() && "Found an empty range");

        // Merge disjoint slots.
        if (!First->overlaps(*Second)) {
          Changed = true;
          First->MergeSegmentsInAsValue(*Second, First->getValNumInfo(0));
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

  return removeAllMarkers();
}
