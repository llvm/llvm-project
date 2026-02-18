//===-- AMDGPUNextUseAnalysis.cpp - Next Use Analysis ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the Next Use Analysis for AMDGPU targets.
///
/// ## Overview
///
/// NextUseAnalysis computes distances from each instruction to the next use of
/// each live virtual register. This information guides register allocation and
/// spilling: registers with distant next uses are better spill candidates.
///
/// ## Storage Model (Memory Optimization)
///
/// A naive implementation would store absolute distances for every
/// (instruction, vreg) pair, requiring O(instructions * live vregs) storage.
/// Instead, we use a **relative offset scheme**:
///
/// 1. During analysis (bottom-up walk), distances are stored as negative
///    offsets relative to a "snapshot point" (the block end).
///
/// 2. Each instruction records its offset from the snapshot point.
///
/// 3. At query time, we "materialize" the actual distance by adding the
///    instruction's snapshot offset to the stored value.
///
/// Example: Block with instructions at positions 0, 1, 2, 3, 4:
///
///   pos 0: ...
///   pos 1: ...
///   pos 2: ...
///   pos 3: ... = op %0      ; %0 is used here
///   pos 4: ...              ; (no use of %0)
///
/// Analysis (bottom-up walk):
///   pos 4 (Offset=0): %0 not in Curr. Snapshot stored.
///   pos 3 (Offset=1): See use of %0, insert(%0, -1). Snapshot stored.
///   pos 2,1,0: %0 -> -1 propagates upward.
///
/// Query results for %0:
///   pos 0 (SnapshotOffset=4): materialize(-1, 4) = 3  (3 instrs to next use)
///   pos 1 (SnapshotOffset=3): materialize(-1, 3) = 2  (2 instrs to next use)
///   pos 2 (SnapshotOffset=2): materialize(-1, 2) = 1  (1 instr to next use)
///   pos 3 (SnapshotOffset=1): materialize(-1, 1) = 0  (use is here)
///   pos 4: %0 NOT IN SNAPSHOT -> returns DeadDistance
///
/// ## Cross-Block Distance Propagation
///
/// When a block has multiple successors, distances are merged using min()
/// semantics: if a vreg is live in multiple successors, we take the closest
/// use (since either path might execute at runtime).
///
/// Example: CFG with diamond shape
///
///       bb.0
///      /    \
///   bb.1    bb.2
///      \    /
///       bb.3
///
/// bb.1 (2 instructions):
///   pos 0: ... = op %0    ; %0 used here (Offset=1 when seen)
///   pos 1: ...
///   EntryOff[bb.1] = 2
///   UpwardNextUses[bb.1]: %0 -> -1 (use 1 before block end)
///
/// bb.2 (3 instructions):
///   pos 0: ...
///   pos 1: ...
///   pos 2: ... = op %0    ; %0 used here (Offset=0 when seen)
///   EntryOff[bb.2] = 3
///   UpwardNextUses[bb.2]: %0 -> 0 (use at block end)
///
/// Processing bb.0 (merging successors):
///   Merge bb.1: %0 stored as -1, rebased: -1 + EntryOff[bb.1] = -1 + 2 = 1
///   Merge bb.2: %0 stored as 0, rebased: 0 + EntryOff[bb.2] = 0 + 3 = 3
///   After merge (min semantics): %0 -> 1 (closer use in bb.1 wins)
///
/// ## Loop-Aware Distance Encoding
///
/// When propagating distances across loop boundaries:
///
/// - **Exiting loop (entering in analysis direction):**
///   Add LoopTag to mark uses as "outside current loop", ensuring in-loop
///   uses always appear closer than post-loop uses.
///
/// - **Entering loop (exiting in analysis direction):**
///   Post-loop uses: subtract LoopTag (become finite in outer context).
///   In-loop uses: reset to preheader distance (natural reload point).
///
/// This ensures spillers using "furthest use first" automatically avoid the
/// costly pattern of spilling inside loop bodies (spill+reload per iteration).
///
/// ## Dataflow Framework
///
/// Next Use Analysis is formulated as a backward dataflow problem over a
/// meet-semilattice.
///
/// **Semilattice Structure:**
///   - Domain: L = integers extended with +Inf (infinity for "dead")
///   - Meet operator: meet(a,b) = min(a,b)
///   - Top element: Top = +Inf (no known use)
///   - Partial order: a <= b iff min(a,b) = a (standard integer ordering)
///
/// **Semilattice Properties of (L, min):**
///   - Commutative: min(a, b) = min(b, a)
///   - Associative: min(min(a, b), c) = min(a, min(b, c))
///   - Idempotent: min(a, a) = a
///
/// **Transfer Functions:**
///   - Pass through instruction: f(d) = d + 1 (increment distance)
///   - Register use: f(d) = 0 (reset to immediate use)
///   - Register def: f(d) = +Inf (kill the value)
///
/// **Monotonicity:** All transfer functions are monotonic:
///   - Addition preserves order: a <= b implies a+1 <= b+1
///   - Constants are trivially monotonic
///
/// **Convergence Guarantee:**
///   The analysis converges to a fixed point because:
///   1. The lattice has finite height for practical programs (bounded by
///      function size + special tags)
///   2. Transfer functions are monotonic
///   3. Meet (min) only decreases values: min(a,b) <= a and min(a,b) <= b
///
/// **Iteration Bounds:**
///   - Reducible CFGs: At most (d+1) iterations, where d is the loop nesting
///     depth. For most programs, 2-3 iterations suffice.
///   - Irreducible CFGs: Still converges, but may require more iterations
///     proportional to the CFG's cyclomatic complexity.
///
/// The implementation uses reverse post-order traversal for efficient
/// propagation, ensuring predecessors are processed after their successors
/// in acyclic regions.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Timer.h"

#define DEBUG_TYPE "amdgpu-next-use"

using namespace llvm;

// === Spill Ranking Constants ===
//
// materializeForSpillRanking() maps internal 64-bit distances to unsigned
// values for spillers. The output space is partitioned into three tiers:
//
//   [0, LoopExitRangeStart)                         Tier 1: Finite distances
//   [LoopExitRangeStart, LoopExitRangeStart + LoopExitRangeSize)  Tier 2:
//   Post-loop DeadDistance                                    Tier 3: Dead
//   registers
//
// Spillers pick highest value to evict, so: Dead > PostLoop > Finite.

/// Start of Tier 2 range for post-loop uses.
static constexpr unsigned LoopExitRangeStart = 60000;

/// Number of distinct values in Tier 2 range for post-loop distance
/// granularity.
static constexpr unsigned LoopExitRangeSize = 5000;

// Note: DeadDistance is defined in the header as
// std::numeric_limits<uint16_t>::max()

// Command-line option to enable timing instrumentation
static cl::opt<bool>
    EnableTimers("amdgpu-next-use-analysis-timers",
                 cl::desc("Enable timing for Next Use Analysis"),
                 cl::init(false), cl::Hidden);

// Command-line option to force analysis and dump all distances.
// When set, ensureAnalyzed() is triggered and all distances are printed.
// Used by lit tests; analogous to the competitor's -dump-distance flag.
static cl::opt<bool>
    DumpDistances("amdgpu-next-use-dump-distance",
                  cl::desc("Force NUA to run and dump all next-use distances"),
                  cl::init(false), cl::Hidden);

// Static timers for performance tracking across all analysis runs
static llvm::TimerGroup TG("amdgpu-next-use", "AMDGPU Next Use Analysis");
static llvm::Timer AnalyzeTimer("analyze", "Time spent in analyze()", TG);
static llvm::Timer GetDistanceTimer("getNextUseDistance",
                                    "Time spent in getNextUseDistance()", TG);

/// Convert stored distance to a spiller-friendly ranking value.
///
/// Transforms the internal 64-bit distance into an unsigned value for
/// "pick highest to spill" algorithms. Output is partitioned:
///
///   Tier 1 [0, LoopExitRangeStart):      Finite distances within current
///   context Tier 2 [LoopExitRangeStart, LoopExitRangeStart +
///   LoopExitRangeSize):
///                                        Post-loop uses (prefer spilling over
///                                        in-loop)
///   Tier 3 = DeadDistance:               Dead registers (always safe to spill)
///
/// \param StoredDistance  Relative distance from analysis (negative offset
///                        from snapshot point)
/// \param SnapshotOffset  Query point's position within block
/// \return Ranking value (higher = better spill candidate)
unsigned
NextUseResult::materializeForSpillRanking(int64_t StoredDistance,
                                          unsigned SnapshotOffset) const {
  int64_t Materialized = materialize(StoredDistance, SnapshotOffset);

  // Tier 3: Dead registers -> highest priority for spilling
  if (Materialized >= DeadTag)
    return DeadDistance;

  // Tier 2: Post-loop uses -> map to [LoopExitRangeStart, LoopExitRangeStart +
  // LoopExitRangeSize)
  if (Materialized >= LoopTag) {
    int64_t DistancePastLoopExit = Materialized - LoopTag;
    unsigned ClampedDistance = static_cast<unsigned>(std::min(
        DistancePastLoopExit, static_cast<int64_t>(LoopExitRangeSize - 1)));
    return LoopExitRangeStart + ClampedDistance;
  }

  // Tier 1: Finite distances
  if (Materialized <= 0)
    return 0;

  return static_cast<unsigned>(Materialized);
}

void NextUseResult::init(const MachineFunction &MF) {
  for (const MachineLoop *L : LI->getLoopsInPreorder()) {
    SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>> Exiting;
    L->getExitEdges(Exiting);
    for (const std::pair<MachineBasicBlock *, MachineBasicBlock *> &P :
         Exiting) {
      LoopExits.insert({P.first->getNumber(), P.second->getNumber()});
    }
  }
}

void NextUseResult::analyze(const MachineFunction &MF) {
  // Upward-exposed distances are only necessary to convey the data flow from
  // the block to its predecessors. No need to store it beyond the analyze
  // function as the analysis users are only interested in the use distances
  // relatively to the given MI or the given block end.
  DenseMap<unsigned, VRegDistances> UpwardNextUses;
  iterator_range<po_iterator<const llvm::MachineFunction *>> POT =
      post_order(&MF);
  if (EnableTimers)
    AnalyzeTimer.startTimer();
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const MachineBasicBlock *MBB : POT) {
      unsigned Offset = 0;
      unsigned MBBNum = MBB->getNumber();
      VRegDistances Curr, Prev;
      DenseMap<unsigned, VRegDistances>::iterator PrevIt =
          UpwardNextUses.find(MBBNum);
      if (PrevIt != UpwardNextUses.end()) {
        Prev = PrevIt->second;
      }

      LLVM_DEBUG({
        dbgs() << "\nMerging successors for "
               << "MBB_" << MBB->getNumber() << "." << MBB->getName() << "\n";
      });

      for (MachineBasicBlock *Succ : successors(MBB)) {
        unsigned SuccNum = Succ->getNumber();

        if (!UpwardNextUses.contains(SuccNum))
          continue;

        VRegDistances SuccDist = UpwardNextUses[SuccNum];
        LLVM_DEBUG({
          dbgs() << "\nMerging "
                 << "MBB_" << Succ->getNumber() << "." << Succ->getName()
                 << "\n";
        });

        // Check if the edge from MBB to Succ goes out of the Loop
        int64_t EdgeWeight = 0;
        if (LoopExits.contains({MBB->getNumber(), SuccNum}))
          EdgeWeight = LoopTag;

        if (LI->getLoopDepth(MBB) < LI->getLoopDepth(Succ)) {
          // MBB->Succ is entering the Succ's loop (analysis exiting the loop)
          // Two transformations:
          // 1. Outside-loop uses (>= LoopTag): subtract LoopTag
          // 2. Inside-loop uses (< LoopTag): reset to preheader position
          //    This models: if spilled before loop, reload at preheader
          for (auto &[VReg, Dists] : SuccDist) {
            VRegDistances::SortedRecords NewDists;
            for (VRegDistances::Record R : Dists) {
              if (R.second >= LoopTag) {
                // Outside-loop use: subtract LoopTag
                R.second -= LoopTag;
              } else {
                // Inside-loop use: reset so distance = 0 at preheader bottom
                R.second = -(int64_t)EntryOff[SuccNum];
              }
              NewDists.insert(R);
            }
            Dists = std::move(NewDists);
          }
        }
        LLVM_DEBUG({
          dbgs() << "\nCurr:";
          printVregDistances(Curr /*, 0 - we're at the block bottom*/);
          if (EdgeWeight != 0)
            dbgs() << "\nSucc (EdgeWeight " << EdgeWeight << " applied):";
          else
            dbgs() << "\nSucc:";
          printVregDistances(SuccDist, EntryOff[SuccNum], EdgeWeight);
        });

        // Add PHI uses for this specific edge (MBB -> Succ).
        // PHI uses are edge-specific and not stored in UpwardNextUses,
        // so we add only the operands relevant to this predecessor.
        for (MachineInstr &PHI : Succ->phis()) {
          for (unsigned OpIdx = 1; OpIdx < PHI.getNumOperands(); OpIdx += 2) {
            const MachineOperand &UseOp = PHI.getOperand(OpIdx);
            const MachineOperand &BlockOp = PHI.getOperand(OpIdx + 1);
            // Only add the operand that comes from current MBB
            if (BlockOp.getMBB() == MBB) {
              if (UseOp.isUndef())
                continue;
              VRegMaskPair PhiVMP(UseOp, TRI, MRI);
              // PHI use is at the block top (offset = EntryOff)
              SuccDist.insert(PhiVMP, -(int64_t)EntryOff[SuccNum],
                              /*ForceCloserToEntry=*/true);
            }
          }
        }

        Curr.merge(SuccDist, EntryOff[SuccNum], EdgeWeight);
        LLVM_DEBUG({
          dbgs() << "\nCurr after merge:";
          printVregDistances(Curr);
        });
      }

      NextUseMap[MBBNum].Bottom = Curr;
      NextUseMap[MBBNum].VRegEvents.clear();
      unsigned Seq = 0;

      for (const MachineInstr &MI : reverse(*MBB)) {
        SmallDenseSet<unsigned, 8> TouchedVRegs;

        for (const MachineOperand &MO : MI.operands()) {

          // Only process virtual register operands
          // Undef operands don't represent real uses
          if (!MO.isReg() || !MO.getReg().isVirtual() || MO.isUndef())
            continue;

          VRegMaskPair P(MO, TRI, MRI);
          if (MO.isUse()) {
            // Skip PHI uses — they are edge-specific and will be
            // added per-edge during successor merge.
            if (!MI.isPHI()) {
              Curr.insert(P, -(int64_t)Offset, /*ForceCloserToEntry=*/true);
              UsedInBlock[MBB->getNumber()].insert(P);
              TouchedVRegs.insert(P.getVReg());
            }
          } else if (MO.isDef()) {
            Curr.clear(P);
            UsedInBlock[MBB->getNumber()].remove(P);
            TouchedVRegs.insert(P.getVReg());
          }
        }

        for (unsigned VReg : TouchedVRegs) {
          VRegDistances::iterator It = Curr.find(VReg);
          if (It != Curr.end())
            NextUseMap[MBBNum].VRegEvents[VReg].push_back(
                {Seq, It->second});
          else
            NextUseMap[MBBNum].VRegEvents[VReg].push_back({Seq, {}});
        }

        NextUseMap[MBBNum].InstrOffset[&MI] = Offset;
        NextUseMap[MBBNum].InstrSeq[&MI] = Seq;
        if (!MI.isPHI())
          ++Offset;
        ++Seq;
      }

      // EntryOff needs the TOTAL instruction count for correct predecessor
      // distances while InstrOffset uses individual instruction offsets for
      // materialization

      LLVM_DEBUG({
        dbgs() << "\nFinal distances for " << printMBBReference(*MBB) << "\n";
        printVregDistances(Curr, Offset);
        dbgs() << "\nPrevious distances for " << printMBBReference(*MBB)
               << "\n";
        printVregDistances(Prev, Offset);
        dbgs() << "\nUsed in block:\n";
        dumpUsedInBlock();
      });

      // EntryOff -offset of the first instruction in the block top-down walk
      EntryOff[MBBNum] = Offset;
      UpwardNextUses[MBBNum] = std::move(Curr);

      bool Changed4MBB = (Prev != UpwardNextUses[MBBNum]);

      Changed |= Changed4MBB;
    }
  }
  // Dump complete analysis results for testing
  LLVM_DEBUG(dumpAllNextUseDistances(MF));
  if (EnableTimers) {
    AnalyzeTimer.stopTimer();
    TG.print(llvm::errs());
  }
}

void NextUseResult::getFromSortedRecords(
    const VRegDistances::SortedRecords &Dists, LaneBitmask Mask,
    unsigned SnapshotOffset, unsigned &D) {
  LLVM_DEBUG({
    dbgs() << "Mask : [" << PrintLaneMask(Mask) << "]  "
           << "SnapshotOffset=" << SnapshotOffset << "\n";
  });

  // Records are sorted by stored value in increasing order. Since all entries
  // in this snapshot share the same SnapshotOffset, ordering by stored value
  // is equivalent to ordering by materialized distance.
  for (const VRegDistances::Record &P : Dists) {
    const LaneBitmask UseMask = P.first;
    LLVM_DEBUG(dbgs() << "  UseMask : [" << PrintLaneMask(UseMask) << "]\n");

    // Check for any overlap between the queried mask and the use mask.
    // This handles both subregister and superregister uses:
    // - If UseMask covers Mask: superregister use (e.g., querying sub0, finding
    // full reg)
    // - If Mask covers UseMask: subregister use (e.g., querying full reg,
    // finding sub0)
    // - If they overlap partially: partial overlap (both are valid uses)
    if ((Mask & UseMask).any()) {
      // Use materializeForSpillRanking for three-tier ranking system
      int64_t Stored = static_cast<int64_t>(P.second);
      D = materializeForSpillRanking(Stored, SnapshotOffset);

      break; // first overlapping record is the nearest for this snapshot
    }
  }
}

const NextUseResult::VRegDistances::SortedRecords *
NextUseResult::resolveVReg(const NextUseInfo &Info, unsigned VReg,
                           unsigned Seq) const {
  auto EventsIt = Info.VRegEvents.find(VReg);
  if (EventsIt != Info.VRegEvents.end()) {
    const auto &Events = EventsIt->second;
    const VRegEvent *Best = nullptr;
    for (const VRegEvent &E : Events) {
      if (E.Seq <= Seq)
        Best = &E;
      else
        break;
    }
    if (Best)
      return Best->Records.empty() ? nullptr : &Best->Records;
  }

  auto BottomIt = Info.Bottom.find(VReg);
  if (BottomIt != Info.Bottom.end())
    return &BottomIt->second;

  return nullptr;
}

// Helper to collect subreg uses from sorted records
static void
collectSubregUses(const NextUseResult::VRegDistances::SortedRecords &Dists,
                  const VRegMaskPair &VMP,
                  SmallVectorImpl<VRegMaskPair> &Result) {
  LLVM_DEBUG(
      { dbgs() << "Mask : [" << PrintLaneMask(VMP.getLaneMask()) << "]\n"; });
  for (const NextUseResult::VRegDistances::Record &P : reverse(Dists)) {
    LaneBitmask UseMask = P.first;
    LLVM_DEBUG(
        { dbgs() << "Used mask : [" << PrintLaneMask(UseMask) << "]\n"; });
    if ((UseMask & VMP.getLaneMask()) == UseMask) {
      Result.push_back({VMP.getVReg(), UseMask});
    }
  }
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock::iterator I,
                                   const VRegMaskPair VMP) {
  ensureAnalyzed();
  SmallVector<VRegMaskPair> Result;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt == NextUseMap.end())
    return Result;
  unsigned Seq = MBBIt->second.InstrSeq.lookup(&*I);
  const VRegDistances::SortedRecords *Records =
      resolveVReg(MBBIt->second, VMP.getVReg(), Seq);
  if (!Records)
    return Result;
  collectSubregUses(*Records, VMP, Result);
  return Result;
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock &MBB,
                                   const VRegMaskPair VMP) {
  ensureAnalyzed();
  SmallVector<VRegMaskPair> Result;
  unsigned MBBNum = MBB.getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt == NextUseMap.end())
    return Result;
  VRegDistances::iterator VRegIt = MBBIt->second.Bottom.find(VMP.getVReg());
  if (VRegIt == MBBIt->second.Bottom.end())
    return Result;
  collectSubregUses(VRegIt->second, VMP, Result);
  return Result;
}

void NextUseResult::dumpUsedInBlock() {
  for (auto &[MBBNum, VMPs] : UsedInBlock) {
    dbgs() << "MBB_" << MBBNum << ":\n";
    for (const VRegMaskPair &VMP : VMPs) {
      dbgs() << "[ " << printReg(VMP.getVReg()) << " : <"
             << PrintLaneMask(VMP.getLaneMask()) << "> ]\n";
    }
  }
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock::iterator I,
                                           const VRegMaskPair VMP) {
  ensureAnalyzed();
  if (EnableTimers)
    GetDistanceTimer.startTimer();

  unsigned Dist = DeadDistance;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt != NextUseMap.end()) {
    unsigned SnapOff = MBBIt->second.InstrOffset.lookup(&*I);
    unsigned Seq = MBBIt->second.InstrSeq.lookup(&*I);
    const VRegDistances::SortedRecords *Records =
        resolveVReg(MBBIt->second, VMP.getVReg(), Seq);
    if (Records)
      getFromSortedRecords(*Records, VMP.getLaneMask(), SnapOff, Dist);
  }

  if (EnableTimers)
    GetDistanceTimer.stopTimer();
  return Dist;
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
                                           const VRegMaskPair VMP) {
  ensureAnalyzed();
  if (EnableTimers)
    GetDistanceTimer.startTimer();

  unsigned Dist = DeadDistance;
  unsigned MBBNum = MBB.getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt != NextUseMap.end()) {
    VRegDistances::iterator VRegIt = MBBIt->second.Bottom.find(VMP.getVReg());
    if (VRegIt != MBBIt->second.Bottom.end()) {
      getFromSortedRecords(VRegIt->second, VMP.getLaneMask(), 0, Dist);
    }
  }

  if (EnableTimers)
    GetDistanceTimer.stopTimer();
  return Dist;
}

AMDGPUNextUseAnalysis::Result
AMDGPUNextUseAnalysis::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  return AMDGPUNextUseAnalysis::Result(MF,
                                       MFAM.getResult<SlotIndexesAnalysis>(MF),
                                       MFAM.getResult<MachineLoopAnalysis>(MF));
}

AnalysisKey AMDGPUNextUseAnalysis::Key;

char AMDGPUNextUseAnalysisWrapper::ID = 0;
char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisWrapper::ID;
INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                      "AMDGPU Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisWrapper, "amdgpu-next-use",
                    "AMDGPU Next Use Analysis", false, false)

bool AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(MachineFunction &MF) {
  NU.Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  NU.LI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  NU.MRI = &MF.getRegInfo();
  NU.TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  assert(NU.MRI->isSSA());
  // Defer init() + analyze() to the first query via ensureAnalyzed().
  // If the spiller finds no register pressure issues, NUA does zero work.
  NU.MF = &MF;
  NU.Analyzed = false;
  if (DumpDistances)
    NU.dumpAllNextUseDistances(MF);
  return false;
}

void NextUseResult::ensureAnalyzed() {
  if (!Analyzed) {
    assert(MF && "MachineFunction not set — was runOnMachineFunction called?");
    init(*MF);
    analyze(*MF);
    Analyzed = true;
  }
}

void AMDGPUNextUseAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

AMDGPUNextUseAnalysisWrapper::AMDGPUNextUseAnalysisWrapper()
    : MachineFunctionPass(ID) {
  initializeAMDGPUNextUseAnalysisWrapperPass(*PassRegistry::getPassRegistry());
}

void NextUseResult::dumpAllNextUseDistances(const MachineFunction &MF) {
  ensureAnalyzed();
  dbgs() << "=== NextUseAnalysis Results for " << MF.getName() << " ===\n";

  for (const MachineBasicBlock &MBB : MF) {
    const unsigned MBBNum = MBB.getNumber();
    dbgs() << "\n--- MBB_" << MBBNum << " ---\n";

    if (!NextUseMap.contains(MBBNum)) {
      dbgs() << "  No analysis data for this block\n";
      continue;
    }

    const NextUseInfo &Info = NextUseMap.at(MBBNum);

    // Collect and sort all VReg keys from Bottom and VRegEvents.
    SmallVector<unsigned, 32> AllVRegs;
    for (const auto &[VReg, Recs] : Info.Bottom)
      AllVRegs.push_back(VReg);
    for (const auto &[VReg, Events] : Info.VRegEvents)
      AllVRegs.push_back(VReg);
    llvm::sort(AllVRegs);
    AllVRegs.erase(llvm::unique(AllVRegs), AllVRegs.end());

    // Per-instruction dump (materialized with per-MI snapshot offset).
    for (MachineBasicBlock::const_iterator II = MBB.begin(), IE = MBB.end();
         II != IE; ++II) {
      const MachineInstr &MI = *II;

      dbgs() << "  Instr: ";
      MI.print(dbgs(), /*IsStandalone=*/false, /*SkipOpers=*/false,
               /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
      dbgs() << "\n";

      dbgs() << "    Next-use distances:\n";
      const unsigned SnapOff = Info.InstrOffset.lookup(&MI);
      const unsigned Seq = Info.InstrSeq.lookup(&MI);
      bool Any = false;
      for (unsigned VReg : AllVRegs) {
        const VRegDistances::SortedRecords *Records =
            resolveVReg(Info, VReg, Seq);
        if (Records)
          Any |= printSortedRecords(*Records, VReg, SnapOff, 0, dbgs(),
                                    "      ");
      }
      if (!Any)
        dbgs() << "      (no register uses)\n";
      dbgs() << "\n";
    }

    // Block-end dump (materialized with offset = 0).
    dbgs() << "  Block End Distances:\n";
    const bool AnyEnd = printVregDistances(Info.Bottom, /*SnapshotOffset=*/0,
                                           /* EdgeWeight */ 0, dbgs(), "    ");
    if (!AnyEnd)
      dbgs() << "    (no registers live at block end)\n";
  }

  dbgs() << "\n=== End NextUseAnalysis Results ===\n";
}
