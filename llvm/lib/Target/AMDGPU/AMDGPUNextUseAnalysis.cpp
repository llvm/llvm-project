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
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"

#include "llvm/ADT/DenseMap.h"
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

// Command-line option to enable timing instrumentation
static cl::opt<bool>
    EnableTimers("amdgpu-next-use-analysis-timers",
                 cl::desc("Enable timing for Next Use Analysis"),
                 cl::init(false), cl::Hidden);

// Static timers for performance tracking across all analysis runs
static llvm::TimerGroup TG("amdgpu-next-use", "AMDGPU Next Use Analysis");
static llvm::Timer AnalyzeTimer("analyze", "Time spent in analyze()", TG);
static llvm::Timer GetDistanceTimer("getNextUseDistance",
                                    "Time spent in getNextUseDistance()", TG);

// Three-tier ranking system for spiller decisions
unsigned NextUseResult::materializeForRank(int64_t Stored,
                                           unsigned SnapshotOffset) const {
  int64_t Mat64 = materialize(Stored, SnapshotOffset);

  // Tier 1: Finite distances (0 to LoopTag-1) → return as-is
  // Tier 2: Loop-exit distances (LoopTag to DeadTag-1) → map to 60000-64999
  // Tier 3: Dead registers (DeadTag+) → return DeadDistance (65535)
  if (Mat64 >= DeadTag)
    return DeadDistance;

  if (Mat64 >= LoopTag) {
    // Tier 2: Loop-exit distances get mapped to high range [60000, 64999]
    int64_t LoopRemainder = Mat64 - LoopTag;
    // Clamp the remainder to fit in available range (5000 values)
    unsigned ClampedRemainder = static_cast<unsigned>(
        std::min(LoopRemainder, static_cast<int64_t>(4999)));
    return 60000 + ClampedRemainder;
  }

  if (Mat64 <= 0)
    return 0; // Tier 1: Zero-distance for immediate uses

  return static_cast<unsigned>(Mat64); // Tier 1: Finite distances as-is
}

void NextUseResult::init(const MachineFunction &MF) {
  for (const MachineLoop *L : LI->getLoopsInPreorder()) {
    SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>> Exiting;
    L->getExitEdges(Exiting);
    for (const std::pair<MachineBasicBlock *, MachineBasicBlock *> &P : Exiting) {
      LoopExits[P.first->getNumber()] = P.second->getNumber();
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
      DenseMap<unsigned, VRegDistances>::iterator PrevIt = UpwardNextUses.find(MBBNum);
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
        DenseMap<unsigned, unsigned>::iterator LoopExitIt = LoopExits.find(MBB->getNumber());
        if (LoopExitIt != LoopExits.end()) {
          if (SuccNum == LoopExitIt->second)
            EdgeWeight = LoopTag;
        }

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

        // Filter out successor's PHI operands with SourceBlock != MBB
        // PHI operands are only live on their specific incoming edge
        for (MachineInstr &PHI : Succ->phis()) {
          // Check each PHI operand pair (value, source block)
          for (unsigned OpIdx = 1; OpIdx < PHI.getNumOperands(); OpIdx += 2) {
            const MachineOperand &UseOp = PHI.getOperand(OpIdx);
            const MachineOperand &BlockOp = PHI.getOperand(OpIdx + 1);

            // Skip if this operand doesn't come from current MBB
            if (BlockOp.getMBB() != MBB) {
              VRegMaskPair PhiVMP(UseOp, TRI, MRI);
              // Remove this PHI operand from the successor distances
              SuccDist.clear(PhiVMP);
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

      for (const MachineInstr &MI : reverse(*MBB)) {

        for (const MachineOperand &MO : MI.operands()) {

          // Only process virtual register operands
          // Undef operands don't represent real uses
          if (!MO.isReg() || !MO.getReg().isVirtual() || MO.isUndef())
            continue;

          VRegMaskPair P(MO, TRI, MRI);
          if (MO.isUse()) {
            Curr.insert(P, -(int64_t)Offset);
            UsedInBlock[MBB->getNumber()].insert(P);
          } else if (MO.isDef()) {
            Curr.clear(P);
            UsedInBlock[MBB->getNumber()].remove(P);
          }
        }
        NextUseMap[MBBNum].InstrDist[&MI] = Curr;
        NextUseMap[MBBNum].InstrOffset[&MI] = Offset;
        if (!MI.isPHI())
          ++Offset;
      }

      // EntryOff needs the TOTAL instruction count for correct predecessor
      // distances while InstrOffset uses individual instruction offsets for
      // materialization

      LLVM_DEBUG({
        dbgs() << "\nFinal distances for " << printMBBReference(*MBB) << "\n";
        printVregDistances(Curr, Offset);
        dbgs() << "\nPrevious distances for " << printMBBReference(*MBB) << "\n";
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
      // Use materializeForRank for three-tier ranking system
      int64_t Stored = static_cast<int64_t>(P.second);
      D = materializeForRank(Stored, SnapshotOffset);

      break; // first overlapping record is the nearest for this snapshot
    }
  }
}

// Helper to collect subreg uses from sorted records
static void collectSubregUses(const NextUseResult::VRegDistances::SortedRecords &Dists,
                              const VRegMaskPair &VMP,
                              SmallVectorImpl<VRegMaskPair> &Result) {
  LLVM_DEBUG({ dbgs() << "Mask : [" << PrintLaneMask(VMP.getLaneMask()) << "]\n"; });
  for (const NextUseResult::VRegDistances::Record &P : reverse(Dists)) {
    LaneBitmask UseMask = P.first;
    LLVM_DEBUG({ dbgs() << "Used mask : [" << PrintLaneMask(UseMask) << "]\n"; });
    if ((UseMask & VMP.getLaneMask()) == UseMask) {
      Result.push_back({VMP.getVReg(), UseMask});
    }
  }
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock::iterator I,
                                   const VRegMaskPair VMP) {
  SmallVector<VRegMaskPair> Result;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt == NextUseMap.end())
    return Result;
  DenseMap<const MachineInstr *, VRegDistances>::iterator InstrIt = MBBIt->second.InstrDist.find(&*I);
  if (InstrIt == MBBIt->second.InstrDist.end())
    return Result;
  VRegDistances::iterator VRegIt = InstrIt->second.find(VMP.getVReg());
  if (VRegIt == InstrIt->second.end())
    return Result;
  collectSubregUses(VRegIt->second, VMP, Result);
  return Result;
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock &MBB,
                                   const VRegMaskPair VMP) {
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
  if (EnableTimers)
    GetDistanceTimer.startTimer();

  unsigned Dist = DeadDistance;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  DenseMap<unsigned, NextUseInfo>::iterator MBBIt = NextUseMap.find(MBBNum);
  if (MBBIt != NextUseMap.end()) {
    DenseMap<const MachineInstr *, VRegDistances>::iterator InstrIt = MBBIt->second.InstrDist.find(&*I);
    if (InstrIt != MBBIt->second.InstrDist.end()) {
      VRegDistances::iterator VRegIt = InstrIt->second.find(VMP.getVReg());
      if (VRegIt != InstrIt->second.end()) {
        unsigned SnapOff = MBBIt->second.InstrOffset.lookup(&*I);
        getFromSortedRecords(VRegIt->second, VMP.getLaneMask(), SnapOff, Dist);
      }
    }
  }

  if (EnableTimers)
    GetDistanceTimer.stopTimer();
  return Dist;
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
                                           const VRegMaskPair VMP) {
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
  NU.init(MF);
  NU.analyze(MF);
  //  LLVM_DEBUG(NU.dump());
  return false;
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
  LLVM_DEBUG(dbgs() << "=== NextUseAnalysis Results for " << MF.getName()
                    << " ===\n");

  for (const MachineBasicBlock &MBB : MF) {
    const unsigned MBBNum = MBB.getNumber();
    LLVM_DEBUG(dbgs() << "\n--- MBB_" << MBBNum << " ---\n");

    if (!NextUseMap.contains(MBBNum)) {
      LLVM_DEBUG(dbgs() << "  No analysis data for this block\n");
      continue;
    }

    const NextUseInfo &Info = NextUseMap.at(MBBNum);

    // Per-instruction dump (materialized with per-MI snapshot offset).
    for (MachineBasicBlock::const_iterator II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      const MachineInstr &MI = *II;

      LLVM_DEBUG(dbgs() << "  Instr: ");
      LLVM_DEBUG(MI.print(dbgs(), /*IsStandalone=*/false, /*SkipOpers=*/false,
                          /*SkipDebugLoc=*/true, /*AddNewLine=*/false));
      LLVM_DEBUG(dbgs() << "\n");

      LLVM_DEBUG(dbgs() << "    Next-use distances:\n");
      DenseMap<const MachineInstr *, VRegDistances>::const_iterator InstrIt = Info.InstrDist.find(&MI);
      if (InstrIt != Info.InstrDist.end()) {
        const VRegDistances &Dists = InstrIt->second;
        const unsigned SnapOff = Info.InstrOffset.lookup(&MI); // 0 if absent
        const bool Any =
            printVregDistances(Dists, SnapOff, 0, dbgs(), "      ");
        if (!Any)
          LLVM_DEBUG(dbgs() << "      (no register uses)\n");
      } else {
        LLVM_DEBUG(dbgs() << "      (no distance data)\n");
      }
      LLVM_DEBUG(dbgs() << "\n");
    }

    // Block-end dump (materialized with offset = 0).
    LLVM_DEBUG(dbgs() << "  Block End Distances:\n");
    const bool AnyEnd = printVregDistances(Info.Bottom, /*SnapshotOffset=*/0,
                                           /* EdgeWeight */ 0, dbgs(), "    ");
    if (!AnyEnd)
      LLVM_DEBUG(dbgs() << "    (no registers live at block end)\n");
  }

  LLVM_DEBUG(dbgs() << "\n=== End NextUseAnalysis Results ===\n");
}
