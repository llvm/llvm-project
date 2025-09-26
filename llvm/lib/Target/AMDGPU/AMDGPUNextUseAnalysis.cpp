#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
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

#include "AMDGPU.h"

#include "AMDGPUNextUseAnalysis.h"

#define DEBUG_TYPE "amdgpu-next-use"

using namespace llvm;

// namespace {

// StoredVal is signed, relative to the block entry of the map it lives in.
static inline int64_t materialize(int64_t storedVal, unsigned snapshotOffset) {
  int64_t v = storedVal + (int64_t)snapshotOffset;
  return v <= 0 ? 0 : v; // (clamp to 0; Infinity handled at call sites)
}

// Three-tier ranking system for spiller decisions
unsigned NextUseResult::materializeForRank(int64_t stored, unsigned snapshotOffset) const {
  int64_t Mat64 = materialize(stored, snapshotOffset);

  // Tier 1: Finite distances (0 to LoopTag-1) → return as-is
  // Tier 2: Loop-exit distances (LoopTag to DeadTag-1) → map to 60000-64999 range
  // Tier 3: Dead registers (DeadTag+) → return Infinity (65535)
  if (Mat64 >= DeadTag) {
    return Infinity;  // Tier 3: Dead registers get maximum distance
  } else if (Mat64 >= LoopTag) {
    // Tier 2: Loop-exit distances get mapped to high range [60000, 64999]
    int64_t LoopRemainder = Mat64 - LoopTag;
    // Clamp the remainder to fit in available range (5000 values)
    unsigned ClampedRemainder = static_cast<unsigned>(
        std::min(LoopRemainder, static_cast<int64_t>(4999)));
    return 60000 + ClampedRemainder;
  } else if (Mat64 <= 0) {
    return 0;  // Tier 1: Zero-distance for immediate uses
  } else {
    return static_cast<unsigned>(Mat64);  // Tier 1: Finite distances as-is
  }
}


void NextUseResult::init(const MachineFunction &MF) {
  TG = new TimerGroup("Next Use Analysis",
                      "Compilation Timers for Next Use Analysis");
  T1 = new Timer("Next Use Analysis", "Time spent in analyse()", *TG);
  T2 = new Timer("Next Use Analysis", "Time spent in computeNextUseDistance()",
                 *TG);
  for (auto L : LI->getLoopsInPreorder()) {
    SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>> Exiting;
    L->getExitEdges(Exiting);
    for (auto P : Exiting) {
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
  T1->startTimer();
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (auto MBB : post_order(&MF)) {
      unsigned Offset = 0;
      unsigned MBBNum = MBB->getNumber();
      VRegDistances Curr, Prev;
      if (UpwardNextUses.contains(MBBNum)) {
        Prev = UpwardNextUses[MBBNum];
      }

      LLVM_DEBUG(dbgs() << "\nMerging successors for "
                        << "MBB_" << MBB->getNumber() << "." << MBB->getName()
                        << "\n";);

      for (auto Succ : successors(MBB)) {
        unsigned SuccNum = Succ->getNumber();

        if (!UpwardNextUses.contains(SuccNum))
          continue;

        VRegDistances SuccDist = UpwardNextUses[SuccNum];
        LLVM_DEBUG(dbgs() << "\nMerging "
                          << "MBB_" << Succ->getNumber() << "."
                          << Succ->getName() << "\n");

        // Check if the edge from MBB to Succ goes out of the Loop
        int64_t EdgeWeight = 0;
        if (LoopExits.contains(MBB->getNumber())) {
          unsigned ExitTo = LoopExits[MBB->getNumber()];
          if (SuccNum == ExitTo)
            EdgeWeight = LoopTag;
        }

        if (LI->getLoopDepth(MBB) < LI->getLoopDepth(Succ)) {
          // MBB->Succ is entering the Succ's loop
          // Clear out the Loop-Exiting weights.
          for (auto &P : SuccDist) {
            auto &Dists = P.second;
            for (auto R : Dists) {
              if (R.second >= LoopTag) {
                std::pair<LaneBitmask, int64_t> New = R;
                New.second -= LoopTag;
                Dists.erase(R);
                Dists.insert(New);
              }
            }
          }
        }
        LLVM_DEBUG(dbgs() << "\nCurr:";
                   printVregDistances(Curr /*, 0 - we're at the block bottom*/);
                   dbgs() << "\nSucc:";
                   printVregDistances(SuccDist, EntryOff[SuccNum], EdgeWeight));

        // Filter out successor's PHI operands with SourceBlock != MBB
        // PHI operands are only live on their specific incoming edge
        for (auto &PHI : Succ->phis()) {
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
        LLVM_DEBUG(dbgs() << "\nCurr after merge:"; printVregDistances(Curr));
      }

      NextUseMap[MBBNum].Bottom = Curr;

      for (auto &MI : make_range(MBB->rbegin(), MBB->rend())) {

        for (auto &MO : MI.operands()) {

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
        // printVregDistances(Curr, Offset);
        if (!MI.isPHI())
          ++Offset;
      }

      // EntryOff needs the TOTAL instruction count for correct predecessor distances
      // while InstrOffset uses individual instruction offsets for materialization

      LLVM_DEBUG(dbgs() << "\nFinal distances for MBB_" << MBB->getNumber()
                        << "." << MBB->getName() << "\n";
                 printVregDistances(Curr, Offset));
      LLVM_DEBUG(dbgs() << "\nPrevious distances for MBB_" << MBB->getNumber()
                        << "." << MBB->getName() << "\n";
                 printVregDistances(Prev, Offset));

      // EntryOff -offset of the first instruction in the block top-down walk
      EntryOff[MBBNum] = Offset;
      UpwardNextUses[MBBNum] = std::move(Curr);

      bool Changed4MBB = (Prev != UpwardNextUses[MBBNum]);

      Changed |= Changed4MBB;
    }
  }
  // dumpUsedInBlock();
  // Dump complete analysis results for testing
  LLVM_DEBUG(dumpAllNextUseDistances(MF));
  T1->stopTimer();
  LLVM_DEBUG(TG->print(llvm::errs()));
}

void NextUseResult::getFromSortedRecords(
    const VRegDistances::SortedRecords &Dists, LaneBitmask Mask,
    unsigned SnapshotOffset, unsigned &D) {
  LLVM_DEBUG(dbgs() << "Mask : [" << PrintLaneMask(Mask) << "]  "
                    << "SnapshotOffset=" << SnapshotOffset << "\n");

  // Records are sorted by stored value in increasing order. Since all entries
  // in this snapshot share the same SnapshotOffset, ordering by stored value
  // is equivalent to ordering by materialized distance.
  for (const auto &P : Dists) {
    const LaneBitmask UseMask = P.first;
    LLVM_DEBUG(dbgs() << "  UseMask : [" << PrintLaneMask(UseMask) << "]\n");

    // Require full coverage: a use contributes only if it covers the queried
    // lanes.
    if ((Mask & UseMask) == Mask) {
      // Use materializeForRank for three-tier ranking system
      int64_t Stored = static_cast<int64_t>(P.second);
      D = materializeForRank(Stored, SnapshotOffset);

      break; // first covering record is the nearest for this snapshot
    }
  }
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock::iterator I,
                                   const VRegMaskPair VMP) {
  SmallVector<VRegMaskPair> Result;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].InstrDist.contains(&*I)) {
    // VRegDistances Dists = NextUseMap[MBBNum].InstrDist[&*I];
    if (NextUseMap[MBBNum].InstrDist[&*I].contains(VMP.getVReg())) {
      VRegDistances::SortedRecords Dists =
          NextUseMap[MBBNum].InstrDist[&*I][VMP.getVReg()];
      LLVM_DEBUG(dbgs() << "Mask : [" << PrintLaneMask(VMP.getLaneMask())
                        << "]\n");
      for (auto P : reverse(Dists)) {
        LaneBitmask UseMask = P.first;
        LLVM_DEBUG(dbgs() << "Used mask : [" << PrintLaneMask(UseMask)
                          << "]\n");
        if ((UseMask & VMP.getLaneMask()) == UseMask) {
          Result.push_back({VMP.getVReg(), UseMask});
        }
      }
    }
  }
  return Result;
}

SmallVector<VRegMaskPair>
NextUseResult::getSortedSubregUses(const MachineBasicBlock &MBB,
                                   const VRegMaskPair VMP) {
  SmallVector<VRegMaskPair> Result;
  unsigned MBBNum = MBB.getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].Bottom.contains(VMP.getVReg())) {
    VRegDistances::SortedRecords Dists =
        NextUseMap[MBBNum].Bottom[VMP.getVReg()];
    LLVM_DEBUG(dbgs() << "Mask : [" << PrintLaneMask(VMP.getLaneMask())
                      << "]\n");
    for (auto P : reverse(Dists)) {
      LaneBitmask UseMask = P.first;
      LLVM_DEBUG(dbgs() << "Used mask : [" << PrintLaneMask(UseMask) << "]\n");
      if ((UseMask & VMP.getLaneMask()) == UseMask) {
        Result.push_back({VMP.getVReg(), UseMask});
      }
    }
  }
  return Result;
}

void NextUseResult::dumpUsedInBlock() {
  LLVM_DEBUG(for (auto P
                  : UsedInBlock) {
    dbgs() << "MBB_" << P.first << ":\n";
    for (auto VMP : P.second) {
      dbgs() << "[ " << printReg(VMP.getVReg()) << " : <"
             << PrintLaneMask(VMP.getLaneMask()) << "> ]\n";
    }
  });
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock::iterator I,
                                           const VRegMaskPair VMP) {
  unsigned Dist = Infinity;
  const MachineBasicBlock *MBB = I->getParent();
  unsigned MBBNum = MBB->getNumber();
  if (NextUseMap.contains(MBBNum) &&
      NextUseMap[MBBNum].InstrDist.contains(&*I)) {
    VRegDistances Dists = NextUseMap[MBBNum].InstrDist[&*I];
    if (NextUseMap[MBBNum].InstrDist[&*I].contains(VMP.getVReg())) {
      // printSortedRecords(Dists[VMP.VReg], VMP.VReg);
      unsigned SnapOff = NextUseMap[MBBNum].InstrOffset[&*I];
      getFromSortedRecords(Dists[VMP.getVReg()], VMP.getLaneMask(),
                           SnapOff, Dist);
    }
  }

  return Dist;
}

unsigned NextUseResult::getNextUseDistance(const MachineBasicBlock &MBB,
                                           const VRegMaskPair VMP) {
  unsigned Dist = Infinity;
  unsigned MBBNum = MBB.getNumber();
  if (NextUseMap.contains(MBBNum)) {
    if (NextUseMap[MBBNum].Bottom.contains(VMP.getVReg())) {
      getFromSortedRecords(NextUseMap[MBBNum].Bottom[VMP.getVReg()],
                           VMP.getLaneMask(), 0, Dist);
    }
  }
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

//} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPUNextUseAnalysisPass",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](MachineFunctionAnalysisManager &MFAM) {
                  MFAM.registerPass([] { return AMDGPUNextUseAnalysis(); });
                });
          }};
}

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

  for (const auto &MBB : MF) {
    const unsigned MBBNum = MBB.getNumber();
    LLVM_DEBUG(dbgs() << "\n--- MBB_" << MBBNum << " ---\n");

    if (!NextUseMap.contains(MBBNum)) {
      LLVM_DEBUG(dbgs() << "  No analysis data for this block\n");
      continue;
    }

    const NextUseInfo &Info = NextUseMap.at(MBBNum);

    // Per-instruction dump (materialized with per-MI snapshot offset).
    for (auto II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      const MachineInstr &MI = *II;

      LLVM_DEBUG(dbgs() << "  Instr: ");
      LLVM_DEBUG(MI.print(dbgs(), /*IsStandalone=*/false, /*SkipOpers=*/false,
                          /*SkipDebugLoc=*/true, /*AddNewLine=*/false));
      LLVM_DEBUG(dbgs() << "\n");

      LLVM_DEBUG(dbgs() << "    Next-use distances:\n");
      if (Info.InstrDist.contains(&MI)) {
        const VRegDistances &Dists = Info.InstrDist.at(&MI);
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
