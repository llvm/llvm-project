//===-- GCNSchedStrategy.h - GCN Scheduler Strategy -*- C++ -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H

#include "GCNRegPressure.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class SIMachineFunctionInfo;
class SIRegisterInfo;
class GCNSubtarget;
class GCNSchedStage;

enum class GCNSchedStageID : unsigned {
  OccInitialSchedule = 0,
  UnclusteredHighRPReschedule = 1,
  ClusteredLowOccupancyReschedule = 2,
  PreRARematerialize = 3,
  ILPInitialSchedule = 4,
  MemoryClauseInitialSchedule = 5
};

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &OS, const GCNSchedStageID &StageID);
#endif

// Tracks the number of cycles that a resource is occupied. Requires top-down
// scheduling.
struct ProcRes {
  unsigned CyclesReserved = 0;

  void reset() { CyclesReserved = 0; }

  void reserve(unsigned Cycles) { CyclesReserved += Cycles; }

  void release(unsigned Cycles) {
    if (Cycles > CyclesReserved)
      CyclesReserved = 0;
    else
      CyclesReserved -= Cycles;
  }
};

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.
class GCNSchedStrategy : public GenericScheduler {
protected:
  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool IsBottomUp);

  void initCandidate(SchedCandidate &Cand, SUnit *SU, bool AtTop,
                     const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI, unsigned SGPRPressure,
                     unsigned VGPRPressure, bool IsBottomUp);

  // If the XDL resource is not occupied, try to schedule a ready MFMA,
  // otherwise, try not to stall XDL.
  bool tryXDL(SchedCandidate &Cand, SchedCandidate &TryCand,
              SchedBoundary *Zone) const;

  std::vector<unsigned> Pressure;

  std::vector<unsigned> MaxPressure;

  unsigned SGPRExcessLimit;

  unsigned VGPRExcessLimit;

  unsigned TargetOccupancy;

  MachineFunction *MF;

  // Scheduling stages for this strategy.
  SmallVector<GCNSchedStageID, 4> SchedStages;

  // Pointer to the current SchedStageID.
  SmallVectorImpl<GCNSchedStageID>::iterator CurrentStage = nullptr;

  // GCN RP Tracker for top-down scheduling
  mutable GCNDownwardRPTracker DownwardTracker;

  // GCN RP Tracker for botttom-up scheduling
  mutable GCNUpwardRPTracker UpwardTracker;

public:
  // schedule() have seen register pressure over the critical limits and had to
  // track register pressure for actual scheduling heuristics.
  bool HasHighPressure;

  // Schedule known to have excess register pressure. Be more conservative in
  // increasing ILP and preserving VGPRs.
  bool KnownExcessRP = false;

  // An error margin is necessary because of poor performance of the generic RP
  // tracker and can be adjusted up for tuning heuristics to try and more
  // aggressively reduce register pressure.
  unsigned ErrorMargin = 3;

  // Bias for SGPR limits under a high register pressure.
  const unsigned HighRPSGPRBias = 7;

  // Bias for VGPR limits under a high register pressure.
  const unsigned HighRPVGPRBias = 7;

  unsigned SGPRCriticalLimit;

  unsigned VGPRCriticalLimit;

  unsigned SGPRLimitBias = 0;

  unsigned VGPRLimitBias = 0;

  // Processor resource for XDL.
  ProcRes XDLProcRes;

  // Use custom resource tracking for scheduling.
  bool CustomResTracking = false;

  GCNSchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  unsigned getTargetOccupancy() { return TargetOccupancy; }

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }

  GCNSchedStageID getCurrentStage();

  // Advances stage. Returns true if there are remaining stages.
  bool advanceStage();

  bool hasNextStage() const;

  GCNSchedStageID getNextStage() const;

  GCNDownwardRPTracker *getDownwardTracker() { return &DownwardTracker; }

  GCNUpwardRPTracker *getUpwardTracker() { return &UpwardTracker; }
};

/// The goal of this scheduling strategy is to maximize kernel occupancy (i.e.
/// maximum number of waves per simd).
class GCNMaxOccupancySchedStrategy final : public GCNSchedStrategy {
public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C,
                               bool IsLegacyScheduler = false);

  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;
};

/// The goal of this scheduling strategy is to maximize ILP for a single wave
/// (i.e. latency hiding).
class GCNMaxILPSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  GCNMaxILPSchedStrategy(const MachineSchedContext *C);
};

/// The goal of this scheduling strategy is to maximize memory clause for a
/// single wave.
class GCNMaxMemoryClauseSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  GCNMaxMemoryClauseSchedStrategy(const MachineSchedContext *C);
};

class ScheduleMetrics {
  unsigned ScheduleLength;
  unsigned BubbleCycles;

public:
  ScheduleMetrics() {}
  ScheduleMetrics(unsigned L, unsigned BC)
      : ScheduleLength(L), BubbleCycles(BC) {}
  unsigned getLength() const { return ScheduleLength; }
  unsigned getBubbles() const { return BubbleCycles; }
  unsigned getMetric() const {
    unsigned Metric = (BubbleCycles * ScaleFactor) / ScheduleLength;
    // Metric is zero if the amount of bubbles is less than 1% which is too
    // small. So, return 1.
    return Metric ? Metric : 1;
  }
  static const unsigned ScaleFactor;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ScheduleMetrics &Sm) {
  dbgs() << "\n Schedule Metric (scaled by "
         << ScheduleMetrics::ScaleFactor
         << " ) is: " << Sm.getMetric() << " [ " << Sm.getBubbles() << "/"
         << Sm.getLength() << " ]\n";
  return OS;
}

class GCNScheduleDAGMILive;
class RegionPressureMap {
  GCNScheduleDAGMILive *DAG;
  // The live in/out pressure as indexed by the first or last MI in the region
  // before scheduling.
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> RegionLiveRegMap;
  // The mapping of RegionIDx to key instruction
  DenseMap<unsigned, MachineInstr *> IdxToInstruction;
  // Whether we are calculating LiveOuts or LiveIns
  bool IsLiveOut;

public:
  RegionPressureMap() {}
  RegionPressureMap(GCNScheduleDAGMILive *GCNDAG, bool LiveOut)
      : DAG(GCNDAG), IsLiveOut(LiveOut) {}
  // Build the Instr->LiveReg and RegionIdx->Instr maps
  void buildLiveRegMap();

  // Retrieve the LiveReg for a given RegionIdx
  GCNRPTracker::LiveRegSet &getLiveRegsForRegionIdx(unsigned RegionIdx) {
    assert(IdxToInstruction.find(RegionIdx) != IdxToInstruction.end());
    MachineInstr *Key = IdxToInstruction[RegionIdx];
    return RegionLiveRegMap[Key];
  }
};

class GCNScheduleDAGMILive final : public ScheduleDAGMILive {
  friend class GCNSchedStage;
  friend class OccInitialScheduleStage;
  friend class UnclusteredHighRPStage;
  friend class ClusteredLowOccStage;
  friend class PreRARematStage;
  friend class ILPInitialScheduleStage;
  friend class RegionPressureMap;

  const GCNSubtarget &ST;

  SIMachineFunctionInfo &MFI;

  // Occupancy target at the beginning of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Vector of regions recorder for later rescheduling
  SmallVector<std::pair<MachineBasicBlock::iterator,
                        MachineBasicBlock::iterator>, 32> Regions;

  // Records if a region is not yet scheduled, or schedule has been reverted,
  // or we generally desire to reschedule it.
  BitVector RescheduleRegions;

  // Record regions with high register pressure.
  BitVector RegionsWithHighRP;

  // Record regions with excess register pressure over the physical register
  // limit. Register pressure in these regions usually will result in spilling.
  BitVector RegionsWithExcessRP;

  // Regions that has the same occupancy as the latest MinOccupancy
  BitVector RegionsWithMinOcc;

  // Regions that have IGLP instructions (SCHED_GROUP_BARRIER or IGLP_OPT).
  BitVector RegionsWithIGLPInstrs;

  // Region live-in cache.
  SmallVector<GCNRPTracker::LiveRegSet, 32> LiveIns;

  // Region pressure cache.
  SmallVector<GCNRegPressure, 32> Pressure;

  // Temporary basic block live-in cache.
  DenseMap<const MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBLiveIns;

  // The map of the initial first region instruction to region live in registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> BBLiveInMap;

  // Calculate the map of the initial first region instruction to region live in
  // registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> getRegionLiveInMap() const;

  // Calculate the map of the initial last region instruction to region live out
  // registers
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
  getRegionLiveOutMap() const;

  // The live out registers per region. These are internally stored as a map of
  // the initial last region instruction to region live out registers, but can
  // be retreived with the regionIdx by calls to getLiveRegsForRegionIdx.
  RegionPressureMap RegionLiveOuts;

  // Return current region pressure.
  GCNRegPressure getRealRegPressure(unsigned RegionIdx) const;

  // Compute and cache live-ins and pressure for all regions in block.
  void computeBlockPressure(unsigned RegionIdx, const MachineBasicBlock *MBB);

  // Update region boundaries when removing MI or inserting NewMI before MI.
  void updateRegionBoundaries(
      SmallVectorImpl<std::pair<MachineBasicBlock::iterator,
                                MachineBasicBlock::iterator>> &RegionBoundaries,
      MachineBasicBlock::iterator MI, MachineInstr *NewMI);

  void runSchedStages();

  std::unique_ptr<GCNSchedStage> createSchedStage(GCNSchedStageID SchedStageID);

public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

// GCNSchedStrategy applies multiple scheduling stages to a function.
class GCNSchedStage {
protected:
  GCNScheduleDAGMILive &DAG;

  GCNSchedStrategy &S;

  MachineFunction &MF;

  SIMachineFunctionInfo &MFI;

  const GCNSubtarget &ST;

  const GCNSchedStageID StageID;

  // The current block being scheduled.
  MachineBasicBlock *CurrentMBB = nullptr;

  // Current region index.
  unsigned RegionIdx = 0;

  // Record the original order of instructions before scheduling.
  std::vector<MachineInstr *> Unsched;

  // RP before scheduling the current region.
  GCNRegPressure PressureBefore;

  // RP after scheduling the current region.
  GCNRegPressure PressureAfter;

  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  GCNSchedStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG);

public:
  // Initialize state for a scheduling stage. Returns false if the current stage
  // should be skipped.
  virtual bool initGCNSchedStage();

  // Finalize state after finishing a scheduling pass on the function.
  virtual void finalizeGCNSchedStage();

  // Setup for scheduling a region. Returns false if the current region should
  // be skipped.
  virtual bool initGCNRegion();

  // Track whether a new region is also a new MBB.
  void setupNewBlock();

  // Finalize state after scheudling a region.
  void finalizeGCNRegion();

  // Check result of scheduling.
  void checkScheduling();

  // computes the given schedule virtual execution time in clocks
  ScheduleMetrics getScheduleMetrics(const std::vector<SUnit> &InputSchedule);
  ScheduleMetrics getScheduleMetrics(const GCNScheduleDAGMILive &DAG);
  unsigned computeSUnitReadyCycle(const SUnit &SU, unsigned CurrCycle,
                                  DenseMap<unsigned, unsigned> &ReadyCycles,
                                  const TargetSchedModel &SM);

  // Returns true if scheduling should be reverted.
  virtual bool shouldRevertScheduling(unsigned WavesAfter);

  // Returns true if current region has known excess pressure.
  bool isRegionWithExcessRP() const {
    return DAG.RegionsWithExcessRP[RegionIdx];
  }

  // The region number this stage is currently working on
  unsigned getRegionIdx() { return RegionIdx; }

  // Returns true if the new schedule may result in more spilling.
  bool mayCauseSpilling(unsigned WavesAfter);

  // Attempt to revert scheduling for this region.
  void revertScheduling();

  void advanceRegion() { RegionIdx++; }

  virtual ~GCNSchedStage() = default;
};

class OccInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  OccInitialScheduleStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class UnclusteredHighRPStage : public GCNSchedStage {
private:
  // Save the initial occupancy before starting this stage.
  unsigned InitialOccupancy;

public:
  bool initGCNSchedStage() override;

  void finalizeGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  UnclusteredHighRPStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

// Retry function scheduling if we found resulting occupancy and it is
// lower than used for other scheduling passes. This will give more freedom
// to schedule low register pressure blocks.
class ClusteredLowOccStage : public GCNSchedStage {
public:
  bool initGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  ClusteredLowOccStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class RematCandidate {
public:
  MachineInstr *Def = nullptr;
  unsigned LoopCost;
  std::set<unsigned> HighRPRegions;
  MachineBasicBlock::iterator InsertPt;

  bool operator<(const RematCandidate &Other) const {
    if (LoopCost < Other.LoopCost)
      return true;

    if (LoopCost == Other.LoopCost) {
      if (Def < Other.Def)
        return true;

      if (Def == Other.Def) {
        return InsertPt->getParent() < Other.InsertPt->getParent();
      }
    }

    return false;
  }

  RematCandidate(MachineInstr *Def, unsigned LoopCost, unsigned HighRPRegion,
                 MachineBasicBlock::iterator InsertPt)
      : Def(Def), LoopCost(LoopCost), InsertPt(InsertPt) {
    HighRPRegions.insert(HighRPRegion);
  }

  RematCandidate(MachineInstr *Def, unsigned LoopCost,
                 std::set<unsigned> HighRPRegions,
                 MachineBasicBlock::iterator InsertPt)
      : Def(Def), LoopCost(LoopCost), HighRPRegions(HighRPRegions),
        InsertPt(InsertPt) {}

private:
  friend Printable print(const RematCandidate R);
};

class RematCandidates {
private:
  std::set<RematCandidate> Entries;
  unsigned MaxLoopCost = 0;

public:
  using iterator = typename std::set<RematCandidate>::iterator;
  using const_iterator = typename std::set<RematCandidate>::const_iterator;
  using reverse_iterator = typename std::set<RematCandidate>::reverse_iterator;
  using const_reverse_iterator =
      typename std::set<RematCandidate>::const_reverse_iterator;

  iterator begin() { return Entries.begin(); }
  const_iterator begin() const { return Entries.begin(); }
  iterator end() { return Entries.end(); }
  const_iterator end() const { return Entries.end(); }

  reverse_iterator rbegin() { return Entries.rbegin(); }
  const_reverse_iterator rbegin() const { return Entries.rbegin(); }
  reverse_iterator rend() { return Entries.rend(); }
  const_reverse_iterator rend() const { return Entries.rend(); }

  SmallVector<RematCandidate, 16> Sorted;
  unsigned getDeferCostThreshold() { return MaxLoopCost; }

  bool empty() const { return Entries.empty(); }

  void insert(const RematCandidate &R) {
    if (R.LoopCost > MaxLoopCost) {
      MaxLoopCost = R.LoopCost;
    }
    Entries.insert(R);
  }
  void clear() { Entries.clear(); }

  void sort(const LiveIntervals *LIS) {
    std::set<RematCandidate> Cache = Entries;
    SmallVector<RematCandidate, 8> Temps;
    for (auto RCand : Entries) {
      auto Def = RCand.Def;

      bool FoundUse = false;
      for (auto MO : Def->operands()) {
        if (!MO.isReg() || !MO.isUse())
          continue;

        auto Reg = MO.getReg();

        for (auto OtherCand : Entries) {
          if (OtherCand.Def->definesRegister(Reg, nullptr)) {
            FoundUse = true;
            break;
          }
        }
        if (FoundUse)
          break;
      }

      if (!FoundUse) {
        Temps.push_back(RCand);
        Cache.erase(RCand);
      }
    }

    std::sort(Temps.begin(), Temps.end(),
              [LIS](RematCandidate A, RematCandidate B) {
                auto R1 = A.Def->getOperand(0).getReg();
                auto R2 = B.Def->getOperand(0).getReg();

                if (R1 != R2)
                  return R1 < R2;

                auto P1 = A.InsertPt->getParent()->getNumber();
                auto P2 = B.InsertPt->getParent()->getNumber();

                if (P1 != P2)
                  return P1 < P2;

                return SlotIndex::isEarlierInstr(
                    LIS->getInstructionIndex(*A.InsertPt),
                    LIS->getInstructionIndex(*B.InsertPt));
              });

    Sorted.append(Temps);
    Temps.clear();

    Entries = Cache;

    while (!Entries.empty()) {
      Cache = Entries;

      for (auto RCand : Entries) {
        auto Def = RCand.Def;
        bool FoundUse = false;
        for (auto MO : Def->operands()) {
          if (!MO.isReg() || !MO.isUse())
            continue;

          auto Reg = MO.getReg();

          for (auto OtherCand : Entries) {
            if (OtherCand.Def->definesRegister(Reg, nullptr)) {
              FoundUse = true;
              break;
            }
          }
          if (FoundUse)
            break;
        }

        if (!FoundUse) {
          Temps.push_back(RCand);
          Cache.erase(RCand);
        }
      }
      std::sort(Temps.begin(), Temps.end(),
                [LIS](RematCandidate A, RematCandidate B) {
                  auto R1 = A.Def->getOperand(0).getReg();
                  auto R2 = B.Def->getOperand(0).getReg();

                  if (R1 != R2)
                    return R1 < R2;

                  auto P1 = A.InsertPt->getParent()->getNumber();
                  auto P2 = B.InsertPt->getParent()->getNumber();

                  if (P1 != P2)
                    return P1 < P2;

                  return SlotIndex::isEarlierInstr(
                      LIS->getInstructionIndex(*A.InsertPt),
                      LIS->getInstructionIndex(*B.InsertPt));
                });

      Sorted.append(Temps);
      Temps.clear();
      Entries = Cache;
    }
  }

  bool hoistToDominator(MachineDominatorTree *PDT, MachineCycleInfo &CI,
                        MachineBasicBlock *TargetBlock) {
    DenseMap<MachineInstr *, SmallVector<RematCandidate, 4>> RematMap;

    for (auto E : Entries) {
      RematMap[E.Def].push_back(E);
    }

    auto isReachableFrom = [](MachineBasicBlock *A, MachineBasicBlock *B) {
      std::set<MachineBasicBlock *> Visited;
      std::list<MachineBasicBlock *> Worklist;

      Worklist.push_back(A);

      while (!Worklist.empty()) {
        MachineBasicBlock *TheBlock = Worklist.front();
        Worklist.pop_front();
        if (TheBlock == B)
          return true;
        if (!Visited.insert(TheBlock).second)
          continue;

        for (auto BB : TheBlock->successors()) {
          Worklist.push_back(BB);
        }
      }
      return false;
    };

    std::set<RematCandidate> Cache;

    // errs() << "HoistToDominator\n";
    for (auto RematInfo : RematMap) {
      // errs() << "\nRemat Inst: "; RematInfo.first->dump();
      std::set<unsigned> HighRPs;
      SmallVector<MachineBasicBlock *> MBBs;
      for (auto R : RematInfo.second) {
        for (auto HRP : R.HighRPRegions) {
          HighRPs.insert(HRP);
        }
        MBBs.push_back(R.InsertPt->getParent());
        // errs() << "Has remat point in: " <<
        // printMBBReference(*R.InsertPt->getParent()) << "\n";
      }

      auto DomBlock = PDT->findNearestCommonDominator(iterator_range(MBBs));
      if (DomBlock && isReachableFrom(TargetBlock, DomBlock)) {
        // errs() << "Found dom block: " << printMBBReference(*DomBlock) <<
        // "\n";
        RematCandidate New(RematInfo.first, CI.getCycleDepth(DomBlock), HighRPs,
                           DomBlock->begin());
        Cache.insert(New);
      } else {
        for (auto R : RematInfo.second) {
          Cache.insert(R);
        }
      }
    }

    // errs() << "Condensed: " << Entries.size() << " into: " << Cache.size() <<
    // "\n";
    Entries.clear();
    Entries = Cache;
    return true;
  }

  bool update(RematCandidate &RNew, const LiveIntervals *LIS) {
    // errs() << "Update: "; RNew.Def->dump();
    ////errs() << "Calling update for cand: ";
    // RNew.Def->dump();
    ////errs() << "With Regions: ";
    // for (auto Regi : RNew.HighRPRegions) {
    //   //errs() << Regi;
    // }
    ////errs() << "\n";
    auto Match = find_if(Entries, [RNew](const RematCandidate &R) {
      if (R.Def == RNew.Def) {
        ////errs() << "equal defs for cand match: \n";

        // R.Def->dump();
        ////errs() << "With Regions: ";
        // for (auto Regi : R.HighRPRegions) {
        //   //errs() << Regi;
        // }
        ////errs() << "\n";

        ////errs() << "RNew parent: " << RNew.InsertPt->getParent()->getName()
        ///<< "\n"; /errs() << "R parent: " <<
        ///R.InsertPt->getParent()->getName() << "\n";
      }
      return R.Def == RNew.Def &&
             RNew.InsertPt->getParent() == R.InsertPt->getParent();
    });
    if (Match != Entries.end()) {
      RematCandidate *TheMatch = const_cast<RematCandidate *>(&*Match);

      for (auto NewRegion : RNew.HighRPRegions)
        TheMatch->HighRPRegions.insert(NewRegion);

      if (SlotIndex::isEarlierInstr(
              LIS->getInstructionIndex(*RNew.InsertPt).getRegSlot(),
              LIS->getInstructionIndex(*Match->InsertPt).getRegSlot())) {

        if (RNew.InsertPt != RNew.InsertPt->getParent()->begin())
          TheMatch->InsertPt = &*std::prev(RNew.InsertPt);
        else {
          TheMatch->InsertPt = RNew.InsertPt;
        }

        return true;
      }
    }
    return false;
  }

  bool updateOrInsert(RematCandidate &RNew, const LiveIntervals *LIS) {
    if (!update(RNew, LIS)) {
      insert(RNew);
    }

    return true;
  }

  void resolveSameBlockUses(const MachineRegisterInfo *MRI,
                            const LiveIntervals *LIS) {
    // errs() << "\nResolve Same Block uses";
    // We may have added remat candidates which are used by other remat
    // candidates -- be sure that we have correct insert points for this
    bool FixedPoint = false;
    while (!FixedPoint) {
      // errs() << "Fixed Point iter\n";
      //  //errs() << "Doling fixed point\n";
      FixedPoint = true;
      for (auto &RematEntry : Entries) {

        MachineInstr *RematInst = RematEntry.Def;
        // errs() << "R: "; RematInst->dump();
        // errs() << "For Regions: ";
        // errs() << "\n";
        MachineBasicBlock::iterator RematPt = RematEntry.InsertPt;
        // for (auto RematInst : RematEntry.second) {
        //   //errs() << "Have Remat Inst: "; RematInst.first->dump();
        // //errs() << "With Insert Point: " <<
        // DAG.LIS->getInstructionIndex(*RematInst.second) << "\n";
        for (auto MO : RematInst->operands()) {
          if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
            continue;
          auto UseReg = MO.getReg();
          if (!UseReg.isVirtual())
            continue;
          // //errs() << "Found UseReg: " << printReg(UseReg) << "\n";
          for (MachineInstr &DefInst : MRI->def_instructions(UseReg)) {

            auto Match =
                find_if(Entries, [&DefInst, &RematPt](const RematCandidate &R) {
                  return R.Def == &DefInst &&
                         RematPt->getParent() == R.InsertPt->getParent();
                });

            if (Match == Entries.end())
              continue;

            RematCandidate R(&DefInst, 0, RematEntry.HighRPRegions, RematPt);
            bool MadeChange = update(R, LIS);
            if (MadeChange)
              FixedPoint = false;
          }
        }
        //}
      }
    }
  }

  RematCandidates() {}
  RematCandidates(std::set<RematCandidate> &Entries) : Entries(Entries) {}
};

class PreRARematStage : public GCNSchedStage {
private:
  // Each region at MinOccupancy will have their own list of trivially
  // rematerializable instructions we can remat to reduce RP. The list maps an
  // instruction to the position we should remat before, usually the MI using
  // the rematerializable instruction.
  MapVector<unsigned, MapVector<MachineInstr *, MachineInstr *>>
      RematerializableInsts;

  RematCandidates Cands;

  RematCandidates RematPlan;

  DenseMap<MachineInstr *, SmallPtrSet<MachineBasicBlock *, 16>> ToDelete;

  BitVector RelevantRegions;

  // Map a trivially rematerializable def to a list of regions at MinOccupancy
  // that has the defined reg as a live-in.
  DenseMap<MachineInstr *, SmallVector<unsigned, 4>> RematDefToLiveInRegions;

  DenseMap<unsigned, int> OptRegionRPReduction;

  MachineCycleInfo CI;
  MachineDominatorTree PDT;

  MachineBasicBlock *TargetBlock = nullptr;

  unsigned LiveThruBias = 40;
  unsigned LiveInBias = 3;

  bool canRemat(Register Reg);

  void collectRematSeeds(bool Aggressive = false);

  bool createRematPlan(bool Aggressive = false);

  bool implementRematPlan(const TargetInstrInfo *TII, bool Aggressive = false);

  bool isTriviallyReMaterializable(const MachineInstr &MI);

  bool eliminateDeadMI();
  bool isDead(MachineInstr *MI);

  bool isReachableFrom(MachineBasicBlock *A, MachineBasicBlock *B) {
    std::set<MachineBasicBlock *> Visited;
    std::list<MachineBasicBlock *> Worklist;

    Worklist.push_back(A);

    while (!Worklist.empty()) {
      MachineBasicBlock *TheBlock = Worklist.front();
      Worklist.pop_front();
      if (TheBlock == B)
        return true;
      if (!Visited.insert(TheBlock).second)
        continue;

      for (auto BB : TheBlock->successors()) {
        Worklist.push_back(BB);
      }
    }
    return false;
  }

public:
  bool initGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  PreRARematStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class ILPInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  ILPInitialScheduleStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class MemoryClauseInitialScheduleStage : public GCNSchedStage {
public:
  bool shouldRevertScheduling(unsigned WavesAfter) override;

  MemoryClauseInitialScheduleStage(GCNSchedStageID StageID,
                                   GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG) {}
};

class GCNPostSchedStrategy : public PostGenericScheduler {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand) override;

  SUnit *pickNode(bool &IsTopNode) override;

  // If the XDL resource is not occupied, try to schedule a ready MFMA,
  // otherwise, try not to stall XDL.
  bool tryXDL(SchedCandidate &Cand, SchedCandidate &TryCand);

public:
  // Processor resource for XDL.
  ProcRes XDLProcRes;

  bool CustomResTracking = false;

  GCNPostSchedStrategy(const MachineSchedContext *C);
};

class GCNPostScheduleDAGMILive final : public ScheduleDAGMI {
private:
  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  bool HasIGLPInstrs = false;

public:
  GCNPostSchedStrategy *S = nullptr;

  void schedule() override;

  void finalizeSchedule() override;

  GCNPostScheduleDAGMILive(MachineSchedContext *C,
                           std::unique_ptr<MachineSchedStrategy> S,
                           bool RemoveKillFlags);
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
