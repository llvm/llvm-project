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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include <cstdint>
#include <limits>

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

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.
class GCNSchedStrategy : public GenericScheduler {
protected:
  SUnit *pickNodeBidirectional(bool &IsTopNode, bool &PickedPending);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool &IsPending,
                         bool IsBottomUp);

  void initCandidate(SchedCandidate &Cand, SUnit *SU, bool AtTop,
                     const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI, unsigned SGPRPressure,
                     unsigned VGPRPressure, bool IsBottomUp);

  /// Evaluates instructions in the pending queue using a subset of scheduling
  /// heuristics.
  ///
  /// Instructions that cannot be issued due to hardware constraints are placed
  /// in the pending queue rather than the available queue, making them normally
  /// invisible to scheduling heuristics. However, in certain scenarios (such as
  /// avoiding register spilling), it may be beneficial to consider scheduling
  /// these not-yet-ready instructions.
  bool tryPendingCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                           SchedBoundary *Zone) const;

  void printCandidateDecision(const SchedCandidate &Current,
                              const SchedCandidate &Preferred);

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
  ScheduleMetrics() = default;
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
  RegionPressureMap() = default;
  RegionPressureMap(GCNScheduleDAGMILive *GCNDAG, bool LiveOut)
      : DAG(GCNDAG), IsLiveOut(LiveOut) {}
  // Build the Instr->LiveReg and RegionIdx->Instr maps
  void buildLiveRegMap();

  // Retrieve the LiveReg for a given RegionIdx
  GCNRPTracker::LiveRegSet &getLiveRegsForRegionIdx(unsigned RegionIdx) {
    assert(IdxToInstruction.contains(RegionIdx));
    MachineInstr *Key = IdxToInstruction[RegionIdx];
    return RegionLiveRegMap[Key];
  }
};

/// A region's boundaries i.e. a pair of instruction bundle iterators. The lower
/// boundary is inclusive, the upper boundary is exclusive.
using RegionBoundaries =
    std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

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
  SmallVector<RegionBoundaries, 32> Regions;

  // Record regions with high register pressure.
  BitVector RegionsWithHighRP;

  // Record regions with excess register pressure over the physical register
  // limit. Register pressure in these regions usually will result in spilling.
  BitVector RegionsWithExcessRP;

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

  void runSchedStages();

  std::unique_ptr<GCNSchedStage> createSchedStage(GCNSchedStageID SchedStageID);

  void deleteMI(unsigned RegionIdx, MachineInstr *MI);

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
  // Save the temporary target occupancy before starting this stage.
  unsigned TempTargetOccupancy;
  // Track whether any region was scheduled by this stage.
  bool IsAnyRegionScheduled;

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

/// Attempts to reduce function spilling or, if there is no spilling, to
/// increase function occupancy by one with respect to register usage by sinking
/// rematerializable instructions to their use. When the stage estimates that
/// reducing spilling or increasing occupancy is possible, it tries to
/// rematerialize as few registers as possible to reduce potential negative
/// effects on function latency.
///
/// The stage only supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual and has a single defining instruction.
/// 2. The single defining instruction is either deemed rematerializable by the
///    target-independent logic, or if not, has no non-constant and
///    non-ignorable physical register use.
/// 3  The register has no virtual register use whose live range would be
///    extended by the rematerialization.
/// 4. The register has a single non-debug user in a different region from its
///    defining region.
/// 5. The register is not used by or using another register that is going to be
///    rematerialized.
class PreRARematStage : public GCNSchedStage {
private:
  /// A rematerializable register.
  struct RematReg {
    /// Single MI defining the rematerializable register.
    MachineInstr *DefMI;
    /// Single user of the rematerializable register.
    MachineInstr *UseMI;
    /// Regions in which the register is live-in/live-out/live anywhere.
    BitVector LiveIn, LiveOut, Live;
    /// The rematerializable register's lane bitmask.
    LaneBitmask Mask;
    /// Defining and using regions.
    unsigned DefRegion, UseRegion;

    RematReg(MachineInstr *DefMI, MachineInstr *UseMI,
             GCNScheduleDAGMILive &DAG,
             const DenseMap<MachineInstr *, unsigned> &MIRegion);

    /// Returns the rematerializable register. Do not call after deleting the
    /// original defining instruction.
    Register getReg() const { return DefMI->getOperand(0).getReg(); }

    /// Determines whether this rematerialization may be beneficial in at least
    /// one target region.
    bool maybeBeneficial(const BitVector &TargetRegions,
                         ArrayRef<GCNRPTarget> RPTargets) const;

    /// Determines if the register is both unused and live-through in region \p
    /// I. This guarantees that rematerializing it will reduce RP in the region.
    bool isUnusedLiveThrough(unsigned I) const {
      assert(I < Live.size() && "region index out of range");
      return LiveIn[I] && LiveOut[I] && I != UseRegion;
    }

    /// Updates internal structures following a MI rematerialization. Part of
    /// the stage instead of the DAG because it makes assumptions that are
    /// specific to the rematerialization process.
    void insertMI(unsigned RegionIdx, MachineInstr *RematMI,
                  GCNScheduleDAGMILive &DAG) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void print() const;
#endif
  };

  /// A scored rematerializable register. Higher scores indicate more beneficial
  /// rematerializations. A null score indicate the rematerialization is
  /// not helpful to reduce RP in target regions.
  struct ScoredRemat {
    /// The rematerializable register under consideration.
    const RematReg *Remat;

    /// Execution frequency information required by scoring heuristics.
    struct FreqInfo {
      /// Per-region execution frequencies, normalized to minimum observed
      /// frequency. 0 when unknown.
      SmallVector<uint64_t> Regions;
      /// Maximum observed frequency, normalized to minimum observed frequency.
      uint64_t MaxFreq = 0;
      /// Rescaling factor for scoring frequency differences in the range [0, 2
      /// * (MaxFreq - 1)].
      uint64_t RescaleFactor = 0;
      /// Whether the rescaling factor should be used as a denominator (when the
      /// maximum frequency is "big") or as a nominator (when the maximum
      /// frequency is "small").
      bool RescaleIsDenom = false;

      FreqInfo(MachineFunction &MF, const GCNScheduleDAGMILive &DAG);
    };

    /// This only initializes state-independent characteristics of \p Remat, not
    /// the actual score.
    ScoredRemat(const RematReg *Remat, const FreqInfo &Freq,
                const GCNScheduleDAGMILive &DAG);

    /// Updates the rematerialization's score w.r.t. the current \p RPTargets.
    /// \p RegionFreq indicates the frequency of each region
    void update(const BitVector &TargetRegions, ArrayRef<GCNRPTarget> RPTargets,
                const FreqInfo &Freq, bool ReduceSpill);

    /// Returns whether the current score is null.
    bool hasNullScore() const { return !Score; }

    bool operator<(const ScoredRemat &O) const {
      // Break ties using pointer to rematerializable register. Since
      // rematerializations are collected in instruction order, registers
      // appearing earlier have a "higher score" than those appearing later.
      if (Score == O.Score)
        return Remat > O.Remat;
      return Score < O.Score;
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void print() const;
#endif

  private:
    /// Bitwidths for score components.
    static constexpr unsigned MaxFreqWidth = 32, FreqDiffWidth = 16,
                              RegionImpactWidth = 16;

    /// Number of 32-bit registers this rematerialization covers.
    const unsigned NumRegs;
    /// Frequency difference between defining and using regions, normalized to
    /// the maximum possible difference and rescaled to the representable range
    /// in the score.
    const uint64_t FreqDiff;

    using ScoreTy = uint64_t;
    /// Rematerialization score. Scoring components are mapped to bit ranges in
    /// the score.
    ///
    /// [63:32] : maximum frequency in benefiting target region (spilling only)
    /// [31:16] : frequency difference between defining and using region
    /// [15: 0] : number of benefiting regions times register size
    ScoreTy Score = 0;

    void setNullScore() { Score = 0; }

    void setMaxFreqScore(ScoreTy MaxFreq) {
      MaxFreq = std::min(
          static_cast<ScoreTy>(std::numeric_limits<uint32_t>::max()), MaxFreq);
      MaxFreq <<= FreqDiffWidth + RegionImpactWidth;

      ScoreTy Mask = ((ScoreTy)1 << (FreqDiffWidth + RegionImpactWidth)) - 1;
      Score = MaxFreq | (Score & Mask);
    }

    void setFreqDiffScore(ScoreTy FreqDiff) {
      FreqDiff = std::min(
          static_cast<ScoreTy>(std::numeric_limits<uint16_t>::max()), FreqDiff);
      FreqDiff <<= RegionImpactWidth;

      ScoreTy Mask = ((ScoreTy)1 << (FreqDiffWidth)) - 1;
      Mask <<= RegionImpactWidth;
      Score = FreqDiff | (Score & ~Mask);
    }

    void setRegionImpactScore(ScoreTy RegionImpact) {
      RegionImpact =
          std::min(static_cast<ScoreTy>(std::numeric_limits<uint16_t>::max()),
                   RegionImpact);

      ScoreTy Mask = ((ScoreTy)1 << (RegionImpactWidth)) - 1;
      Score = RegionImpact | (Score & ~Mask);
    }

    unsigned getNumRegs(const GCNScheduleDAGMILive &DAG) const;

    uint64_t getFreqDiff(const FreqInfo &Freq) const;
  };

  /// Holds enough information to rollback a rematerialization decision post
  /// re-scheduling.
  struct RollbackInfo {
    /// The rematerializable register under consideration.
    const RematReg *Remat;
    /// The rematerialized MI replacing the original defining MI.
    MachineInstr *RematMI;

    RollbackInfo(const RematReg *Remat) : Remat(Remat) {}
  };

  /// Parent MBB to each region, in region order.
  SmallVector<MachineBasicBlock *> RegionBB;

  /// Register pressure targets for all regions.
  SmallVector<GCNRPTarget> RPTargets;
  /// Regions which are above the stage's RP target.
  BitVector TargetRegions;
  /// The target occupancy the set is trying to achieve. Empty when the
  /// objective is spilling reduction.
  std::optional<unsigned> TargetOcc;
  /// Achieved occupancy *only* through rematerializations (pre-rescheduling).
  /// Smaller than or equal to the target occupancy, when it is defined.
  unsigned AchievedOcc;

  /// List of rematerializable registers.
  SmallVector<RematReg, 16> RematRegs;
  /// List of rematerializations to rollback if rematerialization does not end
  /// up being beneficial.
  SmallVector<RollbackInfo> Rollbacks;
  /// After successful stage initialization, indicates which regions should be
  /// rescheduled.
  BitVector RescheduleRegions;

  /// Determines the stage's objective (increasing occupancy or reducing
  /// spilling, set in \ref TargetOcc). Defines \ref RPTargets in all regions to
  /// achieve that objective and mark those that don't achieve it in \ref
  /// TargetRegions. Returns whether there is any target region.
  bool setObjective();

  /// Unsets target regions in \p Regions whose RP target has been reached.
  void unsetSatisifedRPTargets(const BitVector &Regions);

  /// Fully recomputes RP from the DAG in \p Regions. Among those regions, sets
  /// again all \ref TargetRegions that were optimistically marked as satisfied
  /// but are actually not, and returns whether there were any such regions.
  bool updateAndVerifyRPTargets(const BitVector &Regions);

  /// Collects all rematerializable registers and appends them to \ref
  /// RematRegs. \p MIRegion maps MIs to their region. Returns whether any
  /// rematerializable register was found.
  bool collectRematRegs(const DenseMap<MachineInstr *, unsigned> &MIRegion);

  /// Rematerializes \p Remat. This removes the rematerialized register from
  /// live-in/out lists in the DAG and updates RP targets in all affected
  /// regions, which are also marked in \ref RescheduleRegions. Regions in which
  /// RP savings are not guaranteed are set in \p RecomputeRP. When \p Rollback
  /// is non-null, fills it with required information to be able to rollback the
  /// rematerialization post-rescheduling.
  void rematerialize(const RematReg &Remat, BitVector &RecomputeRP,
                     RollbackInfo *Rollback);

  /// Rollbacks the rematerialization decision represented by \p Rollback. This
  /// update live-in/out lists in the DAG but does not update cached register
  /// pressures. Regions in which RP may be impacted are marked in \ref
  /// RecomputeRP.
  void rollback(const RollbackInfo &Rollback, BitVector &RecomputeRP) const;

  /// Whether the MI is rematerializable
  bool isReMaterializable(const MachineInstr &MI);

  /// If remat alone did not increase occupancy to the target one, rollbacks all
  /// rematerializations and resets live-ins/RP in all regions impacted by the
  /// stage to their pre-stage values.
  void finalizeGCNSchedStage() override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printTargetRegions(bool PrintAll = false) const;
#endif
public:
  bool initGCNSchedStage() override;

  bool initGCNRegion() override;

  bool shouldRevertScheduling(unsigned WavesAfter) override;

  PreRARematStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
      : GCNSchedStage(StageID, DAG), TargetRegions(DAG.Regions.size()),
        RescheduleRegions(DAG.Regions.size()) {
    const unsigned NumRegions = DAG.Regions.size();
    RPTargets.reserve(NumRegions);
    RegionBB.reserve(NumRegions);
  }
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

class GCNPostScheduleDAGMILive final : public ScheduleDAGMI {
private:
  std::vector<std::unique_ptr<ScheduleDAGMutation>> SavedMutations;

  bool HasIGLPInstrs = false;

public:
  void schedule() override;

  void finalizeSchedule() override;

  GCNPostScheduleDAGMILive(MachineSchedContext *C,
                           std::unique_ptr<MachineSchedStrategy> S,
                           bool RemoveKillFlags);
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
