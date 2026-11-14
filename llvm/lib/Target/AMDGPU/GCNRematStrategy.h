//===-- GCNRematStrategy.h - Rematerialization strategy -*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Rematerialization strategy, based on the target-independent rematerializer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNREMATSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_GCNREMATSTRATEGY_H

#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/Rematerializer.h"

namespace llvm {

/// Rematerializes registers across code regions to achieve register pressure
/// targets. The strategy updates region boundaries and live-ins/live-outs as it
/// rematerializes. It uses the target independent rematerializers to expose
/// rematerializable registers and perform rematerializations.
///
/// The strategy only supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual and has a single defining instruction.
/// 2. The single defining instruction is either deemed rematerializable by the
///    target-independent logic, or if not, has no non-constant and
///    non-ignorable physical register use.
/// 3  The register has no virtual register use whose live range would be
///    extended by the rematerialization. When reducing spilling under
///    -amdgpu-remat-extend-deps, rematerializable dependencies holding a single
///    value that still reaches the rematerialization points are exempt; their
///    live range is extended instead. This is only allowed when the
///    register's own pressure covers the pressure of those dependencies in
///    every register class, such that the rematerialization cannot increase
///    RP anywhere.
/// 4. The register has at least one user outside its defining region. If
///    rematerialized, one copy is made per using region and the original
///    register stays if the defining region has users.
/// 5. The register is not used by or using another register that is going to be
///    rematerialized.
class GCNRematStrategy {
  using RegionBoundaries = Rematerializer::RegionBoundaries;
  using RegisterIdx = Rematerializer::RegisterIdx;

public:
  /// Creates a strategy that can rematerialize across \p Regions whose live-ins
  /// \p LiveIns and live-outs \p LiveOuts are provided. The underlying
  /// rematerializer is created and initialized.
  GCNRematStrategy(MachineFunction &MF,
                   MutableArrayRef<RegionBoundaries> Regions,
                   MutableArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
                   MutableArrayRef<GCNRPTracker::LiveRegSet> LiveOuts,
                   LiveIntervals &LIS, const MachineBlockFrequencyInfo &MBFI)
      : Regions(Regions, LiveIns, LiveOuts, LIS, MBFI, MF.getRegInfo()),
        Remater(MF, Regions, LIS), MF(MF), MRI(MF.getRegInfo()), LIS(LIS) {}

  /// Rematerializes with the goal of reducing the maximum pressure in all \ref
  /// Regions to \p MaxSGPRs SGPRs and \p MaxVGPRs VGPRs. \p SpillReduction
  /// indicates whether that register pressure target allows to reduce spilling,
  /// which influences rematerialization decisions. Regions which are
  /// rematerialized from/into/over have their corresponding index set in \p
  /// AffectedRegions. When not null, \p MaxPressure provides initial maximum
  /// pressure for all regions and contains updated maximum pressures when the
  /// function returns.
  void rematerializeFor(unsigned MaxSGPRs, unsigned MaxVGPRs,
                        bool SpillReduction, BitVector &AffectedRegions,
                        MutableArrayRef<GCNRegPressure> *MaxPressure = nullptr);

  /// Starts recording rematerializations for eventual rollback.
  void startRecordingRemats() {
    assert(!Recorder && "recorder already defined");
    Recorder = std::make_unique<RematRecorder>(Remater);
  }

  /// Stops recording rematerializations and clears all recorded ones.
  void clearRecordedRemats() {
    assert(Recorder && "recorder not defined");
    Remater.removeListener(&Recorder->Listener);
    Recorder.reset(nullptr);
  }

  /// Rolls back all recorded rematerializations.
  void rollbackRecordedRemats();

private:
  /// Groups information about code regions to rematerialize across.
  struct RematRegions {
    /// Boundaries of regions across which the strategy rematerializes.
    /// FIXME: the rematerializer updates them when necessary, but the strategy
    /// should be responsible for that.
    ArrayRef<RegionBoundaries> Bounds;

    RematRegions(ArrayRef<RegionBoundaries> Regions,
                 MutableArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
                 MutableArrayRef<GCNRPTracker::LiveRegSet> LiveOuts,
                 LiveIntervals &LIS, const MachineBlockFrequencyInfo &MBFI,
                 const MachineRegisterInfo &MRI);

    /// Returns the number of rematerialization regions.
    unsigned size() const { return Bounds.size(); }

    uint64_t getFrequency(unsigned RegionIdx) const {
      return Frequencies[RegionIdx];
    }

    uint64_t getFrequencyOr(unsigned RegionIdx, uint64_t Default = 0) const {
      uint64_t RegionFreq = Frequencies[RegionIdx];
      return RegionFreq ? RegionFreq : Default;
    }

    uint64_t getFrequencyOrMax(unsigned RegionIdx) const {
      return getFrequencyOr(RegionIdx, MaxFreq);
    }

    uint64_t getFrequencyOrMin(unsigned RegionIdx) const {
      return getFrequencyOr(RegionIdx, MinFreq);
    }

    /// Returns up-to-date live-ins for region \p RegionIdx.
    const GCNRPTracker::LiveRegSet &getLiveIns(unsigned RegionIdx) const {
      assert(RegionIdx < size() && "out of bounds");
      return LiveIns[RegionIdx];
    }

    /// Returns up-to-date live-outs for region \p RegionIdx.
    const GCNRPTracker::LiveRegSet &getLiveOuts(unsigned RegionIdx) const {
      assert(RegionIdx < size() && "out of bounds");
      return LiveOuts[RegionIdx];
    }

    /// Computes the up-to-date maximum register pressure in \p RegionIdx.
    GCNRegPressure computeMaxRegPressure(unsigned RegionIdx) const;

    /// Removes register \p Reg from the live-ins of regions set in \p LiveIn
    /// and the live-outs of regions set in \p LiveOut.
    void removeFromLiveMaps(Register Reg, const BitVector &LiveIn,
                            const BitVector &LiveOut);

    /// Adds register \p Reg with mask \p Mask to the live-ins of regions set in
    /// \p LiveIn and the live-outs of regions set in \p LiveOut.
    void addToLiveMaps(Register Reg, LaneBitmask Mask, const BitVector &LiveIn,
                       const BitVector &LiveOut);

    /// Sets the live-in and live-out lanes of register \p Reg in regions set in
    /// \p Regions from its current live interval. Returns the regions in which
    /// \p Reg became live-in or live-out.
    BitVector refreshLiveMaps(Register Reg, const BitVector &Regions);

  private:
    static const uint64_t FreqScaleFactor = 1024;

    /// Current live-ins and live-outs for each region.
    MutableArrayRef<GCNRPTracker::LiveRegSet> LiveIns, LiveOuts;
    /// Scaled per-region execution frequencies. 0 when unknown.
    SmallVector<uint64_t> Frequencies;
    /// Scaled minimum and maximum observed frequencies.
    uint64_t MinFreq, MaxFreq;

    LiveIntervals &LIS;
    const MachineRegisterInfo &MRI;
  };

  /// Tracks progress in reducing register pressure against a target.
  struct RematTargets {
    /// Register pressure targets for all regions.
    SmallVector<GCNRPTarget> RPTargets;
    /// Regions which are above the RP target.
    BitVector ExcessRP;
    /// Regions to rematerialize across.
    const RematRegions &Regions;
    /// Whether the register pressure target allows to reduce spilling.
    const bool SpillReduction;

    /// Sets a target \ref of maximum \p MaxSGPRs SGPRs and \p MaxVGPRs VGPRs
    /// per region. \p SpillReduction indicates whether that register pressure
    /// target allows to reduce spilling. When not null, \p MaxPressure provides
    /// initial maximum pressure for all regions.
    RematTargets(unsigned MaxSGPRs, unsigned MaxVGPRs, bool SpillReduction,
                 const MachineFunction &MF, const RematRegions &Regions,
                 ArrayRef<GCNRegPressure> *MaxPressure = nullptr);

    /// Returns whether the RP target is thought to be achieved in all regions.
    bool allAchieved() const { return ExcessRP.none(); }

    /// In all regions set in \p UpdateRegions, saves pressure \p RPSave and
    /// clear it as a target if its RP target has been reached.
    void updateRPTargets(const BitVector &UpdateRegions,
                         const GCNRegPressure &RPSave);

    /// In all regions set in \p UpdateRegions, adds pressure \p AddedRP and
    /// sets it as a target if its RP target is no longer reached.
    void addRP(const BitVector &UpdateRegions, const GCNRegPressure &AddedRP);

    /// Fully recomputes RP from the DAG in \p UpdateRegions. Among those
    /// regions, sets again all targets that were optimistically marked as
    /// satisfied but are actually not, and returns whether there were any such
    /// regions.
    bool updateAndVerifyRPTargets(const BitVector &UpdateRegions);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    Printable print() const;
#endif
  };

  /// A scored rematerialization candidate. Higher scores indicate more
  /// beneficial rematerializations. A null score indicates the
  /// rematerialization is not helpful to reduce RP in regions with excess RP.
  struct RematCandidate {
    /// The register index handle in the rematerializer.
    RegisterIdx RegIdx;
    /// Rematerializable dependencies whose live range has to be extended to
    /// reach the rematerialization points.
    SmallVector<RegisterIdx, 2> ExtendedDeps;
    /// Regions in which the register is live-in/live-out/live anywhere.
    BitVector LiveIn, LiveOut, Live;
    /// Subset of \ref Live regions in which the rematerialization is not
    /// guaranteed to reduce RP (i.e., regions in which the register is not
    /// live-through and unused).
    BitVector UnpredictableRPSave;
    /// Expected register pressure decrease induced by rematerializing this
    /// candidate.
    GCNRegPressure RPSave;

    RematCandidate(RegisterIdx RegIdx, unsigned NumRegions)
        : RegIdx(RegIdx), LiveIn(NumRegions), LiveOut(NumRegions),
          Live(NumRegions), UnpredictableRPSave(NumRegions) {}

    /// Initializes the candidate with state-independent characteristics.
    /// This doesn't update the actual score (call \ref update for this).
    /// Note: LiveIn/LiveOut must be pre-populated before calling this.
    void init(const RematRegions &RF, const Rematerializer &Remater,
              const MachineRegisterInfo &MRI);

    /// Updates the rematerialization's score w.r.t. the current \p Targets.
    void update(const RematTargets &Targets);

    /// Determines whether this rematerialization may be beneficial in at
    /// least one region with excess RP.
    bool maybeBeneficial(const RematTargets &Targets) const;

    /// Rematerializes the candidate using the \p Remater.
    void rematerialize(Rematerializer &Remater) const;

    /// Returns whether the current score is null, indicating the
    /// rematerialization is useless.
    bool hasNullScore() const { return !RegionImpact; }

    /// Compare score components of non-null scores pair-wise. Scores
    /// shouldn't be null (as defined by \ref hasNullScore).
    bool operator<(const RematCandidate &O) const {
      assert(!hasNullScore() && "this has null score");
      assert(!O.hasNullScore() && "other has null score");
      if (MaxFreq != O.MaxFreq)
        return MaxFreq < O.MaxFreq;
      if (FreqDiff != O.FreqDiff)
        return FreqDiff < O.FreqDiff;
      if (RegionImpact != O.RegionImpact)
        return RegionImpact < O.RegionImpact;
      // Break ties using register index handles. If the two registers are
      // connected in some dependency DAG of rematerializable registers, this
      // will tend to give a higher score to the register further from the
      // dependency DAG's root. If the two registers are disconnected, this
      // will give a higher score to the register with lower virtual register
      // index. In general, within a region, this should prefer registers
      // defined earlier that have longer live ranges in their defining region
      // (since the registers we consider are always live-out in their
      // defining region).
      return RegIdx > O.RegIdx;
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    Printable print() const;
#endif

  private:
    // The three members below are the scoring components, top to bottom from
    // most important to least important when comparing candidates.

    /// Frequency of impacted target region with highest known frequency. This
    /// only matters when the strategy is trying to reduce spilling, so it is
    /// always 0 when it is not.
    uint64_t MaxFreq;
    /// Frequency difference between defining and using regions. Negative
    /// values indicate we are rematerializing to higher frequency regions;
    /// positive values indicate the contrary.
    int64_t FreqDiff;
    /// Expected number of target regions impacted by the rematerialization,
    /// scaled by the size of the register being rematerialized.
    unsigned RegionImpact;
  };

  struct RematRecorder {
    struct LiveMapUpdate {
      /// The register index handle in the rematerializer.
      RegisterIdx RegIdx;
      /// Regions in which the original register was live-in or live-out.
      BitVector LiveIn, LiveOut;

      LiveMapUpdate(RegisterIdx RegIdx, const BitVector &LiveIn,
                    const BitVector &LiveOut)
          : RegIdx(RegIdx), LiveIn(LiveIn), LiveOut(LiveOut) {}
    };

    /// Rematerializer listener which can rollback rematerializations.
    Rollbacker Listener;
    /// Registers removed from live-maps along with bitvectors indicationg the
    /// regions in which they were live-ins and live-outs.
    SmallVector<LiveMapUpdate> LiveMapUpdates;

    /// Attaches the rollback listener to the rematerializer.
    RematRecorder(Rematerializer &Remater) { Remater.addListener(&Listener); }
  };

  /// Code regions to rematerialize across. Region bounds are updated by the
  /// rematerializer as necessary when rematerializations occur at region
  /// boundaries.
  RematRegions Regions;
  /// Rematerializer, initialized with the same rematerialization regions as the
  /// strategy.
  Rematerializer Remater;
  /// Candidates for rematerializations.
  SmallVector<RematCandidate, 0> Candidates;
  /// Whether candidates were already created, implying that the rematerializer
  /// was also fully initialized.
  bool CandidatesCreated = false;
  /// When not null, records all rematerializations.
  std::unique_ptr<RematRecorder> Recorder;

  const MachineFunction &MF;
  const MachineRegisterInfo &MRI;
  LiveIntervals &LIS;

  /// Creates rematerialization candidates. The strategy has more restrictions
  /// on what it can rematerialize compared to the rematerializer. When \p
  /// AllowExtendDeps is set, candidates may have rematerializable dependencies
  /// whose live range has to be extended to reach the rematerialization points.
  void createCandidates(bool AllowExtendDeps);

  /// Determines whether users of rematerialization candidate \p CandIdx are
  /// compatible with the strategy's rematerialization constraints/limitations.
  /// Registers in \p MarkedRegs have already been deemed rematerializable by
  /// the strategy. When \p ExtendedDeps is non-null, a rematerializable
  /// dependency which is not live at a rematerialization point but whose value
  /// still reaches it does not invalidate the candidate; it is instead added to
  /// \p ExtendedDeps.
  bool candidateHasValidUsers(
      RegisterIdx CandIdx, const SmallSet<Register, 4> &MarkedRegs,
      SmallVectorImpl<RegisterIdx> *ExtendedDeps = nullptr) const;

  /// Determines whether the pressure of rematerialization candidate \p CandIdx
  /// covers the summed pressure of \p ExtendedDeps in every register class.
  /// Extended dependencies only become live in regions where the candidate was
  /// live and no longer is after rematerialization. So when this function
  /// returns true the rematerialization does notincrease RP in any of these
  /// regions.
  bool
  candidateSaveCoversExtendedDeps(RegisterIdx CandIdx,
                                  ArrayRef<RegisterIdx> ExtendedDeps) const;

  /// Refreshes live-ins/live-outs of \p Cand's extended dependencies in the
  /// regions where \p Cand was live, and adds the dependencies' pressure to
  /// \p Targets in regions where they became live-in or live-out.
  void updateExtendedDeps(const RematCandidate &Cand, RematTargets &Targets);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREMATSTRATEGY_H
