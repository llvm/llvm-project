//===- AMDGPUSim/SimState.h - Single-wave simulation state -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines mutable timing, resource, dependency, and pending operation state
/// for a single simulated wave.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMSTATE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMSTATE_H

#include "HWModel.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <climits>
#include <deque>
#include <optional>

namespace llvm {
namespace AMDGPUSim {

/// One pending memory operation tracked by its absolute completion cycle.
struct PendingMemOp {
  unsigned CompletionCycle;

  explicit PendingMemOp(unsigned Complete) : CompletionCycle(Complete) {}
};

/// Mutable timing, resource, dependency, and pending operation state for one
/// simulated wave.
///
/// Cycle fields are absolute and pending operation queues preserve issue order.
struct GPUSimState {
  /// Current absolute simulator cycle.
  unsigned CurrentCycle = 0;
  /// First cycle when each functional unit becomes available.
  std::array<unsigned, static_cast<size_t>(FunctionalUnit::NUM_UNITS)>
      UnitBusyUntil = {};

  /// State and stage metadata for a WMMA coexecution window.
  struct WMMACoExecState {
    /// First absolute cycle in the window.
    unsigned StartCycle = 0;
    /// First absolute cycle after the window.
    unsigned EndCycle = 0;
    /// Whether the window is currently active.
    bool Active = false;
    /// Coexecution stages and resource occupancy for the window.
    AMDGPU::CoExecInfo Info;

    /// Return the stage at \p Cycle, or no value outside
    /// [StartCycle, EndCycle).
    std::optional<unsigned> getCurrentStage(unsigned Cycle) const {
      if (!Active || Cycle < StartCycle || Cycle >= EndCycle)
        return std::nullopt;
      return Cycle - StartCycle;
    }
  };

  /// Current WMMA coexecution window state.
  WMMACoExecState ActiveWMMA;

  /// Issue timing for one VALU or TRANS dependency referenced by S_DELAY_ALU.
  struct RecentInst {
    /// Cycle when the instruction issued.
    unsigned IssueCycle;
    /// Latency from issue until the dependency becomes ready.
    unsigned Latency;
  };
  std::deque<RecentInst> RecentVALU;
  std::deque<RecentInst> RecentTRANS;

  /// Most recent SALU issue cycle used to enforce S_DELAY_ALU spacing.
  unsigned LastSALUCycle = 0;

  /// One pending VALU, TRANS, or WMMA result counted by the modeled VaVdst
  /// counter.
  struct PendingVALUWrite {
    /// Cycle when the result becomes ready.
    unsigned ReadyCycle;
  };
  std::deque<PendingVALUWrite> PendingVaVdst;

  /// Most recent VALU resource issue cycle, including scaled WMMA scale reads.
  /// All ones indicates no prior VALU resource issue.
  unsigned LastVALUCycle = ~0u;
  /// Most recent TRANS issue, or all ones when no TRANS has issued.
  unsigned LastTRANSCycle = ~0u;
  /// First cycle when the modeled VALU resource is available.
  unsigned VALUResourceBusyUntil = 0;
  /// First cycle when SALU issue is allowed after a modeled scalar source use.
  unsigned VaSSRCBusyUntil = 0;
  /// First cycle after the current long latency VALU and TRANS hazard.
  unsigned LOLVALUTRANSHazardUntil = 0;

  /// Deferred second S_DELAY_ALU dependency and its remaining skip count.
  struct PendingDelayAlu {
    unsigned DepType;
    unsigned InstructionsLeft;
  };
  std::optional<PendingDelayAlu> PendingInstId1;
  InstClass PreviousInstClass = InstClass::OTHER;

  std::deque<PendingMemOp> PendingDS;
  std::deque<PendingMemOp> PendingVMEMLoad;
  std::deque<PendingMemOp> PendingVMEMStore;
  std::deque<PendingMemOp> PendingSMEM;
  std::deque<PendingMemOp> PendingTDM;
  std::deque<PendingMemOp> PendingXACK;

  /// Return whether a WMMA coexecution window is active.
  bool inWMMAWindow() const { return ActiveWMMA.Active; }

  /// Return the window relative stage at CurrentCycle, if any.
  std::optional<unsigned> getWMMAStage() const {
    return ActiveWMMA.getCurrentStage(CurrentCycle);
  }

  /// Return the first cycle when \p Unit becomes available, or zero for NONE.
  unsigned getUnitBusyUntil(FunctionalUnit Unit) const {
    if (Unit == FunctionalUnit::NONE)
      return 0;
    return UnitBusyUntil[static_cast<size_t>(Unit)];
  }

  /// Return the delay needed to issue WMMA at least two cycles after TRANS.
  unsigned getWMMATRANSStall() const {
    if (LastTRANSCycle == ~0u)
      return 0;
    unsigned EndCycle = LastTRANSCycle + 2;
    return CurrentCycle < EndCycle ? EndCycle - CurrentCycle : 0;
  }

  /// Return the coexecution delay for \p IC at absolute \p Cycle.
  unsigned getCoExecStallAt(InstClass IC, unsigned Cycle) const {
    std::optional<unsigned> Stage = ActiveWMMA.getCurrentStage(Cycle);
    if (!Stage)
      return 0;
    return ActiveWMMA.Info.getStallCycles(getCoExecMask(IC), *Stage);
  }

  /// Return the coexecution delay for \p IC at CurrentCycle.
  unsigned getCoExecStall(InstClass IC) const {
    return getCoExecStallAt(IC, CurrentCycle);
  }

  /// Return whether the active window can absorb a scaled WMMA whose LD_SCALE
  /// stage issues at \p ScaleReadCycle and whose matrix stage follows it.
  bool canAbsorbScaledWMMAAt(unsigned ScaleReadCycle) const {
    if (!ActiveWMMA.Active || ScaleReadCycle < ActiveWMMA.StartCycle ||
        ScaleReadCycle + 1 >= ActiveWMMA.EndCycle)
      return false;

    const AMDGPU::CoExecInfo &CoExec = ActiveWMMA.Info;
    unsigned ScaleStage = ScaleReadCycle - ActiveWMMA.StartCycle;
    unsigned MatrixStage = ScaleStage + 1;
    return CoExec.canCoExec(getCoExecMask(InstClass::VALU), ScaleStage) &&
           CoExec.getType(MatrixStage) == AMDGPU::CoExecStageType::V;
  }

  /// Resolve the earliest absorbable cycle in the active window.
  /// Return \p FromCycle without a window and the window end without a slot.
  unsigned resolveScaledWMMAAbsorbCycle(unsigned FromCycle) const {
    if (!ActiveWMMA.Active || FromCycle >= ActiveWMMA.EndCycle)
      return FromCycle;
    // A window that has not started cannot absorb the complete instruction.
    if (FromCycle < ActiveWMMA.StartCycle)
      return ActiveWMMA.EndCycle;
    for (unsigned Cycle = FromCycle; Cycle < ActiveWMMA.EndCycle; ++Cycle)
      if (canAbsorbScaledWMMAAt(Cycle))
        return Cycle;
    return ActiveWMMA.EndCycle;
  }

  /// Record an LD_SCALE issue at \p ScaleReadCycle on the VALU resource.
  void reserveScaledWMMAScaleRead(unsigned ScaleReadCycle) {
    VALUResourceBusyUntil = std::max(VALUResourceBusyUntil, ScaleReadCycle + 1);
    LastVALUCycle = ScaleReadCycle;
  }

  void setUnitBusyUntil(FunctionalUnit Unit, unsigned Cycle) {
    if (Unit != FunctionalUnit::NONE)
      UnitBusyUntil[static_cast<size_t>(Unit)] = Cycle;
  }

  void advanceCycle(unsigned Count = 1) {
    advanceToCycle(CurrentCycle + Count);
  }

  /// Advance to \p TargetCycle and retire operations that are now complete.
  /// Return the number of cycles advanced.
  unsigned advanceToCycle(unsigned TargetCycle) {
    if (TargetCycle <= CurrentCycle)
      return 0;
    unsigned Delta = TargetCycle - CurrentCycle;
    CurrentCycle = TargetCycle;
    if (ActiveWMMA.Active && CurrentCycle >= ActiveWMMA.EndCycle)
      ActiveWMMA.Active = false;
    retireCompletedMemOps();
    while (!PendingVaVdst.empty() &&
           PendingVaVdst.front().ReadyCycle <= CurrentCycle)
      PendingVaVdst.pop_front();
    return Delta;
  }

  /// Record a VALU issue for telemetry or a TRANS issue for WMMA spacing.
  void trackVALUForWMMA(InstClass IC) {
    if (IC == InstClass::VALU)
      LastVALUCycle = CurrentCycle;
    else if (IC == InstClass::TRANS)
      LastTRANSCycle = CurrentCycle;
  }

  /// Extend VALU occupancy only while a WMMA window is active.
  void holdVALUResourceInWindow(unsigned Cycles) {
    if (inWMMAWindow())
      VALUResourceBusyUntil =
          std::max(VALUResourceBusyUntil, CurrentCycle + Cycles);
  }

  /// Record a VALU issue for later S_DELAY_ALU dependency decoding.
  void trackVALU(unsigned Latency) {
    RecentVALU.push_back({CurrentCycle, Latency});
    if (RecentVALU.size() > 5)
      RecentVALU.pop_front();
  }

  /// Record a TRANS issue for later S_DELAY_ALU dependency decoding.
  void trackTRANS(unsigned Latency) {
    RecentTRANS.push_back({CurrentCycle, Latency});
    if (RecentTRANS.size() > 4)
      RecentTRANS.pop_front();
  }

  /// Return the modeled VaVdst count, capped at its counter maximum.
  unsigned getVaVdst() const {
    unsigned Count = 0;
    for (const PendingVALUWrite &Write : PendingVaVdst)
      if (Write.ReadyCycle > CurrentCycle)
        ++Count;
    return std::min(Count, 15u);
  }

  /// Return the first cycle when the modeled VaVdst count is at most
  /// \p Target. Pending writes retire in issue order.
  unsigned getVaVdstReadyCycle(unsigned Target) const {
    unsigned CurrentCount = getVaVdst();
    if (CurrentCount <= Target)
      return CurrentCycle;

    unsigned ToRetire = CurrentCount - Target;
    unsigned LastRetire = CurrentCycle;
    SmallVector<unsigned, 16> RetireTimes;
    for (const PendingVALUWrite &Write : PendingVaVdst) {
      if (Write.ReadyCycle > CurrentCycle) {
        LastRetire = std::max(LastRetire, Write.ReadyCycle);
        RetireTimes.push_back(LastRetire);
      }
    }
    return ToRetire <= RetireTimes.size() ? RetireTimes[ToRetire - 1]
                                          : CurrentCycle;
  }

  /// Record a VaVdst write ready after \p Latency times \p Multiplier cycles.
  void trackVaVdst(unsigned Latency, unsigned Multiplier) {
    PendingVaVdst.push_back({CurrentCycle + Latency * Multiplier});
  }

  /// Start a window, reserve XDL, and return its occupancy.
  /// Clear the window and return zero when unmodeled.
  unsigned startWMMAWindow(const WMMAProperties &Properties,
                           unsigned MatrixStartCycle) {
    std::optional<AMDGPU::CoExecInfo> Info =
        AMDGPU::getKnownCoExecInfo(Properties);
    if (!Info || Info->TotalWindow == 0) {
      ActiveWMMA = WMMACoExecState();
      return 0;
    }

    const AMDGPU::CoExecInfo &CoExec = *Info;
    ActiveWMMA.StartCycle = MatrixStartCycle;
    ActiveWMMA.EndCycle = MatrixStartCycle + CoExec.TotalWindow;
    ActiveWMMA.Active = true;
    ActiveWMMA.Info = CoExec;
    setUnitBusyUntil(FunctionalUnit::XDL,
                     MatrixStartCycle + CoExec.UnitOccupancy);
    return CoExec.UnitOccupancy;
  }

  /// Return the issue delay until the issue ordered queue has capacity.
  unsigned getFIFOStall(const std::deque<PendingMemOp> &Queue,
                        unsigned MaxInFlight) const {
    if (Queue.size() < MaxInFlight)
      return 0;
    unsigned OldestCompletion = Queue.front().CompletionCycle;
    return OldestCompletion > CurrentCycle ? OldestCompletion - CurrentCycle
                                           : 0;
  }

  /// Return the capacity stall for the combined VMEM load and store queues.
  unsigned getVMEMBufferStall(unsigned MaxInFlight) const {
    if (PendingVMEMLoad.size() + PendingVMEMStore.size() < MaxInFlight)
      return 0;
    unsigned OldestCompletion = UINT_MAX;
    if (!PendingVMEMLoad.empty())
      OldestCompletion =
          std::min(OldestCompletion, PendingVMEMLoad.front().CompletionCycle);
    if (!PendingVMEMStore.empty())
      OldestCompletion =
          std::min(OldestCompletion, PendingVMEMStore.front().CompletionCycle);
    return OldestCompletion > CurrentCycle ? OldestCompletion - CurrentCycle
                                           : 0;
  }

  /// Add an operation that completes \p Latency cycles after CurrentCycle.
  void issueMemOp(std::deque<PendingMemOp> &Queue, unsigned Latency) {
    Queue.emplace_back(CurrentCycle + Latency);
  }

  void issueDS(unsigned Latency) { issueMemOp(PendingDS, Latency); }

  void issueVMEM(unsigned Latency, bool IsLoad) {
    std::deque<PendingMemOp> &Queue =
        IsLoad ? PendingVMEMLoad : PendingVMEMStore;
    issueMemOp(Queue, Latency);
  }

  void issueSMEM(unsigned Latency) { issueMemOp(PendingSMEM, Latency); }

  void issueTDM(unsigned Latency) { issueMemOp(PendingTDM, Latency); }

  void issueXACK(unsigned Latency) { issueMemOp(PendingXACK, Latency); }

  /// Return the cycles until at most \p WaitCount operations remain pending.
  unsigned computeWaitStall(const std::deque<PendingMemOp> &Queue,
                            unsigned WaitCount) const {
    unsigned Pending = Queue.size();
    if (Pending <= WaitCount)
      return 0;
    unsigned WaitForIndex = Pending - WaitCount - 1;
    unsigned CompletionCycle = Queue[WaitForIndex].CompletionCycle;
    return CompletionCycle > CurrentCycle ? CompletionCycle - CurrentCycle : 0;
  }

  void retireCompletedFrom(std::deque<PendingMemOp> &Queue) {
    while (!Queue.empty() && Queue.front().CompletionCycle <= CurrentCycle)
      Queue.pop_front();
  }

  /// Remove oldest tracked operations until \p Queue satisfies \p WaitCount.
  /// This mutates pending state without advancing CurrentCycle.
  void applyWait(std::deque<PendingMemOp> &Queue, unsigned WaitCount) {
    while (Queue.size() > WaitCount)
      Queue.pop_front();
  }

  // The wait methods below return the required delay and apply the requested
  // queue threshold. They mutate pending state without advancing CurrentCycle.
  unsigned waitDS(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingDS, Count);
    applyWait(PendingDS, Count);
    return Stall;
  }
  unsigned waitVMEMLoad(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingVMEMLoad, Count);
    applyWait(PendingVMEMLoad, Count);
    return Stall;
  }
  unsigned waitVMEMStore(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingVMEMStore, Count);
    applyWait(PendingVMEMStore, Count);
    return Stall;
  }
  unsigned waitSMEM(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingSMEM, Count);
    applyWait(PendingSMEM, Count);
    return Stall;
  }
  unsigned waitTensor(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingTDM, Count);
    applyWait(PendingTDM, Count);
    return Stall;
  }
  unsigned waitXCnt(unsigned Count) {
    unsigned Stall = computeWaitStall(PendingXACK, Count);
    applyWait(PendingXACK, Count);
    return Stall;
  }

  void retireCompletedMemOps() {
    retireCompletedFrom(PendingDS);
    retireCompletedFrom(PendingVMEMLoad);
    retireCompletedFrom(PendingVMEMStore);
    retireCompletedFrom(PendingSMEM);
    retireCompletedFrom(PendingTDM);
    retireCompletedFrom(PendingXACK);
  }

  /// Reset all cycle, resource, dependency, and pending operation state.
  void reset() {
    CurrentCycle = 0;
    UnitBusyUntil.fill(0);
    ActiveWMMA = WMMACoExecState();
    RecentVALU.clear();
    RecentTRANS.clear();
    LastSALUCycle = 0;
    PendingVaVdst.clear();
    LastVALUCycle = ~0u;
    LastTRANSCycle = ~0u;
    VALUResourceBusyUntil = 0;
    VaSSRCBusyUntil = 0;
    LOLVALUTRANSHazardUntil = 0;
    PendingInstId1.reset();
    PreviousInstClass = InstClass::OTHER;
    PendingDS.clear();
    PendingVMEMLoad.clear();
    PendingVMEMStore.clear();
    PendingSMEM.clear();
    PendingTDM.clear();
    PendingXACK.clear();
  }
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
