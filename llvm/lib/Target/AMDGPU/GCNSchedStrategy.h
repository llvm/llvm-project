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
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class SIMachineFunctionInfo;
class SIRegisterInfo;
class GCNSubtarget;

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that GCNSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.  Its goal is to
/// maximize kernel occupancy (i.e. maximum number of waves per simd).
class GCNMaxOccupancySchedStrategy final : public GenericScheduler {
  friend class GCNScheduleDAGMILive;

  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand);

  void initCandidate(SchedCandidate &Cand, SUnit *SU,
                     bool AtTop, const RegPressureTracker &RPTracker,
                     const SIRegisterInfo *SRI,
                     unsigned SGPRPressure, unsigned VGPRPressure);

  std::vector<unsigned> Pressure;
  std::vector<unsigned> MaxPressure;

  unsigned SGPRExcessLimit;
  unsigned VGPRExcessLimit;
  unsigned SGPRCriticalLimit;
  unsigned VGPRCriticalLimit;

  unsigned TargetOccupancy;

  // schedule() have seen a clustered memory operation. Set it to false
  // before a region scheduling to know if the region had such clusters.
  bool HasClusteredNodes;

  // schedule() have seen an excess register pressure and had to track
  // register pressure for actual scheduling heuristics.
  bool HasExcessPressure;

  MachineFunction *MF;

public:
  GCNMaxOccupancySchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }
};

class GCNScheduleDAGMILive final : public ScheduleDAGMILive {

  enum : unsigned {
    Collect,
    InitialSchedule,
    UnclusteredReschedule,
    ClusteredLowOccupancyReschedule,
    PreRARematerialize,
    LastStage = PreRARematerialize
  };

  const GCNSubtarget &ST;

  SIMachineFunctionInfo &MFI;

  // Occupancy target at the beginning of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Scheduling stage number.
  unsigned Stage;

  // Current region index.
  size_t RegionIdx;

  // Vector of regions recorder for later rescheduling
  SmallVector<std::pair<MachineBasicBlock::iterator,
                        MachineBasicBlock::iterator>, 32> Regions;

  // Records if a region is not yet scheduled, or schedule has been reverted,
  // or we generally desire to reschedule it.
  BitVector RescheduleRegions;

  // Record regions which use clustered loads/stores.
  BitVector RegionsWithClusters;

  // Record regions with high register pressure.
  BitVector RegionsWithHighRP;

  // Regions that has the same occupancy as the latest MinOccupancy
  BitVector RegionsWithMinOcc;

  // Region live-in cache.
  SmallVector<GCNRPTracker::LiveRegSet, 32> LiveIns;

  // Region pressure cache.
  SmallVector<GCNRegPressure, 32> Pressure;

  // Each region at MinOccupancy will have their own list of trivially
  // rematerializable instructions we can remat to reduce RP. The list maps an
  // instruction to the position we should remat before, usually the MI using
  // the rematerializable instruction.
  MapVector<unsigned, MapVector<MachineInstr *, MachineInstr *>>
      RematerializableInsts;

  // Map a trivially remateriazable def to a list of regions at MinOccupancy
  // that has the defined reg as a live-in.
  DenseMap<MachineInstr *, SmallVector<unsigned, 4>> RematDefToLiveInRegions;

  // Temporary basic block live-in cache.
  DenseMap<const MachineBasicBlock*, GCNRPTracker::LiveRegSet> MBBLiveIns;

  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> BBLiveInMap;
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> getBBLiveInMap() const;

  // Collect all trivially rematerializable VGPR instructions with a single def
  // and single use outside the defining block into RematerializableInsts.
  void collectRematerializableInstructions();

  bool isTriviallyReMaterializable(const MachineInstr &MI, AAResults *AA);

  // TODO: Should also attempt to reduce RP of SGPRs and AGPRs
  // Attempt to reduce RP of VGPR by sinking trivially rematerializable
  // instructions. Returns true if we were able to sink instruction(s).
  bool sinkTriviallyRematInsts(const GCNSubtarget &ST,
                               const TargetInstrInfo *TII);

  // Return current region pressure.
  GCNRegPressure getRealRegPressure() const;

  // Compute and cache live-ins and pressure for all regions in block.
  void computeBlockPressure(const MachineBasicBlock *MBB);

  // Update region boundaries when removing MI or inserting NewMI before MI.
  void updateRegionBoundaries(
      SmallVectorImpl<std::pair<MachineBasicBlock::iterator,
                                MachineBasicBlock::iterator>> &RegionBoundaries,
      MachineBasicBlock::iterator MI, MachineInstr *NewMI,
      bool Removing = false);

public:
  GCNScheduleDAGMILive(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSCHEDSTRATEGY_H
