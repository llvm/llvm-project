//==- SystemZMachineScheduler.h - SystemZ Scheduler Interface ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// -------------------------- Pre RA scheduling ----------------------------- //
//
// SystemZPreRASchedStrategy keeps track of currently live registers and first
// tries to reduce live ranges by scheduling e.g. a load of a live register
// immediately (bottom-up). It also aims to preserve the scheduled latency.
// Small regions (up to 10 instructions) are mostly left alone as the input
// order is usually then preferred.
//
// -------------------------- Post RA scheduling ---------------------------- //
//
// SystemZPostRASchedStrategy is a scheduling strategy which is plugged into
// the MachineScheduler. It has a sorted Available set of SUs and a pickNode()
// implementation that looks to optimize decoder grouping and balance the
// usage of processor resources. Scheduler states are saved for the end
// region of each MBB, so that a successor block can learn from it.
//
//----------------------------------------------------------------------------//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H

#include "SystemZHazardRecognizer.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include <set>

namespace llvm {

/// A MachineSchedStrategy implementation for SystemZ pre RA  scheduling.
class SystemZPreRASchedStrategy : public GenericScheduler {
  // The FP/Vector registers are prioritized during scheduling.
  std::set<unsigned> PrioRegClasses;
  void initializePrioRegClasses(const TargetRegisterInfo *TRI);
  bool isPrioVirtReg(Register Reg, const MachineRegisterInfo *MRI) const {
    return (Reg.isVirtual() &&
            PrioRegClasses.count(MRI->getRegClass(Reg)->getID()));
  }

  // A TinyRegion has up to 10 instructions and is scheduled differently.
  bool TinyRegion;

  // Num instructions left to schedule.
  unsigned NumLeft;

  // True if latency scheduling is enabled.
  bool ShouldReduceLatency;

  // Keep track of currently live registers.
  class VRegSet {
    std::set<Register> Regs;

  public:
    void clear() { Regs.clear(); }
    void insert(Register Reg);
    void erase(Register Reg);
    bool count(Register Reg) const;
    void dump() const;
  } LiveRegs;

  // True if MI is also using the register it defines.
  std::vector<bool> IsRedefining;

  // Only call computeRemLatency() once per scheduled node.
  mutable unsigned RemLat;
  unsigned getRemLat(SchedBoundary *Zone) const;

  // A large group of stores at the bottom is spread upwards.
  std::set<const SUnit *> StoresGroup;
  bool FirstStoreInGroupScheduled;
  void initializeStoresGroup();

  // Compute the effect on register liveness by scheduling C next. An
  // instruction that defines a live register without causing any other
  // register to become live reduces liveness, while a store of a non-live
  // register would increase it.
  int computeSULivenessScore(SchedCandidate &C, ScheduleDAGMILive *DAG,
                             SchedBoundary *Zone) const;

protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

public:
  SystemZPreRASchedStrategy(const MachineSchedContext *C)
      : GenericScheduler(C) {
    initializePrioRegClasses(C->MF->getRegInfo().getTargetRegisterInfo());
  }

  void initPolicy(MachineBasicBlock::iterator Begin,
                  MachineBasicBlock::iterator End,
                  unsigned NumRegionInstrs) override;
  void initialize(ScheduleDAGMI *dag) override;
  void schedNode(SUnit *SU, bool IsTopNode) override;
};

/// A MachineSchedStrategy implementation for SystemZ post RA scheduling.
class SystemZPostRASchedStrategy : public MachineSchedStrategy {

  const MachineLoopInfo *MLI;
  const SystemZInstrInfo *TII;

  // A SchedModel is needed before any DAG is built while advancing past
  // non-scheduled instructions, so it would not always be possible to call
  // DAG->getSchedClass(SU).
  TargetSchedModel SchedModel;

  /// A candidate during instruction evaluation.
  struct Candidate {
    SUnit *SU = nullptr;

    /// The decoding cost.
    int GroupingCost = 0;

    /// The processor resources cost.
    int ResourcesCost = 0;

    Candidate() = default;
    Candidate(SUnit *SU_, SystemZHazardRecognizer &HazardRec);

    // Compare two candidates.
    bool operator<(const Candidate &other);

    // Check if this node is free of cost ("as good as any").
    bool noCost() const {
      return (GroupingCost <= 0 && !ResourcesCost);
    }

#ifndef NDEBUG
    void dumpCosts() {
      if (GroupingCost != 0)
        dbgs() << "  Grouping cost:" << GroupingCost;
      if (ResourcesCost != 0)
        dbgs() << "  Resource cost:" << ResourcesCost;
    }
#endif
  };

  // A sorter for the Available set that makes sure that SUs are considered
  // in the best order.
  struct SUSorter {
    bool operator() (SUnit *lhs, SUnit *rhs) const {
      if (lhs->isScheduleHigh && !rhs->isScheduleHigh)
        return true;
      if (!lhs->isScheduleHigh && rhs->isScheduleHigh)
        return false;

      if (lhs->getHeight() > rhs->getHeight())
        return true;
      else if (lhs->getHeight() < rhs->getHeight())
        return false;

      return (lhs->NodeNum < rhs->NodeNum);
    }
  };
  // A set of SUs with a sorter and dump method.
  struct SUSet : std::set<SUnit*, SUSorter> {
    #ifndef NDEBUG
    void dump(SystemZHazardRecognizer &HazardRec) const;
    #endif
  };

  /// The set of available SUs to schedule next.
  SUSet Available;

  /// Current MBB
  MachineBasicBlock *MBB;

  /// Maintain hazard recognizers for all blocks, so that the scheduler state
  /// can be maintained past BB boundaries when appropariate.
  typedef std::map<MachineBasicBlock*, SystemZHazardRecognizer*> MBB2HazRec;
  MBB2HazRec SchedStates;

  /// Pointer to the HazardRecognizer that tracks the scheduler state for
  /// the current region.
  SystemZHazardRecognizer *HazardRec;

  /// Update the scheduler state by emitting (non-scheduled) instructions
  /// up to, but not including, NextBegin.
  void advanceTo(MachineBasicBlock::iterator NextBegin);

public:
  SystemZPostRASchedStrategy(const MachineSchedContext *C);
  virtual ~SystemZPostRASchedStrategy();

  /// Called for a region before scheduling.
  void initPolicy(MachineBasicBlock::iterator Begin,
                  MachineBasicBlock::iterator End,
                  unsigned NumRegionInstrs) override;

  /// PostRA scheduling does not track pressure.
  bool shouldTrackPressure() const override { return false; }

  // Process scheduling regions top-down so that scheduler states can be
  // transferrred over scheduling boundaries.
  bool doMBBSchedRegionsTopDown() const override { return true; }

  void initialize(ScheduleDAGMI *dag) override;

  /// Tell the strategy that MBB is about to be processed.
  void enterMBB(MachineBasicBlock *NextMBB) override;

  /// Tell the strategy that current MBB is done.
  void leaveMBB() override;

  /// Pick the next node to schedule, or return NULL.
  SUnit *pickNode(bool &IsTopNode) override;

  /// ScheduleDAGMI has scheduled an instruction - tell HazardRec
  /// about it.
  void schedNode(SUnit *SU, bool IsTopNode) override;

  /// SU has had all predecessor dependencies resolved. Put it into
  /// Available.
  void releaseTopNode(SUnit *SU) override;

  /// Currently only scheduling top-down, so this method is empty.
  void releaseBottomNode(SUnit *SU) override {};
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H
