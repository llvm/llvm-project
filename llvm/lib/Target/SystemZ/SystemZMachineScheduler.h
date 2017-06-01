//==-- SystemZMachineScheduler.h - SystemZ Scheduler Interface -*- C++ -*---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// -------------------------- Post RA scheduling ---------------------------- //
// SystemZPostRASchedStrategy is a scheduling strategy which is plugged into
// the MachineScheduler. It has a sorted Available set of SUs and a pickNode()
// implementation that looks to optimize decoder grouping and balance the
// usage of processor resources.
//===----------------------------------------------------------------------===//

#include "SystemZInstrInfo.h"
#include "SystemZHazardRecognizer.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H

using namespace llvm;

namespace llvm {
  
/// A MachineSchedStrategy implementation for SystemZ post RA scheduling.
class SystemZPostRASchedStrategy : public MachineSchedStrategy {
    ScheduleDAGMI *DAG;
  
  /// A candidate during instruction evaluation.
  struct Candidate {
    SUnit *SU;

    /// The decoding cost.
    int GroupingCost;

    /// The processor resources cost.
    int ResourcesCost;

    Candidate() : SU(nullptr), GroupingCost(0), ResourcesCost(0) {}
    Candidate(SUnit *SU_, SystemZHazardRecognizer &HazardRec);

    // Compare two candidates.
    bool operator<(const Candidate &other);

    // Check if this node is free of cost ("as good as any").
    bool inline noCost() {
      return (GroupingCost <= 0 && !ResourcesCost);
    }
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
    void dump(SystemZHazardRecognizer &HazardRec);
    #endif
  };

  /// The set of available SUs to schedule next.
  SUSet Available;

  // HazardRecognizer that tracks the scheduler state for the current
  // region.
  SystemZHazardRecognizer HazardRec;
  
 public:
  SystemZPostRASchedStrategy(const MachineSchedContext *C);

  /// PostRA scheduling does not track pressure.
  bool shouldTrackPressure() const override { return false; }

  /// Initialize the strategy after building the DAG for a new region.
  void initialize(ScheduleDAGMI *dag) override;

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

} // namespace llvm

#endif /* LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINESCHEDULER_H */
