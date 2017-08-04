//-- SystemZMachineScheduler.cpp - SystemZ Scheduler Interface -*- C++ -*---==//
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

#include "SystemZMachineScheduler.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

#ifndef NDEBUG
// Print the set of SUs
void SystemZPostRASchedStrategy::SUSet::
dump(SystemZHazardRecognizer &HazardRec) const {
  dbgs() << "{";
  for (auto &SU : *this) {
    HazardRec.dumpSU(SU, dbgs());
    if (SU != *rbegin())
      dbgs() << ",  ";
  }
  dbgs() << "}\n";
}
#endif

SystemZPostRASchedStrategy::
SystemZPostRASchedStrategy(const MachineSchedContext *C)
  : DAG(nullptr), HazardRec(C) {}

void SystemZPostRASchedStrategy::initialize(ScheduleDAGMI *dag) {
  DAG = dag;
  HazardRec.setDAG(dag);
  HazardRec.Reset();
}

// Pick the next node to schedule.
SUnit *SystemZPostRASchedStrategy::pickNode(bool &IsTopNode) {
  // Only scheduling top-down.
  IsTopNode = true;

  if (Available.empty())
    return nullptr;

  // If only one choice, return it.
  if (Available.size() == 1) {
    DEBUG (dbgs() << "+++ Only one: ";
           HazardRec.dumpSU(*Available.begin(), dbgs()); dbgs() << "\n";);
    return *Available.begin();
  }

  // All nodes that are possible to schedule are stored by in the
  // Available set.
  DEBUG(dbgs() << "+++ Available: "; Available.dump(HazardRec););

  Candidate Best;
  for (auto *SU : Available) {

    // SU is the next candidate to be compared against current Best.
    Candidate c(SU, HazardRec);

    // Remeber which SU is the best candidate.
    if (Best.SU == nullptr || c < Best) {
      Best = c;
      DEBUG(dbgs() << "+++ Best sofar: ";
            HazardRec.dumpSU(Best.SU, dbgs());
            if (Best.GroupingCost != 0)
              dbgs() << "\tGrouping cost:" << Best.GroupingCost;
            if (Best.ResourcesCost != 0)
              dbgs() << " Resource cost:" << Best.ResourcesCost;
            dbgs() << " Height:" << Best.SU->getHeight();
            dbgs() << "\n";);
    }

    // Once we know we have seen all SUs that affect grouping or use unbuffered
    // resources, we can stop iterating if Best looks good.
    if (!SU->isScheduleHigh && Best.noCost())
      break;
  }

  assert (Best.SU != nullptr);
  return Best.SU;
}

SystemZPostRASchedStrategy::Candidate::
Candidate(SUnit *SU_, SystemZHazardRecognizer &HazardRec) : Candidate() {
  SU = SU_;

  // Check the grouping cost. For a node that must begin / end a
  // group, it is positive if it would do so prematurely, or negative
  // if it would fit naturally into the schedule.
  GroupingCost = HazardRec.groupingCost(SU);

    // Check the resources cost for this SU.
  ResourcesCost = HazardRec.resourcesCost(SU);
}

bool SystemZPostRASchedStrategy::Candidate::
operator<(const Candidate &other) {

  // Check decoder grouping.
  if (GroupingCost < other.GroupingCost)
    return true;
  if (GroupingCost > other.GroupingCost)
    return false;

  // Compare the use of resources.
  if (ResourcesCost < other.ResourcesCost)
    return true;
  if (ResourcesCost > other.ResourcesCost)
    return false;

  // Higher SU is otherwise generally better.
  if (SU->getHeight() > other.SU->getHeight())
    return true;
  if (SU->getHeight() < other.SU->getHeight())
    return false;

  // If all same, fall back to original order.
  if (SU->NodeNum < other.SU->NodeNum)
    return true;

  return false;
}

void SystemZPostRASchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  DEBUG(dbgs() << "+++ Scheduling SU(" << SU->NodeNum << ")\n";);

  // Remove SU from Available set and update HazardRec.
  Available.erase(SU);
  HazardRec.EmitInstruction(SU);
}

void SystemZPostRASchedStrategy::releaseTopNode(SUnit *SU) {
  // Set isScheduleHigh flag on all SUs that we want to consider first in
  // pickNode().
  const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
  bool AffectsGrouping = (SC->isValid() && (SC->BeginGroup || SC->EndGroup));
  SU->isScheduleHigh = (AffectsGrouping || SU->isUnbuffered);

  // Put all released SUs in the Available set.
  Available.insert(SU);
}
