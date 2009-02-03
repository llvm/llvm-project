//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/Statistic.h"
#include <climits>
using namespace llvm;

STATISTIC(NumNoops , "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");

static RegisterScheduler
  tdListDAGScheduler("list-td", "Top-down list scheduler",
                     createTDListDAGScheduler);
   
namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGList - The actual list scheduler implementation.  This supports
/// top-down scheduling.
///
class VISIBILITY_HIDDEN ScheduleDAGList : public ScheduleDAGSDNodes {
private:
  /// AvailableQueue - The priority queue to use for the available SUnits.
  ///
  SchedulingPriorityQueue *AvailableQueue;
  
  /// PendingQueue - This contains all of the instructions whose operands have
  /// been issued, but their results are not ready yet (due to the latency of
  /// the operation).  Once the operands become available, the instruction is
  /// added to the AvailableQueue.
  std::vector<SUnit*> PendingQueue;

  /// HazardRec - The hazard recognizer to use.
  ScheduleHazardRecognizer *HazardRec;

public:
  ScheduleDAGList(MachineFunction &mf,
                  SchedulingPriorityQueue *availqueue,
                  ScheduleHazardRecognizer *HR)
    : ScheduleDAGSDNodes(mf),
      AvailableQueue(availqueue), HazardRec(HR) {
    }

  ~ScheduleDAGList() {
    delete HazardRec;
    delete AvailableQueue;
  }

  void Schedule();

private:
  void ReleaseSucc(SUnit *SU, const SDep &D);
  void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
  void ListScheduleTopDown();
};
}  // end anonymous namespace

/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGList::Schedule() {
  DOUT << "********** List Scheduling **********\n";
  
  // Build the scheduling graph.
  BuildSchedGraph();

  AvailableQueue->initNodes(SUnits);
  
  ListScheduleTopDown();
  
  AvailableQueue->releaseState();
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGList::ReleaseSucc(SUnit *SU, const SDep &D) {
  SUnit *SuccSU = D.getSUnit();
  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  SuccSU->setDepthToAtLeast(SU->getDepth() + D.getLatency());
  
  if (SuccSU->NumPredsLeft == 0) {
    PendingQueue.push_back(SuccSU);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));
  
  Sequence.push_back(SU);
  assert(CurCycle >= SU->getDepth() && "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);

  // Top down: release successors.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    assert(!I->isAssignedRegDep() &&
           "The list-td scheduler doesn't yet support physreg dependencies!");

    ReleaseSucc(SU, *I);
  }

  SU->isScheduled = true;
  AvailableQueue->ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue->empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i]->getDepth() == CurCycle) {
        AvailableQueue->push(PendingQueue[i]);
        PendingQueue[i]->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else {
        assert(PendingQueue[i]->getDepth() > CurCycle && "Negative latency?");
      }
    }
    
    // If there are no instructions available, don't try to issue anything, and
    // don't advance the hazard recognizer.
    if (AvailableQueue->empty()) {
      ++CurCycle;
      continue;
    }

    SUnit *FoundSUnit = 0;
    
    bool HasNoopHazards = false;
    while (!AvailableQueue->empty()) {
      SUnit *CurSUnit = AvailableQueue->pop();
      
      ScheduleHazardRecognizer::HazardType HT =
        HazardRec->getHazardType(CurSUnit);
      if (HT == ScheduleHazardRecognizer::NoHazard) {
        FoundSUnit = CurSUnit;
        break;
      }
    
      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == ScheduleHazardRecognizer::NoopHazard;
      
      NotReady.push_back(CurSUnit);
    }
    
    // Add the nodes that aren't ready back onto the available list.
    if (!NotReady.empty()) {
      AvailableQueue->push_all(NotReady);
      NotReady.clear();
    }

    // If we found a node to schedule, do it now.
    if (FoundSUnit) {
      ScheduleNodeTopDown(FoundSUnit, CurCycle);
      HazardRec->EmitInstruction(FoundSUnit);

      // If this is a pseudo-op node, we don't want to increment the current
      // cycle.
      if (FoundSUnit->Latency)  // Don't increment CurCycle for pseudo-ops!
        ++CurCycle;        
    } else if (!HasNoopHazards) {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DOUT << "*** Advancing cycle, no work to do\n";
      HazardRec->AdvanceCycle();
      ++NumStalls;
      ++CurCycle;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DOUT << "*** Emitting noop\n";
      HazardRec->EmitNoop();
      Sequence.push_back(0);   // NULL here means noop
      ++NumNoops;
      ++CurCycle;
    }
  }

#ifndef NDEBUG
  VerifySchedule(/*isBottomUp=*/false);
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

/// createTDListDAGScheduler - This creates a top-down list scheduler with a
/// new hazard recognizer. This scheduler takes ownership of the hazard
/// recognizer and deletes it when done.
ScheduleDAG* llvm::createTDListDAGScheduler(SelectionDAGISel *IS,
                                            bool Fast) {
  return new ScheduleDAGList(*IS->MF,
                             new LatencyPriorityQueue(),
                             IS->CreateTargetHazardRecognizer());
}
