//===-- HexagonMachineScheduler.h - Custom Hexagon MI scheduler.      ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Custom Hexagon MI scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONMACHINESCHEDULER_H

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ResourcePriorityQueue.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

namespace llvm {
//===----------------------------------------------------------------------===//
// ConvergingVLIWScheduler - Implementation of the standard
// MachineSchedStrategy.
//===----------------------------------------------------------------------===//

class VLIWResourceModel {
  /// ResourcesModel - Represents VLIW state.
  /// Not limited to VLIW targets per say, but assumes
  /// definition of DFA by a target.
  DFAPacketizer *ResourcesModel;

  const TargetSchedModel *SchedModel;

  /// Local packet/bundle model. Purely
  /// internal to the MI schedulre at the time.
  std::vector<SUnit*> Packet;

  /// Total packets created.
  unsigned TotalPackets;

public:
  /// Save the last formed packet.
  std::vector<SUnit*> OldPacket;

public:
  VLIWResourceModel(const TargetSubtargetInfo &STI, const TargetSchedModel *SM)
      : SchedModel(SM), TotalPackets(0) {
  ResourcesModel = STI.getInstrInfo()->CreateTargetScheduleState(STI);

    // This hard requirement could be relaxed,
    // but for now do not let it proceed.
    assert(ResourcesModel && "Unimplemented CreateTargetScheduleState.");

    Packet.resize(SchedModel->getIssueWidth());
    Packet.clear();
    OldPacket.resize(SchedModel->getIssueWidth());
    OldPacket.clear();
    ResourcesModel->clearResources();
  }

  ~VLIWResourceModel() {
    delete ResourcesModel;
  }

  void resetPacketState() {
    Packet.clear();
  }

  void resetDFA() {
    ResourcesModel->clearResources();
  }

  void reset() {
    Packet.clear();
    ResourcesModel->clearResources();
  }

  bool isResourceAvailable(SUnit *SU);
  bool reserveResources(SUnit *SU);
  void savePacket();
  unsigned getTotalPackets() const { return TotalPackets; }

  bool isInPacket(SUnit *SU) const {
    return std::find(Packet.begin(), Packet.end(), SU) != Packet.end();
  }
};

/// Extend the standard ScheduleDAGMI to provide more context and override the
/// top-level schedule() driver.
class VLIWMachineScheduler : public ScheduleDAGMILive {
public:
  VLIWMachineScheduler(MachineSchedContext *C,
                       std::unique_ptr<MachineSchedStrategy> S)
      : ScheduleDAGMILive(C, std::move(S)) {}

  /// Schedule - This is called back from ScheduleDAGInstrs::Run() when it's
  /// time to do some work.
  void schedule() override;
};

/// ConvergingVLIWScheduler shrinks the unscheduled zone using heuristics
/// to balance the schedule.
class ConvergingVLIWScheduler : public MachineSchedStrategy {

  /// Store the state used by ConvergingVLIWScheduler heuristics, required
  ///  for the lifetime of one invocation of pickNode().
  struct SchedCandidate {
    // The best SUnit candidate.
    SUnit *SU;

    // Register pressure values for the best candidate.
    RegPressureDelta RPDelta;

    // Best scheduling cost.
    int SCost;

    SchedCandidate(): SU(nullptr), SCost(0) {}
  };
  /// Represent the type of SchedCandidate found within a single queue.
  enum CandResult {
    NoCand, NodeOrder, SingleExcess, SingleCritical, SingleMax, MultiPressure,
    BestCost};

  /// Each Scheduling boundary is associated with ready queues. It tracks the
  /// current cycle in whichever direction at has moved, and maintains the state
  /// of "hazards" and other interlocks at the current cycle.
  struct VLIWSchedBoundary {
    VLIWMachineScheduler *DAG;
    const TargetSchedModel *SchedModel;

    ReadyQueue Available;
    ReadyQueue Pending;
    bool CheckPending;

    ScheduleHazardRecognizer *HazardRec;
    VLIWResourceModel *ResourceModel;

    unsigned CurrCycle;
    unsigned IssueCount;

    /// MinReadyCycle - Cycle of the soonest available instruction.
    unsigned MinReadyCycle;

    // Remember the greatest min operand latency.
    unsigned MaxMinLatency;

    /// Pending queues extend the ready queues with the same ID and the
    /// PendingFlag set.
    VLIWSchedBoundary(unsigned ID, const Twine &Name):
      DAG(nullptr), SchedModel(nullptr), Available(ID, Name+".A"),
      Pending(ID << ConvergingVLIWScheduler::LogMaxQID, Name+".P"),
      CheckPending(false), HazardRec(nullptr), ResourceModel(nullptr),
      CurrCycle(0), IssueCount(0),
      MinReadyCycle(UINT_MAX), MaxMinLatency(0) {}

    ~VLIWSchedBoundary() {
      delete ResourceModel;
      delete HazardRec;
    }

    void init(VLIWMachineScheduler *dag, const TargetSchedModel *smodel) {
      DAG = dag;
      SchedModel = smodel;
      IssueCount = 0;
    }

    bool isTop() const {
      return Available.getID() == ConvergingVLIWScheduler::TopQID;
    }

    bool checkHazard(SUnit *SU);

    void releaseNode(SUnit *SU, unsigned ReadyCycle);

    void bumpCycle();

    void bumpNode(SUnit *SU);

    void releasePending();

    void removeReady(SUnit *SU);

    SUnit *pickOnlyChoice();
  };

  VLIWMachineScheduler *DAG;
  const TargetSchedModel *SchedModel;

  // State of the top and bottom scheduled instruction boundaries.
  VLIWSchedBoundary Top;
  VLIWSchedBoundary Bot;

public:
  /// SUnit::NodeQueueId: 0 (none), 1 (top), 2 (bot), 3 (both)
  enum {
    TopQID = 1,
    BotQID = 2,
    LogMaxQID = 2
  };

  ConvergingVLIWScheduler()
    : DAG(nullptr), SchedModel(nullptr), Top(TopQID, "TopQ"),
      Bot(BotQID, "BotQ") {}

  void initialize(ScheduleDAGMI *dag) override;

  SUnit *pickNode(bool &IsTopNode) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  void releaseTopNode(SUnit *SU) override;

  void releaseBottomNode(SUnit *SU) override;

  unsigned ReportPackets() {
    return Top.ResourceModel->getTotalPackets() +
           Bot.ResourceModel->getTotalPackets();
  }

protected:
  SUnit *pickNodeBidrectional(bool &IsTopNode);

  int SchedulingCost(ReadyQueue &Q,
                     SUnit *SU, SchedCandidate &Candidate,
                     RegPressureDelta &Delta, bool verbose);

  CandResult pickNodeFromQueue(ReadyQueue &Q,
                               const RegPressureTracker &RPTracker,
                               SchedCandidate &Candidate);
#ifndef NDEBUG
  void traceCandidate(const char *Label, const ReadyQueue &Q, SUnit *SU,
                      int Cost, PressureChange P = PressureChange());

  void readyQueueVerboseDump(const RegPressureTracker &RPTracker,
                             SchedCandidate &Candidate, ReadyQueue &Q);
#endif
};

} // namespace


#endif
