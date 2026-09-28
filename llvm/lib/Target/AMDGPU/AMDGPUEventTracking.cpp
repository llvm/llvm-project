//===- AMDGPUEventTracking.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUEventTracking.h"
#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntUtils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <optional>

#define DEBUG_TYPE "amdgpu-event-tracking"

namespace llvm {

/// Mimic legacy (coarse) tracking of counter state.
static cl::opt<bool> MimicLegacyTracking("amdgpu-event-legacy-tracking",
                                         cl::init(false));

#ifndef NDEBUG
static cl::opt<bool> EventTrackerPrintAll(
    "amdgpu-event-tracker-print-all", cl::init(false),
    cl::desc("When using -debug, print the full set of live events every time "
             "an event is added or removed"));
#endif

namespace AMDGPU {
namespace eventtracking {

namespace {
static bool greaterThan(const EventTrackerRecord &A,
                        const EventTrackerRecord &B) {
  return A.getScore() > B.getScore();
}
} // namespace

EventTrackerRecord::EventTrackerRecord(EventTrackingContext &Ctx,
                                       MachineInstr *MI, SingleHWEvent Kind,
                                       uint32_t Score)
    : EventTrackerRecord(Ctx.nextDynamicInstanceID(), MI, Kind, Score) {}

void EventTrackerRecord::print(raw_ostream &OS, bool PrintMI,
                               unsigned Indent) const {
  OS.indent(Indent) << "#" << ID << " " << Kind << " (Score=" << Score << "): ";
  if (PrintMI && MI)
    OS << *MI;
  else
    OS << "MI@" << (void *)MI << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTrackerRecord::dump() const {
  dbgs() << "\n";
  print(dbgs(), /*PrintMI=*/true);
  dbgs() << "\n";
}
#endif

EventTracker::EventTracker(MachineBasicBlock &MBB, EventTrackingContext &ETC)
    : MBB(&MBB), Ctx(&ETC) {
  Counters.resize(ETC.counters().size());
  for (const CounterInfo &Info : ETC.counters())
    Counters[Info.CounterT].CI = &Info;
}

void EventTracker::enterBlock() {
  LLVM_DEBUG(dbgs() << "\n[EventTracker] Entering ";
             MBB->printAsOperand(dbgs()); dbgs() << "\n");

  // FIXME: This is a bit hacky, but we need to save the old state in case the
  // MBB is also its own predecessor. Revisit when the design and clients of
  // this class are set in stone.

  SmallVector<EventTracker *> Preds;
  bool IsSelfPred = false;
  if (Preds.empty() && !MBB->pred_empty()) {
    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      if (Pred == MBB)
        IsSelfPred = true;
      else
        Preds.push_back(&(*Ctx)[Pred]);
    }
  }

  if (IsSelfPred) {
    EventTracker SelfCopy = *this;
    Preds.push_back(&SelfCopy);
    clear();
    recordIncomings(*Ctx, Preds);
  } else {
    clear();
    recordIncomings(*Ctx, Preds);
  }
}

void EventTracker::leaveBlock() {
  LLVM_DEBUG(dbgs() << "[EventTracker] Leaving "; MBB->printAsOperand(dbgs());
             dbgs() << "\n");
}

void EventTracker::record(MachineInstr &MI, SingleHWEvent Event) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Recording " << Event << ": " << MI);

  EventTrackerRecord Rec = EventTrackerRecord(*Ctx, &MI, Event);
  [[maybe_unused]] bool FoundMatch = false;
  for (auto &CD : Counters) {
    if (!CD.CI->Events.contains(Event))
      continue;

    FoundMatch = true;
    ++CD.Count;
    CD.LegacyPendingEvents |= Event;

    // Do not age records if we are out-of-order.
    if (!CD.IsOutOfOrder) {
      // NB: There is an intentional tradeoff here. We could avoid this loop by
      // instead storing a timestamp in each record, and having a
      // constantly-increasing clock to infer the score (clock-timestamp is
      // score). However, it'd:
      //  - Complexify fetching the score (`EventTrackerRecord` cannot answer it
      //    on its own anymore and we need a separate query/wrapper).
      //  - Make merge of incoming records a bit more annoying (we'd need to
      //    rebase the `clock`).
      //  - Potentially demand (much) more space in EventTrackerRecord to store
      //    bigger numbers.
      //
      // All in all, I think this small loop is fine for now, but we can still
      // change the system if we have data backed up by profiling to
      // justify the change.
      for (auto &Live : CD.LiveRecords)
        Live.setScore(Live.getScore() + 1); // Age all existing events.
    }

    CD.LiveRecords.push_back(Rec);

#ifndef NDEBUG
    LLVM_DEBUG(if (EventTrackerPrintAll) {
      dbgs().indent(2) << "Updated Timeline:\n";
      print(dbgs(), CD, /*Indent=*/4);
    });
#endif
  }

  assert(FoundMatch && "Event has no matching InstCounterType!");
}

void EventTracker::wait(InstCounterType T, unsigned N) {
  auto &CD = get(T);
  LLVM_DEBUG(dbgs() << "[EventTracker] Wait on " << getInstCounterName(T)
                    << " for " << N << "\n");

  // Fast path for clearing the counter
  if (N == 0) {
    CD.LiveRecords.clear();
    CD.Count = 0;
    CD.IsIndeterminate = false;
    CD.IsOutOfOrder = false;
    CD.LegacyPendingEvents = HWEvents();
    return;
  }

  CD.Count = std::min(CD.Count, N);

  // Don't bother erasing stuff if we are out-of-order. All records have a score
  // of zero in such cases.
  if (!CD.IsOutOfOrder) {
    auto *RmIt = remove_if(CD.LiveRecords, [&](EventTrackerRecord &E) {
      if (E.getScore() < N)
        return false;
      LLVM_DEBUG(dbgs() << "  | Removing "; E.print(dbgs()));
      return true;
    });
    CD.LiveRecords.erase(RmIt, CD.LiveRecords.end());
  }

  LLVM_DEBUG(dbgs() << "  | => Updated Count:" << CD.Count << "\n");

#ifndef NDEBUG
  LLVM_DEBUG(if (EventTrackerPrintAll) {
    dbgs().indent(2) << "Updated Timeline:\n";
    print(dbgs(), CD, /*Indent=*/4);
  });
#endif
}

void EventTracker::markIndeterminate(InstCounterType T) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Marking " << getInstCounterName(T)
                    << " as indeterminate!\n");
  auto &CD = get(T);
  CD.IsIndeterminate = true;
  markOutOfOrder(T);
}

void EventTracker::markOutOfOrder(InstCounterType T) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Marking " << getInstCounterName(T)
                    << " as out-of-order!\n");
  auto &CD = get(T);
  CD.IsOutOfOrder = true;
  for (auto &Rec : CD.LiveRecords)
    Rec.setScore(0);
}

std::optional<unsigned> EventTracker::count(InstCounterType T) const {
  auto &CD = get(T);
  if (CD.IsIndeterminate)
    return std::nullopt;
  return get(T).Count;
}

bool EventTracker::isIndeterminate(InstCounterType T) const {
  return get(T).IsIndeterminate;
}

bool EventTracker::isOutOfOrder(InstCounterType T) const {
  return get(T).IsOutOfOrder;
}

HWEvents EventTracker::getPendingEvents(InstCounterType T) const {
  const auto &CD = get(T);
  if (CD.IsIndeterminate)
    return CD.CI->Events; // return all events

  if (MimicLegacyTracking)
    return CD.LegacyPendingEvents;

  HWEvents Res;
  for (const auto &E : get(T).LiveRecords)
    Res |= E.getKind();
  return Res;
}

ArrayRef<EventTrackerRecord>
EventTracker::getLiveRecords(InstCounterType T) const {
  return get(T).LiveRecords;
}

void EventTracker::print(raw_ostream &OS, InstCounterType T) const {
  print(OS, get(T));
}

bool EventTracker::mimicsLegacyTracking() { return MimicLegacyTracking; }

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
void EventTracker::verify() const {
  assert(MBB && Ctx && "Invalid internal state!");

  for (auto &C : Counters) {
    const auto OnError = [&]() {
      dbgs() << "EventTracker verification error\n";
      print(dbgs(), C);
    };

    if (C.IsIndeterminate) {
      if (!C.IsOutOfOrder) {
        OnError();
        assert(false && "IsIndeterminate but not IsOutOfOrder");
      }
      continue;
    }

    if (C.IsOutOfOrder) {
      if (!all_of(C.LiveRecords, [](auto &R) { return R.getScore() == 0; })) {
        OnError();
        assert(false &&
               "IsOutOfOrder but some records do not have a score of 0!");
      }
    }

    if (C.Count > C.LiveRecords.size() && !mimicsLegacyTracking()) {
      OnError();
      assert(false &&
             "'Count' is inconsistent with the number of live records");
    }

    for (auto &E : C.LiveRecords) {
      if (E.getScore() > C.Count) {
        OnError();
        dbgs() << "Concerning Record:";
        E.print(dbgs());
        assert(false && "record score is out of range");
      }
    }

    // Check live records are sorted
    if (!is_sorted(C.LiveRecords, greaterThan)) {
      OnError();
      assert(false && "live records are not sorted!");
    }
  }
}
#endif

void EventTracker::print(raw_ostream &OS, bool IgnoreEmpty,
                         unsigned Indent) const {
  OS.indent(Indent) << "EventTracker for ";
  MBB->printAsOperand(OS);
  OS << ":";
  if (IgnoreEmpty) {
    if (all_of(Counters, [](const auto &CD) { return CD.Count == 0; })) {
      OS << " (empty)\n";
      return;
    }
  }

  OS << "\n";
  for (auto &C : Counters) {
    if (IgnoreEmpty && C.Count == 0)
      continue;
    print(dbgs(), C, Indent + 2);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTracker::dump() const {
  dbgs() << "\n";
  print(dbgs());
  dbgs() << "\n";
}
#endif

void EventTracker::clear() {
  for (auto &C : Counters) {
    C.LiveRecords.clear();
    C.Count = 0;
    C.IsIndeterminate = false;
    C.IsOutOfOrder = false;
  }
}

void EventTracker::recordIncomings(EventTrackingContext &ETC,
                                   ArrayRef<EventTracker *> Preds) {
  LLVM_DEBUG(if (!Preds.empty()) {
    dbgs() << "[EventTracker] Recording incoming events (merge) from "
              "predecessors:\n";
    for (EventTracker *Pred : Preds) {
      Pred->print(dbgs(), /*IgnoreEmpty=*/true, /*Indent=*/2);
    }
  });

  /// Iterate over all counters that are available to us.
  for (auto &CI : ETC.counters()) {
    auto &CData = Counters[CI.CounterT];
    assert(CData.LiveRecords.empty());

    DenseMap<EventTrackerRecord::DynamicInstanceID, EventTrackerRecord> Acc;

    for (EventTracker *Pred : Preds) {
      auto &PredCData = Pred->Counters[CI.CounterT];

      // Merge domain for the count value:
      CData.Count = std::max(CData.Count, PredCData.Count);
      // Merge domain for the legacy pending events.
      CData.LegacyPendingEvents |= PredCData.LegacyPendingEvents;
      // Merge domain for the indeterminate state.
      CData.IsIndeterminate |= PredCData.IsIndeterminate;
      // Merge domain for the out-of-order state.
      CData.IsOutOfOrder |= PredCData.IsOutOfOrder;

      for (EventTrackerRecord &PredEntry : PredCData.LiveRecords) {
        EventTrackerRecord::DynamicInstanceID ID = PredEntry.getID();
        auto It = Acc.find(ID);
        if (It != Acc.end()) {
          auto &AccVal = It->second;
          AccVal.setScore(std::min(AccVal.getScore(), PredEntry.getScore()));
          assert(
              PredEntry.getMI() == AccVal.getMI() &&
              PredEntry.getKind() == AccVal.getKind() &&
              "EventTrackerRecord have same DynamicInstanceID, but different "
              "MachineInstr/HWEvent kind, which should not be possible");
        } else
          Acc.insert({ID, PredEntry});
      }
    }

    if (MimicLegacyTracking) {
      CData.PersistentUpperBound =
          std::max(CData.PersistentUpperBound, CData.Count);
      CData.Count = CData.PersistentUpperBound;
    }

    auto AccVals = Acc.values();
    CData.LiveRecords.append(AccVals.begin(), AccVals.end());

    // Sort records by Score (descending) for consistent iteration.
    stable_sort(CData.LiveRecords, greaterThan);
  }

  LLVM_DEBUG(if (!Preds.empty()) {
    dbgs() << "[EventTracker] Timeline after recording incomings:\n";
    print(dbgs(), /*IgnoreEmpty=*/true, /*Indent=*/2);
  });
}

void EventTracker::print(raw_ostream &OS, const CounterData &CD,
                         unsigned Indent) {
  OS.indent(Indent) << getInstCounterName(CD.CI->CounterT)
                    << " (Count=" << CD.Count
                    << ", PersistentUpperBound=" << CD.PersistentUpperBound
                    << ", LiveRecords=" << CD.LiveRecords.size()
                    << ", IsOutOfOrder=" << CD.IsOutOfOrder
                    << ", IsIndeterminate=" << CD.IsIndeterminate << ")\n";
  for (const auto &E : CD.LiveRecords) {
    OS.indent(Indent + 2);
    E.print(OS);
  }
}

EventTrackingContext::EventTrackingContext(MachineFunction &MF,
                                           ArrayRef<CounterInfo> Counters)
    : CounterInfos(Counters) {
  LLVM_DEBUG(dbgs() << "\n[EventTrackingContext] CounterInfos for "
                    << MF.getName() << "\n";
             for (const auto &CI
                  : CounterInfos) {
               dbgs().indent(2)
                   << AMDGPU::getInstCounterName(CI.CounterT) << " ";
               if (CI.Events.none()) {
                 dbgs() << " (unused - no HWEvents assigned)\n";
               } else {
                 dbgs() << "(Limit=" << CI.Limit << ") " << CI.Events << "\n";
               }
             });

  Trackers.reserve(MF.size());
  for (MachineBasicBlock &MBB : MF)
    Trackers[&MBB] = std::make_unique<EventTracker>(MBB, *this);
}

EventTracker &EventTrackingContext::operator[](MachineBasicBlock *MBB) {
  assert(MBB);
  return *Trackers.at(MBB);
}

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
void EventTrackingContext::verify() const {
  for (const auto &[MBB, Tracker] : Trackers) {
    assert(MBB && "Unexpected nullptr entry!");
    Tracker->verify();
  }

  // Check CounterInfos is sane.
  for (auto [Idx, CI] : enumerate(CounterInfos)) {
    assert(Idx == CI.CounterT && "CounterInfo is in wrong position!");
    assert(CI.Events.any() &&
           "InstCounterType has no event associated with it!");
  }
}
#endif

void EventTrackingContext::print(raw_ostream &OS) const {
  for (const auto &[MBB, Tracker] : Trackers) {
    MBB->printAsOperand(OS);
    OS << ":\n";
    Tracker->print(OS, /*Indent=*/2);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTrackingContext::dump() const {
  dbgs() << "\n";
  print(dbgs());
  dbgs() << "\n";
}
#endif

} // namespace eventtracking
} // namespace AMDGPU

} // namespace llvm
