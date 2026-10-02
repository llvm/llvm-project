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

void EventTrackerRecord::print(raw_ostream &OS, bool PrintMI,
                               unsigned Indent) const {
  OS.indent(Indent) << "Record Kind=" << Kind << " Height=" << Height << ": ";
  if (PrintMI && MI)
    OS << *MI;
  else
    OS << "MI@" << (void *)MI << '\n';
}

hash_code EventTrackerRecord::getIdentity() const {
  return hash_combine((void *)MI, Kind.rawValue());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTrackerRecord::dump() const {
  dbgs() << '\n';
  print(dbgs(), /*PrintMI=*/true);
  dbgs() << '\n';
}
#endif

EventTracker::EventTracker(const MachineBasicBlock &MBB,
                           ArrayRef<CounterInfo> CounterInfos)
    : MBB(&MBB) {
  Counters.resize(CounterInfos.size());
  for (const CounterInfo &Info : CounterInfos)
    Counters[Info.CounterT].CI = &Info;
}

void EventTracker::enterBlock(GetEventTrackerFn EventTrackerGetter) {
  LLVM_DEBUG(dbgs() << "\n[EventTracker] Entering ";
             MBB->printAsOperand(dbgs()); dbgs() << '\n');

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
        Preds.push_back(&EventTrackerGetter(*Pred));
    }
  }

  if (IsSelfPred) {
    EventTracker SelfCopy = *this;
    Preds.push_back(&SelfCopy);
    clear();
    recordIncomings(Preds);
  } else {
    clear();
    recordIncomings(Preds);
  }
}

void EventTracker::leaveBlock() {
  LLVM_DEBUG(dbgs() << "[EventTracker] Leaving "; MBB->printAsOperand(dbgs());
             dbgs() << '\n');
}

void EventTracker::record(MachineInstr &MI, SingleHWEvent Event) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Recording " << Event << ": " << MI);

  EventTrackerRecord Rec = EventTrackerRecord(&MI, Event);
  [[maybe_unused]] bool FoundMatch = false;
  for (CounterData &CD : Counters) {
    if (!CD.CI->Events.contains(Event))
      continue;

    FoundMatch = true;
    ++CD.Count;
    CD.LegacyPendingEvents |= Event;

    // Do not age records if we are out-of-order.
    if (!CD.IsOutOfOrder) {
      // NB: There is an intentional tradeoff here. We could avoid this loop by
      // instead storing a timestamp in each record, and having a
      // constantly-increasing clock to infer the height (clock-timestamp is
      // height). However, it'd:
      //  - Complexify fetching the height (`EventTrackerRecord` cannot answer
      //  it
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
        Live.setHeight(Live.getHeight() + 1);
    }

    // NOTE: We do not merge with a previous record that has the same (MI+Kind),
    // unlike in recordIncomings. We are okay with having 2 separate records
    // with the same identity (MI+Kind), if one is carried over from a backedge
    // and one is from a more recent iteration.
    CD.LiveRecords.push_back(Rec);

#ifndef NDEBUG
    LLVM_DEBUG(if (EventTrackerPrintAll) {
      dbgs().indent(2) << "Updated Timeline:\n";
      print(dbgs(), CD, /*Indent=*/4);
    });
#endif
  }

  assert(FoundMatch && "Event has no matching InstCounterType!");

#ifdef EXPENSIVE_CHECKS
  verify();
#endif
}

void EventTracker::wait(InstCounterType T, unsigned N) {
  CounterData &CD = Counters[T];
  LLVM_DEBUG(dbgs() << "[EventTracker] Wait on " << getInstCounterName(T)
                    << " for " << N << '\n');

  // Fast path for clearing the counter
  if (N == 0) {
    CD.LiveRecords.clear();
    CD.Count = 0;
    CD.IsIndeterminate = false;
    CD.IsOutOfOrder = false;
    CD.LegacyPendingEvents = HWEvents();
  } else {
    CD.Count = std::min(CD.Count, N);

    // Don't bother erasing stuff if we are out-of-order. All records have a
    // height of zero in such cases.
    if (!CD.IsOutOfOrder) {
      auto *RmIt = remove_if(CD.LiveRecords, [&](EventTrackerRecord &E) {
        if (E.getHeight() < N)
          return false;
        LLVM_DEBUG(dbgs() << "  | Removing "; E.print(dbgs()));
        return true;
      });
      CD.LiveRecords.erase(RmIt, CD.LiveRecords.end());
    }

    LLVM_DEBUG(dbgs() << "  | => Updated Count:" << CD.Count << '\n');

#ifndef NDEBUG
    LLVM_DEBUG(if (EventTrackerPrintAll) {
      dbgs().indent(2) << "Updated Timeline:\n";
      print(dbgs(), CD, /*Indent=*/4);
    });
#endif
  }

#ifdef EXPENSIVE_CHECKS
  verify();
#endif
}

void EventTracker::markIndeterminate(InstCounterType T) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Marking " << getInstCounterName(T)
                    << " as indeterminate!\n");
  CounterData &CD = Counters[T];
  CD.IsIndeterminate = true;
  markOutOfOrder(T);
}

void EventTracker::markOutOfOrder(InstCounterType T) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Marking " << getInstCounterName(T)
                    << " as out-of-order!\n");
  CounterData &CD = Counters[T];
  CD.IsOutOfOrder = true;
  for (auto &Rec : CD.LiveRecords)
    Rec.setHeight(0);
}

std::optional<unsigned> EventTracker::count(InstCounterType T) const {
  const CounterData &CD = Counters[T];
  if (CD.IsIndeterminate)
    return std::nullopt;
  return Counters[T].Count;
}

bool EventTracker::isIndeterminate(InstCounterType T) const {
  return Counters[T].IsIndeterminate;
}

bool EventTracker::isOutOfOrder(InstCounterType T) const {
  return Counters[T].IsOutOfOrder;
}

HWEvents EventTracker::getPendingEvents(InstCounterType T) const {
  const CounterData &CD = Counters[T];
  if (CD.IsIndeterminate)
    return CD.CI->Events; // return all events

  if (MimicLegacyTracking)
    return CD.LegacyPendingEvents;

  HWEvents Res;
  for (const auto &E : Counters[T].LiveRecords)
    Res |= E.getKind();
  return Res;
}

ArrayRef<EventTrackerRecord>
EventTracker::getLiveRecords(InstCounterType T) const {
  return Counters[T].LiveRecords;
}

void EventTracker::print(raw_ostream &OS, InstCounterType T) const {
  print(OS, Counters[T]);
}

bool EventTracker::mimicsLegacyTracking() { return MimicLegacyTracking; }

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
void EventTracker::verify() const {
  if (!MBB)
    llvm_unreachable("EventTracker has no MBB!");

  for (const CounterData &C : Counters) {
    const auto OnError = [&]() {
      dbgs() << "EventTracker verification error\n";
      print(dbgs(), C);
    };

    if (C.IsIndeterminate) {
      if (!C.IsOutOfOrder) {
        OnError();
        llvm_unreachable("IsIndeterminate but not IsOutOfOrder");
      }
      continue;
    }

    if (C.IsOutOfOrder) {
      if (!all_of(C.LiveRecords, [](auto &R) { return R.getHeight() == 0; })) {
        OnError();
        llvm_unreachable(
            "IsOutOfOrder but some records do not have a height of 0!");
      }
    }

    // Check some basic invariants
    //  - Height of a record cannot exceed the value of the counter
    //  - We cannot have two records with same "identity" at the same height.
    //    e.g. We can't have 2 VMEM_READ_ACCESS at the same instruction at
    //    the same height. This is because `recordIncomings` merges based on
    //    identity, and unless there is a bug, `record` should increment all
    //    pre-existing records when a new one is inserted.
    DenseSet<std::pair<hash_code, unsigned>> RecIdentityCheck;
    for (const EventTrackerRecord &E : C.LiveRecords) {
      if (E.getHeight() > C.Count) {
        OnError();
        dbgs() << "Concerning Record:";
        E.print(dbgs());
        llvm_unreachable("record height is out of range");
      }

      auto [It, Inserted] =
          RecIdentityCheck.insert({E.getIdentity(), E.getHeight()});
      if (!Inserted) {
        OnError();
        dbgs() << "Concerning Record:";
        E.print(dbgs());
        llvm_unreachable("The timeline cannot have two records with the same "
                         "MachineInstr, Kind and Height at the same time!");
      }
    }

    // Check live records are sorted
    if (!is_sorted(C.LiveRecords, compareRecords)) {
      OnError();
      llvm_unreachable("live records are not sorted!");
    }
  }
}
#endif

void EventTracker::print(raw_ostream &OS, bool IgnoreEmpty,
                         unsigned Indent) const {
  OS.indent(Indent) << "EventTracker for ";
  MBB->printAsOperand(OS);
  OS << ':';
  if (IgnoreEmpty) {
    if (all_of(Counters, [](const CounterData &CD) { return CD.Count == 0; })) {
      OS << " (empty)\n";
      return;
    }
  }

  OS << '\n';
  for (const CounterData &C : Counters) {
    if (IgnoreEmpty && C.Count == 0)
      continue;
    print(dbgs(), C, Indent + 2);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTracker::dump() const {
  dbgs() << '\n';
  print(dbgs());
  dbgs() << '\n';
}
#endif

bool EventTracker::compareRecords(const EventTrackerRecord &A,
                                  const EventTrackerRecord &B) {
  // Different height
  if (A.getHeight() != B.getHeight())
    return A.getHeight() > B.getHeight();

  // Same height but different kind.
  unsigned AKindV = A.getKind().rawValue();
  unsigned BKindV = B.getKind().rawValue();
  if (AKindV != BKindV)
    return AKindV > BKindV;

  const MachineInstr *AMI = A.getMI();
  const MachineInstr *BMI = B.getMI();
  if (AMI == BMI)
    return false;

  // Same height/kind but MIs are in different BBs:
  // The one in the earlier BB comes first.
  const MachineBasicBlock *Bbb = BMI->getParent();
  const MachineBasicBlock *Abb = AMI->getParent();
  if (Abb != Bbb)
    return Abb->getNumber() < Bbb->getNumber();

  llvm_unreachable("We cannot have two distinct instructions from the same "
                   "MBB, at the same height and with the same event kind!");
}

void EventTracker::clear() {
  for (CounterData &C : Counters) {
    C.LiveRecords.clear();
    C.Count = 0;
    C.IsIndeterminate = false;
    C.IsOutOfOrder = false;
  }
}

void EventTracker::recordIncomings(ArrayRef<EventTracker *> Preds) {
  LLVM_DEBUG(if (!Preds.empty()) {
    dbgs() << "[EventTracker] Recording incoming events (merge) from "
              "predecessors:\n";
    for (EventTracker *Pred : Preds) {
      Pred->print(dbgs(), /*IgnoreEmpty=*/true, /*Indent=*/2);
    }
  });

  /// Iterate over all counters that are available to us.
  for (CounterData &CData : Counters) {
    assert(CData.LiveRecords.empty());

    DenseMap<hash_code, EventTrackerRecord> Acc;

    for (EventTracker *Pred : Preds) {
      auto &PredCData = Pred->Counters[CData.CI->CounterT];

      // Merge domain for the count value:
      CData.Count = std::max(CData.Count, PredCData.Count);
      // Merge domain for the legacy pending events.
      CData.LegacyPendingEvents |= PredCData.LegacyPendingEvents;
      // Merge domain for the indeterminate state.
      CData.IsIndeterminate |= PredCData.IsIndeterminate;
      // Merge domain for the out-of-order state.
      CData.IsOutOfOrder |= PredCData.IsOutOfOrder;

      for (EventTrackerRecord &PredEntry : PredCData.LiveRecords) {
        // At a join, we collapse records from all predecessors with the same
        // MI+Kind to a single entry with the Height of the entry being the
        // minimum across predecessors.
        hash_code Identity = PredEntry.getIdentity();
        auto It = Acc.find(Identity);
        if (It != Acc.end()) {
          auto &AccVal = It->second;
          AccVal.setHeight(std::min(AccVal.getHeight(), PredEntry.getHeight()));
        } else
          Acc.insert({Identity, PredEntry});
      }
    }

    if (MimicLegacyTracking) {
      CData.PersistentUpperBound =
          std::max(CData.PersistentUpperBound, CData.Count);
      CData.Count = CData.PersistentUpperBound;
    }

    auto AccVals = Acc.values();
    CData.LiveRecords.append(AccVals.begin(), AccVals.end());

    // Sort records by Height (descending) for consistent iteration.
    stable_sort(CData.LiveRecords, compareRecords);
  }

#ifdef EXPENSIVE_CHECKS
  verify();
#endif

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
  for (const EventTrackerRecord &E : CD.LiveRecords) {
    OS.indent(Indent + 2);
    E.print(OS);
  }
}

} // namespace eventtracking
} // namespace AMDGPU

} // namespace llvm
