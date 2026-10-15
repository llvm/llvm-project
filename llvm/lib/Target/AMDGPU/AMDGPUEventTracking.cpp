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

#define DEBUG_TYPE "amdgpu-event-tracking"

namespace llvm {

#ifndef NDEBUG
static cl::opt<bool> EventTrackerPrintAll(
    "amdgpu-event-tracker-print-all", cl::init(false),
    cl::desc("When using -debug, print the updated timeline every time a "
             "record is added or removed"));
#endif

namespace AMDGPU {
namespace eventtracking {

void EventTrackerRecord::print(raw_ostream &OS, bool PrintMI,
                               unsigned Indent) const {
  OS.indent(Indent) << "Record Kind=" << Kind << ": ";
  if (PrintMI && MI)
    OS << *MI;
  else
    OS << "MI@" << (void *)MI << '\n';
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

void EventTracker::record(MachineInstr &MI, SingleHWEvent Event) {
  LLVM_DEBUG(dbgs() << "[EventTracker] Recording " << Event << ": " << MI);

  EventTrackerRecord Rec = EventTrackerRecord(&MI, Event);
  [[maybe_unused]] bool FoundMatch = false;
  for (CounterData &CD : Counters) {
    if (!CD.CI->Events.contains(Event))
      continue;

    FoundMatch = true;
    ++CD.Count;
    CD.Timeline.push_back(Rec);

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

void EventTracker::drain(InstCounterType T, unsigned N) {
  CounterData &CD = Counters[T];
  LLVM_DEBUG(dbgs() << "[EventTracker] Drain " << getInstCounterName(T)
                    << " at " << N << '\n');

  // Fast path for clearing the counter.
  if (N == 0) {
    CD.Timeline.clear();
    CD.Count = 0;
  } else {
    CD.Count = std::min(CD.Count, N);

    // Only keep the N youngest records.
    if (CD.Timeline.size() > N) {
      auto *RmEnd = CD.Timeline.end() - N;
      LLVM_DEBUG(for (const EventTrackerRecord &E
                      : make_range(CD.Timeline.begin(), RmEnd)) {
        dbgs() << "  | Removing ";
        E.print(dbgs());
      });
      CD.Timeline.erase(CD.Timeline.begin(), RmEnd);
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

unsigned EventTracker::count(InstCounterType T) const {
  return Counters[T].Count;
}

HWEvents EventTracker::getPendingEvents(InstCounterType T) const {
  HWEvents Res;
  for (const auto &E : Counters[T].Timeline)
    Res |= E.getKind();
  return Res;
}

ArrayRef<EventTrackerRecord>
EventTracker::getTimeline(InstCounterType T) const {
  return Counters[T].Timeline;
}

void EventTracker::print(raw_ostream &OS, InstCounterType T) const {
  print(OS, Counters[T]);
}

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
void EventTracker::verify() const {
  if (!MBB)
    llvm_unreachable("EventTracker has no MBB!");

  for (const CounterData &C : Counters) {
    const auto OnError = [&]() {
      dbgs() << "EventTracker verification error\n";
      print(dbgs(), C);
    };

    // The timeline cannot have more records than the value of the counter.
    if (C.Timeline.size() > C.Count) {
      OnError();
      llvm_unreachable("timeline has more records than the counter value");
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
    print(OS, C, Indent + 2);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void EventTracker::dump() const {
  dbgs() << '\n';
  print(dbgs());
  dbgs() << '\n';
}
#endif

void EventTracker::print(raw_ostream &OS, const CounterData &CD,
                         unsigned Indent) {
  OS.indent(Indent) << getInstCounterName(CD.CI->CounterT)
                    << " (Count=" << CD.Count
                    << ", Timeline=" << CD.Timeline.size() << ")\n";
  for (const EventTrackerRecord &E : CD.Timeline) {
    OS.indent(Indent + 2);
    E.print(OS);
  }
}

} // namespace eventtracking
} // namespace AMDGPU

} // namespace llvm
