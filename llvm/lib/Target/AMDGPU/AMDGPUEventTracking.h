//===- AMDGPUEventTracking.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Minimal API to track in-flight events that increment hardware instruction
/// counters (\ref InstCounterType, e.g. LOAD_CNT, DS_CNT). Intended as a
/// building block for waitcnt insertion: it tracks which events are
/// outstanding and how far back they are, but does not directly compute waits.
///
///   - \ref CounterInfo: constant per-counter description (events, limit).
///   - \ref EventTrackerRecord: a (MachineInstr, SingleHWEvent) pair with a
///     height. A wait that brings the counter down to the height (or lower)
///     guarantees the event has completed.
///   - \ref EventTracker: per-MBB state; for each counter, a count and a
///     timeline of records sorted by descending height. Driven by the client
///     during an RPO dataflow analysis: \ref EventTracker::enterBlock merges
///     predecessor state, \ref EventTracker::record adds events, and
///     \ref EventTracker::drain models waits.
///
/// This is a low-level API. It can be used directly for trivial waitcnt
/// calculation (e.g. wait on a specific MI), but the typical usage is for
/// clients to build helpers on top of it that combine the timeline with
/// additional information (e.g. alias analysis, subtarget-specific behavior)
/// to compute more precise waits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUEVENTTRACKING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUEVENTTRACKING_H

#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class raw_ostream;
class MachineBasicBlock;

namespace AMDGPU {
namespace eventtracking {

class EventTracker;

/// FIXME: Move this to a common file and make it a generic utility.
///
/// Represents target-specific information about an available
/// \ref InstCounterType. This information is constant and inferred from the
/// subtarget.
struct CounterInfo {
  CounterInfo() = default;

  constexpr CounterInfo(InstCounterType T, HWEvents Events, unsigned Limit)
      : CounterT(T), Events(Events), Limit(Limit) {}

  /// Type of counter this is.
  InstCounterType CounterT;
  /// HWEvents for this counter.
  HWEvents Events;
  /// Hardware limit for this counter.
  unsigned Limit;
};

/// Represents a record within the \ref EventTracker's timeline
/// for one \ref InstCounterType. A record is a pair of \ref MachineInstr
/// and \ref SingleHWEvent with a height in the timeline.
///
/// The height is a conservative minimum estimate of *when* the event it
/// represents occurred. It is used to emit a wait instruction that ensures the
/// event has completed (e.g. Height = 2 means "wait 2" on the counter ensures
/// completion of the event across all control-flow paths).
///
/// Two examples illustrate how the height works:
///
/// If we are iterating a single block where there are no records inherited from
/// incoming basic blocks, then the height of a record is trivial: if there are
/// N records, the oldest is at N-1, while the youngest is at 0.
/// The height of each existing record increases whenever a new one is added.
///
/// However, if one record is carried into multiple successor blocks and merged
/// later (e.g. in a diamond), then the divergence (the same record could be at
/// a different height depending on the CFG path taken) is reconciled by merging
/// the records with the same \ref MachineInstr and \ref SingleHWEvent into a
/// single record whose height is the minimum across the predecessors.
///
/// The second example shows that the height is decoupled from a record's
/// position in the timeline. For example, a timeline could have 4 records, 3
/// of them at height 0 and the remaining one at height 1.
class EventTrackerRecord {
public:
  EventTrackerRecord(MachineInstr *MI, SingleHWEvent Kind, uint32_t Height = 0)
      : MI(MI), Kind(Kind) {
    setHeight(Height);
  }

  /// \returns the MachineInstr that originated this record.
  MachineInstr *getMI() const { return MI; }

  /// \returns the kind of record this is, as a \ref SingleHWEvent.
  SingleHWEvent getKind() const { return Kind; }

  /// \returns the height of this record.
  uint32_t getHeight() const { return Height; }

  void print(raw_ostream &OS, bool PrintMI = true, unsigned Indent = 0) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// \returns the "identity" of this record, which is a hash of its kind and
  /// MI.
  hash_code getIdentity() const;

private:
  friend class EventTracker;

  /// Sets the height of this record to \p NewHeight.
  void setHeight(uint32_t NewHeight) {
    Height = NewHeight;
    assert(Height == NewHeight && "Height overflow!");
  }

  MachineInstr *MI;
  SingleHWEvent Kind;
  // Height should never exceed the uint8_t limit in normal circumstances, as
  // most counter types use at most 6 bits to encode waitcnts. 16 bits is a
  // very generous limit; we can probably shrink it once we refine how we
  // handle overflowing inst counters.
  uint16_t Height;
};

/// This assert serves as a reminder to be mindful of the size of the object.
static_assert(sizeof(EventTrackerRecord) == 16,
              "EventTrackerRecord should remain small to optimize its layout "
              "within cache lines, for maximum iteration speed");

/// Per-MBB tracking context which tracks the current value (count) of each
/// instruction counter and the timeline of \ref EventTrackerRecord
/// for each instruction counter.
///
/// This class is only responsible for tracking records for every
/// \ref InstCounterType. It does not deal with calculating the waitcnts needed,
/// or doing more advanced reasoning over the timeline for specific queries
/// (e.g. finding an aliasing store). These responsibilities are left to
/// helpers built on top of this class (see the file header).
///
/// It is intended to be used during a reverse postorder dataflow analysis of a
/// function. The expected usage pattern is to call
/// \ref EventTracker::enterBlock, then visit the MBB in instruction order, call
/// \ref EventTracker::record whenever an interesting event occurs, and call
/// \ref EventTracker::drain whenever a wait on a counter is performed.
/// In between those operations, the user of the class can use all other methods
/// to query the current state of a counter, iterate its timeline, etc.
///
/// Allocation/storage/mapping of EventTrackers to MBBs is left to the user of
/// the class via \ref EventTracker::GetEventTrackerFn.
///
/// Whenever the dataflow analysis re-visits a block, the user of the class is
/// expected to re-use the same EventTracker (which is required in case the MBB
/// is its own predecessor) and call \ref EventTracker::enterBlock again, which
/// clears the state and re-imports all incoming state.
class EventTracker {
public:
  using GetEventTrackerFn =
      function_ref<EventTracker &(const MachineBasicBlock &)>;

  /// \p MBB is the MachineBasicBlock for this tracker.
  /// \p CounterInfos describes the available \ref InstCounterType. It must
  /// contain exactly one entry for each counter type in
  /// [0, CounterInfos.size()), and must outlive this class as its elements are
  /// stored by reference.
  EventTracker(const MachineBasicBlock &MBB,
               ArrayRef<CounterInfo> CounterInfos);

  /// Prepare this EventTracker for iteration through its basic block.
  ///
  /// Clears the current state and merges the state of the EventTrackers of
  /// predecessor MBBs. Records are merged using their identity (see
  /// \ref EventTrackerRecord::getIdentity) and only the lowest height is kept.
  /// Counts are merged by taking the maximum.
  ///
  /// \param EventTrackerGetter maps a MachineBasicBlock to its EventTracker.
  void enterBlock(GetEventTrackerFn EventTrackerGetter);

  /// Record an event of type \p Event at a MachineInstr \p MI, which will
  /// affect all counters that have \p Event in their event set.
  /// \pre \p Event belongs to at least one \ref InstCounterType.
  void record(MachineInstr &MI, SingleHWEvent Event);

  /// Drains the timeline of \p T such that:
  /// - The \ref count of \p T is no higher than \p N.
  /// - There are no records in the timeline with a height >= \p N.
  void drain(InstCounterType T, unsigned N = 0);

  /// \returns a conservative estimate of the current value of the counter \p T
  /// (a maximum value across all possible CFG paths).
  unsigned count(InstCounterType T) const;

  /// \returns the set of pending HWEvents for \p T.
  HWEvents getPendingEvents(InstCounterType T) const;

  /// \returns the current timeline for \p T, which contains all known in-flight
  /// (not drained) records. The timeline is always sorted by descending
  /// height.
  /// The returned ArrayRef is invalidated whenever the timeline changes (e.g.
  /// \ref record, \ref drain, \ref enterBlock).
  ArrayRef<EventTrackerRecord> getTimeline(InstCounterType T) const;

  /// Prints a dump of the internal tracking state of this class for \p T to the
  /// stream \p OS.
  void print(raw_ostream &OS, InstCounterType T) const;

  /// Prints a dump of all internal tracking state of this class to the stream
  /// \p OS. If \p IgnoreEmpty is true, do not print counters with a count of 0.
  void print(raw_ostream &OS, bool IgnoreEmpty = false,
             unsigned Indent = 0) const;

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
  /// Verifies that the invariants of this class hold.
  void verify() const;
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  /// FIXME: I do not think it is relevant to expose this class like we expose
  /// \ref EventTrackerRecord at this time. However, if this becomes complex
  /// enough, we could clean it up and expose it if it leads to a better API.
  struct CounterData {
    const CounterInfo *CI = nullptr;

    SmallVector<EventTrackerRecord, 16> Timeline;

    /// The current value of the counter. This is a max (upper bound) across
    /// all possible execution paths at runtime. It cannot be inferred from the
    /// Timeline alone and is thus a separate tracking domain.
    ///
    /// FIXME: We shouldn't need it for correctness so perhaps it should be
    /// removed entirely, or be marked as debug.
    uint32_t Count = 0;

    // TODO: We could imagine storing the per-predecessor height for incoming
    // events. We could achieve that by storing that as a map of ((ID, Pred),
    // Height). This would allow identifying events that are "deep" in one
    // branch but "shallow" in another, e.g. an event needing a waitcnt 1 for
    // one pred, but a waitcnt 8 for another. Not sure if we can exploit that
    // though?
  };

  /// Orders records by descending height. Ties are broken by kind, then by
  /// parent MBB number, so that the order is deterministic.
  static bool compareRecords(const EventTrackerRecord &A,
                             const EventTrackerRecord &B);

  /// Clears the tracked data, used when entering a block.
  void clear();

  /// Import all records from the predecessor trackers in \p Preds and
  /// reconcile divergence at joins.
  void recordIncomings(ArrayRef<EventTracker *> Preds);

  static void print(raw_ostream &OS, const CounterData &CD,
                    unsigned Indent = 0);

  const MachineBasicBlock *MBB;

  // NB: This, combined with the inline storage of Timeline, can lead to this
  // class becoming quite big - verify the size of this object whenever a change
  // is made.
  SmallVector<CounterData, InstCounterType::NUM_INST_CNTS> Counters;
};

} // namespace eventtracking
} // namespace AMDGPU

} // namespace llvm

#endif
