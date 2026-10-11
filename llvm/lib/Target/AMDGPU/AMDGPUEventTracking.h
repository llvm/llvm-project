//===- AMDGPUEventTracking.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Tracks a per-InstCounterType event timeline which preserves
/// information about previously encountered MachineInstr and HWEvents.
///
/// The goal of this file is to implement a minimal API to preserve information
/// about events that increment an instruction counter. Such events are
/// contained in records called \ref EventTrackerRecord, which include:
///   - The event kind, as a `HWEvent`
///   - The instruction that triggered the event
///   - The height of the record in the overall timeline of the counter, see
///     \ref EventTrackerRecord for more details.
///
/// The "timeline" represents the state of an InstCounterType at a
/// given point in time for a given MBB during a reverse postorder dataflow
/// analysis of a function. Such a dataflow analysis is expected to pass over
/// blocks as many times as necessary until a "fix point" is reached, which is a
/// state in which the client (the pass that uses the APIs) decides it has
/// enough information to safely perform its duties (generally, inserting
/// `s_waitcnt` instructions).
///
/// See \ref EventTracker for more information as well.
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
/// Represents target-specific information about available \ref InstCounterType
/// and their specifities. This is handled by reference and is expected to
/// outlive \ref EventTracker
struct CounterInfo {
  CounterInfo() = default;

  constexpr CounterInfo(InstCounterType T, HWEvents Events, unsigned Limit)
      : CounterT(T), Events(Events), Limit(Limit) {}

  /// Type of counter this is.
  /// This must match the index in the container, e.g. CounterT=2 must be at
  /// index 2.
  InstCounterType CounterT;
  /// HWEvents for this counter.
  HWEvents Events;
  /// Hardware limit for this counter.
  unsigned Limit;
};

/// Information about an event recorded within the \ref EventTracker's timeline
/// for one \ref InstCounterType. An event (see \ref HWEvents) has an
/// accompanying \ref MachineInstr, and exists at a fixed height in the
/// timeline.
///
/// An event's height is a conservative minimum estimate of *when* the event
/// occured. There are two examples that help illustrate this concept:
///
/// If we are iterating a single block where there are no records inherited from
/// incoming basic blocks, then the height of an event is trivial: if there are
/// N events, the oldest event is at N-1, while the youngest one is at 0.
/// The height of each existing event increases whenever a new one is recorded.
///
/// However, in cases where one record is carried over into multiple successor
/// blocks, and then merged later (in a diamond pattern for example), then the
/// divergence (the same record could be at a different height depending on the
/// CFG path taken) is reconciled by merging the records with the same
/// \ref MachineInstr and \ref SingleHWEvent into a single record, and the
/// height of the record is the minimum across the predecessors.
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

  /// Sets the height of this record to \p NewHeight.
  void setHeight(uint32_t NewHeight) {
    Height = NewHeight;
    assert(Height == NewHeight && "Height overflow!");
  }

  void print(raw_ostream &OS, bool PrintMI = true, unsigned Indent = 0) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// \returns the "identity" of a this record which is a hash of its kind and
  /// MI.
  hash_code getIdentity() const;

private:
  MachineInstr *MI;
  SingleHWEvent Kind;
  // Height should never exceed uint8_t limit in normal circumstances as
  // most counter types only use up to 6 bits encoding for the waitcnts. 16 bit
  // is a very generous limit, we can probably shrink that at some point once we
  // refine how we handle overflowing inst counters.
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
/// (e.g. finding an aliasing store). These responsibilities are for
/// utils/wrappers/users of the class.
///
/// As stated earlier in the file, this is a per-MBB state, it is intended to be
/// used during a reverse postorder dataflow analysis of a function.
/// The expected usage pattern is to visit a MBB in instruction order, call
/// \ref EventTracker::record whenever an interesting event occurs, call
/// \ref EventTracker::drain whenever a wait on a counter is performed.
/// In between those operations, the user of the class can use all other methods
/// to query the current state of a counter, iterate its timeline, etc.
///
/// Allocation/storage/mapping of EventTrackers to MBBs is left to the user of
/// the class via the \ref EventTracker::GetEventTrackerFn
///
/// Whenever dataflow analysis leads to re-visiting a block, the user of the
/// class is expected to re-use the previous state (which is required in case
/// the MBB is its own predecessor) and call the \ref EventTracker::enterBlock
/// function, which will clear the state and re-import all incoming state.
class EventTracker {
public:
  using GetEventTrackerFn =
      function_ref<EventTracker &(const MachineBasicBlock &)>;

  /// \ref MBB the MachineBasicBlock for this tracker
  /// \ref CounterInfos Information about available \ref InstCounterType -
  /// expected to outlive this class as it'll be stored by reference.
  EventTracker(const MachineBasicBlock &MBB,
               ArrayRef<CounterInfo> CounterInfos);

  /// Prepare this EventTracker for iteration through its basic block.
  ///
  /// When entering a block, we merge state from the EventTrackers of
  /// incoming MBBs. Records from incoming MBBs are merged using the `identity`
  /// of the \ref EventTrackerRecord and only the record with the lowest height
  /// is kept.
  ///
  /// \param EventTrackerGetter Is a function that map a MachineBasicBlock to
  /// the EventTracker used for that MachineBasicBlock.
  void enterBlock(GetEventTrackerFn EventTrackerGetter);

  /// Record an event of type \p Event at a MachineInstr \p MI, which will
  /// affect all counters that have \p Event in their event set.
  /// \pre \ref Event is expected to belong to at least one \ref InstCounterType
  void record(MachineInstr &MI, SingleHWEvent Event);

  /// Drains the timeline of \p T such that:
  /// - The \ref count of \p T is no higher than \p N
  /// - There are no records in the timelime where the height of the record is
  ///   >= \p N.
  void drain(InstCounterType T, unsigned N = 0);

  /// \returns a conservative estimate of the current value of the counter \p T
  /// (a maximum value across all possible CFG paths).
  unsigned count(InstCounterType T) const;

  /// \returns the set of pending HWEvents for \p T
  HWEvents getPendingEvents(InstCounterType T) const;

  /// \returns the current timeline for \p T, which contains all known in-flight
  /// records (= records not resolved/drained). The timeline is always sorted by
  /// the descending height of the records.
  /// This ArrayRef is not safe for storage as it'll be invalidated when the
  /// timeline is changed via \ref record, \ref drain or by other means.
  ArrayRef<EventTrackerRecord> getTimeline(InstCounterType T) const;

  /// Prints a dump of the internal tracking state of this class for \p T to the
  /// stream \p OS.
  void print(raw_ostream &OS, InstCounterType T) const;

  /// Prints a dump of all internal tracking state of this class to the stream
  /// \p OS. If \p IgnoreEmpty is true, do not print counters with a count of 0.
  void print(raw_ostream &OS, bool IgnoreEmpty = false,
             unsigned Indent = 0) const;

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
  /// Verifies invariants of this class are respected.
  void verify() const;
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  /// FIXME: I do not think it is relevant to expose this class like we expose
  /// \ref EventTrackerRecord at this time. However if this goes complex enough,
  /// we could clean it up and expose it if it leads to a better API.
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
    // Score). This would allow identifying events that are "deep" in one branch
    // but "shallow" in another, e.g. an event needing a waitcnt 1 for one pred,
    // but a waitcnt 8 for another. Not sure if we can exploit that though?
  };

  /// Compare records in order to achieve stable sorting by descending height.
  static bool compareRecords(const EventTrackerRecord &A,
                             const EventTrackerRecord &B);

  /// Clears the tracked data, used when entering a block.
  void clear();

  /// Import all events from the incoming basic blocks in \p Preds and reconcile
  /// divergence at joints.
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
