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
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUEVENTTRACKING_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUEVENTTRACKING_H

#include "AMDGPUHWEvents.h"
#include "AMDGPUWaitcntUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <memory>

namespace llvm {
class raw_ostream;
class MachineBasicBlock;
class MachineOperand;
class MachineFunction;

namespace AMDGPU {
namespace eventtracking {

class EventTrackingContext;
class EventTracker;

/// FIXME: Make this a generic util?
struct CounterInfo {
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

/// Represents entries in the \ref EventTracker.
///
/// Records are all uniquely identified by a \ref DynamicInstanceID. For each
/// unique \ref DynamicInstanceID value, all \ref EventTrackerRecord that use
/// that ID should have the same MI and Kind. This is enforced by exposing these
/// as read-only, and making the constructor assign a new \ref DynamicInstanceID
/// every time.
///
/// Only the score can change as it may be unique to each instance
/// of \ref EventTracker that carry it.
class EventTrackerRecord {
public:
  /// An always-increasing counter used to represent a dynamic instance of a
  /// record.
  ///
  /// Whenever we add a new \ref EventTrackerRecord, even if it's one we already
  /// have seen in a previous dataflow iteration, this counter is increased so
  /// that the new record has a unique `DynamicInstanceID`.
  using DynamicInstanceID = uint32_t;

  EventTrackerRecord(EventTrackingContext &Ctx, MachineInstr *MI,
                     SingleHWEvent Kind, uint32_t Score = 0);

  /// \returns the ID uniquely identifying this record across an entire
  /// EventTrackingContext. Whenever we revisit an instruction (when iterating
  /// until a fixpoint is reached), we give it a new ID. This is used to
  /// represent records carried over from previous iterations of the same basic
  /// block.
  DynamicInstanceID getID() const { return ID; }

  /// \returns the MachineInstr that originated this record.
  MachineInstr *getMI() const { return MI; }

  /// \returns the kind of record this is, as a \ref SingleHWEvent.
  SingleHWEvent getKind() const { return Kind; }

  /// \returns the score of this record.
  uint32_t getScore() const { return Score; }

  /// Sets the score of this record to \p NewScore.
  void setScore(uint32_t NewScore) {
    Score = NewScore;
    assert(Score == NewScore && "Score overflow!");
  }

  void print(raw_ostream &OS, bool PrintMI = true, unsigned Indent = 0) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  EventTrackerRecord(DynamicInstanceID ID, MachineInstr *MI, SingleHWEvent Kind,
                     uint32_t Score = 0)
      : MI(MI), ID(ID), Kind(Kind) {
    setScore(Score);
  }

  MachineInstr *MI;
  DynamicInstanceID ID;
  SingleHWEvent Kind;
  // Score should already never exceed uint8_t limit in normal circumstances as
  // most counter types only use up to 6 bits encoding for the waitcnts. 16 bit
  // is a very generous limit, we can probably shrink that at some point.
  uint16_t Score;
};

/// This assert serves as a reminder to be mindful of the size of the object.
static_assert(sizeof(EventTrackerRecord) == 16,
              "EventTrackerRecord should remain small to optimize its layout "
              "within cache lines, for maximum iteration speed");

/// Per-MBB tracking context.
///
/// Tracks data across the following domains:
///   - Current value (count) of each instruction counter.
///   - In-flight (alive) \ref EventTrackerRecord of each instruction counter.
///
/// This class is only responsible for tracking records for every
/// InstCounterType. It does not deal with calculating the waitcnts needed, or
/// doing more advance reasoning over the timeline for specific queries (e.g.
/// finding an aliasing store). These responsibilities are for
/// utils/wrappers/users of the class.
///
/// The API should be kept as simple and clear as possible.
class EventTracker {
public:
  EventTracker(MachineBasicBlock &MBB, EventTrackingContext &ET);

  /// \defgroup MachineBasicBlock entry and exit
  /// \{

  /// Notify this EventTracker that we are going to begin recording events.
  /// In case this is not the first time we are going through this block, this
  /// clears the internal state of the tracker and re-imports all incoming
  /// tracking state from the predecessors.
  void enterBlock();

  /// Notify this EventTracker that we are done recording events.
  void leaveBlock();

  /// \}

  /// \defgroup InstCounters Tracking Entrypoints
  /// Methods update the state of the InstCounters by adding/removing events
  /// or signaling certain special conditions.
  /// \{

  /// Record an event of type \p Event at a MachineInstr \p MI, which will
  /// affect all counters that have \p Event in their event set.
  void record(MachineInstr &MI, SingleHWEvent Event);

  /// Notify that we waited until the counter \p T reached the value \p N before
  /// continuing execution of the program (and recording more events).
  ///
  /// This affects the count of \p T, an removes all records that have a score
  /// greater than or equal to \p N.
  ///
  /// If \p N is zero, then \p T will no longer be in an indeterminate or
  /// out-of-order state afterwards if it previously was in such a state.
  void wait(InstCounterType T, unsigned N = 0);

  /// Mark the counter \p T as being in an indeterminate state. This means that
  /// we no longer accurately track \p T because there may be more records we do
  /// not know about. This primarily affects \ref getPendingEvents and
  /// \ref count.
  ///
  /// Implies \ref markOutOfOrder for \p T as well.
  void markIndeterminate(InstCounterType T);

  /// Mark the counter \p T as being "out-of-order", meaning records may retire
  /// in any order. This sets the score of all records to zero.
  void markOutOfOrder(InstCounterType T);

  /// \}

  /// \defgroup InstCounters Tracking Queries
  /// Query the current state of each InstCounter without modifying it.
  /// \{

  /// \returns the current value of the counter \p T at this point in time, or
  /// std::nullopt if \p T is in the indeterminate state.
  std::optional<unsigned> count(InstCounterType T) const;

  /// \returns true if the counter \p T is in an indeterminate state.
  bool isIndeterminate(InstCounterType T) const;

  /// \returns true if the counter \p T is out-of-order
  bool isOutOfOrder(InstCounterType T) const;

  /// \returns the set of pending HWEvents for \p T. If \p T is in an
  /// indeterminate state, returns a conservative set of pending events instead.
  HWEvents getPendingEvents(InstCounterType T) const;

  /// \returns the set of live records recorded for \p T. This is the list of
  /// all instructions in-flight for that counter.
  /// Note that if \p T is indeterminate, then this set is non-exhaustive. It
  /// only contains the records this class knows about.
  ArrayRef<EventTrackerRecord> getLiveRecords(InstCounterType T) const;

  /// \}

  /// \defgroup Miscellaneous helpers
  /// \{

  /// Prints a dump of the internal tracking state of this class for \p T to the
  /// stream \p OS.
  void print(raw_ostream &OS, InstCounterType T) const;

  /// Prints a dump of all internal tracking state of this class to the stream
  /// \p OS. If \p IgnoreEmpty is true, do not print counters with a count of 0.
  void print(raw_ostream &OS, bool IgnoreEmpty = false,
             unsigned Indent = 0) const;

  /// \returns true if the option to mimic legacy (SIInsertWaitcnts
  ///          scoreboard-style) tracking of counters and pending events.
  /// TODO: Remove in the future when legacy tracking is no longer needed.
  static bool mimicsLegacyTracking();

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
  /// Verifies invariants of this class are respected.
  void verify() const;
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// \}

private:
  struct CounterData {
    const CounterInfo *CI = nullptr;

    /// Set of live records (events) that make up the `Count`.
    SmallVector<EventTrackerRecord, 16> LiveRecords;
    /// The current value of the counter. This is a max (upper bound) across
    /// all possible execution paths at runtime. It cannot be inferred from the
    /// LiveRecords alone and is thus a separate tracking domain.
    uint32_t Count = 0;
    /// An upper bound that persists across fixpoint iterations. This is only
    /// used when \ref mimicsLegacyTracking returns true.
    uint32_t PersistentUpperBound = 0;
    /// Whether this counter is in an indeterminate state, which means that both
    /// the set of LiveRecords and the Count are imprecise. This implies that
    /// the counter is out-of-order as well.
    bool IsIndeterminate = false;
    /// Whether this counter is out-of-order, meaning records may retire in any
    /// order and they all exist at a score of zero.
    bool IsOutOfOrder = false;
    /// Legacy-style tracking of pending events that is coarse and does not
    /// leverage the live set of records. Only used when
    /// \ref mimicsLegacyTracking returns true and not cleared between
    /// iterations.
    HWEvents LegacyPendingEvents;

    // TODO: We could imagine storing the per-predecessor score for incoming
    // events. We could achieve that by storing that as a map of ((ID, Pred),
    // Score). This would allow identifying events that are "deep" in one branch
    // but "shallow" in another, e.g. an event needing a waitcnt 1 for one pred,
    // but a waitcnt 8 for another. Not sure if we can exploit that though?
  };

  CounterData &get(InstCounterType T) {
    assert(Counters.size() > T && "T is out of range!");
    return Counters[T];
  }

  const CounterData &get(InstCounterType T) const {
    assert(Counters.size() > T && "T is out of range!");
    return Counters[T];
  }

  /// Clears the tracked data, used when entering a block.
  void clear();

  /// Import all events from the incoming basic blocks in \p Preds and reconcile
  /// divergence at joints.
  void recordIncomings(EventTrackingContext &ETC,
                       ArrayRef<EventTracker *> Preds);

  static void print(raw_ostream &OS, const CounterData &CD,
                    unsigned Indent = 0);

  MachineBasicBlock *MBB;
  EventTrackingContext *Ctx;

  // NB: This, combined with the inline storage of LiveRecords, can lead to this
  // class becoming quite big - verify the size of this object whenever a change
  // is made.
  SmallVector<CounterData, InstCounterType::NUM_INST_CNTS> Counters;
};

/// Per-MF Tracking Context.
///
/// This owns all \ref EventTrackers and keeps track of state that persists
/// across dataflow analysis iterations, such as the current value of
/// \ref DynamicInstanceID.
class EventTrackingContext {
public:
  /// \param MF Machine Function
  /// \param Counters The counters available to \p MF on this target.
  EventTrackingContext(MachineFunction &MF, ArrayRef<CounterInfo> Counters);

  /// Fetch the \ref MBBEventTracker of \p MBB.
  EventTracker &operator[](MachineBasicBlock *MBB);

  /// \returns the list of counters available to the current target.
  ArrayRef<CounterInfo> counters() const { return CounterInfos; }

#if !defined(NDEBUG) || defined(EXPENSIVE_CHECKS)
  /// Verifies invariants of this class are respected.
  void verify() const;
#endif

  void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  friend class EventTrackerRecord;

  /// \returns a new, unique \ref DynamicInstanceID - only for use by
  /// \ref EventTrackerRecord.
  EventTrackerRecord::DynamicInstanceID nextDynamicInstanceID() {
    assert(NextDynID + 1 > NextDynID && "DynamicInstanceIDs overflow!");
    return ++NextDynID;
  }

  SmallVector<CounterInfo> CounterInfos;

  EventTrackerRecord::DynamicInstanceID NextDynID = 0;
  DenseMap<MachineBasicBlock *, std::unique_ptr<EventTracker>> Trackers;
};

} // namespace eventtracking
} // namespace AMDGPU

} // namespace llvm

#endif
