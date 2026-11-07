//===- FailedReassignCache.h - Cache failed canReassign answers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remembers, per live interval, that no other allocatable physical register
// was free for it, so repeated queries can skip the full search. An entry
// stays valid until an unassign frees a physical register across the
// interval's live range. The cache then records that register on the entry,
// so the next query only checks the registers that were freed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_FAILEDREASSIGNCACHE_H
#define LLVM_LIB_CODEGEN_FAILEDREASSIGNCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCRegister.h"

namespace llvm {
class LiveInterval;
class LiveIntervals;
class LiveRegMatrix;
class MachineRegisterInfo;
class RegisterClassInfo;

/// Result of consulting the failed-reassign cache for one canReassign query.
enum class Reuse {
  Miss,         ///< No usable entry. Run the full search.
  HitFalse,     ///< The failure is unchanged.
  RecheckFalse, ///< Freed registers still interfere.
  RecheckTrue   ///< A freed register is available.
};

class Cache {
public:
  static const unsigned BucketShift = 12;

  /// Default cap on freed physregs recorded for one entry. Past this the
  /// entry is saturated and the next query runs the full search.
  static const unsigned DefaultMaxFreedRegs = 64;

  Cache(LiveIntervals &LIS, LiveRegMatrix &Matrix, MachineRegisterInfo &MRI,
        const TargetRegisterInfo &TRI, const RegisterClassInfo &RegClassInfo,
        unsigned MaxFreedRegs = DefaultMaxFreedRegs);

  void clear();
  void erase(unsigned VirtRegId);

  /// Record that canReassign failed for \p LI. If \p HintOnly is true, only
  /// the preferred-register subset was searched; discard any existing entry
  /// without recording a failure, since the hint set can change independently.
  void remember(const LiveInterval &LI, bool HintOnly);

  /// Look up a prior canReassign failure for \p LI (currently on \p FromReg).
  ///
  /// \returns
  /// - Miss: no entry, fingerprint mismatch (entry erased), saturated, or
  ///   slot indexes were renumbered since the entry was recorded (the cache
  ///   is dropped).
  ///   Caller runs the full scan.
  /// - HitFalse: entry is valid and FreedRegs is empty; failure still holds.
  /// - RecheckFalse: FreedRegs were tested (and aliases in the order); none
  ///   are free. FreedRegs is cleared so later queries can HitFalse.
  /// - RecheckTrue: a freed reg (or alias) in the order is free; entry erased.
  Reuse reuse(const LiveInterval &LI, MCRegister FromReg);

  /// Notify that \p PhysReg was unassigned from \p Unassigned. The unassigned
  /// interval's own entry is erased so the next query recomputes it; interior
  /// segment bounds are not stored. Other overlapping entries append
  /// \p PhysReg to FreedRegs. \p ClearAllReferencingSegments, an empty or
  /// invalid free, or a slot-index renumber since the entries were recorded
  /// clears the whole cache.
  void noteUnassign(const LiveInterval &Unassigned, MCRegister PhysReg,
                    bool ClearAllReferencingSegments);

  bool empty() const { return Entries.empty(); }
  bool contains(unsigned VirtRegId) const {
    return Entries.contains(VirtRegId);
  }
  bool isSaturated(unsigned VirtRegId) const;
  ArrayRef<MCPhysReg> freedRegs(unsigned VirtRegId) const;

private:
  struct Entry {
    unsigned SegCount = 0;  ///< Segment count at remember time (fingerprint).
    SlotIndex Begin;        ///< Live range begin (fingerprint).
    SlotIndex End;          ///< Live range end (fingerprint).
    bool Saturated = false; ///< FreedRegs hit MaxFreedRegs; treat as Miss.
    SmallVector<MCPhysReg, 4>
        FreedRegs; ///< Physregs freed across [Begin, End).
  };

  LiveIntervals &LIS;
  LiveRegMatrix &Matrix;
  MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  const RegisterClassInfo &RegClassInfo;
  unsigned MaxFreedRegs;

  /// Cached failures keyed by virtual register id. At most one entry per
  /// virtreg: the live-interval fingerprint and physical registers freed
  /// across that range since the failure was recorded.
  DenseMap<unsigned, Entry> Entries;

  /// Reverse index from slot-index bucket (`index >> BucketShift`) to the
  /// virtreg ids whose cached ranges cover that bucket. Unassign walks only
  /// the buckets that overlap the freed segments, then checks real overlap
  /// before updating an entry. Valid only while SlotIndexes::getIndexEpoch()
  /// equals IndexEpochSeen; a mismatch drops the cache.
  DenseMap<unsigned, SmallVector<unsigned, 8>> Buckets;
  unsigned IndexEpochSeen = 0;

  /// Lazy bitvectors of `RegClassInfo.getOrder(RC)`, one per register class
  /// touched by a recheck. Used to test whether a freed physreg (or alias)
  /// is a legal canReassign destination without scanning the order each time.
  DenseMap<const TargetRegisterClass *, BitVector> OrderBits;

  static bool hasFingerprint(const Entry &E, const LiveInterval &LI);
  /// Drop every entry when slot indexes have been renumbered since the last
  /// insert. Bucket keys are distances, so they would miss freed ranges.
  void dropIfRenumbered();
  /// Call \p Fn with each bucket overlapped by the half-open range
  /// [\p Begin, \p End). An invalid or empty range calls nothing.
  void forEachBucket(SlotIndex Begin, SlotIndex End,
                     function_ref<void(unsigned)> Fn) const;
  void addToBuckets(unsigned VirtReg, const Entry &E);
  void removeFromBuckets(unsigned VirtReg, const Entry &E);
  bool inOrder(const LiveInterval &LI, MCRegister Reg);
  bool isFree(const LiveInterval &LI, MCRegister Reg) const;
};
} // namespace llvm

#endif // LLVM_LIB_CODEGEN_FAILEDREASSIGNCACHE_H
