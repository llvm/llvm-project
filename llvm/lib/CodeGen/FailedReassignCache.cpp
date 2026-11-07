//===- FailedReassignCache.cpp - Cache failed canReassign answers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FailedReassignCache.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <cassert>

using namespace llvm;

namespace llvm::reassign {

Cache::Cache(LiveIntervals &LIS, LiveRegMatrix &Matrix,
             MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
             const RegisterClassInfo &RegClassInfo, unsigned MaxFreedRegs)
    : LIS(LIS), Matrix(Matrix), MRI(MRI), TRI(TRI), RegClassInfo(RegClassInfo),
      MaxFreedRegs(MaxFreedRegs),
      IndexEpochSeen(LIS.getSlotIndexes()->getIndexEpoch()) {
  assert(MaxFreedRegs > 0 && "the freed-reg list needs room for one register");
}

void Cache::clear() {
  Entries.clear();
  Buckets.clear();
  IndexEpochSeen = LIS.getSlotIndexes()->getIndexEpoch();
}

void Cache::dropIfRenumbered() {
  unsigned Epoch = LIS.getSlotIndexes()->getIndexEpoch();
  if (Epoch == IndexEpochSeen)
    return;
  clear();
}

bool Cache::isSaturated(unsigned VirtRegId) const {
  auto It = Entries.find(VirtRegId);
  return It != Entries.end() && It->second.Saturated;
}

ArrayRef<MCPhysReg> Cache::freedRegs(unsigned VirtRegId) const {
  auto It = Entries.find(VirtRegId);
  if (It == Entries.end())
    return {};
  return It->second.FreedRegs;
}

bool Cache::hasFingerprint(const Entry &E, const LiveInterval &LI) {
  return !LI.empty() && E.SegCount == LI.segments.size() &&
         E.Begin == LI.beginIndex() && E.End == LI.endIndex();
}

void Cache::forEachBucket(SlotIndex Begin, SlotIndex End,
                          function_ref<void(unsigned)> Fn) const {
  if (!Begin.isValid() || !End.isValid() || Begin >= End)
    return;
  // SlotIndex::getIndex() is private. The function's zero index is 0, so
  // distance from it is the same integer.
  SlotIndex Zero = LIS.getSlotIndexes()->getZeroIndex();
  int BeginDist = Zero.distance(Begin);
  int EndDist = Zero.distance(End);
  if (BeginDist < 0 || EndDist <= BeginDist)
    return;
  unsigned First = static_cast<unsigned>(BeginDist) >> BucketShift;
  unsigned Last = static_cast<unsigned>(EndDist - 1) >> BucketShift;
  for (unsigned Bucket = First; Bucket <= Last; ++Bucket)
    Fn(Bucket);
}

void Cache::addToBuckets(unsigned VirtReg, const Entry &E) {
  forEachBucket(E.Begin, E.End,
                [&](unsigned Bucket) { Buckets[Bucket].push_back(VirtReg); });
}

void Cache::removeFromBuckets(unsigned VirtReg, const Entry &E) {
  forEachBucket(E.Begin, E.End, [&](unsigned Bucket) {
    auto It = Buckets.find(Bucket);
    if (It == Buckets.end())
      return;
    llvm::erase(It->second, VirtReg);
    if (It->second.empty())
      Buckets.erase(It);
  });
}

void Cache::erase(unsigned VirtRegId) {
  // Renumbering stales every bucket key. Dropping the cache removes entries
  // without searching those keys.
  dropIfRenumbered();
  auto It = Entries.find(VirtRegId);
  if (It == Entries.end())
    return;
  removeFromBuckets(VirtRegId, It->second);
  Entries.erase(It);
}

bool Cache::inOrder(const LiveInterval &LI, MCRegister Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(LI.reg());
  if (!RC->contains(Reg))
    return false;
  BitVector &Bits = OrderBits[RC];
  if (Bits.empty()) {
    Bits.resize(TRI.getNumRegs());
    for (MCPhysReg OrderReg : RegClassInfo.getOrder(RC))
      Bits.set(OrderReg);
  }
  return Reg.id() < Bits.size() && Bits.test(Reg.id());
}

bool Cache::isFree(const LiveInterval &LI, MCRegister Reg) const {
  assert(LI.reg().isVirtual() && Reg.isPhysical());
  auto Interferes = [&](MCRegUnit Unit) {
    LiveIntervalUnion::Query Q(
        LI, Matrix.getLiveUnions()[static_cast<unsigned>(Unit)]);
    return Q.checkInterference();
  };
  return none_of(TRI.regunits(Reg), Interferes);
}

void Cache::remember(const LiveInterval &LI, bool HintOnly) {
  if (LI.empty())
    return;
  dropIfRenumbered();
  unsigned Id = LI.reg().id();
  auto [It, Inserted] = Entries.try_emplace(Id);
  Entry &E = It->second;
  if (!Inserted) {
    removeFromBuckets(Id, E);
    E = Entry();
  }
  E.SegCount = LI.segments.size();
  E.Begin = LI.beginIndex();
  E.End = LI.endIndex();
  E.HintOnly = HintOnly;
  addToBuckets(Id, E);
}

Reuse Cache::reuse(const LiveInterval &LI, MCRegister FromReg) {
  dropIfRenumbered();
  auto It = Entries.find(LI.reg().id());
  if (It == Entries.end())
    return Reuse::Miss;
  Entry &E = It->second;
  if (!hasFingerprint(E, LI)) {
    erase(LI.reg().id());
    return Reuse::Miss;
  }
  if (isSaturated(LI.reg().id()) || E.HintOnly)
    return Reuse::Miss;
  if (freedRegs(LI.reg().id()).empty())
    return Reuse::HitFalse;

  SmallDenseSet<MCPhysReg, 16> Seen;
  for (MCPhysReg Freed : E.FreedRegs) {
    for (MCRegAliasIterator Alias(Freed, &TRI, /*IncludeSelf=*/true);
         Alias.isValid(); ++Alias) {
      MCRegister Reg(*Alias);
      if (Reg == FromReg || !inOrder(LI, Reg) || !Seen.insert(Reg.id()).second)
        continue;
      if (isFree(LI, Reg)) {
        erase(LI.reg().id());
        return Reuse::RecheckTrue;
      }
    }
  }
  E.FreedRegs.clear();
  return Reuse::RecheckFalse;
}

void Cache::noteUnassign(const LiveInterval &Unassigned, MCRegister PhysReg,
                         bool ClearAllReferencingSegments) {
  // A pass that never caches, such as SGPR allocation, must not walk buckets.
  if (empty())
    return;
  if (!PhysReg || ClearAllReferencingSegments || Unassigned.empty()) {
    // The freed range is not the interval's current segments. Drop every
    // failure rather than keep an answer that may now be true.
    clear();
    return;
  }
  // Bucket keys are distances captured at remember time. After a renumber
  // they no longer cover the freed segments, so drop every failure and let
  // the next query scan instead of keeping a stale false.
  dropIfRenumbered();
  // SegCount and the outer bounds do not record interior edges. Drop this
  // interval's failure before those edges change, and recompute next time.
  if (Unassigned.reg().isVirtual())
    erase(Unassigned.reg().id());
  if (empty())
    return;
  MCPhysReg Freed = PhysReg.id();
  for (const LiveRange::Segment &Seg : Unassigned.segments) {
    forEachBucket(Seg.start, Seg.end, [&](unsigned Bucket) {
      auto BucketIt = Buckets.find(Bucket);
      if (BucketIt == Buckets.end())
        return;
      for (unsigned Id : BucketIt->second) {
        auto EntryIt = Entries.find(Id);
        if (EntryIt == Entries.end())
          continue;
        Entry &E = EntryIt->second;
        if (!(E.Begin < Seg.end && Seg.start < E.End) || E.Saturated)
          continue;
        if (is_contained(E.FreedRegs, Freed))
          continue;
        if (E.FreedRegs.size() >= MaxFreedRegs) {
          E.Saturated = true;
          E.FreedRegs.clear();
          continue;
        }
        E.FreedRegs.push_back(Freed);
      }
    });
  }
}

} // namespace llvm::reassign
