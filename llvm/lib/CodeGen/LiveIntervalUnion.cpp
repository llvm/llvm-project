//===-- LiveIntervalUnion.cpp - Live interval union data structure --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LiveIntervalUnion represents a coalesced set of live intervals. This may be
// used during coalescing to represent a congruence class, or during register
// allocation to model liveness of a physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "LiveIntervalUnion.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineLoopRanges.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <algorithm>

using namespace llvm;


// Merge a LiveInterval's segments. Guarantee no overlaps.
void LiveIntervalUnion::unify(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;
  ++Tag;

  // Insert each of the virtual register's live segments into the map.
  LiveInterval::iterator RegPos = VirtReg.begin();
  LiveInterval::iterator RegEnd = VirtReg.end();
  SegmentIter SegPos = Segments.find(RegPos->start);

  while (SegPos.valid()) {
    SegPos.insert(RegPos->start, RegPos->end, &VirtReg);
    if (++RegPos == RegEnd)
      return;
    SegPos.advanceTo(RegPos->start);
  }

  // We have reached the end of Segments, so it is no longer necessary to search
  // for the insertion position.
  // It is faster to insert the end first.
  --RegEnd;
  SegPos.insert(RegEnd->start, RegEnd->end, &VirtReg);
  for (; RegPos != RegEnd; ++RegPos, ++SegPos)
    SegPos.insert(RegPos->start, RegPos->end, &VirtReg);
}

// Remove a live virtual register's segments from this union.
void LiveIntervalUnion::extract(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;
  ++Tag;

  // Remove each of the virtual register's live segments from the map.
  LiveInterval::iterator RegPos = VirtReg.begin();
  LiveInterval::iterator RegEnd = VirtReg.end();
  SegmentIter SegPos = Segments.find(RegPos->start);

  for (;;) {
    assert(SegPos.value() == &VirtReg && "Inconsistent LiveInterval");
    SegPos.erase();
    if (!SegPos.valid())
      return;

    // Skip all segments that may have been coalesced.
    RegPos = VirtReg.advanceTo(RegPos, SegPos.start());
    if (RegPos == RegEnd)
      return;

    SegPos.advanceTo(RegPos->start);
  }
}

void
LiveIntervalUnion::print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
  if (empty()) {
    OS << " empty\n";
    return;
  }
  for (LiveSegments::const_iterator SI = Segments.begin(); SI.valid(); ++SI) {
    OS << " [" << SI.start() << ' ' << SI.stop() << "):"
       << PrintReg(SI.value()->reg, TRI);
  }
  OS << '\n';
}

#ifndef NDEBUG
// Verify the live intervals in this union and add them to the visited set.
void LiveIntervalUnion::verify(LiveVirtRegBitSet& VisitedVRegs) {
  for (SegmentIter SI = Segments.begin(); SI.valid(); ++SI)
    VisitedVRegs.set(SI.value()->reg);
}
#endif //!NDEBUG

// Scan the vector of interfering virtual registers in this union. Assume it's
// quite small.
bool LiveIntervalUnion::Query::isSeenInterference(LiveInterval *VirtReg) const {
  SmallVectorImpl<LiveInterval*>::const_iterator I =
    std::find(InterferingVRegs.begin(), InterferingVRegs.end(), VirtReg);
  return I != InterferingVRegs.end();
}

// Collect virtual registers in this union that interfere with this
// query's live virtual register.
//
// The query state is one of:
//
// 1. CheckedFirstInterference == false: Iterators are uninitialized.
// 2. SeenAllInterferences == true: InterferingVRegs complete, iterators unused.
// 3. Iterators left at the last seen intersection.
//
unsigned LiveIntervalUnion::Query::
collectInterferingVRegs(unsigned MaxInterferingRegs) {
  // Fast path return if we already have the desired information.
  if (SeenAllInterferences || InterferingVRegs.size() >= MaxInterferingRegs)
    return InterferingVRegs.size();

  // Set up iterators on the first call.
  if (!CheckedFirstInterference) {
    CheckedFirstInterference = true;

    // Quickly skip interference check for empty sets.
    if (VirtReg->empty() || LiveUnion->empty()) {
      SeenAllInterferences = true;
      return 0;
    }

    // In most cases, the union will start before VirtReg.
    VirtRegI = VirtReg->begin();
    LiveUnionI.setMap(LiveUnion->getMap());
    LiveUnionI.find(VirtRegI->start);
  }

  LiveInterval::iterator VirtRegEnd = VirtReg->end();
  LiveInterval *RecentReg = 0;
  while (LiveUnionI.valid()) {
    assert(VirtRegI != VirtRegEnd && "Reached end of VirtReg");

    // Check for overlapping interference.
    while (VirtRegI->start < LiveUnionI.stop() &&
           VirtRegI->end > LiveUnionI.start()) {
      // This is an overlap, record the interfering register.
      LiveInterval *VReg = LiveUnionI.value();
      if (VReg != RecentReg && !isSeenInterference(VReg)) {
        RecentReg = VReg;
        InterferingVRegs.push_back(VReg);
        if (InterferingVRegs.size() >= MaxInterferingRegs)
          return InterferingVRegs.size();
      }
      // This LiveUnion segment is no longer interesting.
      if (!(++LiveUnionI).valid()) {
        SeenAllInterferences = true;
        return InterferingVRegs.size();
      }
    }

    // The iterators are now not overlapping, LiveUnionI has been advanced
    // beyond VirtRegI.
    assert(VirtRegI->end <= LiveUnionI.start() && "Expected non-overlap");

    // Advance the iterator that ends first.
    VirtRegI = VirtReg->advanceTo(VirtRegI, LiveUnionI.start());
    if (VirtRegI == VirtRegEnd)
      break;

    // Detect overlap, handle above.
    if (VirtRegI->start < LiveUnionI.stop())
      continue;

    // Still not overlapping. Catch up LiveUnionI.
    LiveUnionI.advanceTo(VirtRegI->start);
  }
  SeenAllInterferences = true;
  return InterferingVRegs.size();
}

bool LiveIntervalUnion::Query::checkLoopInterference(MachineLoopRange *Loop) {
  // VirtReg is likely live throughout the loop, so start by checking LIU-Loop
  // overlaps.
  IntervalMapOverlaps<LiveIntervalUnion::Map, MachineLoopRange::Map>
    Overlaps(LiveUnion->getMap(), Loop->getMap());
  if (!Overlaps.valid())
    return false;

  // The loop is overlapping an LIU assignment. Check VirtReg as well.
  LiveInterval::iterator VRI = VirtReg->find(Overlaps.start());

  for (;;) {
    if (VRI == VirtReg->end())
      return false;
    if (VRI->start < Overlaps.stop())
      return true;

    Overlaps.advanceTo(VRI->start);
    if (!Overlaps.valid())
      return false;
    if (Overlaps.start() < VRI->end)
      return true;

    VRI = VirtReg->advanceTo(VRI, Overlaps.start());
  }
}

void LiveIntervalUnion::Array::init(LiveIntervalUnion::Allocator &Alloc,
                                    unsigned NSize) {
  // Reuse existing allocation.
  if (NSize == Size)
    return;
  clear();
  Size = NSize;
  LIUs = static_cast<LiveIntervalUnion*>(
    malloc(sizeof(LiveIntervalUnion)*NSize));
  for (unsigned i = 0; i != Size; ++i)
    new(LIUs + i) LiveIntervalUnion(Alloc);
}

void LiveIntervalUnion::Array::clear() {
  if (!LIUs)
    return;
  for (unsigned i = 0; i != Size; ++i)
    LIUs[i].~LiveIntervalUnion();
  free(LIUs);
  Size =  0;
  LIUs = 0;
}
