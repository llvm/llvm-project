//===- MachineLoopRanges.cpp - Ranges of machine loops --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the MachineLoopRanges analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLoopRanges.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

char MachineLoopRanges::ID = 0;
INITIALIZE_PASS_BEGIN(MachineLoopRanges, "machine-loop-ranges",
                "Machine Loop Ranges", true, true)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineLoopRanges, "machine-loop-ranges",
                "Machine Loop Ranges", true, true)

char &llvm::MachineLoopRangesID = MachineLoopRanges::ID;

void MachineLoopRanges::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<SlotIndexes>();
  AU.addRequiredTransitive<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// runOnMachineFunction - Don't do much, loop ranges are computed on demand.
bool MachineLoopRanges::runOnMachineFunction(MachineFunction &) {
  releaseMemory();
  Indexes = &getAnalysis<SlotIndexes>();
  return false;
}

void MachineLoopRanges::releaseMemory() {
  DeleteContainerSeconds(Cache);
  Cache.clear();
}

MachineLoopRange *MachineLoopRanges::getLoopRange(const MachineLoop *Loop) {
  MachineLoopRange *&Range = Cache[Loop];
  if (!Range)
    Range = new MachineLoopRange(Loop, Allocator, *Indexes);
  return Range;
}

/// Create a MachineLoopRange, only accessible to MachineLoopRanges.
MachineLoopRange::MachineLoopRange(const MachineLoop *loop,
                                   MachineLoopRange::Allocator &alloc,
                                   SlotIndexes &Indexes)
  : Loop(loop), Intervals(alloc), Area(0) {
  // Compute loop coverage.
  for (MachineLoop::block_iterator I = Loop->block_begin(),
         E = Loop->block_end(); I != E; ++I) {
    const std::pair<SlotIndex, SlotIndex> &Range = Indexes.getMBBRange(*I);
    Intervals.insert(Range.first, Range.second, 1u);
    Area += Range.first.distance(Range.second);
  }
}

/// overlaps - Return true if this loop overlaps the given range of machine
/// instructions.
bool MachineLoopRange::overlaps(SlotIndex Start, SlotIndex Stop) {
  Map::const_iterator I = Intervals.find(Start);
  return I.valid() && Stop > I.start();
}

unsigned MachineLoopRange::getNumber() const {
  return Loop->getHeader()->getNumber();
}

/// byNumber - Comparator for array_pod_sort that sorts a list of
/// MachineLoopRange pointers by number.
int MachineLoopRange::byNumber(const void *pa, const void *pb) {
  const MachineLoopRange *a = *static_cast<MachineLoopRange *const *>(pa);
  const MachineLoopRange *b = *static_cast<MachineLoopRange *const *>(pb);
  unsigned na = a->getNumber();
  unsigned nb = b->getNumber();
  if (na < nb)
    return -1;
  if (na > nb)
    return 1;
  return 0;
}

/// byAreaDesc - Comparator for array_pod_sort that sorts a list of
/// MachineLoopRange pointers by:
/// 1. Descending area.
/// 2. Ascending number.
int MachineLoopRange::byAreaDesc(const void *pa, const void *pb) {
  const MachineLoopRange *a = *static_cast<MachineLoopRange *const *>(pa);
  const MachineLoopRange *b = *static_cast<MachineLoopRange *const *>(pb);
  if (a->getArea() != b->getArea())
    return a->getArea() > b->getArea() ? -1 : 1;
  return byNumber(pa, pb);
}

void MachineLoopRange::print(raw_ostream &OS) const {
  OS << "Loop#" << getNumber() << " =";
  for (Map::const_iterator I = Intervals.begin(); I.valid(); ++I)
    OS << " [" << I.start() << ';' << I.stop() << ')';
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const MachineLoopRange &MLR) {
  MLR.print(OS);
  return OS;
}
