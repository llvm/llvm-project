//===- SafeStackColoring.h - SafeStack frame coloring ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SAFESTACKCOLORING_H
#define LLVM_LIB_CODEGEN_SAFESTACKCOLORING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <utility>

namespace llvm {

class AllocaInst;
class BasicBlock;
class Function;
class Instruction;

namespace safestack {

/// Compute live ranges of allocas.
/// Live ranges are represented as sets of "interesting" instructions, which are
/// defined as instructions that may start or end an alloca's lifetime. These
/// are:
/// * lifetime.start and lifetime.end intrinsics
/// * first instruction of any basic block
/// Interesting instructions are numbered in the depth-first walk of the CFG,
/// and in the program order inside each basic block.
class StackColoring {
  /// A class representing liveness information for a single basic block.
  /// Each bit in the BitVector represents the liveness property
  /// for a different stack slot.
  struct BlockLifetimeInfo {
    explicit BlockLifetimeInfo(unsigned Size)
        : Begin(Size), End(Size), LiveIn(Size), LiveOut(Size) {}

    /// Which slots BEGINs in each basic block.
    BitVector Begin;

    /// Which slots ENDs in each basic block.
    BitVector End;

    /// Which slots are marked as LIVE_IN, coming into each basic block.
    BitVector LiveIn;

    /// Which slots are marked as LIVE_OUT, coming out of each basic block.
    BitVector LiveOut;
  };

public:
  /// This class represents a set of interesting instructions where an alloca is
  /// live.
  class LiveRange {
    BitVector Bits;
    friend raw_ostream &operator<<(raw_ostream &OS,
                                   const StackColoring::LiveRange &R);

  public:
    LiveRange(unsigned Size, bool Set = false) : Bits(Size, Set) {}
    void addRange(unsigned Start, unsigned End) { Bits.set(Start, End); }

    bool overlaps(const LiveRange &Other) const {
      return Bits.anyCommon(Other.Bits);
    }

    void join(const LiveRange &Other) { Bits |= Other.Bits; }
  };

private:
  const Function &F;

  /// Maps active slots (per bit) for each basic block.
  using LivenessMap = DenseMap<const BasicBlock *, BlockLifetimeInfo>;
  LivenessMap BlockLiveness;

  /// Number of interesting instructions.
  int NumInst = -1;

  /// Numeric ids for interesting instructions.
  DenseMap<const IntrinsicInst *, unsigned> InstructionNumbering;

  /// A range [Start, End) of instruction ids for each basic block.
  /// Instructions inside each BB have monotonic and consecutive ids.
  DenseMap<const BasicBlock *, std::pair<unsigned, unsigned>> BlockInstRange;

  ArrayRef<const AllocaInst *> Allocas;
  unsigned NumAllocas;
  DenseMap<const AllocaInst *, unsigned> AllocaNumbering;

  /// LiveRange for allocas.
  SmallVector<LiveRange, 8> LiveRanges;

  /// The set of allocas that have at least one lifetime.start. All other
  /// allocas get LiveRange that corresponds to the entire function.
  BitVector InterestingAllocas;

  struct Marker {
    unsigned AllocaNo;
    bool IsStart;
  };

  /// List of {InstNo, {AllocaNo, IsStart}} for each BB, ordered by InstNo.
  DenseMap<const BasicBlock *, SmallVector<std::pair<unsigned, Marker>, 4>>
      BBMarkers;

  void dumpAllocas() const;
  void dumpBlockLiveness() const;
  void dumpLiveRanges() const;

  void collectMarkers();
  void calculateLocalLiveness();
  void calculateLiveIntervals();

public:
  StackColoring(const Function &F, ArrayRef<const AllocaInst *> Allocas);

  void run();
  std::vector<const IntrinsicInst *> getMarkers() const;

  /// Returns a set of "interesting" instructions where the given alloca is
  /// live. Not all instructions in a function are interesting: we pick a set
  /// that is large enough for LiveRange::Overlaps to be correct.
  const LiveRange &getLiveRange(const AllocaInst *AI) const;

  /// Returns a live range that represents an alloca that is live throughout the
  /// entire function.
  LiveRange getFullLiveRange() const {
    assert(NumInst >= 0);
    return LiveRange(NumInst, true);
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS, const BitVector &V) {
  OS << "{";
  int Idx = V.find_first();
  bool First = true;
  while (Idx >= 0) {
    if (!First) {
      OS << ", ";
    }
    First = false;
    OS << Idx;
    Idx = V.find_next(Idx);
  }
  OS << "}";
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS,
                               const StackColoring::LiveRange &R) {
  return OS << R.Bits;
}

} // end namespace safestack

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_SAFESTACKCOLORING_H
