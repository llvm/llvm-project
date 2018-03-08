//===------------------------- LSUnit.h --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A Load/Store unit class that models load/store queues and that implements
/// a simple weak memory consistency model.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_LSUNIT_H
#define LLVM_TOOLS_LLVM_MCA_LSUNIT_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

#define DEBUG_TYPE "llvm-mca"

namespace mca {

/// \brief A Load/Store Unit implementing a load and store queues.
///
/// This class implements a load queue and a store queue to emulate the
/// out-of-order execution of memory operations.
/// Each load (or store) consumes an entry in the load (or store) queue.
///
/// Rules are:
/// 1) A younger load is allowed to pass an older load only if there are no
///    stores nor barriers in between the two loads.
/// 2) An younger store is not allowed to pass an older store.
/// 3) A younger store is not allowed to pass an older load.
/// 4) A younger load is allowed to pass an older store only if the load does
///    not alias with the store.
///
/// This class optimistically assumes that loads don't alias store operations.
/// Under this assumption, younger loads are always allowed to pass older
/// stores (this would only affects rule 4).
/// Essentially, this LSUnit doesn't attempt to run any sort alias analysis to
/// predict when loads and stores don't alias with eachother.
///
/// To enforce aliasing between loads and stores, flag `AssumeNoAlias` must be
/// set to `false` by the constructor of LSUnit.
///
/// In the case of write-combining memory, rule 2. could be relaxed to allow
/// reordering of non-aliasing store operations. At the moment, this is not
/// allowed.
/// To put it in another way, there is no option to specify a different memory
/// type for memory operations (example: write-through, write-combining, etc.).
/// Also, there is no way to weaken the memory model, and this unit currently
/// doesn't support write-combining behavior.
///
/// No assumptions are made on the size of the store buffer.
/// As mentioned before, this class doesn't perform alias analysis.
/// Consequently,  LSUnit doesn't know how to identify cases where
/// store-to-load forwarding may occur.
///
/// LSUnit doesn't attempt to predict whether a load or store hits or misses
/// the L1 cache. To be more specific, LSUnit doesn't know anything about
/// the cache hierarchy and memory types.
/// It only knows if an instruction "mayLoad" and/or "mayStore". For loads, the
/// scheduling model provides an "optimistic" load-to-use latency (which usually
/// matches the load-to-use latency for when there is a hit in the L1D).
///
/// Class MCInstrDesc in LLVM doesn't know about serializing operations, nor
/// memory-barrier like instructions.
/// LSUnit conservatively assumes that an instruction which `mayLoad` and has
/// `unmodeled side effects` behave like a "soft" load-barrier. That means, it
/// serializes loads without forcing a flush of the load queue.
/// Similarly, instructions that both `mayStore` and have `unmodeled side
/// effects` are treated like store barriers. A full memory
/// barrier is a 'mayLoad' and 'mayStore' instruction with unmodeled side
/// effects. This is obviously inaccurate, but this is the best that we can do
/// at the moment.
///
/// Each load/store barrier consumes one entry in the load/store queue. A
/// load/store barrier enforces ordering of loads/stores:
///  - A younger load cannot pass a load barrier.
///  - A younger store cannot pass a store barrier.
///
/// A younger load has to wait for the memory load barrier to execute.
/// A load/store barrier is "executed" when it becomes the oldest entry in
/// the load/store queue(s). That also means, all the older loads/stores have
/// already been executed.
class LSUnit {
  // Load queue size.
  // LQ_Size == 0 means that there are infinite slots in the load queue.
  unsigned LQ_Size;

  // Store queue size.
  // SQ_Size == 0 means that there are infinite slots in the store queue.
  unsigned SQ_Size;

  // If true, loads will never alias with stores. This is the default.
  bool NoAlias;

  std::set<unsigned> LoadQueue;
  std::set<unsigned> StoreQueue;

  void assignLQSlot(unsigned Index);
  void assignSQSlot(unsigned Index);
  bool isReadyNoAlias(unsigned Index) const;

  // An instruction that both 'mayStore' and 'HasUnmodeledSideEffects' is
  // conservatively treated as a store barrier. It forces older store to be
  // executed before newer stores are issued.
  std::set<unsigned> StoreBarriers;

  // An instruction that both 'MayLoad' and 'HasUnmodeledSideEffects' is
  // conservatively treated as a load barrier. It forces older loads to execute
  // before newer loads are issued.
  std::set<unsigned> LoadBarriers;

public:
  LSUnit(unsigned LQ = 0, unsigned SQ = 0, bool AssumeNoAlias = false)
      : LQ_Size(LQ), SQ_Size(SQ), NoAlias(AssumeNoAlias) {}

#ifndef NDEBUG
  void dump() const;
#endif

  bool isSQEmpty() const { return StoreQueue.empty(); }
  bool isLQEmpty() const { return LoadQueue.empty(); }
  bool isSQFull() const { return SQ_Size != 0 && StoreQueue.size() == SQ_Size; }
  bool isLQFull() const { return LQ_Size != 0 && LoadQueue.size() == LQ_Size; }

  void reserve(unsigned Index, bool MayLoad, bool MayStore, bool IsMemBarrier) {
    if (!MayLoad && !MayStore)
      return;
    if (MayLoad) {
      if (IsMemBarrier)
        LoadBarriers.insert(Index);
      assignLQSlot(Index);
    }
    if (MayStore) {
      if (IsMemBarrier)
        StoreBarriers.insert(Index);
      assignSQSlot(Index);
    }
  }

  // The rules are:
  // 1. A store may not pass a previous store.
  // 2. A load may not pass a previous store unless flag 'NoAlias' is set.
  // 3. A load may pass a previous load.
  // 4. A store may not pass a previous load (regardless of flag 'NoAlias').
  // 5. A load has to wait until an older load barrier is fully executed.
  // 6. A store has to wait until an older store barrier is fully executed.
  bool isReady(unsigned Index) const;
  void onInstructionExecuted(unsigned Index);
};

} // namespace mca

#endif
