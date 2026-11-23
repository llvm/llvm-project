//===- MemProfRadixTree.cpp - Radix tree encoded callstacks ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains logic that implements a space efficient radix tree
// encoding for callstacks used by MemProf.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProfRadixTree.h"

namespace llvm {
namespace memprof {
// Encode a call stack into RadixArray.  Return the starting index within
// RadixArray.  For each call stack we encode, we emit two or three components
// into RadixArray.  If a given call stack doesn't have a common prefix relative
// to the previous one, we emit:
//
// - the frames in the given call stack in the root-to-leaf order
//
// - the length of the given call stack
//
// If a given call stack has a non-empty common prefix relative to the previous
// one, we emit:
//
// - the relative location of the common prefix, encoded as a negative number.
//
// - a portion of the given call stack that's beyond the common prefix
//
// - the length of the given call stack, including the length of the common
//   prefix.
//
// The resulting RadixArray requires a somewhat unintuitive backward traversal
// to reconstruct a call stack -- read the call stack length and scan backward
// while collecting frames in the leaf to root order.  build, the caller of this
// function, reverses RadixArray in place so that we can reconstruct a call
// stack as if we were deserializing an array in a typical way -- the call stack
// length followed by the frames in the leaf-to-root order except that we need
// to handle pointers to parents along the way.
//
// To quickly determine the location of the common prefix within RadixArray,
// Indexes caches the indexes of the previous call stack's frames within
// RadixArray.
template <typename FrameIdTy>
LinearCallStackId CallStackRadixTreeBuilder<FrameIdTy>::encodeCallStack(
    const llvm::SmallVector<FrameIdTy> *CallStack,
    const llvm::SmallVector<FrameIdTy> *Prev,
    const llvm::DenseMap<FrameIdTy, LinearFrameId> *MemProfFrameIndexes) {
  // Compute the length of the common root prefix between Prev and CallStack.
  uint32_t CommonLen = 0;
  if (Prev) {
    auto Pos = std::mismatch(Prev->rbegin(), Prev->rend(), CallStack->rbegin(),
                             CallStack->rend());
    CommonLen = std::distance(CallStack->rbegin(), Pos.second);
  }

  // Drop the portion beyond CommonLen.
  assert(CommonLen <= Indexes.size());
  Indexes.resize(CommonLen);

  // Append a pointer to the parent.
  if (CommonLen) {
    uint32_t CurrentIndex = RadixArray.size();
    uint32_t ParentIndex = Indexes.back();
    // The offset to the parent must be negative because we are pointing to an
    // element we've already added to RadixArray.
    assert(ParentIndex < CurrentIndex);
    RadixArray.push_back(ParentIndex - CurrentIndex);
  }

  // Copy the part of the call stack beyond the common prefix to RadixArray.
  assert(CommonLen <= CallStack->size());
  for (FrameIdTy F : llvm::drop_begin(llvm::reverse(*CallStack), CommonLen)) {
    // Remember the index of F in RadixArray.
    Indexes.push_back(RadixArray.size());
    RadixArray.push_back(
        MemProfFrameIndexes ? MemProfFrameIndexes->find(F)->second : F);
  }
  assert(CallStack->size() == Indexes.size());

  // End with the call stack length.
  RadixArray.push_back(CallStack->size());

  // Return the index within RadixArray where we can start reconstructing a
  // given call stack from.
  return RadixArray.size() - 1;
}

template <typename FrameIdTy>
void CallStackRadixTreeBuilder<FrameIdTy>::build(
    llvm::MapVector<CallStackId, llvm::SmallVector<FrameIdTy>>
        &&MemProfCallStackData,
    const llvm::DenseMap<FrameIdTy, LinearFrameId> *MemProfFrameIndexes,
    llvm::DenseMap<FrameIdTy, FrameStat> &FrameHistogram) {
  // Take the vector portion of MemProfCallStackData.  The vector is exactly
  // what we need to sort.  Also, we no longer need its lookup capability.
  llvm::SmallVector<CSIdPair, 0> CallStacks = MemProfCallStackData.takeVector();

  // Return early if we have no work to do.
  if (CallStacks.empty()) {
    RadixArray.clear();
    CallStackPos.clear();
    return;
  }

  // Sorting the list of call stacks in the dictionary order is sufficient to
  // maximize the length of the common prefix between two adjacent call stacks
  // and thus minimize the length of RadixArray.  However, we go one step
  // further and try to reduce the number of times we follow pointers to parents
  // during deserilization.  Consider a poorly encoded radix tree:
  //
  // CallStackId 1:  f1 -> f2 -> f3
  //                  |
  // CallStackId 2:   +--- f4 -> f5
  //                        |
  // CallStackId 3:         +--> f6
  //
  // Here, f2 and f4 appear once and twice, respectively, in the call stacks.
  // Once we encode CallStackId 1 into RadixArray, every other call stack with
  // common prefix f1 ends up pointing to CallStackId 1.  Since CallStackId 3
  // share "f1 f4" with CallStackId 2, CallStackId 3 needs to follow pointers to
  // parents twice.
  //
  // We try to alleviate the situation by sorting the list of call stacks by
  // comparing the popularity of frames rather than the integer values of
  // FrameIds.  In the example above, f4 is more popular than f2, so we sort the
  // call stacks and encode them as:
  //
  // CallStackId 2:  f1 -- f4 -> f5
  //                  |     |
  // CallStackId 3:   |     +--> f6
  //                  |
  // CallStackId 1:   +--> f2 -> f3
  //
  // Notice that CallStackId 3 follows a pointer to a parent only once.
  //
  // All this is a quick-n-dirty trick to reduce the number of jumps.  The
  // proper way would be to compute the weight of each radix tree node -- how
  // many call stacks use a given radix tree node, and encode a radix tree from
  // the heaviest node first.  We do not do so because that's a lot of work.
  llvm::sort(CallStacks, [&](const CSIdPair &L, const CSIdPair &R) {
    // Call stacks are stored from leaf to root.  Perform comparisons from the
    // root.
    return std::lexicographical_compare(
        L.second.rbegin(), L.second.rend(), R.second.rbegin(), R.second.rend(),
        [&](FrameIdTy F1, FrameIdTy F2) {
          uint64_t H1 = FrameHistogram[F1].Count;
          uint64_t H2 = FrameHistogram[F2].Count;
          // Popular frames should come later because we encode call stacks from
          // the last one in the list.
          if (H1 != H2)
            return H1 < H2;
          // For sort stability.
          return F1 < F2;
        });
  });

  // Reserve some reasonable amount of storage.
  RadixArray.clear();
  RadixArray.reserve(CallStacks.size() * 8);

  // Indexes will grow as long as the longest call stack.
  Indexes.clear();
  Indexes.reserve(512);

  // CallStackPos will grow to exactly CallStacks.size() entries.
  CallStackPos.clear();
  CallStackPos.reserve(CallStacks.size());

  // Compute the radix array.  We encode one call stack at a time, computing the
  // longest prefix that's shared with the previous call stack we encode.  For
  // each call stack we encode, we remember a mapping from CallStackId to its
  // position within RadixArray.
  //
  // As an optimization, we encode from the last call stack in CallStacks to
  // reduce the number of times we follow pointers to the parents.  Consider the
  // list of call stacks that has been sorted in the dictionary order:
  //
  // Call Stack 1: F1
  // Call Stack 2: F1 -> F2
  // Call Stack 3: F1 -> F2 -> F3
  //
  // If we traversed CallStacks in the forward order, we would end up with a
  // radix tree like:
  //
  // Call Stack 1:  F1
  //                |
  // Call Stack 2:  +---> F2
  //                      |
  // Call Stack 3:        +---> F3
  //
  // Notice that each call stack jumps to the previous one.  However, if we
  // traverse CallStacks in the reverse order, then Call Stack 3 has the
  // complete call stack encoded without any pointers.  Call Stack 1 and 2 point
  // to appropriate prefixes of Call Stack 3.
  const llvm::SmallVector<FrameIdTy> *Prev = nullptr;
  for (const auto &[CSId, CallStack] : llvm::reverse(CallStacks)) {
    LinearCallStackId Pos =
        encodeCallStack(&CallStack, Prev, MemProfFrameIndexes);
    CallStackPos.insert({CSId, Pos});
    Prev = &CallStack;
  }

  // "RadixArray.size() - 1" below is problematic if RadixArray is empty.
  assert(!RadixArray.empty());

  // Reverse the radix array in place.  We do so mostly for intuitive
  // deserialization where we would read the length field and then the call
  // stack frames proper just like any other array deserialization, except
  // that we have occasional jumps to take advantage of prefixes.
  for (size_t I = 0, J = RadixArray.size() - 1; I < J; ++I, --J)
    std::swap(RadixArray[I], RadixArray[J]);

  // "Reverse" the indexes stored in CallStackPos.
  for (auto &[K, V] : CallStackPos)
    V = RadixArray.size() - 1 - V;
}

// Explicitly instantiate class with the utilized FrameIdTy.
template class LLVM_EXPORT_TEMPLATE CallStackRadixTreeBuilder<FrameId>;
template class LLVM_EXPORT_TEMPLATE CallStackRadixTreeBuilder<LinearFrameId>;

template <typename FrameIdTy>
llvm::DenseMap<FrameIdTy, FrameStat>
computeFrameHistogram(llvm::MapVector<CallStackId, llvm::SmallVector<FrameIdTy>>
                          &MemProfCallStackData) {
  llvm::DenseMap<FrameIdTy, FrameStat> Histogram;

  for (const auto &KV : MemProfCallStackData) {
    const auto &CS = KV.second;
    for (unsigned I = 0, E = CS.size(); I != E; ++I) {
      auto &S = Histogram[CS[I]];
      ++S.Count;
      S.PositionSum += I;
    }
  }
  return Histogram;
}

// Explicitly instantiate function with the utilized FrameIdTy.
template LLVM_ABI llvm::DenseMap<FrameId, FrameStat>
computeFrameHistogram<FrameId>(
    llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>>
        &MemProfCallStackData);
template LLVM_ABI llvm::DenseMap<LinearFrameId, FrameStat>
computeFrameHistogram<LinearFrameId>(
    llvm::MapVector<CallStackId, llvm::SmallVector<LinearFrameId>>
        &MemProfCallStackData);
} // namespace memprof
} // namespace llvm
