//===- MemProfRadixTree.h - MemProf format support ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A custom Radix Tree builder for memprof data to optimize for space.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_MEMPROFRADIXTREE_H
#define LLVM_PROFILEDATA_MEMPROFRADIXTREE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ProfileData/IndexedMemProfData.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Compiler.h"

#include <optional>

namespace llvm {
namespace memprof {
namespace detail {
// "Dereference" the iterator from DenseMap or OnDiskChainedHashTable.  We have
// to do so in one of two different ways depending on the type of the hash
// table.
template <typename value_type, typename IterTy>
value_type DerefIterator(IterTy Iter) {
  using deref_type = llvm::remove_cvref_t<decltype(*Iter)>;
  if constexpr (std::is_same_v<deref_type, value_type>)
    return *Iter;
  else
    return Iter->second;
}
} // namespace detail

// A function object that returns a frame for a given FrameId.
template <typename MapTy> struct FrameIdConverter {
  std::optional<FrameId> LastUnmappedId;
  MapTy &Map;

  FrameIdConverter() = delete;
  FrameIdConverter(MapTy &Map) : Map(Map) {}

  // Delete the copy constructor and copy assignment operator to avoid a
  // situation where a copy of FrameIdConverter gets an error in LastUnmappedId
  // while the original instance doesn't.
  FrameIdConverter(const FrameIdConverter &) = delete;
  FrameIdConverter &operator=(const FrameIdConverter &) = delete;

  Frame operator()(FrameId Id) {
    auto Iter = Map.find(Id);
    if (Iter == Map.end()) {
      LastUnmappedId = Id;
      return Frame();
    }
    return detail::DerefIterator<Frame>(Iter);
  }
};

// A function object that returns a call stack for a given CallStackId.
template <typename MapTy> struct CallStackIdConverter {
  std::optional<CallStackId> LastUnmappedId;
  MapTy &Map;
  llvm::function_ref<Frame(FrameId)> FrameIdToFrame;

  CallStackIdConverter() = delete;
  CallStackIdConverter(MapTy &Map,
                       llvm::function_ref<Frame(FrameId)> FrameIdToFrame)
      : Map(Map), FrameIdToFrame(FrameIdToFrame) {}

  // Delete the copy constructor and copy assignment operator to avoid a
  // situation where a copy of CallStackIdConverter gets an error in
  // LastUnmappedId while the original instance doesn't.
  CallStackIdConverter(const CallStackIdConverter &) = delete;
  CallStackIdConverter &operator=(const CallStackIdConverter &) = delete;

  std::vector<Frame> operator()(CallStackId CSId) {
    std::vector<Frame> Frames;
    auto CSIter = Map.find(CSId);
    if (CSIter == Map.end()) {
      LastUnmappedId = CSId;
    } else {
      llvm::SmallVector<FrameId> CS =
          detail::DerefIterator<llvm::SmallVector<FrameId>>(CSIter);
      Frames.reserve(CS.size());
      for (FrameId Id : CS)
        Frames.push_back(FrameIdToFrame(Id));
    }
    return Frames;
  }
};

// A function object that returns a Frame stored at a given index into the Frame
// array in the profile.
struct LinearFrameIdConverter {
  const unsigned char *FrameBase;

  LinearFrameIdConverter() = delete;
  LinearFrameIdConverter(const unsigned char *FrameBase)
      : FrameBase(FrameBase) {}

  Frame operator()(LinearFrameId LinearId) {
    uint64_t Offset = static_cast<uint64_t>(LinearId) * Frame::serializedSize();
    return Frame::deserialize(FrameBase + Offset);
  }
};

// A function object that returns a call stack stored at a given index into the
// call stack array in the profile.
struct LinearCallStackIdConverter {
  const unsigned char *CallStackBase;
  llvm::function_ref<Frame(LinearFrameId)> FrameIdToFrame;

  LinearCallStackIdConverter() = delete;
  LinearCallStackIdConverter(
      const unsigned char *CallStackBase,
      llvm::function_ref<Frame(LinearFrameId)> FrameIdToFrame)
      : CallStackBase(CallStackBase), FrameIdToFrame(FrameIdToFrame) {}

  std::vector<Frame> operator()(LinearCallStackId LinearCSId) {
    std::vector<Frame> Frames;

    const unsigned char *Ptr =
        CallStackBase +
        static_cast<uint64_t>(LinearCSId) * sizeof(LinearFrameId);
    uint32_t NumFrames =
        support::endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
    Frames.reserve(NumFrames);
    for (; NumFrames; --NumFrames) {
      LinearFrameId Elem =
          support::endian::read<LinearFrameId, llvm::endianness::little>(Ptr);
      // Follow a pointer to the parent, if any.  See comments below on
      // CallStackRadixTreeBuilder for the description of the radix tree format.
      if (static_cast<std::make_signed_t<LinearFrameId>>(Elem) < 0) {
        Ptr += (-Elem) * sizeof(LinearFrameId);
        Elem =
            support::endian::read<LinearFrameId, llvm::endianness::little>(Ptr);
      }
      // We shouldn't encounter another pointer.
      assert(static_cast<std::make_signed_t<LinearFrameId>>(Elem) >= 0);
      Frames.push_back(FrameIdToFrame(Elem));
      Ptr += sizeof(LinearFrameId);
    }

    return Frames;
  }
};

// Used to extract caller-callee pairs from the call stack array.  The leaf
// frame is assumed to call a heap allocation function with GUID 0.  The
// resulting pairs are accumulated in CallerCalleePairs.  Users can take it
// with:
//
//   auto Pairs = std::move(Extractor.CallerCalleePairs);
struct CallerCalleePairExtractor {
  // The base address of the radix tree array.
  const unsigned char *CallStackBase;
  // A functor to convert a linear FrameId to a Frame.
  llvm::function_ref<Frame(LinearFrameId)> FrameIdToFrame;
  // A map from caller GUIDs to lists of call sites in respective callers.
  DenseMap<uint64_t, SmallVector<CallEdgeTy, 0>> CallerCalleePairs;

  // The set of linear call stack IDs that we've visited.
  BitVector Visited;

  CallerCalleePairExtractor() = delete;
  CallerCalleePairExtractor(
      const unsigned char *CallStackBase,
      llvm::function_ref<Frame(LinearFrameId)> FrameIdToFrame,
      unsigned RadixTreeSize)
      : CallStackBase(CallStackBase), FrameIdToFrame(FrameIdToFrame),
        Visited(RadixTreeSize) {}

  void operator()(LinearCallStackId LinearCSId) {
    const unsigned char *Ptr =
        CallStackBase +
        static_cast<uint64_t>(LinearCSId) * sizeof(LinearFrameId);
    uint32_t NumFrames =
        support::endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
    // The leaf frame calls a function with GUID 0.
    uint64_t CalleeGUID = 0;
    for (; NumFrames; --NumFrames) {
      LinearFrameId Elem =
          support::endian::read<LinearFrameId, llvm::endianness::little>(Ptr);
      // Follow a pointer to the parent, if any.  See comments below on
      // CallStackRadixTreeBuilder for the description of the radix tree format.
      if (static_cast<std::make_signed_t<LinearFrameId>>(Elem) < 0) {
        Ptr += (-Elem) * sizeof(LinearFrameId);
        Elem =
            support::endian::read<LinearFrameId, llvm::endianness::little>(Ptr);
      }
      // We shouldn't encounter another pointer.
      assert(static_cast<std::make_signed_t<LinearFrameId>>(Elem) >= 0);

      // Add a new caller-callee pair.
      Frame F = FrameIdToFrame(Elem);
      uint64_t CallerGUID = F.Function;
      LineLocation Loc(F.LineOffset, F.Column);
      CallerCalleePairs[CallerGUID].emplace_back(Loc, CalleeGUID);

      // Keep track of the indices we've visited.  If we've already visited the
      // current one, terminate the traversal.  We will not discover any new
      // caller-callee pair by continuing the traversal.
      unsigned Offset =
          std::distance(CallStackBase, Ptr) / sizeof(LinearFrameId);
      if (Visited.test(Offset))
        break;
      Visited.set(Offset);

      Ptr += sizeof(LinearFrameId);
      CalleeGUID = CallerGUID;
    }
  }
};

// A convenience wrapper around FrameIdConverter and CallStackIdConverter for
// tests.
struct IndexedCallstackIdConverter {
  IndexedCallstackIdConverter() = delete;
  IndexedCallstackIdConverter(IndexedMemProfData &MemProfData)
      : FrameIdConv(MemProfData.Frames),
        CSIdConv(MemProfData.CallStacks, FrameIdConv) {}

  // Delete the copy constructor and copy assignment operator to avoid a
  // situation where a copy of IndexedCallstackIdConverter gets an error in
  // LastUnmappedId while the original instance doesn't.
  IndexedCallstackIdConverter(const IndexedCallstackIdConverter &) = delete;
  IndexedCallstackIdConverter &
  operator=(const IndexedCallstackIdConverter &) = delete;

  std::vector<Frame> operator()(CallStackId CSId) { return CSIdConv(CSId); }

  FrameIdConverter<decltype(IndexedMemProfData::Frames)> FrameIdConv;
  CallStackIdConverter<decltype(IndexedMemProfData::CallStacks)> CSIdConv;
};

struct FrameStat {
  // The number of occurrences of a given FrameId.
  uint64_t Count = 0;
  // The sum of indexes where a given FrameId shows up.
  uint64_t PositionSum = 0;
};

// Compute a histogram of Frames in call stacks.
template <typename FrameIdTy>
llvm::DenseMap<FrameIdTy, FrameStat>
computeFrameHistogram(llvm::MapVector<CallStackId, llvm::SmallVector<FrameIdTy>>
                          &MemProfCallStackData);

// Construct a radix tree of call stacks.
//
// A set of call stacks might look like:
//
// CallStackId 1:  f1 -> f2 -> f3
// CallStackId 2:  f1 -> f2 -> f4 -> f5
// CallStackId 3:  f1 -> f2 -> f4 -> f6
// CallStackId 4:  f7 -> f8 -> f9
//
// where each fn refers to a stack frame.
//
// Since we expect a lot of common prefixes, we can compress the call stacks
// into a radix tree like:
//
// CallStackId 1:  f1 -> f2 -> f3
//                       |
// CallStackId 2:        +---> f4 -> f5
//                             |
// CallStackId 3:              +---> f6
//
// CallStackId 4:  f7 -> f8 -> f9
//
// Now, we are interested in retrieving call stacks for a given CallStackId, so
// we just need a pointer from a given call stack to its parent.  For example,
// CallStackId 2 would point to CallStackId 1 as a parent.
//
// We serialize the radix tree above into a single array along with the length
// of each call stack and pointers to the parent call stacks.
//
// Index:              0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
// Array:             L3 f9 f8 f7 L4 f6 J3 L4 f5 f4 J3 L3 f3 f2 f1
//                     ^           ^        ^           ^
//                     |           |        |           |
// CallStackId 4:  0 --+           |        |           |
// CallStackId 3:  4 --------------+        |           |
// CallStackId 2:  7 -----------------------+           |
// CallStackId 1: 11 -----------------------------------+
//
// - LN indicates the length of a call stack, encoded as ordinary integer N.
//
// - JN indicates a pointer to the parent, encoded as -N.
//
// The radix tree allows us to reconstruct call stacks in the leaf-to-root
// order as we scan the array from left ro right while following pointers to
// parents along the way.
//
// For example, if we are decoding CallStackId 2, we start a forward traversal
// at Index 7, noting the call stack length of 4 and obtaining f5 and f4.  When
// we see J3 at Index 10, we resume a forward traversal at Index 13 = 10 + 3,
// picking up f2 and f1.  We are done after collecting 4 frames as indicated at
// the beginning of the traversal.
//
// On-disk IndexedMemProfRecord will refer to call stacks by their indexes into
// the radix tree array, so we do not explicitly encode mappings like:
// "CallStackId 1 -> 11".
template <typename FrameIdTy> class CallStackRadixTreeBuilder {
  // The radix tree array.
  std::vector<LinearFrameId> RadixArray;

  // Mapping from CallStackIds to indexes into RadixArray.
  llvm::DenseMap<CallStackId, LinearCallStackId> CallStackPos;

  // In build, we partition a given call stack into two parts -- the prefix
  // that's common with the previously encoded call stack and the frames beyond
  // the common prefix -- the unique portion.  Then we want to find out where
  // the common prefix is stored in RadixArray so that we can link the unique
  // portion to the common prefix.  Indexes, declared below, helps with our
  // needs.  Intuitively, Indexes tells us where each of the previously encoded
  // call stack is stored in RadixArray.  More formally, Indexes satisfies:
  //
  //   RadixArray[Indexes[I]] == Prev[I]
  //
  // for every I, where Prev is the the call stack in the root-to-leaf order
  // previously encoded by build.  (Note that Prev, as passed to
  // encodeCallStack, is in the leaf-to-root order.)
  //
  // For example, if the call stack being encoded shares 5 frames at the root of
  // the call stack with the previously encoded call stack,
  // RadixArray[Indexes[0]] is the root frame of the common prefix.
  // RadixArray[Indexes[5 - 1]] is the last frame of the common prefix.
  std::vector<LinearCallStackId> Indexes;

  using CSIdPair = std::pair<CallStackId, llvm::SmallVector<FrameIdTy>>;

  // Encode a call stack into RadixArray.  Return the starting index within
  // RadixArray.
  LinearCallStackId encodeCallStack(
      const llvm::SmallVector<FrameIdTy> *CallStack,
      const llvm::SmallVector<FrameIdTy> *Prev,
      const llvm::DenseMap<FrameIdTy, LinearFrameId> *MemProfFrameIndexes);

public:
  CallStackRadixTreeBuilder() = default;

  // Build a radix tree array.
  void
  build(llvm::MapVector<CallStackId, llvm::SmallVector<FrameIdTy>>
            &&MemProfCallStackData,
        const llvm::DenseMap<FrameIdTy, LinearFrameId> *MemProfFrameIndexes,
        llvm::DenseMap<FrameIdTy, FrameStat> &FrameHistogram);

  ArrayRef<LinearFrameId> getRadixArray() const { return RadixArray; }

  llvm::DenseMap<CallStackId, LinearCallStackId> takeCallStackPos() {
    return std::move(CallStackPos);
  }
};

// Defined in MemProfRadixTree.cpp
extern template class LLVM_TEMPLATE_ABI CallStackRadixTreeBuilder<FrameId>;
extern template class LLVM_TEMPLATE_ABI
    CallStackRadixTreeBuilder<LinearFrameId>;

} // namespace memprof
} // namespace llvm
#endif // LLVM_PROFILEDATA_MEMPROFRADIXTREE_H
