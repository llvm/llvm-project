//===-- Support/FoldingSet.cpp - Uniquing Hash Set --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a hash set that can be used to remove duplication of
// nodes in a graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SwapByteOrder.h"
#include <cassert>
#include <cstring>
using namespace llvm;

//===----------------------------------------------------------------------===//
// FoldingSetNodeIDRef Implementation

bool FoldingSetNodeIDRef::operator==(FoldingSetNodeIDRef RHS) const {
  if (Size != RHS.Size)
    return false;
  return memcmp(Data, RHS.Data, Size * sizeof(*Data)) == 0;
}

bool FoldingSetNodeIDRef::operator<(FoldingSetNodeIDRef RHS) const {
  if (Size != RHS.Size)
    return Size < RHS.Size;
  return memcmp(Data, RHS.Data, Size * sizeof(*Data)) < 0;
}

//===----------------------------------------------------------------------===//
// FoldingSetNodeID Implementation

void FoldingSetNodeID::AddString(StringRef String) {
  unsigned Size = String.size();

  unsigned NumInserts = 1 + divideCeil(Size, 4);
  Bits.reserve(Bits.size() + NumInserts);

  Bits.push_back(Size);
  if (!Size)
    return;

  unsigned Units = Size / 4;
  unsigned Pos = 0;
  const unsigned *Base = (const unsigned *)String.data();

  // If the string is aligned do a bulk transfer.
  if (!((intptr_t)Base & 3)) {
    Bits.append(Base, Base + Units);
    Pos = (Units + 1) * 4;
  } else {
    // Otherwise do it the hard way.
    // To be compatible with above bulk transfer, we need to take endianness
    // into account.
    static_assert(sys::IsBigEndianHost || sys::IsLittleEndianHost,
                  "Unexpected host endianness");
    if (sys::IsBigEndianHost) {
      for (Pos += 4; Pos <= Size; Pos += 4) {
        unsigned V = ((unsigned char)String[Pos - 4] << 24) |
                     ((unsigned char)String[Pos - 3] << 16) |
                     ((unsigned char)String[Pos - 2] << 8) |
                     (unsigned char)String[Pos - 1];
        Bits.push_back(V);
      }
    } else { // Little-endian host
      for (Pos += 4; Pos <= Size; Pos += 4) {
        unsigned V = ((unsigned char)String[Pos - 1] << 24) |
                     ((unsigned char)String[Pos - 2] << 16) |
                     ((unsigned char)String[Pos - 3] << 8) |
                     (unsigned char)String[Pos - 4];
        Bits.push_back(V);
      }
    }
  }

  // With the leftover bits.
  unsigned V = 0;
  // Pos will have overshot size by 4 - #bytes left over.
  // No need to take endianness into account here - this is always executed.
  switch (Pos - Size) {
  case 1:
    V = (V << 8) | (unsigned char)String[Size - 3];
    [[fallthrough]];
  case 2:
    V = (V << 8) | (unsigned char)String[Size - 2];
    [[fallthrough]];
  case 3:
    V = (V << 8) | (unsigned char)String[Size - 1];
    break;
  default:
    return; // Nothing left.
  }

  Bits.push_back(V);
}

void FoldingSetNodeID::AddNodeID(const FoldingSetNodeID &ID) {
  Bits.append(ID.Bits.begin(), ID.Bits.end());
}

bool FoldingSetNodeID::operator==(const FoldingSetNodeID &RHS) const {
  return *this == FoldingSetNodeIDRef(RHS.Bits.data(), RHS.Bits.size());
}

bool FoldingSetNodeID::operator==(FoldingSetNodeIDRef RHS) const {
  return FoldingSetNodeIDRef(Bits.data(), Bits.size()) == RHS;
}

bool FoldingSetNodeID::operator<(const FoldingSetNodeID &RHS) const {
  return *this < FoldingSetNodeIDRef(RHS.Bits.data(), RHS.Bits.size());
}

bool FoldingSetNodeID::operator<(FoldingSetNodeIDRef RHS) const {
  return FoldingSetNodeIDRef(Bits.data(), Bits.size()) < RHS;
}

FoldingSetNodeIDRef
FoldingSetNodeID::Intern(BumpPtrAllocator &Allocator) const {
  unsigned *New = Allocator.Allocate<unsigned>(Bits.size());
  llvm::uninitialized_copy(Bits, New);
  return FoldingSetNodeIDRef(New, Bits.size());
}

//===----------------------------------------------------------------------===//
// FoldingSetBase Theory of Operations
//
// FoldingSet is implemented as an open-addressing Swiss Table hash set.
//
// Memory Layout:
// A single heap allocation holds both the control bytes (Ctrl) and the bucket
// pointers (Buckets) sequentially:
//   [ Ctrl (NumBuckets + GroupWidth bytes) ]
//   [ Buckets (NumBuckets * sizeof(void *) bytes) ]
//
// Control Byte Encoding:
// - 0x80 (Empty): The slot has never been occupied.
// - 0xFE (Deleted): The slot previously held an element that was removed.
// - 0x00..0x7F (Occupied): Stores H2(Hash) = Hash & 0x7F (the low 7 bits).
//
// Control Byte Mirroring:
// The first GroupWidth (8) control bytes are mirrored at the end of the
// control array (Ctrl[NumBuckets .. NumBuckets + GroupWidth - 1]). This
// allows 8-byte group loads to read past the end of the table without wrapping
// or branching.
//
// Hash Splitting and Probing:
// - H1(Hash) = Hash >> 7 determines the initial group index.
// - H2(Hash) = Hash & 0x7F is stored in the control byte for fast filtering.
// Probing proceeds triangularly in increments of GroupWidth (8 slots at a
// time), checking control bytes in parallel via 64-bit SWAR bitmask operations.
//
// Hash Caching:
// Nodes cache their 32-bit hash value to avoid recomputing profiles during
// table rehashing and node removal.
//
// Load Factor:
// The table doubles in capacity when (NumNodes + NumDeleted + 1) * 8 exceeds
// NumBuckets * 7 (a maximum load factor of 87.5%).

//===----------------------------------------------------------------------===//
/// Helper functions for FoldingSetBase.

// Extract the 7-bit tag stored in the control byte for fast filtering.
static inline uint8_t H2(uint32_t Hash) {
  return static_cast<uint8_t>(Hash & 0x7F);
}

// Extract the initial group index for probing.
static inline unsigned H1(uint32_t Hash) { return Hash >> 7; }

// Ensure the hash value never wraps to nullptr when encoded.
static inline uint32_t sanitizeHash(uint32_t Hash) {
  if (Hash == UINT32_MAX)
    return 0;
  return Hash;
}

// Encode a 32-bit hash into a non-null opaque pointer token.
static inline void *encodeHash(uint32_t Hash) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(Hash) + 1);
}

// Decode a 32-bit hash from an opaque pointer token.
static inline uint32_t decodeHash(void *InsertPos) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(InsertPos) - 1);
}

namespace {

// A bitmask representing matching slots within a probe group.
struct BitMask {
  uint64_t Mask;
  explicit BitMask(uint64_t M) : Mask(M) {}
  explicit operator bool() const { return Mask != 0; }
  int lowestSetBit() const { return llvm::countr_zero(Mask) >> 3; }
  BitMask removeLowestBit() const { return BitMask(Mask & (Mask - 1)); }
};

// An 8-byte group of control metadata that performs parallel slot matching.
struct Group {
  static_assert(FoldingSetBase::GroupWidth == 8,
                "Group SWAR matching requires a group width of 8 bytes.");
  static constexpr uint64_t Lsbs = 0x0101010101010101ULL;
  static constexpr uint64_t Msbs = 0x8080808080808080ULL;

  uint64_t Ctrl;

  explicit Group(const uint8_t *Ptr) : Ctrl(support::endian::read64le(Ptr)) {}

  BitMask matchByte(uint8_t Byte) const {
    uint64_t X = Ctrl ^ (Lsbs * Byte);
    return BitMask((X - Lsbs) & ~X & Msbs);
  }

  BitMask matchEmptyOrDeleted() const { return BitMask(Ctrl & Msbs); }

  BitMask matchEmpty() const { return matchByte(FoldingSetBase::Empty); }
};

// Generate triangular probing offsets across group boundaries.
struct ProbeSequence {
  unsigned Mask;
  unsigned Offset;
  unsigned Step = 0;

  ProbeSequence(uint32_t Hash, unsigned NumBuckets)
      : Mask(NumBuckets - 1), Offset(H1(Hash) & Mask) {}

  unsigned offset() const { return Offset; }
  unsigned slot(int Pos) const { return (Offset + Pos) & Mask; }

  void next() {
    Step += FoldingSetBase::GroupWidth;
    Offset = (Offset + Step) & Mask;
  }
};

} // namespace

/// AllocateBuckets - Allocate and initialize storage for Ctrl and Buckets.
static std::pair<uint8_t *, void **> AllocateBuckets(unsigned NumBuckets) {
  size_t CtrlBytes = NumBuckets + FoldingSetBase::GroupWidth;
  size_t BucketsBytes = NumBuckets * sizeof(void *);
  uint8_t *Ctrl = static_cast<uint8_t *>(safe_malloc(CtrlBytes + BucketsBytes));
  void **Buckets = reinterpret_cast<void **>(Ctrl + CtrlBytes);
  memset(Ctrl, FoldingSetBase::Empty, CtrlBytes);
  return {Ctrl, Buckets};
}

//===----------------------------------------------------------------------===//
// FoldingSetBase Implementation

FoldingSetBase::FoldingSetBase(unsigned Log2InitSize) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  std::tie(Ctrl, Buckets) = AllocateBuckets(NumBuckets);
}

FoldingSetBase::FoldingSetBase(FoldingSetBase &&Arg)
    : Ctrl(Arg.Ctrl), Buckets(Arg.Buckets), NumBuckets(Arg.NumBuckets),
      NumNodes(Arg.NumNodes), NumDeleted(Arg.NumDeleted) {
  Arg.incrementEpoch();
  Arg.Ctrl = const_cast<uint8_t *>(EmptyGroup);
  Arg.Buckets = nullptr;
  Arg.NumBuckets = 0;
  Arg.NumNodes = 0;
  Arg.NumDeleted = 0;
}

FoldingSetBase &FoldingSetBase::operator=(FoldingSetBase &&RHS) {
  if (this == &RHS)
    return *this;

  incrementEpoch();
  RHS.incrementEpoch();

  if (NumBuckets)
    free(Ctrl);

  Ctrl = RHS.Ctrl;
  Buckets = RHS.Buckets;
  NumBuckets = RHS.NumBuckets;
  NumNodes = RHS.NumNodes;
  NumDeleted = RHS.NumDeleted;
  RHS.Ctrl = const_cast<uint8_t *>(EmptyGroup);
  RHS.Buckets = nullptr;
  RHS.NumBuckets = 0;
  RHS.NumNodes = 0;
  RHS.NumDeleted = 0;
  return *this;
}

FoldingSetBase::~FoldingSetBase() {
  if (NumBuckets)
    free(Ctrl);
}

void FoldingSetBase::clear() {
  incrementEpoch();
  if (NumBuckets == 0)
    return;
  memset(Ctrl, Empty, NumBuckets + GroupWidth);
  NumNodes = 0;
  NumDeleted = 0;
}

void FoldingSetBase::insertImpl(void *N, uint32_t Hash) {
  incrementEpoch();
  uint8_t TargetH2 = H2(Hash);

  for (ProbeSequence Seq(Hash, NumBuckets);; Seq.next()) {
    Group G(Ctrl + Seq.offset());
    if (BitMask Candidates = G.matchEmptyOrDeleted()) {
      unsigned SlotIdx = Seq.slot(Candidates.lowestSetBit());
      if (Ctrl[SlotIdx] == Deleted)
        --NumDeleted;
      setCtrlMirrored(SlotIdx, TargetH2);
      Buckets[SlotIdx] = N;
      static_cast<Node *>(N)->setFoldingSetHash(Hash);
      ++NumNodes;
      return;
    }
  }
}

void FoldingSetBase::GrowBucketCount(unsigned NewBucketCount,
                                     const FoldingSetInfo &Info) {
  assert((NewBucketCount > NumBuckets) &&
         "Can't shrink a folding set with GrowBucketCount");
  assert(isPowerOf2_32(NewBucketCount) && "Bad bucket count!");

  FoldingSetBase Tmp(llvm::Log2_32(NewBucketCount));
  for (unsigned i = 0; i != NumBuckets; ++i) {
    if (isEmptyOrDeleted(Ctrl[i]))
      continue;
    Node *N = static_cast<Node *>(Buckets[i]);
    Tmp.insertImpl(N, N->getFoldingSetHash());
  }

  *this = std::move(Tmp);
}

void FoldingSetBase::reserve(unsigned EltCount, const FoldingSetInfo &Info) {
  if (EltCount <= capacity())
    return;
  unsigned RequiredBuckets = std::max((EltCount * 8 + 6) / 7, GroupWidth);
  GrowBucketCount(llvm::bit_ceil(RequiredBuckets), Info);
}

FoldingSetBase::Node *FoldingSetBase::FindNodeOrInsertPos(
    const FoldingSetNodeID &ID, void *&InsertPos, const FoldingSetInfo &Info) {
  uint32_t IDHash = sanitizeHash(ID.ComputeHash());
  uint8_t TargetH2 = H2(IDHash);
  FoldingSetNodeID TempID;
  for (ProbeSequence Seq(IDHash, NumBuckets);; Seq.next()) {
    Group G(Ctrl + Seq.offset());
    for (BitMask Matches = G.matchByte(TargetH2); Matches;
         Matches = Matches.removeLowestBit()) {
      unsigned SlotIdx = Seq.slot(Matches.lowestSetBit());
      Node *Candidate = static_cast<Node *>(Buckets[SlotIdx]);
      if (Info.NodeEquals(this, Candidate, ID, IDHash, TempID)) {
        InsertPos = nullptr;
        return Candidate;
      }
      TempID.clear();
    }

    if (G.matchEmpty())
      break;
  }

  // Didn't find the node, return null with the encoded hash as the InsertPos.
  InsertPos = encodeHash(IDHash);
  return nullptr;
}

void FoldingSetBase::InsertNode(Node *N, void *InsertPos,
                                const FoldingSetInfo &Info) {
  // Do we need to grow the hashtable?
  if (NumNodes + NumDeleted + 1 > capacity())
    GrowBucketCount(NumBuckets * 2, Info);

  assert(InsertPos && "Invalid InsertPos!");
  insertImpl(N, decodeHash(InsertPos));
}

bool FoldingSetBase::RemoveNode(Node *N) {
  uint32_t Hash = N->getFoldingSetHash();
  uint8_t TargetH2 = H2(Hash);

  for (ProbeSequence Seq(Hash, NumBuckets);; Seq.next()) {
    Group G(Ctrl + Seq.offset());
    for (BitMask Matches = G.matchByte(TargetH2); Matches;
         Matches = Matches.removeLowestBit()) {
      unsigned SlotIdx = Seq.slot(Matches.lowestSetBit());
      if (Buckets[SlotIdx] == N) {
        incrementEpoch();
        setCtrlMirrored(SlotIdx, Deleted);
        Buckets[SlotIdx] = nullptr;
        --NumNodes;
        ++NumDeleted;
        return true;
      }
    }

    if (G.matchEmpty())
      return false;
  }
}

FoldingSetBase::Node *
FoldingSetBase::GetOrInsertNode(Node *N, const FoldingSetInfo &Info) {
  FoldingSetNodeID ID;
  Info.GetNodeProfile(this, N, ID);
  void *IP;
  if (Node *E = FindNodeOrInsertPos(ID, IP, Info))
    return E;
  InsertNode(N, IP, Info);
  return N;
}

//===----------------------------------------------------------------------===//
// FoldingSetIteratorImpl Implementation

FoldingSetIteratorImpl::FoldingSetIteratorImpl(const DebugEpochBase *Epoch,
                                               const FoldingSetBase *Set,
                                               unsigned Index)
    : DebugEpochBase::HandleBase(Epoch), Set(Set), Index(Index) {
  assert(Set && "Set cannot be null!");
  // Fast-forward to end() when the set is empty.
  if (Set->empty()) {
    this->Index = Set->NumBuckets;
    return;
  }
  while (this->Index < Set->NumBuckets &&
         FoldingSetBase::isEmptyOrDeleted(Set->Ctrl[this->Index]))
    ++this->Index;
}

void FoldingSetIteratorImpl::advance() {
  assert(isHandleInSync() && "invalid iterator access!");
  ++Index;
  while (Index < Set->NumBuckets &&
         FoldingSetBase::isEmptyOrDeleted(Set->Ctrl[Index]))
    ++Index;
}
