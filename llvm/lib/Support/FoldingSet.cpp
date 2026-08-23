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
// FoldingSet is implemented as an open-addressing hash set with linear probing
// and Knuth TAOCP 6.4 Algorithm R gap-closing deletion (tombstone-free).
//
// Memory Layout:
// A single heap allocation holds both the bucket pointers (Buckets) and the
// packed 1-bit occupancy array (Used) sequentially:
//   [ Buckets (NumBuckets * sizeof(void *) bytes) ]
//   [ Used ((NumBuckets + 31) / 32 * sizeof(uint32_t) bytes) ]
//
// Linear Probing & Equality Check:
// Probing proceeds linearly starting from Hash & (NumBuckets - 1).
// For each occupied slot, the full 32-bit cached hash on FoldingSetNode is
// compared. If the hashes match, NodeEquals is called to perform the full
// profile comparison.
//
// Tombstone-Free Deletion (Algorithm R):
// When a node is erased, subsequent elements in the linear-probe cluster are
// inspected. Any element whose home position lies before or at the hole is
// shifted backward to close the gap, ensuring chains remain contiguous without
// tombstones.
//
// Hash Caching:
// Nodes cache their 32-bit hash value to avoid recomputing profiles during
// table rehashing, node lookups, and node removal.
//
// Load Factor:
// The table doubles in capacity when (NumNodes + 1) * 4 exceeds NumBuckets * 3
// (a maximum load factor of 75%).

//===----------------------------------------------------------------------===//
/// Helper functions for FoldingSetBase.

namespace {
using UsedT = uint32_t;

// Number of used words backing N buckets where N is zero or a power of two.
constexpr size_t usedWords(size_t N) {
  assert((N == 0 || isPowerOf2_64(N)) &&
         "bucket count must be zero or a power of two");
  return (N + 31) / 32;
}

inline bool used(const UsedT *U, size_t I) {
  return (U[I >> 5] >> (I & 31)) & 1;
}

inline void setUsed(UsedT *U, size_t I) { U[I >> 5] |= UsedT(1) << (I & 31); }

inline void unsetUsed(UsedT *U, size_t I) {
  U[I >> 5] &= ~(UsedT(1) << (I & 31));
}

template <typename Fn>
LLVM_ATTRIBUTE_ALWAYS_INLINE void forEachUsed(const UsedT *U, unsigned N,
                                              Fn Func) {
  const unsigned NW = usedWords(N);
  for (unsigned W = 0; W != NW; ++W) {
    UsedT Bits = U[W];
    while (Bits) {
      Func((W << 5) + llvm::countr_zero(Bits));
      Bits &= Bits - 1;
    }
  }
}
} // namespace

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

/// AllocateBuckets - Allocate and initialize storage for Buckets and Used.
static std::pair<void **, UsedT *> AllocateBuckets(unsigned NumBuckets) {
  size_t BucketsBytes = NumBuckets * sizeof(void *);
  size_t UsedBytes = usedWords(NumBuckets) * sizeof(UsedT);
  void **Buckets = static_cast<void **>(safe_malloc(BucketsBytes + UsedBytes));
  UsedT *Used = reinterpret_cast<UsedT *>(reinterpret_cast<char *>(Buckets) +
                                          BucketsBytes);
  memset(Buckets, 0, BucketsBytes);
  memset(Used, 0, UsedBytes);
  return {Buckets, Used};
}

//===----------------------------------------------------------------------===//
// FoldingSetBase Implementation

FoldingSetBase::FoldingSetBase(unsigned Log2InitSize) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  std::tie(Buckets, Used) = AllocateBuckets(NumBuckets);
}

FoldingSetBase::FoldingSetBase(FoldingSetBase &&Arg)
    : Buckets(Arg.Buckets), Used(Arg.Used), NumBuckets(Arg.NumBuckets),
      NumNodes(Arg.NumNodes) {
  Arg.incrementEpoch();
  Arg.Buckets = nullptr;
  Arg.Used = nullptr;
  Arg.NumBuckets = 0;
  Arg.NumNodes = 0;
}

FoldingSetBase &FoldingSetBase::operator=(FoldingSetBase &&RHS) {
  if (this == &RHS)
    return *this;

  incrementEpoch();
  RHS.incrementEpoch();

  if (NumBuckets)
    free(Buckets);

  Buckets = RHS.Buckets;
  Used = RHS.Used;
  NumBuckets = RHS.NumBuckets;
  NumNodes = RHS.NumNodes;
  RHS.Buckets = nullptr;
  RHS.Used = nullptr;
  RHS.NumBuckets = 0;
  RHS.NumNodes = 0;
  return *this;
}

FoldingSetBase::~FoldingSetBase() {
  if (NumBuckets)
    free(Buckets);
}

void FoldingSetBase::clear() {
  incrementEpoch();
  if (NumBuckets == 0)
    return;
  memset(Buckets, 0, NumBuckets * sizeof(void *));
  memset(Used, 0, usedWords(NumBuckets) * sizeof(UsedT));
  NumNodes = 0;
}

void FoldingSetBase::insertImpl(void *N, uint32_t Hash) {
  incrementEpoch();
  const unsigned Mask = NumBuckets - 1;
  unsigned BucketNo = Hash & Mask;
  while (used(Used, BucketNo))
    BucketNo = (BucketNo + 1) & Mask;

  Buckets[BucketNo] = N;
  setUsed(Used, BucketNo);
  static_cast<Node *>(N)->setFoldingSetHash(Hash);
  ++NumNodes;
}

void FoldingSetBase::GrowBucketCount(unsigned NewBucketCount,
                                     const FoldingSetInfo &Info) {
  assert((NewBucketCount > NumBuckets) &&
         "Can't shrink a folding set with GrowBucketCount");
  assert(isPowerOf2_32(NewBucketCount) && "Bad bucket count!");

  FoldingSetBase Tmp(llvm::Log2_32(NewBucketCount));
  forEachUsed(Used, NumBuckets, [&](unsigned I) {
    Node *N = static_cast<Node *>(Buckets[I]);
    Tmp.insertImpl(N, N->getFoldingSetHash());
  });

  *this = std::move(Tmp);
}

void FoldingSetBase::reserve(unsigned EltCount, const FoldingSetInfo &Info) {
  if (EltCount <= capacity())
    return;
  unsigned RequiredBuckets = (EltCount * 4 + 2) / 3;
  GrowBucketCount(llvm::bit_ceil(RequiredBuckets), Info);
}

FoldingSetBase::Node *FoldingSetBase::FindNodeOrInsertPos(
    const FoldingSetNodeID &ID, void *&InsertPos, const FoldingSetInfo &Info) {
  uint32_t IDHash = sanitizeHash(ID.ComputeHash());
  const unsigned Mask = NumBuckets - 1;
  unsigned BucketNo = IDHash & Mask;
  FoldingSetNodeID TempID;

  while (true) {
    if (LLVM_LIKELY(!used(Used, BucketNo))) {
      InsertPos = encodeHash(IDHash);
      return nullptr;
    }

    Node *Candidate = static_cast<Node *>(Buckets[BucketNo]);
    if (LLVM_LIKELY(Candidate->getFoldingSetHash() == IDHash)) {
      if (LLVM_LIKELY(Info.NodeEquals(this, Candidate, ID, IDHash, TempID))) {
        InsertPos = nullptr;
        return Candidate;
      }
      TempID.clear();
    }

    BucketNo = (BucketNo + 1) & Mask;
  }
}

void FoldingSetBase::InsertNode(Node *N, void *InsertPos,
                                const FoldingSetInfo &Info) {
  if (NumNodes + 1 > capacity())
    GrowBucketCount(NumBuckets * 2, Info);

  assert(InsertPos && "Invalid InsertPos!");
  insertImpl(N, decodeHash(InsertPos));
}

bool FoldingSetBase::RemoveNode(Node *N) {
  uint32_t Hash = N->getFoldingSetHash();
  const unsigned Mask = NumBuckets - 1;
  unsigned BucketNo = Hash & Mask;

  while (true) {
    if (!used(Used, BucketNo))
      return false;
    if (Buckets[BucketNo] == N)
      break;
    BucketNo = (BucketNo + 1) & Mask;
  }

  incrementEpoch();
  --NumNodes;
  unsigned I = BucketNo;
  unsigned J = I;
  while (true) {
    J = (J + 1) & Mask;
    if (!used(Used, J))
      break;

    Node *NJ = static_cast<Node *>(Buckets[J]);
    auto Ideal = NJ->getFoldingSetHash();

    // If the hole (I) lies on the linear-probe chain from the home bucket
    // (Ideal) to J, shift J into the hole and make J the new hole.
    if (((I - Ideal) & Mask) < ((J - Ideal) & Mask)) {
      Buckets[I] = NJ;
      I = J;
    }
  }

  unsetUsed(Used, I);
  Buckets[I] = nullptr;
  return true;
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
  while (this->Index < Set->NumBuckets && !used(Set->Used, this->Index))
    ++this->Index;
}

void FoldingSetIteratorImpl::advance() {
  assert(isHandleInSync() && "invalid iterator access!");
  ++Index;
  while (Index < Set->NumBuckets && !used(Set->Used, Index))
    ++Index;
}
