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
// FoldingSetBase Implementation
//
// FoldingSet is an open-addressed hash set using linear probing. One
// allocation holds the bucket array followed by a parallel array of the
// buckets' 32-bit hashes:
//   [ Buckets (NumBuckets * sizeof(void *)) ][ Hashes (NumBuckets * 4) ]
// A null bucket marks an empty slot, and the hash array rejects mismatches
// before the profile compare, so walking a probe chain touches no nodes.
// Nodes cache their own hash as well, which is what lets RemoveNode() find a
// node without re-running Profile().
//
// Removal uses Knuth TAOCP vol. 3 6.4 Algorithm R, as StringMap, DenseMap and
// SmallPtrSet do: it closes the hole rather than leaving a tombstone, so the
// table stays sized to the live node count under insert/erase churn.

/// Encode a hash into the non-null token FindNodeOrInsertPos hands back.
/// Unlike a bucket address the token survives intervening insertions.
static void *encodeHash(uint32_t Hash) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(Hash) + 1);
}

static uint32_t decodeHash(void *InsertPos) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(InsertPos) - 1);
}

/// AllocateBuckets - Allocate zeroed bucket and hash arrays.
static void **AllocateBuckets(unsigned NumBuckets) {
  return static_cast<void **>(
      safe_calloc(NumBuckets, sizeof(void *) + sizeof(uint32_t)));
}

FoldingSetBase::FoldingSetBase(unsigned Log2InitSize) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  Buckets = AllocateBuckets(NumBuckets);
}

FoldingSetBase::FoldingSetBase(FoldingSetBase &&Arg)
    : Buckets(Arg.Buckets), NumBuckets(Arg.NumBuckets), NumNodes(Arg.NumNodes) {
  Arg.incrementEpoch();
  Arg.Buckets = nullptr;
  Arg.NumBuckets = 0;
  Arg.NumNodes = 0;
}

FoldingSetBase &FoldingSetBase::operator=(FoldingSetBase &&RHS) {
  incrementEpoch();
  RHS.incrementEpoch();
  free(Buckets); // This may be null if the set is in a moved-from state.
  Buckets = RHS.Buckets;
  NumBuckets = RHS.NumBuckets;
  NumNodes = RHS.NumNodes;
  RHS.Buckets = nullptr;
  RHS.NumBuckets = 0;
  RHS.NumNodes = 0;
  return *this;
}

FoldingSetBase::~FoldingSetBase() { free(Buckets); }

void FoldingSetBase::clear() {
  incrementEpoch();
  // Stale hashes are unreachable, so only the occupancy needs resetting.
  if (NumBuckets)
    memset(Buckets, 0, NumBuckets * sizeof(void *));
  NumNodes = 0;
}

void FoldingSetBase::insertImpl(Node *N, uint32_t Hash) {
  incrementEpoch();
  unsigned Mask = NumBuckets - 1;
  unsigned I = Hash & Mask;
  while (Buckets[I])
    I = (I + 1) & Mask;
  Buckets[I] = N;
  getHashes()[I] = Hash;
  N->setFoldingSetHash(Hash);
  ++NumNodes;
}

void FoldingSetBase::GrowBucketCount(unsigned NewBucketCount,
                                     const FoldingSetInfo &Info) {
  assert((NewBucketCount > NumBuckets) &&
         "Can't shrink a folding set with GrowBucketCount");
  assert(isPowerOf2_32(NewBucketCount) && "Bad bucket count!");

  FoldingSetBase Tmp(llvm::Log2_32(NewBucketCount));
  const uint32_t *Hashes = getHashes();
  for (unsigned I = 0; I != NumBuckets; ++I)
    if (void *N = Buckets[I])
      Tmp.insertImpl(static_cast<Node *>(N), Hashes[I]);

  *this = std::move(Tmp);
}

void FoldingSetBase::reserve(unsigned EltCount, const FoldingSetInfo &Info) {
  if (EltCount <= capacity())
    return;
  uint64_t Required = divideCeil(uint64_t(EltCount) * 4, 3);
  GrowBucketCount(
      static_cast<unsigned>(llvm::bit_ceil(std::max<uint64_t>(Required, 64))),
      Info);
}

FoldingSetBase::Node *FoldingSetBase::FindNodeOrInsertPos(
    const FoldingSetNodeID &ID, void *&InsertPos, const FoldingSetInfo &Info) {
  unsigned IDHash = ID.ComputeHash();
  const uint32_t *Hashes = getHashes();
  unsigned Mask = NumBuckets - 1;

  FoldingSetNodeID TempID;
  for (unsigned I = IDHash & Mask; Buckets[I]; I = (I + 1) & Mask) {
    // Reject on the hash first: the common case only reads the bucket and hash
    // arrays, which matters for cache locality.
    if (Hashes[I] != IDHash)
      continue;
    Node *N = static_cast<Node *>(Buckets[I]);
    if (Info.NodeEquals(this, N, ID, IDHash, TempID)) {
      InsertPos = nullptr;
      return N;
    }
    TempID.clear();
  }

  // Didn't find the node, hand back the hash so that InsertNode can place it.
  InsertPos = encodeHash(IDHash);
  return nullptr;
}

void FoldingSetBase::InsertNode(Node *N, void *InsertPos,
                                const FoldingSetInfo &Info) {
  assert(InsertPos && "Invalid InsertPos!");
  if (NumNodes + 1 > capacity())
    GrowBucketCount(NumBuckets * 2, Info);
  insertImpl(N, decodeHash(InsertPos));
}

bool FoldingSetBase::RemoveNode(Node *N) {
  uint32_t *Hashes = getHashes();
  unsigned Mask = NumBuckets - 1;

  unsigned I = N->getFoldingSetHash() & Mask;
  while (Buckets[I] != N) {
    if (!Buckets[I])
      return false; // Not in folding set.
    I = (I + 1) & Mask;
  }

  incrementEpoch();

  // Knuth TAOCP 6.4 Algorithm R: walk forward sliding each following entry
  // whose probe path crosses the hole.
  for (unsigned J = (I + 1) & Mask; Buckets[J]; J = (J + 1) & Mask) {
    unsigned Ideal = Hashes[J];
    if (((I - Ideal) & Mask) < ((J - Ideal) & Mask)) {
      Buckets[I] = Buckets[J];
      Hashes[I] = Hashes[J];
      I = J;
    }
  }
  Buckets[I] = nullptr;
  --NumNodes;
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

FoldingSetIteratorImpl::FoldingSetIteratorImpl(const FoldingSetBase *Set,
                                               unsigned Index)
    : DebugEpochBase::HandleBase(Set), Set(Set), Index(Index) {
  while (this->Index < Set->NumBuckets && !Set->Buckets[this->Index])
    ++this->Index;
}

void FoldingSetIteratorImpl::advance() {
  assert(isHandleInSync() && "invalid iterator access!");
  do
    ++Index;
  while (Index < Set->NumBuckets && !Set->Buckets[Index]);
}
