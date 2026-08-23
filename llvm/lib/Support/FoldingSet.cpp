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
#include <algorithm>
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

/// Encode a hash as the token FindNodeOrInsertPos hands back. Never null, and
/// unlike a bucket address it survives intervening insertions.
static void *encodeHash(uint32_t Hash) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(Hash));
}

static uint32_t decodeHash(void *InsertPos) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(InsertPos));
}

FoldingSetBase::FoldingSetBase(unsigned Log2InitSize) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  Buckets = static_cast<void **>(
      safe_calloc(NumBuckets, sizeof(void *)));
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

void FoldingSetBase::placeNode(Node *N, uint32_t Hash) {
  unsigned Mask = NumBuckets - 1;
  unsigned I = Hash & Mask;
  while (Buckets[I]) {
    // A second copy of N hashes here too, so the probe always passes it.
    assert(Buckets[I] != N && "Node already in the folding set");
    I = (I + 1) & Mask;
  }
  Buckets[I] = N;
  ++NumNodes;
}

void FoldingSetBase::grow(unsigned MinNumBuckets) {
  // The floor is the smallest size the constructor accepts.
  unsigned NewBucketCount = std::max(64u, llvm::bit_ceil(MinNumBuckets));
  assert(NewBucketCount > NumBuckets && "Can't shrink a folding set");

  FoldingSetBase Tmp(llvm::Log2_32(NewBucketCount));
  for (unsigned I = 0; I != NumBuckets; ++I)
    if (void *N = Buckets[I])
      Tmp.placeNode(static_cast<Node *>(N),
                    static_cast<Node *>(N)->getFoldingSetHash());

  *this = std::move(Tmp);
}

void FoldingSetBase::reserve(unsigned N) {
  if (N * 4 <= NumBuckets * 3)
    return;
  // N + (N + 2) / 3 is ceil(4N/3).
  grow(N + (N + 2) / 3);
}

LLVM_ATTRIBUTE_NOINLINE bool
FoldingSetBase::nodeEquals(const FoldingSetInfo &Info,
                           const FoldingSetBase *Self, Node *N,
                           const FoldingSetNodeID &ID, unsigned IDHash) {
  FoldingSetNodeID TempID;
  return Info.NodeEquals(Self, N, ID, IDHash, TempID);
}

FoldingSetBase::Node *FoldingSetBase::FindNodeOrInsertPos(
    const FoldingSetNodeID &ID, void *&InsertPos, const FoldingSetInfo &Info) {
  unsigned IDHash = ID.ComputeHash();
  unsigned Mask = NumBuckets - 1;
  for (unsigned I = IDHash & Mask; Buckets[I]; I = (I + 1) & Mask) {
    // Reject on the hash first, so a probe step touches no node.
    Node *N = static_cast<Node *>(Buckets[I]);
    if (N->getFoldingSetHash() != IDHash)
      continue;
    if (nodeEquals(Info, this, N, ID, IDHash)) {
      InsertPos = nullptr;
      return N;
    }
  }

  InsertPos = encodeHash(IDHash);
  return nullptr;
}

void FoldingSetBase::InsertNode(Node *N, void *InsertPos) {
  assert(InsertPos && "Invalid InsertPos!");
  incrementEpoch();
  if (LLVM_UNLIKELY((NumNodes + 1) * 4 > NumBuckets * 3))
    grow(NumBuckets * 2);
  uint32_t Hash = decodeHash(InsertPos);
  placeNode(N, Hash);
  N->setFoldingSetHash(Hash);
}

bool FoldingSetBase::RemoveNode(Node *N) {
  uint32_t Hash = N->getFoldingSetHash();
  if (Hash == FoldingSetNodeIDRef::NotAHash)
    return false; // Never inserted.

  unsigned Mask = NumBuckets - 1;
  unsigned I = Hash & Mask;
  while (Buckets[I] != N) {
    if (!Buckets[I])
      return false; // Not in folding set.
    I = (I + 1) & Mask;
  }

  incrementEpoch();

  // Knuth TAOCP 6.4 Algorithm R: walk forward sliding each following entry
  // whose probe path crosses the hole.
  for (unsigned J = (I + 1) & Mask; Buckets[J]; J = (J + 1) & Mask) {
    unsigned Ideal = static_cast<Node *>(Buckets[J])->getFoldingSetHash();
    if (((I - Ideal) & Mask) < ((J - Ideal) & Mask)) {
      Buckets[I] = Buckets[J];
      I = J;
    }
  }
  Buckets[I] = nullptr;
  N->setFoldingSetHash(FoldingSetNodeIDRef::NotAHash);
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
  InsertNode(N, IP);
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
