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
/// Helper functions for FoldingSetBase.

/// GetNextPtr - In order to save space, each bucket is a
/// singly-linked-list. In order to make deletion more efficient, we make
/// the list circular, so we can delete a node without computing its hash.
/// The problem with this is that the start of the hash buckets are not
/// Nodes. If NextInBucketPtr is a bucket pointer, this method returns null:
/// use GetBucketPtr when this happens.
static FoldingSetBase::Node *GetNextPtr(void *NextInBucketPtr) {
  // The low bit is set if this is the pointer back to the bucket.
  if (reinterpret_cast<intptr_t>(NextInBucketPtr) & 1)
    return nullptr;

  return static_cast<FoldingSetBase::Node *>(NextInBucketPtr);
}

/// GetBucketPtr - Provides a casting of a bucket pointer for isNode
/// testing.
static void **GetBucketPtr(void *NextInBucketPtr) {
  intptr_t Ptr = reinterpret_cast<intptr_t>(NextInBucketPtr);
  assert((Ptr & 1) && "Not a bucket pointer");
  return reinterpret_cast<void **>(Ptr & ~intptr_t(1));
}

/// GetBucketFor - Hash the specified node ID and return the hash bucket for
/// the specified ID.
static void **GetBucketFor(unsigned Hash, void **Buckets, unsigned NumBuckets) {
  // NumBuckets is always a power of 2.
  unsigned BucketNum = Hash & (NumBuckets - 1);
  return Buckets + BucketNum;
}

/// AllocateBuckets - Allocate initialized bucket memory.
static void **AllocateBuckets(unsigned NumBuckets) {
  void **Buckets =
      static_cast<void **>(safe_calloc(NumBuckets + 1, sizeof(void *)));
  // Set the very last bucket to be a non-null "pointer".
  Buckets[NumBuckets] = reinterpret_cast<void *>(-1);
  return Buckets;
}

//===----------------------------------------------------------------------===//
// FoldingSetBase Implementation

FoldingSetBase::FoldingSetBase(unsigned Log2InitSize) {
  assert(5 < Log2InitSize && Log2InitSize < 32 &&
         "Initial hash table size out of range");
  NumBuckets = 1 << Log2InitSize;
  Buckets = AllocateBuckets(NumBuckets);
  NumNodes = 0;
}

FoldingSetBase::FoldingSetBase(FoldingSetBase &&Arg)
    : Buckets(Arg.Buckets), NumBuckets(Arg.NumBuckets), NumNodes(Arg.NumNodes) {
  Arg.Buckets = nullptr;
  Arg.NumBuckets = 0;
  Arg.NumNodes = 0;
}

FoldingSetBase &FoldingSetBase::operator=(FoldingSetBase &&RHS) {
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
  // Set all but the last bucket to null pointers.
  memset(Buckets, 0, NumBuckets * sizeof(void *));

  // Set the very last bucket to be a non-null "pointer".
  Buckets[NumBuckets] = reinterpret_cast<void *>(-1);

  // Reset the node count to zero.
  NumNodes = 0;
}

void FoldingSetBase::GrowBucketCount(unsigned NewBucketCount,
                                     const FoldingSetInfo &Info) {
  assert((NewBucketCount > NumBuckets) &&
         "Can't shrink a folding set with GrowBucketCount");
  assert(isPowerOf2_32(NewBucketCount) && "Bad bucket count!");

  FoldingSetBase Tmp(llvm::Log2_32(NewBucketCount));
  FoldingSetNodeID TempID;
  for (unsigned i = 0; i != NumBuckets; ++i) {
    void *Probe = Buckets[i];
    if (!Probe)
      continue;
    while (Node *NodeInBucket = GetNextPtr(Probe)) {
      // Figure out the next link, remove NodeInBucket from the old link.
      Probe = NodeInBucket->getNextInBucket();
      NodeInBucket->SetNextInBucket(nullptr);

      // Insert the node into the new bucket, after recomputing the hash.
      Tmp.InsertNode(
          NodeInBucket,
          GetBucketFor(Info.ComputeNodeHash(this, NodeInBucket, TempID),
                       Tmp.Buckets, Tmp.NumBuckets),
          Info);
      TempID.clear();
    }
  }

  *this = std::move(Tmp);
}

void FoldingSetBase::reserve(unsigned EltCount, const FoldingSetInfo &Info) {
  // This will give us somewhere between EltCount / 2 and
  // EltCount buckets.  This puts us in the load factor
  // range of 1.0 - 2.0.
  if (EltCount <= capacity())
    return;
  GrowBucketCount(llvm::bit_floor(EltCount), Info);
}

FoldingSetBase::Node *FoldingSetBase::FindNodeOrInsertPos(
    const FoldingSetNodeID &ID, void *&InsertPos, const FoldingSetInfo &Info) {
  unsigned IDHash = ID.ComputeHash();
  void **Bucket = GetBucketFor(IDHash, Buckets, NumBuckets);
  void *Probe = *Bucket;

  InsertPos = nullptr;

  FoldingSetNodeID TempID;
  while (Node *NodeInBucket = GetNextPtr(Probe)) {
    if (Info.NodeEquals(this, NodeInBucket, ID, IDHash, TempID))
      return NodeInBucket;
    TempID.clear();

    Probe = NodeInBucket->getNextInBucket();
  }

  // Didn't find the node, return null with the bucket as the InsertPos.
  InsertPos = Bucket;
  return nullptr;
}

void FoldingSetBase::InsertNode(Node *N, void *InsertPos,
                                const FoldingSetInfo &Info) {
  assert(!N->getNextInBucket());
  // Do we need to grow the hashtable?
  if (NumNodes + 1 > capacity()) {
    GrowBucketCount(NumBuckets * 2, Info);
    FoldingSetNodeID TempID;
    InsertPos = GetBucketFor(Info.ComputeNodeHash(this, N, TempID), Buckets,
                             NumBuckets);
  }

  ++NumNodes;

  /// The insert position is actually a bucket pointer.
  void **Bucket = static_cast<void **>(InsertPos);

  void *Next = *Bucket;

  // If this is the first insertion into this bucket, its next pointer will be
  // null.  Pretend as if it pointed to itself, setting the low bit to indicate
  // that it is a pointer to the bucket.
  if (!Next)
    Next = reinterpret_cast<void *>(reinterpret_cast<intptr_t>(Bucket) | 1);

  // Set the node's next pointer, and make the bucket point to the node.
  N->SetNextInBucket(Next);
  *Bucket = N;
}

bool FoldingSetBase::RemoveNode(Node *N) {
  // Because each bucket is a circular list, we don't need to compute N's hash
  // to remove it.
  void *Ptr = N->getNextInBucket();
  if (!Ptr)
    return false; // Not in folding set.

  --NumNodes;
  N->SetNextInBucket(nullptr);

  // Remember what N originally pointed to, either a bucket or another node.
  void *NodeNextPtr = Ptr;

  // Chase around the list until we find the node (or bucket) which points to N.
  while (true) {
    if (Node *NodeInBucket = GetNextPtr(Ptr)) {
      // Advance pointer.
      Ptr = NodeInBucket->getNextInBucket();

      // We found a node that points to N, change it to point to N's next node,
      // removing N from the list.
      if (Ptr == N) {
        NodeInBucket->SetNextInBucket(NodeNextPtr);
        return true;
      }
    } else {
      void **Bucket = GetBucketPtr(Ptr);
      Ptr = *Bucket;

      // If we found that the bucket points to N, update the bucket to point to
      // whatever is next.
      if (Ptr == N) {
        *Bucket = NodeNextPtr;
        return true;
      }
    }
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

FoldingSetIteratorImpl::FoldingSetIteratorImpl(void **Bucket) {
  // Skip to the first non-null non-self-cycle bucket.
  while (*Bucket != reinterpret_cast<void *>(-1) &&
         (!*Bucket || !GetNextPtr(*Bucket)))
    ++Bucket;

  NodePtr = static_cast<FoldingSetNode *>(*Bucket);
}

void FoldingSetIteratorImpl::advance() {
  // If there is another link within this bucket, go to it.
  void *Probe = NodePtr->getNextInBucket();

  if (FoldingSetNode *NextNodeInBucket = GetNextPtr(Probe))
    NodePtr = NextNodeInBucket;
  else {
    // Otherwise, this is the last link in this bucket.
    void **Bucket = GetBucketPtr(Probe);

    // Skip to the next non-null non-self-cycle bucket.
    do {
      ++Bucket;
    } while (*Bucket != reinterpret_cast<void *>(-1) &&
             (!*Bucket || !GetNextPtr(*Bucket)));

    NodePtr = static_cast<FoldingSetNode *>(*Bucket);
  }
}
