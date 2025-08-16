//===- llvm/ADT/SmallPtrSet.cpp - 'Normally small' pointer set ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallPtrSet class.  See SmallPtrSet.h for an
// overview of the algorithm.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>

using namespace llvm;

void SmallPtrSetImplBase::shrink_and_clear() {
  assert(!isSmall() && "Can't shrink a small set!");
  free(CurArray);

  // Reduce the number of buckets.
  unsigned Size = size();
  CurArraySize = Size > 16 ? 1 << (Log2_32_Ceil(Size) + 1) : 32;
  NumEntries = NumTombstones = 0;

  // Install the new array.  Clear all the buckets to empty.
  CurArray = (const void**)safe_malloc(sizeof(void*) * CurArraySize);

  memset(CurArray, -1, CurArraySize*sizeof(void*));
}

std::pair<const void *const *, bool>
SmallPtrSetImplBase::insert_imp_big(const void *Ptr) {
  if (LLVM_UNLIKELY(size() * 4 >= CurArraySize * 3)) {
    // If more than 3/4 of the array is full, grow.
    Grow(CurArraySize < 64 ? 128 : CurArraySize * 2);
  } else if (LLVM_UNLIKELY(CurArraySize - NumEntries - NumTombstones <
                           CurArraySize / 8)) {
    // If fewer of 1/8 of the array is empty (meaning that many are filled with
    // tombstones), rehash.
    Grow(CurArraySize);
  }

  // Okay, we know we have space.  Find a hash bucket.
  const void **Bucket = const_cast<const void**>(FindBucketFor(Ptr));
  if (*Bucket == Ptr)
    return std::make_pair(Bucket, false); // Already inserted, good.

  // Otherwise, insert it!
  if (*Bucket == getTombstoneMarker())
    --NumTombstones;
  ++NumEntries;
  *Bucket = Ptr;
  incrementEpoch();
  return std::make_pair(Bucket, true);
}

const void *const *SmallPtrSetImplBase::doFind(const void *Ptr) const {
  unsigned BucketNo =
      DenseMapInfo<void *>::getHashValue(Ptr) & (CurArraySize - 1);
  unsigned ProbeAmt = 1;
  while (true) {
    const void *const *Bucket = CurArray + BucketNo;
    if (LLVM_LIKELY(*Bucket == Ptr))
      return Bucket;
    if (LLVM_LIKELY(*Bucket == getEmptyMarker()))
      return nullptr;

    // Otherwise, it's a hash collision or a tombstone, continue quadratic
    // probing.
    BucketNo += ProbeAmt++;
    BucketNo &= CurArraySize - 1;
  }
}

const void *const *SmallPtrSetImplBase::FindBucketFor(const void *Ptr) const {
  unsigned Bucket = DenseMapInfo<void *>::getHashValue(Ptr) & (CurArraySize-1);
  unsigned ArraySize = CurArraySize;
  unsigned ProbeAmt = 1;
  const void *const *Array = CurArray;
  const void *const *Tombstone = nullptr;
  while (true) {
    // If we found an empty bucket, the pointer doesn't exist in the set.
    // Return a tombstone if we've seen one so far, or the empty bucket if
    // not.
    if (LLVM_LIKELY(Array[Bucket] == getEmptyMarker()))
      return Tombstone ? Tombstone : Array+Bucket;

    // Found Ptr's bucket?
    if (LLVM_LIKELY(Array[Bucket] == Ptr))
      return Array+Bucket;

    // If this is a tombstone, remember it.  If Ptr ends up not in the set, we
    // prefer to return it than something that would require more probing.
    if (Array[Bucket] == getTombstoneMarker() && !Tombstone)
      Tombstone = Array+Bucket;  // Remember the first tombstone found.

    // It's a hash collision or a tombstone. Reprobe.
    Bucket = (Bucket + ProbeAmt++) & (ArraySize-1);
  }
}

/// Grow - Allocate a larger backing store for the buckets and move it over.
///
void SmallPtrSetImplBase::Grow(unsigned NewSize) {
  auto OldBuckets = buckets();
  bool WasSmall = isSmall();

  // Install the new array.  Clear all the buckets to empty.
  const void **NewBuckets = (const void**) safe_malloc(sizeof(void*) * NewSize);

  // Reset member only if memory was allocated successfully
  CurArray = NewBuckets;
  CurArraySize = NewSize;
  memset(CurArray, -1, NewSize*sizeof(void*));

  // Copy over all valid entries.
  for (const void *&Bucket : OldBuckets) {
    // Copy over the element if it is valid.
    if (Bucket != getTombstoneMarker() && Bucket != getEmptyMarker())
      *const_cast<void **>(FindBucketFor(Bucket)) = const_cast<void *>(Bucket);
  }

  if (!WasSmall)
    free(OldBuckets.begin());
  NumTombstones = 0;
  IsSmall = false;
}

SmallPtrSetImplBase::SmallPtrSetImplBase(const void **SmallStorage,
                                         const SmallPtrSetImplBase &that) {
  IsSmall = that.isSmall();
  if (IsSmall) {
    // If we're becoming small, prepare to insert into our stack space
    CurArray = SmallStorage;
  } else {
    // Otherwise, allocate new heap space (unless we were the same size)
    CurArray = (const void**)safe_malloc(sizeof(void*) * that.CurArraySize);
  }

  // Copy over the that array.
  copyHelper(that);
}

SmallPtrSetImplBase::SmallPtrSetImplBase(const void **SmallStorage,
                                         unsigned SmallSize,
                                         const void **RHSSmallStorage,
                                         SmallPtrSetImplBase &&that) {
  moveHelper(SmallStorage, SmallSize, RHSSmallStorage, std::move(that));
}

void SmallPtrSetImplBase::copyFrom(const void **SmallStorage,
                                   const SmallPtrSetImplBase &RHS) {
  assert(&RHS != this && "Self-copy should be handled by the caller.");

  if (isSmall() && RHS.isSmall())
    assert(CurArraySize == RHS.CurArraySize &&
           "Cannot assign sets with different small sizes");

  // If we're becoming small, prepare to insert into our stack space
  if (RHS.isSmall()) {
    if (!isSmall())
      free(CurArray);
    CurArray = SmallStorage;
    IsSmall = true;
    // Otherwise, allocate new heap space (unless we were the same size)
  } else if (CurArraySize != RHS.CurArraySize) {
    if (isSmall())
      CurArray = (const void**)safe_malloc(sizeof(void*) * RHS.CurArraySize);
    else {
      const void **T = (const void**)safe_realloc(CurArray,
                                             sizeof(void*) * RHS.CurArraySize);
      CurArray = T;
    }
    IsSmall = false;
  }

  copyHelper(RHS);
}

void SmallPtrSetImplBase::copyHelper(const SmallPtrSetImplBase &RHS) {
  // Copy over the new array size
  CurArraySize = RHS.CurArraySize;

  // Copy over the contents from the other set
  llvm::copy(RHS.buckets(), CurArray);

  NumEntries = RHS.NumEntries;
  NumTombstones = RHS.NumTombstones;
}

void SmallPtrSetImplBase::moveFrom(const void **SmallStorage,
                                   unsigned SmallSize,
                                   const void **RHSSmallStorage,
                                   SmallPtrSetImplBase &&RHS) {
  if (!isSmall())
    free(CurArray);
  moveHelper(SmallStorage, SmallSize, RHSSmallStorage, std::move(RHS));
}

void SmallPtrSetImplBase::moveHelper(const void **SmallStorage,
                                     unsigned SmallSize,
                                     const void **RHSSmallStorage,
                                     SmallPtrSetImplBase &&RHS) {
  assert(&RHS != this && "Self-move should be handled by the caller.");

  if (RHS.isSmall()) {
    // Copy a small RHS rather than moving.
    CurArray = SmallStorage;
    llvm::copy(RHS.small_buckets(), CurArray);
  } else {
    CurArray = RHS.CurArray;
    RHS.CurArray = RHSSmallStorage;
  }

  // Copy the rest of the trivial members.
  CurArraySize = RHS.CurArraySize;
  NumEntries = RHS.NumEntries;
  NumTombstones = RHS.NumTombstones;
  IsSmall = RHS.IsSmall;

  // Make the RHS small and empty.
  RHS.CurArraySize = SmallSize;
  RHS.NumEntries = 0;
  RHS.NumTombstones = 0;
  RHS.IsSmall = true;
}

void SmallPtrSetImplBase::swap(const void **SmallStorage,
                               const void **RHSSmallStorage,
                               SmallPtrSetImplBase &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither set is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->CurArray, RHS.CurArray);
    std::swap(this->CurArraySize, RHS.CurArraySize);
    std::swap(this->NumEntries, RHS.NumEntries);
    std::swap(this->NumTombstones, RHS.NumTombstones);
    return;
  }

  // FIXME: From here on we assume that both sets have the same small size.

  // If only RHS is small, copy the small elements into LHS and move the pointer
  // from LHS to RHS.
  if (!this->isSmall() && RHS.isSmall()) {
    llvm::copy(RHS.small_buckets(), SmallStorage);
    std::swap(RHS.CurArraySize, this->CurArraySize);
    std::swap(this->NumEntries, RHS.NumEntries);
    std::swap(this->NumTombstones, RHS.NumTombstones);
    RHS.CurArray = this->CurArray;
    RHS.IsSmall = false;
    this->CurArray = SmallStorage;
    this->IsSmall = true;
    return;
  }

  // If only LHS is small, copy the small elements into RHS and move the pointer
  // from RHS to LHS.
  if (this->isSmall() && !RHS.isSmall()) {
    llvm::copy(this->small_buckets(), RHSSmallStorage);
    std::swap(RHS.CurArraySize, this->CurArraySize);
    std::swap(RHS.NumEntries, this->NumEntries);
    std::swap(RHS.NumTombstones, this->NumTombstones);
    this->CurArray = RHS.CurArray;
    this->IsSmall = false;
    RHS.CurArray = RHSSmallStorage;
    RHS.IsSmall = true;
    return;
  }

  // Both a small, just swap the small elements.
  assert(this->isSmall() && RHS.isSmall());
  unsigned MinEntries = std::min(this->NumEntries, RHS.NumEntries);
  std::swap_ranges(this->CurArray, this->CurArray + MinEntries, RHS.CurArray);
  if (this->NumEntries > MinEntries) {
    std::copy(this->CurArray + MinEntries, this->CurArray + this->NumEntries,
              RHS.CurArray + MinEntries);
  } else {
    std::copy(RHS.CurArray + MinEntries, RHS.CurArray + RHS.NumEntries,
              this->CurArray + MinEntries);
  }
  assert(this->CurArraySize == RHS.CurArraySize);
  std::swap(this->NumEntries, RHS.NumEntries);
  std::swap(this->NumTombstones, RHS.NumTombstones);
}
