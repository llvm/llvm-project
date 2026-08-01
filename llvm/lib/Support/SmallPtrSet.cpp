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
  NumEntries = 0;

  // Install the new array.  Clear all the buckets to empty.
  CurArray = (const void**)safe_malloc(sizeof(void*) * CurArraySize);

  memset(CurArray, -1, CurArraySize*sizeof(void*));
}

std::pair<const void *const *, bool>
SmallPtrSetImplBase::insert_imp_big(const void *Ptr) {
  if (LLVM_UNLIKELY(size() * 3 >= CurArraySize * 2)) {
    // If more than 2/3 of the array is full, grow.
    Grow(CurArraySize < 64 ? 128 : CurArraySize * 2);
  }

  // Find the first empty bucket or Ptr itself on the probe chain.
  unsigned Mask = CurArraySize - 1;
  unsigned I = DenseMapInfo<void *>::getHashValue(Ptr) & Mask;
  const void **Array = CurArray;
  while (Array[I] != getEmptyMarker()) {
    if (Array[I] == Ptr)
      return {Array + I, false};
    I = (I + 1) & Mask;
  }

  // Insert into the empty bucket.
  ++NumEntries;
  Array[I] = Ptr;
  incrementEpoch();
  return {Array + I, true};
}

const void *const *SmallPtrSetImplBase::doFind(const void *Ptr) const {
  unsigned Mask = CurArraySize - 1;
  unsigned BucketNo = DenseMapInfo<void *>::getHashValue(Ptr) & Mask;
  while (true) {
    const void *const *Bucket = CurArray + BucketNo;
    if (LLVM_LIKELY(*Bucket == Ptr))
      return Bucket;
    if (LLVM_LIKELY(*Bucket == getEmptyMarker()))
      return nullptr;
    BucketNo = (BucketNo + 1) & Mask;
  }
}

void SmallPtrSetImplBase::eraseFromBucket(const void **Bucket) {
  // Knuth TAOCP 6.4 Algorithm R: walk forward sliding each following entry
  // whose probe path crosses the hole.
  unsigned Mask = CurArraySize - 1;
  unsigned I = Bucket - CurArray;
  unsigned J = I;
  const void *Empty = getEmptyMarker();
  while ((J = (J + 1) & Mask), CurArray[J] != Empty) {
    auto Ideal = DenseMapInfo<void *>::getHashValue(CurArray[J]);
    if (((I - Ideal) & Mask) < ((J - Ideal) & Mask)) {
      CurArray[I] = CurArray[J];
      I = J;
    }
  }
  CurArray[I] = Empty;
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
  unsigned Mask = CurArraySize - 1;
  for (const void *Ptr : OldBuckets) {
    if (Ptr == getEmptyMarker())
      continue;
    // Find the first empty bucket on this key's probe chain; there is no equal
    // key, so nothing to compare against.
    unsigned I = DenseMapInfo<void *>::getHashValue(Ptr) & Mask;
    while (NewBuckets[I] != getEmptyMarker())
      I = (I + 1) & Mask;
    NewBuckets[I] = Ptr;
  }

  if (!WasSmall)
    free(OldBuckets.begin());
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
  IsSmall = RHS.IsSmall;

  // Make the RHS small and empty.
  RHS.CurArraySize = SmallSize;
  RHS.NumEntries = 0;
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
    return;
  }

  // FIXME: From here on we assume that both sets have the same small size.

  // Both a small, just swap the small elements.
  if (this->isSmall() && RHS.isSmall()) {
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
    return;
  }

  // If only one side is small, copy the small elements into the large side and
  // move the pointer from the large side to the small side.
  SmallPtrSetImplBase &SmallSide = this->isSmall() ? *this : RHS;
  SmallPtrSetImplBase &LargeSide = this->isSmall() ? RHS : *this;
  const void **LargeSideInlineStorage =
      this->isSmall() ? RHSSmallStorage : SmallStorage;
  llvm::copy(SmallSide.small_buckets(), LargeSideInlineStorage);
  std::swap(LargeSide.CurArraySize, SmallSide.CurArraySize);
  std::swap(LargeSide.NumEntries, SmallSide.NumEntries);
  SmallSide.CurArray = LargeSide.CurArray;
  SmallSide.IsSmall = false;
  LargeSide.CurArray = LargeSideInlineStorage;
  LargeSide.IsSmall = true;
}
