//===- llvm/ADT/SmallPtrSet.cpp - 'Normally small' pointer set ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallPtrSet class.  See SmallPtrSet.h for an
// overview of the algorithm.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <cstdlib>

using namespace llvm;

void SmallPtrSetImpl::shrink_and_clear() {
  assert(!isSmall() && "Can't shrink a small set!");
  free(CurArray);

  // Reduce the number of buckets.
  CurArraySize = NumElements > 16 ? 1 << (Log2_32_Ceil(NumElements) + 1) : 32;
  NumElements = NumTombstones = 0;

  // Install the new array.  Clear all the buckets to empty.
  CurArray = (const void**)malloc(sizeof(void*) * (CurArraySize+1));
  assert(CurArray && "Failed to allocate memory?");
  memset(CurArray, -1, CurArraySize*sizeof(void*));
  
  // The end pointer, always valid, is set to a valid element to help the
  // iterator.
  CurArray[CurArraySize] = 0;
}

bool SmallPtrSetImpl::insert_imp(const void * Ptr) {
  if (isSmall()) {
    // Check to see if it is already in the set.
    for (const void **APtr = SmallArray, **E = SmallArray+NumElements;
         APtr != E; ++APtr)
      if (*APtr == Ptr)
        return false;
    
    // Nope, there isn't.  If we stay small, just 'pushback' now.
    if (NumElements < CurArraySize-1) {
      SmallArray[NumElements++] = Ptr;
      return true;
    }
    // Otherwise, hit the big set case, which will call grow.
  }
  
  // If more than 3/4 of the array is full, grow.
  if (NumElements*4 >= CurArraySize*3 ||
      CurArraySize-(NumElements+NumTombstones) < CurArraySize/8)
    Grow();
  
  // Okay, we know we have space.  Find a hash bucket.
  const void **Bucket = const_cast<const void**>(FindBucketFor(Ptr));
  if (*Bucket == Ptr) return false; // Already inserted, good.
  
  // Otherwise, insert it!
  if (*Bucket == getTombstoneMarker())
    --NumTombstones;
  *Bucket = Ptr;
  ++NumElements;  // Track density.
  return true;
}

bool SmallPtrSetImpl::erase_imp(const void * Ptr) {
  if (isSmall()) {
    // Check to see if it is in the set.
    for (const void **APtr = SmallArray, **E = SmallArray+NumElements;
         APtr != E; ++APtr)
      if (*APtr == Ptr) {
        // If it is in the set, replace this element.
        *APtr = E[-1];
        E[-1] = getEmptyMarker();
        --NumElements;
        return true;
      }
    
    return false;
  }
  
  // Okay, we know we have space.  Find a hash bucket.
  void **Bucket = const_cast<void**>(FindBucketFor(Ptr));
  if (*Bucket != Ptr) return false;  // Not in the set?

  // Set this as a tombstone.
  *Bucket = getTombstoneMarker();
  --NumElements;
  ++NumTombstones;
  return true;
}

const void * const *SmallPtrSetImpl::FindBucketFor(const void *Ptr) const {
  unsigned Bucket = Hash(Ptr);
  unsigned ArraySize = CurArraySize;
  unsigned ProbeAmt = 1;
  const void *const *Array = CurArray;
  const void *const *Tombstone = 0;
  while (1) {
    // Found Ptr's bucket?
    if (Array[Bucket] == Ptr)
      return Array+Bucket;
    
    // If we found an empty bucket, the pointer doesn't exist in the set.
    // Return a tombstone if we've seen one so far, or the empty bucket if
    // not.
    if (Array[Bucket] == getEmptyMarker())
      return Tombstone ? Tombstone : Array+Bucket;
    
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
void SmallPtrSetImpl::Grow() {
  // Allocate at twice as many buckets, but at least 128.
  unsigned OldSize = CurArraySize;
  unsigned NewSize = OldSize < 64 ? 128 : OldSize*2;
  
  const void **OldBuckets = CurArray;
  bool WasSmall = isSmall();
  
  // Install the new array.  Clear all the buckets to empty.
  CurArray = (const void**)malloc(sizeof(void*) * (NewSize+1));
  assert(CurArray && "Failed to allocate memory?");
  CurArraySize = NewSize;
  memset(CurArray, -1, NewSize*sizeof(void*));
  
  // The end pointer, always valid, is set to a valid element to help the
  // iterator.
  CurArray[NewSize] = 0;
  
  // Copy over all the elements.
  if (WasSmall) {
    // Small sets store their elements in order.
    for (const void **BucketPtr = OldBuckets, **E = OldBuckets+NumElements;
         BucketPtr != E; ++BucketPtr) {
      const void *Elt = *BucketPtr;
      *const_cast<void**>(FindBucketFor(Elt)) = const_cast<void*>(Elt);
    }
  } else {
    // Copy over all valid entries.
    for (const void **BucketPtr = OldBuckets, **E = OldBuckets+OldSize;
         BucketPtr != E; ++BucketPtr) {
      // Copy over the element if it is valid.
      const void *Elt = *BucketPtr;
      if (Elt != getTombstoneMarker() && Elt != getEmptyMarker())
        *const_cast<void**>(FindBucketFor(Elt)) = const_cast<void*>(Elt);
    }
    
    free(OldBuckets);
    NumTombstones = 0;
  }
}

SmallPtrSetImpl::SmallPtrSetImpl(const SmallPtrSetImpl& that) {
  // If we're becoming small, prepare to insert into our stack space
  if (that.isSmall()) {
    CurArray = &SmallArray[0];
  // Otherwise, allocate new heap space (unless we were the same size)
  } else {
    CurArray = (const void**)malloc(sizeof(void*) * (that.CurArraySize+1));
    assert(CurArray && "Failed to allocate memory?");
  }
  
  // Copy over the new array size
  CurArraySize = that.CurArraySize;

  // Copy over the contents from the other set
  memcpy(CurArray, that.CurArray, sizeof(void*)*(CurArraySize+1));
  
  NumElements = that.NumElements;
  NumTombstones = that.NumTombstones;
}

/// CopyFrom - implement operator= from a smallptrset that has the same pointer
/// type, but may have a different small size.
void SmallPtrSetImpl::CopyFrom(const SmallPtrSetImpl &RHS) {
  if (isSmall() && RHS.isSmall())
    assert(CurArraySize == RHS.CurArraySize &&
           "Cannot assign sets with different small sizes");
           
  // If we're becoming small, prepare to insert into our stack space
  if (RHS.isSmall()) {
    if (!isSmall())
      free(CurArray);
    CurArray = &SmallArray[0];
  // Otherwise, allocate new heap space (unless we were the same size)
  } else if (CurArraySize != RHS.CurArraySize) {
    if (isSmall())
      CurArray = (const void**)malloc(sizeof(void*) * (RHS.CurArraySize+1));
    else
      CurArray = (const void**)realloc(CurArray, sizeof(void*)*(RHS.CurArraySize+1));
    assert(CurArray && "Failed to allocate memory?");
  }
  
  // Copy over the new array size
  CurArraySize = RHS.CurArraySize;

  // Copy over the contents from the other set
  memcpy(CurArray, RHS.CurArray, sizeof(void*)*(CurArraySize+1));
  
  NumElements = RHS.NumElements;
  NumTombstones = RHS.NumTombstones;
}

SmallPtrSetImpl::~SmallPtrSetImpl() {
  if (!isSmall())
    free(CurArray);
}
