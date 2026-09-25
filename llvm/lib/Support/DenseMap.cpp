//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemAlloc.h"
#include <cstring>

using namespace llvm;
using namespace llvm::densemap;
using namespace llvm::densemap::detail;

// A nonzero FixedSize turns the bucket copy into a couple of stores.
template <size_t FixedSize, bool InlinePtrHash>
static void rehashLoop(void *DstBuckets, UsedT *DstUsed, unsigned Mask,
                       const void *SrcBuckets, const UsedT *SrcUsed,
                       unsigned SrcNumBuckets, size_t RuntimeSize,
                       BucketHasher Hasher) {
  const size_t BucketSize = FixedSize ? FixedSize : RuntimeSize;
  char *Dst = static_cast<char *>(DstBuckets);
  const char *Src = static_cast<const char *>(SrcBuckets);
  forEachUsed(SrcUsed, SrcNumBuckets, [&](unsigned I) {
    const char *SrcBucket = Src + static_cast<size_t>(I) * BucketSize;
    unsigned Hash;
    if constexpr (InlinePtrHash) {
      void *Key;
      std::memcpy(&Key, SrcBucket, sizeof(Key));
      Hash = DenseMapInfo<void *>::getHashValue(Key);
    } else {
      Hash = Hasher(SrcBucket);
    }
    unsigned BucketNo = Hash & Mask;
    while (used(DstUsed, BucketNo))
      BucketNo = (BucketNo + 1) & Mask;
    std::memcpy(Dst + static_cast<size_t>(BucketNo) * BucketSize, SrcBucket,
                BucketSize);
    setUsed(DstUsed, BucketNo);
  });
}

template <bool InlinePtrHash>
static void rehashBySize(void *Dst, UsedT *DstUsed, unsigned Mask,
                         const void *Src, const UsedT *SrcUsed,
                         unsigned SrcNumBuckets, size_t BucketSize,
                         BucketHasher Hasher) {
  // The bucket sizes of 95% of the grow instantiations in an LLVM build.
  switch (BucketSize) {
#define REHASH_CASE(N)                                                         \
  case N:                                                                      \
    return rehashLoop<N, InlinePtrHash>(Dst, DstUsed, Mask, Src, SrcUsed,      \
                                        SrcNumBuckets, BucketSize, Hasher);
    REHASH_CASE(4)
    REHASH_CASE(8)
    REHASH_CASE(12)
    REHASH_CASE(16)
    REHASH_CASE(24)
    REHASH_CASE(32)
    REHASH_CASE(40)
    REHASH_CASE(48)
#undef REHASH_CASE
  default:
    return rehashLoop<0, InlinePtrHash>(Dst, DstUsed, Mask, Src, SrcUsed,
                                        SrcNumBuckets, BucketSize, Hasher);
  }
}

void densemap::detail::rehashRelocatable(void *Dst, UsedT *DstUsed,
                                         unsigned DstNumBuckets,
                                         const void *Src, const UsedT *SrcUsed,
                                         unsigned SrcNumBuckets,
                                         size_t BucketSize,
                                         BucketHasher Hasher) {
  const unsigned Mask = DstNumBuckets - 1;
  if (!Hasher)
    return rehashBySize<true>(Dst, DstUsed, Mask, Src, SrcUsed, SrcNumBuckets,
                              BucketSize, Hasher);
  return rehashBySize<false>(Dst, DstUsed, Mask, Src, SrcUsed, SrcNumBuckets,
                             BucketSize, Hasher);
}

void *densemap::detail::growRelocatable(void *OldBuckets, const UsedT *OldUsed,
                                        unsigned OldNumBuckets,
                                        unsigned NewNumBuckets,
                                        size_t BucketSize, size_t Align,
                                        BucketHasher Hasher, bool FreeOld) {
  void *Storage = allocate_buffer(allocBytes(BucketSize, NewNumBuckets), Align);
  UsedT *NewUsed = usedFor(Storage, BucketSize, NewNumBuckets);
  clearUsed(NewUsed, NewNumBuckets);
  if (OldNumBuckets) {
    rehashRelocatable(Storage, NewUsed, NewNumBuckets, OldBuckets, OldUsed,
                      OldNumBuckets, BucketSize, Hasher);
    if (FreeOld)
      deallocate_buffer(OldBuckets, allocBytes(BucketSize, OldNumBuckets),
                        Align);
  }
  return Storage;
}
