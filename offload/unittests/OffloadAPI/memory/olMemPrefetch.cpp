//===------- Offload API tests - olMemPrefetch ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemPrefetchTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemPrefetchTest);

TEST_P(olMemPrefetchTest, SuccessHostToDevice) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  std::memset(Alloc, 0x42, Size);

  const void *Mems[] = {Alloc};
  const size_t Sizes[] = {Size};
  ASSERT_SUCCESS(olMemPrefetch(Queue, 1, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(static_cast<uint8_t *>(Alloc)[I], 0x42);

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, SuccessDeviceToHost) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  std::memset(Alloc, 0x21, Size);

  const void *Mems[] = {Alloc};
  const size_t Sizes[] = {Size};
  // Migrate to the device first, then bring it back.
  ASSERT_SUCCESS(olMemPrefetch(Queue, 1, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olMemPrefetch(Queue, 1, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_DEVICE_TO_HOST));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(static_cast<uint8_t *>(Alloc)[I], 0x21);

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, SuccessMultiple) {
  constexpr size_t NumAllocs = 3;
  constexpr size_t Size = 1024;
  void *Allocs[NumAllocs];
  const void *Mems[NumAllocs];
  size_t Sizes[NumAllocs];
  for (size_t I = 0; I < NumAllocs; I++) {
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Allocs[I]));
    std::memset(Allocs[I], static_cast<int>(0x10 + I), Size);
    Mems[I] = Allocs[I];
    Sizes[I] = Size;
  }

  ASSERT_SUCCESS(olMemPrefetch(Queue, NumAllocs, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t I = 0; I < NumAllocs; I++) {
    for (size_t J = 0; J < Size; J++)
      ASSERT_EQ(static_cast<uint8_t *>(Allocs[I])[J],
                static_cast<uint8_t>(0x10 + I));
    ASSERT_SUCCESS(olMemFree(Allocs[I]));
  }
}

TEST_P(olMemPrefetchTest, SuccessZeroCount) {
  ASSERT_SUCCESS(olMemPrefetch(Queue, 0, nullptr, nullptr,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));
}

TEST_P(olMemPrefetchTest, SuccessZeroSize) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  const void *Mems[] = {Alloc};
  const size_t Sizes[] = {0};
  ASSERT_SUCCESS(olMemPrefetch(Queue, 1, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, SuccessUnsupportedAllocType) {
  // Prefetching a non-managed allocation is not meaningful, but per the API
  // contract the hint must be silently ignored and the call must still succeed.
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));

  const void *Mems[] = {Alloc};
  const size_t Sizes[] = {Size};
  ASSERT_SUCCESS(olMemPrefetch(Queue, 1, Mems, Sizes,
                               OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, InvalidFlags) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  const void *Mems[] = {Alloc};
  const size_t Sizes[] = {Size};
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olMemPrefetch(Queue, 1, Mems, Sizes, 0xdeadbeef));

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, InvalidNullMems) {
  const size_t Sizes[] = {0};
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemPrefetch(Queue, 1, nullptr, Sizes,
                             OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));
}

TEST_P(olMemPrefetchTest, InvalidNullSizes) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  const void *Mems[] = {Alloc};
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemPrefetch(Queue, 1, Mems, nullptr,
                             OL_MEM_MIGRATION_FLAG_HOST_TO_DEVICE));

  ASSERT_SUCCESS(olMemFree(Alloc));
}
