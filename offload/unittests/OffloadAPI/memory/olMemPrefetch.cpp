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

  ASSERT_SUCCESS(olMemPrefetch(Queue, Alloc, Size,
                               OL_USM_MIGRATION_FLAG_HOST_TO_DEVICE));
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

  // Migrate to the device first, then bring it back.
  ASSERT_SUCCESS(olMemPrefetch(Queue, Alloc, Size,
                               OL_USM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olMemPrefetch(Queue, Alloc, Size,
                               OL_USM_MIGRATION_FLAG_DEVICE_TO_HOST));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(static_cast<uint8_t *>(Alloc)[I], 0x21);

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, SuccessZeroSize) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  ASSERT_SUCCESS(olMemPrefetch(Queue, Alloc, 0,
                               OL_USM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, SuccessUnsupportedAllocType) {
  // Prefetching a non-managed allocation is not meaningful, but per the API
  // contract the hint must be silently ignored and the call must still succeed.
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));

  ASSERT_SUCCESS(olMemPrefetch(Queue, Alloc, Size,
                               OL_USM_MIGRATION_FLAG_HOST_TO_DEVICE));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemPrefetchTest, InvalidFlags) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olMemPrefetch(Queue, Alloc, Size, 0xdeadbeef));

  ASSERT_SUCCESS(olMemFree(Alloc));
}
