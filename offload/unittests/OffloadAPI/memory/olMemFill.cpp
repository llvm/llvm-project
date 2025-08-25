//===------- Offload API tests - olMemFill --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemFillTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFillTest);

TEST_P(olMemFillTest, Success8) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint8_t Pattern = 0x42;
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    uint8_t *AllocPtr = reinterpret_cast<uint8_t *>(Alloc);
    ASSERT_EQ(AllocPtr[i], Pattern);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, Success16) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint16_t Pattern = 0x4242;
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    uint16_t *AllocPtr = reinterpret_cast<uint16_t *>(Alloc);
    ASSERT_EQ(AllocPtr[i], Pattern);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, Success32) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint32_t Pattern = 0xDEADBEEF;
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    uint32_t *AllocPtr = reinterpret_cast<uint32_t *>(Alloc);
    ASSERT_EQ(AllocPtr[i], Pattern);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessLarge) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct PatternT {
    uint64_t A;
    uint64_t B;
  } Pattern{UINT64_MAX, UINT64_MAX};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessLargeByteAligned) {
  constexpr size_t Size = 17 * 64;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct __attribute__((packed)) PatternT {
    uint64_t A;
    uint64_t B;
    uint8_t C;
  } Pattern{UINT64_MAX, UINT64_MAX, 255};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);

  size_t N = Size / sizeof(Pattern);
  for (size_t i = 0; i < N; i++) {
    PatternT *AllocPtr = reinterpret_cast<PatternT *>(Alloc);
    ASSERT_EQ(AllocPtr[i].A, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].B, UINT64_MAX);
    ASSERT_EQ(AllocPtr[i].C, 255);
  }

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, InvalidPatternSize) {
  constexpr size_t Size = 1025;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint16_t Pattern = 0x4242;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);
  olMemFree(Alloc);
}
