//===------- Offload API tests - olMemFill --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <array>
#include <gtest/gtest.h>
#include <vector>

struct olMemFillTest : OffloadQueueTest {
  void SetUp() override { RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp()); }

  template <typename PatternTy, PatternTy PatternVal, size_t Size,
            bool Block = false>
  void test_body() {
    ManuallyTriggeredTask Manual;

    // Block/enqueue tests ensure that the test has been enqueued to a queue
    // (rather than being done synchronously if the queue happens to be empty)
    if constexpr (Block) {
      ASSERT_SUCCESS(Manual.enqueue(Queue));
    }

    void *Alloc;
    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

    PatternTy Pattern = PatternVal;
    ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

    if constexpr (Block) {
      ASSERT_SUCCESS(Manual.trigger());
    }
    olSyncQueue(Queue);

    size_t N = Size / sizeof(Pattern);
    for (size_t i = 0; i < N; i++) {
      PatternTy *AllocPtr = reinterpret_cast<PatternTy *>(Alloc);
      ASSERT_EQ(AllocPtr[i], Pattern);
    }

    olMemFree(Alloc);
  }
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFillTest);

TEST_P(olMemFillTest, Success8) { test_body<uint8_t, 0x42, 1024>(); }
TEST_P(olMemFillTest, Success8NotMultiple4) {
  test_body<uint8_t, 0x42, 1023>();
}
TEST_P(olMemFillTest, Success8Enqueue) {
  test_body<uint8_t, 0x42, 1024, true>();
}
TEST_P(olMemFillTest, Success8NotMultiple4Enqueue) {
  test_body<uint8_t, 0x42, 1023, true>();
}

TEST_P(olMemFillTest, Success16) { test_body<uint8_t, 0x42, 1024>(); }
TEST_P(olMemFillTest, Success16NotMultiple4) {
  test_body<uint16_t, 0x4243, 1022>();
}
TEST_P(olMemFillTest, Success16Enqueue) {
  test_body<uint8_t, 0x42, 1024, true>();
}
TEST_P(olMemFillTest, Success16NotMultiple4Enqueue) {
  test_body<uint16_t, 0x4243, 1022, true>();
}

TEST_P(olMemFillTest, Success32) { test_body<uint32_t, 0xDEADBEEF, 1024>(); }
TEST_P(olMemFillTest, Success32Enqueue) {
  test_body<uint32_t, 0xDEADBEEF, 1024, true>();
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

TEST_P(olMemFillTest, SuccessLargeEnqueue) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct PatternT {
    uint64_t A;
    uint64_t B;
  } Pattern{UINT64_MAX, UINT64_MAX};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  Manual.trigger();
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

TEST_P(olMemFillTest, SuccessLargeByteAlignedEnqueue) {
  SKIP_KNOWN_FAILURE(LevelZero{"unsupported feature"});
  constexpr size_t Size = 17 * 64;
  void *Alloc;
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  struct __attribute__((packed)) PatternT {
    uint64_t A;
    uint64_t B;
    uint8_t C;
  } Pattern{UINT64_MAX, UINT64_MAX, 255};

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  Manual.trigger();
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

TEST_P(olMemFillTest, InvalidSizeNotMultipleOfPatternSize) {
  constexpr size_t Size = 1025;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint16_t Pattern = 0x4242;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);
  olMemFree(Alloc);
}

TEST_P(olMemFillTest, InvalidPatternSizeLargerThanFillSize) {
  constexpr size_t Size = 4;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  uint64_t Pattern = 0x4242424242424242;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemFill(Queue, Alloc, sizeof(Pattern), &Pattern, Size));

  olSyncQueue(Queue);
  olMemFree(Alloc);
}

// Even though L0, CUDA and HSA do not support non-power-of-two patterns,
// plugins are currently expected to handle arbitrary pattern sizes.
// The following tests are intended to cover the fallback paths
// for non-power-of-two patterns.
static constexpr std::array<unsigned char, 3> FallbackPattern = {0x11, 0x22,
                                                                 0x33};

TEST_P(olMemFillTest, SuccessNonPow2PatternManaged) {
  constexpr size_t Size = FallbackPattern.size() * 1000;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, FallbackPattern.size(),
                           FallbackPattern.data(), Size));
  olSyncQueue(Queue);

  auto *AllocPtr = reinterpret_cast<unsigned char *>(Alloc);
  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(AllocPtr[I], FallbackPattern[I % FallbackPattern.size()]);

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessNonPow2PatternManagedEnqueue) {
  constexpr size_t Size = FallbackPattern.size() * 1000;
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));

  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, Size, &Alloc));

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, FallbackPattern.size(),
                           FallbackPattern.data(), Size));
  ASSERT_SUCCESS(Manual.trigger());
  olSyncQueue(Queue);

  auto *AllocPtr = reinterpret_cast<unsigned char *>(Alloc);
  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(AllocPtr[I], FallbackPattern[I % FallbackPattern.size()]);

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessNonPow2PatternDevice) {
  constexpr size_t Size = FallbackPattern.size() * 1000;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, FallbackPattern.size(),
                           FallbackPattern.data(), Size));

  std::vector<unsigned char> HostBuf(Size);
  ASSERT_SUCCESS(olMemcpy(Queue, HostBuf.data(), Host, Alloc, Device, Size));
  olSyncQueue(Queue);

  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(HostBuf[I], FallbackPattern[I % FallbackPattern.size()]);

  olMemFree(Alloc);
}

TEST_P(olMemFillTest, SuccessNonPow2PatternDeviceSmall) {
  constexpr size_t Size = FallbackPattern.size() * 2;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));

  ASSERT_SUCCESS(olMemFill(Queue, Alloc, FallbackPattern.size(),
                           FallbackPattern.data(), Size));

  std::vector<unsigned char> HostBuf(Size);
  ASSERT_SUCCESS(olMemcpy(Queue, HostBuf.data(), Host, Alloc, Device, Size));
  olSyncQueue(Queue);

  for (size_t I = 0; I < Size; I++)
    ASSERT_EQ(HostBuf[I], FallbackPattern[I % FallbackPattern.size()]);

  olMemFree(Alloc);
}
