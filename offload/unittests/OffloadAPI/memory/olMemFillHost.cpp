//===------- Offload API tests - host olMemFill remainder -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// olMemFill rejects a fill size that is not a multiple of the pattern size.
// This suite skips implicit olInit so it can disable that check and reach the
// host plugin's dataFill with a remainder, which used to write past the
// destination.

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <cstdlib>
#include <gtest/gtest.h>

struct olMemFillHostRemainderTest : ::testing::Test {
  void SetUp() override {
    ASSERT_EQ(setenv("OFFLOAD_DISABLE_VALIDATION", "1", 1), 0);

    ol_init_args_t Args = OL_INIT_ARGS_INIT;
    ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_HOST};
    Args.NumPlatforms = 1;
    Args.Platforms = Backends;
    ASSERT_SUCCESS(olInit(&Args));
    Initialized = true;

    Device = TestEnvironment::getHostDevice();
    if (!Device)
      GTEST_SKIP() << "No host device.";

    ASSERT_SUCCESS(olCreateContext(1, &Device, &Context));
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  }

  void TearDown() override {
    if (Queue)
      olDestroyQueue(Queue);
    if (Context)
      olDestroyContext(Context);
    if (Initialized)
      ASSERT_SUCCESS(olShutDown());
    unsetenv("OFFLOAD_DISABLE_VALIDATION");
  }

  bool Initialized = false;
  ol_device_handle_t Device = nullptr;
  ol_context_handle_t Context = nullptr;
  ol_queue_handle_t Queue = nullptr;
};

TEST_F(olMemFillHostRemainderTest, RemainderDoesNotWritePastFillSize) {
  constexpr size_t AllocSize = 16;
  constexpr size_t FillSize = 5;
  constexpr unsigned char Canary = 0xAA;
  const unsigned char Pattern[4] = {1, 2, 3, 4};

  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, AllocSize, &Alloc));
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, 1, &Canary, AllocSize));
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), Pattern, FillSize));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  auto *Bytes = static_cast<unsigned char *>(Alloc);
  for (size_t I = 0; I < FillSize; ++I)
    ASSERT_EQ(Bytes[I], Pattern[I % sizeof(Pattern)]);
  for (size_t I = FillSize; I < AllocSize; ++I)
    ASSERT_EQ(Bytes[I], Canary) << "wrote past FillSize at byte " << I;

  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemFillHostRemainderTest, PatternLargerThanFillSize) {
  constexpr size_t AllocSize = 16;
  constexpr size_t FillSize = 3;
  constexpr unsigned char Canary = 0xAA;
  const unsigned char Pattern[4] = {1, 2, 3, 4};

  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, AllocSize, &Alloc));
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, 1, &Canary, AllocSize));
  ASSERT_SUCCESS(olMemFill(Queue, Alloc, sizeof(Pattern), Pattern, FillSize));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  auto *Bytes = static_cast<unsigned char *>(Alloc);
  for (size_t I = 0; I < FillSize; ++I)
    ASSERT_EQ(Bytes[I], Pattern[I]);
  for (size_t I = FillSize; I < AllocSize; ++I)
    ASSERT_EQ(Bytes[I], Canary) << "wrote past FillSize at byte " << I;

  ASSERT_SUCCESS(olMemFree(Alloc));
}
