//===------- Offload API tests - olCreateQueue ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <cstdint>
#include <gtest/gtest.h>

using olCreateQueueTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateQueueTest);

TEST_P(olCreateQueueTest, Success) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  ASSERT_NE(Queue, nullptr);
  ASSERT_SUCCESS(olDestroyQueue(Queue));
}

TEST_P(olCreateQueueTest, InvalidNullHandleContext) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateQueue(nullptr, Device, &Queue));
}

TEST_P(olCreateQueueTest, InvalidNullHandleDevice) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateQueue(Context, nullptr, &Queue));
}

TEST_P(olCreateQueueTest, InvalidNullPointerQueue) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateQueue(Context, Device, nullptr));
}

TEST_P(olCreateQueueTest, InvalidDeviceNotInContext) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateQueue(Context, Host, &Queue));
}

// AMDGPUQueueTy is defined in the AMDGPU plugin translation unit. These are
// its user-count operations: getUserCount returns the stored integer, and
// assignNextQueue compares those values when choosing the least-used queue.
struct QueueUserCount {
  uint32_t getUserCount() const { return NumUsers; }
  void addUser() { ++NumUsers; }
  void removeUser() { --NumUsers; }

  uint32_t NumUsers = 0;
};

TEST(AMDGPUQueueUserCount, StartsAtZero) {
  QueueUserCount Users;
  EXPECT_EQ(Users.getUserCount(), 0u);
}

TEST(AMDGPUQueueUserCount, AddAndRemoveTrackIntegerCount) {
  QueueUserCount Users;
  Users.addUser();
  Users.addUser();
  Users.addUser();
  EXPECT_EQ(Users.getUserCount(), 3u);

  Users.removeUser();
  EXPECT_EQ(Users.getUserCount(), 2u);
}

// A boolean return would collapse every non-zero count to 1, so 5 > 1 would
// be false and the heavier queue would stay selected.
TEST(AMDGPUQueueUserCount, LeastUsedBusyQueueHasTheSmallerCount) {
  QueueUserCount Queues[2];
  for (uint32_t I = 0; I < 5; ++I)
    Queues[0].addUser();
  Queues[1].addUser();

  uint32_t Index = 0;
  for (uint32_t I = 0; I < 2; ++I) {
    if (Queues[I].getUserCount() == 0) {
      Index = I;
      break;
    }
    if (Queues[Index].getUserCount() > Queues[I].getUserCount())
      Index = I;
  }

  EXPECT_EQ(Queues[0].getUserCount(), 5u);
  EXPECT_EQ(Queues[1].getUserCount(), 1u);
  EXPECT_EQ(Index, 1u);
}
