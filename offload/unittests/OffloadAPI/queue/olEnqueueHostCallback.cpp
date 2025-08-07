//===------- Offload API tests - olEnqueueHostCallback --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olEnqueueHostCallbackTest : OffloadQueueTest {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olEnqueueHostCallbackTest);

TEST_P(olEnqueueHostCallbackTest, Success) {
  ASSERT_SUCCESS(olEnqueueHostCallback(Queue, [](void *) {}, nullptr));
}

TEST_P(olEnqueueHostCallbackTest, SuccessSequence) {
  uint32_t Buff[16] = {1, 1};

  for (auto BuffPtr = &Buff[2]; BuffPtr != &Buff[16]; BuffPtr++) {
    ASSERT_SUCCESS(olEnqueueHostCallback(
        Queue,
        [](void *BuffPtr) {
          uint32_t *AsU32 = reinterpret_cast<uint32_t *>(BuffPtr);
          AsU32[0] = AsU32[-1] + AsU32[-2];
        },
        BuffPtr));
  }

  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (uint32_t i = 2; i < 16; i++) {
    ASSERT_EQ(Buff[i], Buff[i - 1] + Buff[i - 2]);
  }
}

TEST_P(olEnqueueHostCallbackTest, InvalidNullCallback) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olEnqueueHostCallback(Queue, nullptr, nullptr));
}

TEST_P(olEnqueueHostCallbackTest, InvalidNullQueue) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olEnqueueHostCallback(nullptr, [](void *) {}, nullptr));
}
