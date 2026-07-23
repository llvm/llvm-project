//===------- Offload API tests - olQueryQueue ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olQueryQueueTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olQueryQueueTest);

TEST_P(olQueryQueueTest, SuccessEmptyAsyncQueue) {
  ASSERT_SUCCESS(olQueryQueue(Queue, nullptr));
}

TEST_P(olQueryQueueTest, SuccessEmptyAsyncQueueCheckResult) {
  bool IsQueueWorkCompleted;
  ASSERT_SUCCESS(olQueryQueue(Queue, &IsQueueWorkCompleted));
  ASSERT_TRUE(IsQueueWorkCompleted);
}
