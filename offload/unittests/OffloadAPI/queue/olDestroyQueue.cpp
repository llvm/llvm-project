//===------- Offload API tests - olDestroyQueue ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyQueueTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyQueueTest);

TEST_P(olDestroyQueueTest, Success) {
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
}

TEST_P(olDestroyQueueTest, SuccessDelayedResolution) {
  ManuallyTriggeredTask Manual;
  ASSERT_SUCCESS(Manual.enqueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;

  ASSERT_SUCCESS(Manual.trigger());
}

TEST_P(olDestroyQueueTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyQueue(nullptr));
}
