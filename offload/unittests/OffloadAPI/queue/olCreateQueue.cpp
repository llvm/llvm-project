//===------- Offload API tests - olCreateQueue ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCreateQueueTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateQueueTest);

TEST_P(olCreateQueueTest, Success) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  ASSERT_NE(Queue, nullptr);
}

TEST_P(olCreateQueueTest, InvalidNullHandleDevice) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olCreateQueue(nullptr, &Queue));
}

TEST_P(olCreateQueueTest, InvalidNullPointerQueue) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olCreateQueue(Device, nullptr));
}
