//===------- Offload API tests - olGetQueueInfoSize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetQueueInfoSizeTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetQueueInfoSizeTest);

TEST_P(olGetQueueInfoSizeTest, SuccessDevice) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetQueueInfoSize(Queue, OL_QUEUE_INFO_DEVICE, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_handle_t));
}

TEST_P(olGetQueueInfoSizeTest, SuccessEmpty) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetQueueInfoSize(Queue, OL_QUEUE_INFO_EMPTY, &Size));
  ASSERT_EQ(Size, sizeof(bool));
}

TEST_P(olGetQueueInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetQueueInfoSize(nullptr, OL_QUEUE_INFO_DEVICE, &Size));
}

TEST_P(olGetQueueInfoSizeTest, InvalidQueueInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetQueueInfoSize(Queue, OL_QUEUE_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetQueueInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetQueueInfoSize(Queue, OL_QUEUE_INFO_DEVICE, nullptr));
}
