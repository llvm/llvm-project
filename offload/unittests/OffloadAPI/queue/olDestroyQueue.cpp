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

TEST_F(olDestroyQueueTest, Success) {
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
}

TEST_F(olDestroyQueueTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyQueue(nullptr));
}
