//===------- Offload API tests - olSyncQueue ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olSyncQueueTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olSyncQueueTest);

TEST_P(olSyncQueueTest, SuccessEmptyQueue) {
  ASSERT_SUCCESS(olSyncQueue(Queue));
}
