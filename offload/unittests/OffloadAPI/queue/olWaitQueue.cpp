//===------- Offload API tests - olWaitQueue ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olWaitQueueTest = offloadQueueTest;

TEST_F(olWaitQueueTest, SuccessEmptyQueue) {
  ASSERT_SUCCESS(olWaitQueue(Queue));
}
