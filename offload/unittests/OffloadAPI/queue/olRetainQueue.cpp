//===------- Offload API tests - olRetainQueue ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olRetainQueueTest = offloadQueueTest;

// TODO: When we can fetch queue info we can check the reference count is
// changing in the expected way. In the meantime just check the entry point
// doesn't blow up.
TEST_F(olRetainQueueTest, Success) { ASSERT_SUCCESS(olRetainQueue(Queue)); }
