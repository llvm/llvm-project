//===------- Offload tests - AMDGPU queue user count ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUQueueUserCount.h"
#include <cstdint>
#include <gtest/gtest.h>

using llvm::omp::target::plugin::AMDGPUQueueUserCount;

TEST(AMDGPUQueueUserCount, StartsAtZero) {
  AMDGPUQueueUserCount Users;
  EXPECT_EQ(Users.getUserCount(), 0u);
}

TEST(AMDGPUQueueUserCount, AddAndRemoveTrackIntegerCount) {
  AMDGPUQueueUserCount Users;
  Users.addUser();
  Users.addUser();
  Users.addUser();
  EXPECT_EQ(Users.getUserCount(), 3u);

  Users.removeUser();
  EXPECT_EQ(Users.getUserCount(), 2u);
}

// assignNextQueue picks the least-used busy queue by comparing getUserCount().
// A boolean conversion would collapse every non-zero count to 1, so 5 > 1
// would be false.
TEST(AMDGPUQueueUserCount, BusyQueuesCompareByIntegerCount) {
  AMDGPUQueueUserCount Heavy;
  AMDGPUQueueUserCount Light;
  for (uint32_t I = 0; I < 5; ++I)
    Heavy.addUser();
  Light.addUser();

  EXPECT_EQ(Heavy.getUserCount(), 5u);
  EXPECT_EQ(Light.getUserCount(), 1u);
  EXPECT_GT(Heavy.getUserCount(), Light.getUserCount());
}
