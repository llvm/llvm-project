//===------- Offload API tests - olMemIsAccessible -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

using olMemIsAccessibleTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemIsAccessibleTest);

TEST_P(olMemIsAccessibleTest, SuccessDeviceAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessHostAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(Context, Device, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessInteriorRange) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A range that ends exactly at the end of the allocation is accessible.
  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(
      Context, Device, static_cast<char *>(Ptr) + 512, 512, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessNoContext) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // Without a context the query is answered by the plugin's own pointer
  // tracking rather than by the context's allocation info. Allocations made
  // in a context are not necessarily tracked by the device, so the answer is
  // backend specific; only require that the query itself succeeds.
  bool IsAccessible = false;
  ASSERT_SUCCESS(
      olMemIsAccessible(/*Context=*/nullptr, Device, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleHostMemory) {
  // Plain host memory is not accessible from the device unless the system
  // provides unified shared memory.
  SKIP_KNOWN_FAILURE(AMDGPU("host memory is accessible when XNACK is enabled"));

  std::vector<char> HostBuffer(1024);

  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, HostBuffer.data(),
                                   HostBuffer.size(), &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  // The same pointer without a context must give the same answer.
  IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(/*Context=*/nullptr, Device,
                                   HostBuffer.data(), HostBuffer.size(),
                                   &IsAccessible));
  ASSERT_FALSE(IsAccessible);
}

TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleBeyondAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A range extending past the end of the allocation is not accessible.
  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device, Ptr, 1024 * 1024 * 1024,
                                   &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  // Same when starting from an interior pointer.
  IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(Context, Device,
                                   static_cast<char *>(Ptr) + 512,
                                   1024 * 1024 * 1024, &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessNotAccessibleSizeOverflow) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // A size large enough to overflow `Ptr + Size` must not be reported as
  // accessible.
  bool IsAccessible = true;
  ASSERT_SUCCESS(olMemIsAccessible(
      Context, Device, Ptr, std::numeric_limits<size_t>::max(), &IsAccessible));
  ASSERT_FALSE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, InvalidNullPointer) {
  bool IsAccessible = false;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olMemIsAccessible(Context, Device, nullptr, 1024, &IsAccessible));
}

TEST_P(olMemIsAccessibleTest, InvalidNullOutputPointer) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemIsAccessible(Context, Device, Ptr, 1024, nullptr));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, InvalidNullDevice) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemIsAccessible(Context, nullptr, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}

TEST_P(olMemIsAccessibleTest, InvalidDeviceNotInContext) {
  if (!Host)
    GTEST_SKIP() << "No host device available.";

  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  // The host device does not belong to the context under test.
  bool IsAccessible = false;
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE,
               olMemIsAccessible(Context, Host, Ptr, 1024, &IsAccessible));

  ASSERT_SUCCESS(olMemFree(Context, Ptr));
}
