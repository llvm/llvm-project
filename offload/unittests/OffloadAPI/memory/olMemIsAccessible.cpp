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

using olMemIsAccessibleTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemIsAccessibleTest);

TEST_P(olMemIsAccessibleTest, SuccessDeviceAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Ptr));
}

TEST_P(olMemIsAccessibleTest, SuccessHostAllocation) {
  void *Ptr = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(Device, 1024, &Ptr));
  ASSERT_NE(Ptr, nullptr);

  bool IsAccessible = false;
  ASSERT_SUCCESS(olMemIsAccessible(Device, Ptr, 1024, &IsAccessible));
  ASSERT_TRUE(IsAccessible);

  ASSERT_SUCCESS(olMemFree(Ptr));
}

TEST_P(olMemIsAccessibleTest, InvalidNullPointer) {
  bool IsAccessible = false;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemIsAccessible(Device, nullptr, 1024, &IsAccessible));
}
